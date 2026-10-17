"""T8 人工验收脚本的离线回归。

覆盖 CLI 数据清理边界、并发预算预留、失败结算和统一通过判定；
全部使用内存 Fake，不连接 Redis／ChromaDB，也不调用真实模型。
"""
from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import types
from typing import Any, Dict, Iterable, List

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "t8_check_real_model", ROOT / "tests" / "check_real_model.py"
)
assert _SPEC and _SPEC.loader
harness = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(harness)


@pytest.fixture
def configured(tmp_path: pathlib.Path):
    args = harness._parse_args(
        [
            "--budget-cny",
            "0.1",
            "--run-id",
            "t8_offline_test",
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    # 测试不启动服务、不改全局环境变量，只初始化记录器与用例日志。
    harness.configure(args, apply_env=False)
    return harness


def _response(text: str = "ok", stop_reason: str = "end_turn") -> Any:
    return types.SimpleNamespace(
        usage=types.SimpleNamespace(
            input_tokens=10,
            output_tokens=10,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        content=[types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        model="fake-model",
        id="fake-response",
    )


def _budget_kwargs() -> Dict[str, Any]:
    return {
        "model": "fake-model",
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "并发预算测试" * 300}],
    }


class _SlowMessages:
    def __init__(self) -> None:
        self.attempts = 0

    async def create(self, **kwargs: Any) -> Any:
        self.attempts += 1
        await asyncio.sleep(0.05)
        return _response()


def test_concurrent_requests_reserve_budget(configured) -> None:
    real = _SlowMessages()
    messages = configured.RecordingMessages(real)
    kwargs = _budget_kwargs()
    each_worst = configured.Recorder.estimate_worst_cny(kwargs)
    assert each_worst * 2 > configured.RECORDER.budget_cny

    async def scenario() -> BaseException | None:
        first = asyncio.create_task(messages.create(**kwargs))
        await asyncio.sleep(0.01)
        error: BaseException | None = None
        try:
            await messages.create(**kwargs)
        except BaseException as exc:  # BudgetExceeded 继承 BaseException
            error = exc
        await first
        return error

    error = asyncio.run(scenario())
    assert isinstance(error, configured.BudgetExceeded)
    # 第二个请求没有越过预算守卫，因而没有真正发出。
    assert real.attempts == 1
    assert configured.RECORDER.reserved_cny == 0.0
    assert configured.RECORDER.spent_cny < configured.RECORDER.budget_cny


class _FailingMessages:
    async def create(self, **kwargs: Any) -> Any:
        raise RuntimeError("controlled failure")


def test_failed_request_releases_reservation_and_charges_worst_case(
    configured,
) -> None:
    messages = configured.RecordingMessages(_FailingMessages())
    kwargs = _budget_kwargs()
    worst = configured.Recorder.estimate_worst_cny(kwargs)

    with pytest.raises(RuntimeError):
        asyncio.run(messages.create(**kwargs))

    recorder = configured.RECORDER
    assert recorder.reserved_cny == 0.0
    assert recorder.spent_cny == pytest.approx(worst)
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["failed"] is True
    assert recorder.calls[0]["cost_cny_conservative"] == round(worst, 6)


def test_client_kwargs_disable_sdk_retries() -> None:
    constrained = harness._constrained_client_kwargs({"api_key": "test"})
    assert constrained["max_retries"] == 0
    assert constrained["api_key"] == "test"


class _FakeRedis:
    def __init__(self, keys: Iterable[str]):
        self.data = {key: "value" for key in keys}
        self.closed = False

    def scan_iter(self, match: str) -> List[str]:
        prefix = match.rstrip("*")
        return [key for key in self.data if key.startswith(prefix)]

    def delete(self, *keys: str) -> None:
        for key in keys:
            self.data.pop(key, None)

    def close(self) -> None:
        self.closed = True


class _FakeCollection:
    def __init__(
        self,
        ids: Iterable[str] = (),
        records: Iterable[Dict[str, Any]] = (),
    ):
        self.ids = set(ids)
        self.records = [dict(record) for record in records]
        self.deleted_ids: List[str] = []
        self.deleted_where: List[Dict[str, Any]] = []

    def delete(
        self,
        where: Dict[str, Any] | None = None,
        ids: List[str] | None = None,
    ) -> None:
        if ids:
            self.deleted_ids.extend(ids)
            for item in ids:
                self.ids.discard(item)
        if where:
            self.deleted_where.append(where)
            self.records = [
                record for record in self.records if not _matches(record, where)
            ]


class _FakeChroma:
    def __init__(self, collections: Dict[str, _FakeCollection]):
        self.collections = collections

    def get_collection(
        self, name: str, embedding_function: Any = None
    ) -> _FakeCollection:
        return self.collections[name]


def _matches(record: Dict[str, Any], where: Dict[str, Any]) -> bool:
    if "$and" in where:
        return all(_matches(record, clause) for clause in where["$and"])
    return all(record.get(key) == value for key, value in where.items())


def test_cleanup_preserves_shared_cli_profile_and_old_sessions(configured) -> None:
    new_conv = "isolated-new-conversation"
    old_conv = "old-conversation"
    redis = _FakeRedis(
        [
            f"wm:cli_user:{new_conv}",
            f"summary:cli_user:{new_conv}",
            f"wm:cli_user:{old_conv}",
            f"summary:cli_user:{old_conv}",
            "wm:cli_user:http-session",
        ]
    )
    episodic = _FakeCollection(
        records=[
            {"user_id": "cli_user", "conv_id": new_conv},
            {"user_id": "cli_user", "conv_id": old_conv},
            {"user_id": "other_user", "conv_id": new_conv},
        ]
    )
    profiles = _FakeCollection(
        ids=["user_profile:cli_user", "user_profile:other_user"]
    )
    chroma = _FakeChroma(
        {"episodic": episodic, "user_profile": profiles}
    )

    harness._cleanup_user(
        "cli_user",
        new_conv,
        delete_profile=False,
        redis_client=redis,
        chroma_client=chroma,
    )

    assert f"wm:cli_user:{new_conv}" not in redis.data
    assert f"summary:cli_user:{new_conv}" not in redis.data
    assert f"wm:cli_user:{old_conv}" in redis.data
    assert f"summary:cli_user:{old_conv}" in redis.data
    assert "wm:cli_user:http-session" in redis.data
    assert {"user_id": "cli_user", "conv_id": new_conv} not in episodic.records
    assert {"user_id": "cli_user", "conv_id": old_conv} in episodic.records
    assert {"user_id": "other_user", "conv_id": new_conv} in episodic.records
    assert profiles.ids == {"user_profile:cli_user", "user_profile:other_user"}
    assert profiles.deleted_ids == []


def test_cleanup_deletes_run_owned_profile_when_requested(configured) -> None:
    profiles = _FakeCollection(ids=["user_profile:run_user", "user_profile:keep"])
    chroma = _FakeChroma(
        {"episodic": _FakeCollection(), "user_profile": profiles}
    )

    harness._cleanup_user(
        "run_user",
        "conversation",
        delete_profile=True,
        redis_client=_FakeRedis([]),
        chroma_client=chroma,
    )

    assert profiles.deleted_ids == ["user_profile:run_user"]
    assert profiles.ids == {"user_profile:keep"}


def _call(
    stage: str,
    stop_reason: str,
    block_types: List[str],
    text_preview: str = "",
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "stop_reason": stop_reason,
        "content_block_types": block_types,
        "text_preview": text_preview,
        "input_tokens": 10,
        "output_tokens": 10,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost_cny_conservative": 0.0001,
    }


def _valid_calls() -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for stage in harness.REQUIRED_MODEL_STAGES:
        if stage == "agent":
            calls.append(_call(stage, "tool_use", ["tool_use"]))
            calls.append(_call(stage, "end_turn", ["thinking", "text"], "完成"))
        else:
            calls.append(
                _call(stage, "end_turn", ["thinking", "text"], "有效输出")
            )
    return calls


def test_case_status_uses_required_assertions_only(configured, tmp_path) -> None:
    log = configured.CaseLog(tmp_path / "cases.jsonl")
    passed = log.add(
        case="V2-billing",
        assertions={"http_200": True, "invalid_arg_retry": False},
        required=["http_200"],
    )
    failed = log.add(
        case="V6-multi-agent-composer",
        assertions={"http_200": True, "mentions_401": False},
        required=["http_200", "mentions_401"],
    )

    assert passed["status"] == "pass"
    assert passed["informational_failed"] == ["invalid_arg_retry"]
    assert failed["status"] == "fail"
    assert failed["required_failed"] == ["mentions_401"]


@pytest.mark.parametrize(
    ("case_name", "missing_key"),
    [
        ("V6-multi-agent-composer", "mentions_401"),
        ("V4-context-and-profile", "profile_has_preference"),
        ("V7-handoff-and-refund-boundary", "handoff_no_ticket_claim"),
    ],
)
def test_missing_required_assertion_fails_run(
    configured, tmp_path, case_name: str, missing_key: str
) -> None:
    log = configured.CaseLog(tmp_path / "cases.jsonl")
    case = log.add(
        case=case_name,
        assertions={"http_200": True, missing_key: False},
        required=["http_200", missing_key],
    )

    result = configured.validate_run([case], _valid_calls())

    assert result["ok"] is False
    assert any(
        reason["code"] == "case_required_failed"
        and reason["case"] == case_name
        and reason["required_failed"] == [missing_key]
        for reason in result["reasons"]
    )


def test_structured_truncation_fails_run(configured) -> None:
    calls = _valid_calls()
    for call in calls:
        if call["stage"] == "rerank":
            call["stop_reason"] = "max_tokens"

    result = configured.validate_run([], calls)

    assert result["ok"] is False
    assert any(
        reason["code"] == "truncated_response"
        and reason["stage"] == "rerank"
        for reason in result["reasons"]
    )


def test_composer_degradation_fails_run(configured) -> None:
    result = configured.validate_run(
        [],
        _valid_calls(),
        [{"code": "composer_fallback", "message": "Response Composer 失败"}],
    )

    assert result["ok"] is False
    assert any(
        reason["code"] == "degradation"
        and reason["detail"].get("code") == "composer_fallback"
        and reason["detail"].get("message") == "Response Composer 失败"
        for reason in result["reasons"]
    )


def test_legal_roundtrip_and_shutdown_pass(configured, tmp_path) -> None:
    log = configured.CaseLog(tmp_path / "cases.jsonl")
    case = log.add(
        case="V11-shutdown",
        assertions={
            "thread_alive": False,
            "thread_stopped": True,
            "orchestrator_cleared": True,
            "memory_cleared": True,
            "monitor_cleared": True,
        },
        required=[
            "thread_stopped",
            "orchestrator_cleared",
            "memory_cleared",
            "monitor_cleared",
        ],
    )

    result = configured.validate_run([case], _valid_calls())

    assert result["ok"] is True
    assert result["reasons"] == []


def test_main_returns_zero_for_valid_run(configured, monkeypatch) -> None:
    configured.CASES.add(
        case="V2-general",
        assertions={"http_200": True},
        required=["http_200"],
    )
    configured.RECORDER.calls = _valid_calls()

    async def fake_run() -> Dict[str, Any]:
        return {"ready": True, "cases": configured.CASES.items}

    monkeypatch.setattr(configured, "run_acceptance", fake_run)
    assert configured.main() == 0


def test_main_returns_nonzero_for_injected_failure(configured, monkeypatch) -> None:
    configured.CASES.add(
        case="V6-multi-agent-composer",
        assertions={"http_200": True, "mentions_401": False},
        required=["http_200", "mentions_401"],
    )
    configured.RECORDER.calls = _valid_calls()

    async def fake_run() -> Dict[str, Any]:
        return {"ready": True, "cases": configured.CASES.items}

    monkeypatch.setattr(configured, "run_acceptance", fake_run)
    assert configured.main() == 2
