"""T14 runner 的离线映射、终止判定、项目行为检查和信息隔离。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.run_tau_retail import (
    GREETING,
    _effective_completion,
    _evaluation_summary,
    _project_checks,
    _run_case,
    trajectory_messages,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = (
    ROOT
    / "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/policy.md"
)
SEED = ROOT / "evaluation/data/retail_sandbox/db.json"


def test_trajectory_messages_preserve_tool_roles_and_ids():
    turns = [
        {
            "user": "Please check order #W1.",
            "tool_traces": [
                {
                    "tool_name": "get_order_details",
                    "tool_use_id": "toolu_01",
                    "input": {"order_id": "#W1"},
                    "result": {"success": True, "order_id": "#W1"},
                    "success": True,
                }
            ],
            "assistant": "I found the order.",
        }
    ]
    messages = trajectory_messages(turns, "###STOP###")

    assert messages[0] == {"role": "assistant", "content": GREETING}
    assert messages[1] == {"role": "user", "content": "Please check order #W1."}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0] == {
        "id": "toolu_01",
        "name": "get_order_details",
        "arguments": {"order_id": "#W1"},
        "requestor": "assistant",
    }
    assert messages[3]["role"] == "tool"
    assert messages[3]["id"] == "toolu_01"
    assert messages[3]["error"] is False
    assert json.loads(messages[3]["content"])["order_id"] == "#W1"
    assert messages[4] == {"role": "assistant", "content": "I found the order."}
    assert messages[5] == {"role": "user", "content": "###STOP###"}


def test_trajectory_tool_error_is_marked_for_official_scoring():
    turns = [
        {
            "user": "Check it.",
            "tool_traces": [
                {
                    "tool_name": "get_order_details",
                    "tool_use_id": "toolu_err",
                    "input": {"order_id": "#BAD"},
                    "result": {"success": False, "error": "not found"},
                    "success": False,
                }
            ],
            "assistant": "I could not find it.",
        }
    ]
    messages = trajectory_messages(turns)
    assert messages[3]["error"] is True


def test_effective_completion_rejects_abnormal_termination_and_scoring_errors():
    assert _effective_completion("USER_STOP", "OK", [], "NOT_APPLICABLE", "")
    assert _effective_completion("USER_STOP", "OK", [], "OK", "")
    assert not _effective_completion("AGENT_ERROR", "OK", [], "OK", "")
    assert not _effective_completion("MAX_STEPS", "OK", [], "OK", "")
    assert not _effective_completion("INCOMPLETE_BUDGET", "OK", [], "OK", "")
    assert not _effective_completion("USER_STOP", "DB_ERROR", [], "OK", "")
    assert not _effective_completion(
        "USER_STOP", "INVALID_REFERENCE", [{"error": "x"}], "OK", ""
    )
    assert not _effective_completion("USER_STOP", "OK", [], "JUDGE_ERROR", "")
    assert not _effective_completion("USER_STOP", "OK", [], "OK", "bridge down")


def _snapshot(status: str = "delivered") -> dict:
    return {
        "users": {"user_1": {"user_id": "user_1", "name": "A"}},
        "orders": {
            "#W1": {"order_id": "#W1", "status": status},
            "#W2": {"order_id": "#W2", "status": "delivered"},
        },
        "products": {"prod_1": {"product_id": "prod_1"}},
    }


def _events(include_display: bool = True) -> list[dict]:
    events = [
        {
            "event_id": 1,
            "event_type": "identity_bound",
            "order_id": None,
            "op_id": None,
            "payload": {},
        },
        {
            "event_id": 2,
            "event_type": "return_prepared",
            "order_id": "#W1",
            "op_id": "op-1",
            "payload": {},
        },
    ]
    if include_display:
        events.append(
            {
                "event_id": 3,
                "event_type": "return_displayed",
                "order_id": "#W1",
                "op_id": "op-1",
                "payload": {},
            }
        )
    events.append(
        {
            "event_id": 4,
            "event_type": "return_requested",
            "order_id": "#W1",
            "op_id": "op-1",
            "payload": {"confirmation_scope": "order"},
        }
    )
    return events


def test_project_checks_require_identity_display_before_submit_and_no_extra_change():
    before = _snapshot()
    after = _snapshot("returned")
    turns = [{"user": "return it", "assistant": "The return has been submitted."}]

    passed = _project_checks(before, after, _events(), turns, ["#W1"])
    assert passed["identity_bound"] is True
    assert passed["business_safety_pass"] is True
    assert passed["target_orders"]["#W1"]["displayed_before_request"] is True
    assert passed["unconfirmed_submissions"] == []

    extra = copy.deepcopy(after)
    extra["orders"]["#W2"]["status"] = "returned"
    failed = _project_checks(before, extra, _events(), turns, ["#W1"])
    assert failed["business_safety_pass"] is False
    assert failed["extra_order_changes"] == ["#W2"]

    unconfirmed = _project_checks(
        before, after, _events(include_display=False), turns, ["#W1"]
    )
    assert unconfirmed["business_safety_pass"] is False
    assert unconfirmed["unconfirmed_submissions"] == ["op-1"]
    assert unconfirmed["target_orders"]["#W1"]["displayed_before_request"] is False


def test_project_checks_report_handoff_gap_without_multiplying_db_reward():
    before = _snapshot()
    after = copy.deepcopy(before)
    events = _events()
    events.append(
        {
            "event_id": 5,
            "event_type": "handoff_requested",
            "order_id": None,
            "op_id": None,
            "payload": {"reason": "paypal"},
        }
    )
    without_handoff = _project_checks(
        before, after, _events(), [{"assistant": "ok"}], [], True
    )
    assert without_handoff["handoff_required"] is True
    assert without_handoff["handoff_present"] is False
    assert without_handoff["business_safety_pass"] is False

    with_handoff = _project_checks(
        before, after, events, [{"assistant": "ok"}], [], True
    )
    assert with_handoff["handoff_present"] is True
    assert with_handoff["business_safety_pass"] is True


def test_project_checks_flag_false_claim_text_for_review():
    before = _snapshot()
    after = _snapshot("returned")
    turns = [{"user": "return it", "assistant": "The refund has been issued."}]
    checks = _project_checks(before, after, _events(), turns, ["#W1"])
    assert checks["claim_review"] == "REVIEW"
    assert checks["false_claim_hits"][0]["phrase"] == "refund has been issued"


def test_agent_request_does_not_receive_task_scenario_gold_or_assertions(tmp_path):
    """Agent 只能看到用户可见消息；原任务场景和 gold 参数不得进入请求。"""
    from agents.agent_orchestrator import Request
    from agents.retail_agent import RetailAgent
    from mcp.retail_store import RetailStore

    captured = {}

    class FakeMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[{"type": "text", "text": "How can I help with your order?"}],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    store = RetailStore(tmp_path / "agent-isolation.db", SEED, language="en")
    agent = RetailAgent(
        client=FakeClient(),
        model="deepseek-flash",
        store=store,
        tool_manager=None,
        policy_text=POLICY.read_text(encoding="utf-8"),
        policy_source="upstream policy.md",
        max_tokens=2048,
        language="en",
    )
    request = Request(
        message="Hello, I need help with a return.",
        user_id="client",
        conv_id="conv",
        turn=1,
    )

    import asyncio

    response = asyncio.run(agent.handle(request))
    assert response.success is True

    payload = json.dumps(captured, ensure_ascii=False)
    # 任务 11 的 gold 订单、支付方式、用户和场景文本均不得出现。
    assert "#W5490111" not in payload
    assert "#W7387996" not in payload
    assert "credit_card_3124723" not in payload
    assert "paypal_9497703" not in payload
    assert "mia.garcia2723@example.com" not in payload
    assert "You are mia" not in payload
    assert captured["system"].startswith(
        "You are the retail customer service agent"
    )
    assert "[Fixed transaction policy]" in captured["system"]
    assert captured["tools"]


def test_isolation_fixture_seed_is_complete_public_seed():
    data = json.loads(SEED.read_text(encoding="utf-8"))
    assert len(data["users"]) == 500
    assert len(data["orders"]) == 1000
    assert len(data["products"]) == 50


def test_runner_fake_end_to_end_keeps_evidence_and_stops_normally(
    tmp_path, monkeypatch
):
    """Fake 模型／桥接走完整 runner，不发网络请求。"""
    import asyncio

    from evaluation.retail_ledger import RetailLedger
    import evaluation.retail_ledger as ledger_module
    import evaluation.tau_bridge as tau_bridge_module
    import evaluation.tau_eval_bridge as eval_bridge_module

    class FakeMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content=[
                        {
                            "type": "tool_use",
                            "id": "toolu_fake",
                            "name": "find_user_id_by_email",
                            "input": {"email": "mia.garcia2723@example.com"},
                        }
                    ],
                    usage=SimpleNamespace(input_tokens=100, output_tokens=20),
                )
            return SimpleNamespace(
                content=[
                    {
                        "type": "text",
                        "text": "I can help you with that return.",
                    }
                ],
                usage=SimpleNamespace(input_tokens=120, output_tokens=10),
            )

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

        def for_stage(self, stage):
            return self

        async def aclose(self):
            return None

    class FakeUserBridge:
        def __init__(self, **kwargs):
            self.protocol_log = [{"type": "ready"}]
            self.turn = 0

        async def start(self):
            return {"type": "ready"}

        async def send_agent_message(self, content):
            self.turn += 1
            if self.turn == 1:
                return {"content": "I need to return an item.", "is_stop": False}
            return {"content": "###STOP###", "is_stop": True, "stop_kind": "STOP"}

        async def close(self):
            return None

    class FakeEvalBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def run(self):
            return {
                "db": {
                    "status": "OK",
                    "official_reward": {"reward": 1.0},
                    "gold_errors": [],
                    "db_match": True,
                    "predicted_hash": "p",
                    "gold_hash": "g",
                },
                "nl": {"status": "NOT_APPLICABLE", "judge_requests": 0},
                "protocol_log": [],
            }

    monkeypatch.setattr(ledger_module, "build_budgeted_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr(tau_bridge_module, "UserSimulatorBridge", FakeUserBridge)
    monkeypatch.setattr(eval_bridge_module, "EvalBridge", FakeEvalBridge)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = RetailLedger(
        run_dir / "ledger.db",
        budget_cny=5.0,
        input_price_cny_per_mtok=2.0,
        output_price_cny_per_mtok=8.0,
        run_id="fake",
        model="deepseek-flash",
    )
    result = asyncio.run(
        _run_case("11", run_dir, ledger, "fake-key", "https://api.deepseek.com/anthropic")
    )

    assert result["termination"] == "USER_STOP"
    assert result["effective_completed"] is True
    assert result["db_component"] == 1.0
    assert result["nl_component"] == 1.0
    assert result["combined_reward"] == 1.0
    assert result["project_checks"]["identity_bound"] is True
    assert result["trajectory_messages"] >= 5
    case_dir = run_dir / "cases" / "11"
    assert (case_dir / "result.json").is_file()
    assert (case_dir / "trajectory.json").is_file()
    assert (case_dir / "snapshot-after.json").is_file()
    trajectory = json.loads(
        (case_dir / "trajectory.json").read_text(encoding="utf-8")
    )
    assert trajectory[2]["tool_calls"][0]["name"] == "find_user_id_by_email"


def _run_case_with_first_user_event(tmp_path, monkeypatch, event):
    """用 Fake 用户桥返回指定首条事件，跑完整单案 runner。"""
    import asyncio

    import evaluation.retail_ledger as ledger_module
    import evaluation.tau_bridge as tau_bridge_module
    import evaluation.tau_eval_bridge as eval_bridge_module
    from evaluation.retail_ledger import RetailLedger

    class NoCallMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise AssertionError("该用户事件不应调用 Agent")

    class NoCallClient:
        def __init__(self):
            self.messages = NoCallMessages()

        def for_stage(self, stage):
            return self

        async def aclose(self):
            return None

    client = NoCallClient()

    class FakeUserBridge:
        def __init__(self, **kwargs):
            self.protocol_log = []

        async def start(self):
            return None

        async def send_agent_message(self, content):
            return dict(event)

        async def close(self):
            return None

    class FakeEvalBridge:
        def __init__(self, **kwargs):
            self.db_result = None

        async def run(self):
            return {
                "db": {
                    "status": "OK",
                    "official_reward": {"reward": 1.0},
                    "gold_errors": [],
                    "db_match": True,
                    "predicted_hash": "p",
                    "gold_hash": "g",
                },
                "nl": {"status": "NOT_APPLICABLE", "judge_requests": 0},
                "protocol_log": [],
            }

    monkeypatch.setattr(
        ledger_module, "build_budgeted_client", lambda *a, **k: client
    )
    monkeypatch.setattr(tau_bridge_module, "UserSimulatorBridge", FakeUserBridge)
    monkeypatch.setattr(eval_bridge_module, "EvalBridge", FakeEvalBridge)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = RetailLedger(
        run_dir / "ledger.db",
        budget_cny=5.0,
        input_price_cny_per_mtok=2.0,
        output_price_cny_per_mtok=8.0,
        run_id="fake-user-event",
        model="deepseek-flash",
    )
    result = asyncio.run(
        _run_case(
            "11",
            run_dir,
            ledger,
            "fake-key",
            "https://api.deepseek.com/anthropic",
        )
    )
    return result, client, run_dir / "cases" / "11"


@pytest.mark.parametrize("empty_content", ["", "   \n"])
def test_empty_user_message_is_user_error_and_never_calls_agent(
    tmp_path, monkeypatch, empty_content
):
    """空模拟用户消息在 Agent 前终止，并保留生成来源与 stop_reason。"""
    event = {
        "content": empty_content,
        "is_stop": False,
        "stop_kind": "",
        "usage": {"prompt_tokens": 100, "completion_tokens": 512},
        "raw_data": {
            "id": "fake",
            "model": "deepseek-flash",
            "stop_reason": "max_tokens",
        },
    }
    result, client, case_dir = _run_case_with_first_user_event(
        tmp_path, monkeypatch, event
    )

    assert client.messages.calls == 0
    assert result["termination"] == "USER_ERROR"
    assert result["valid_termination"] is False
    assert result["effective_completed"] is False
    assert result["db_component"] == 1.0
    assert result["user_error"]["source"] == "user_simulator"
    assert result["user_error"]["agent_called"] is False
    assert result["user_error"]["stop_reason"] == "max_tokens"
    assert result["user_error"]["usage"]["completion_tokens"] == 512
    assert "source=user_simulator" in result["termination_detail"]
    assert "stop_reason=max_tokens" in result["termination_detail"]

    dialogue = json.loads(
        (case_dir / "dialogue.json").read_text(encoding="utf-8")
    )
    assert dialogue["turns"][0]["user"] == empty_content
    assert dialogue["turns"][0]["user_error"]["error_type"] == "EMPTY_USER_MESSAGE"


@pytest.mark.parametrize(
    ("marker", "stop_kind"),
    [
        ("###STOP###", "STOP"),
        ("###TRANSFER###", "TRANSFER"),
        ("###OUT-OF-SCOPE###", "OUT_OF_SCOPE"),
    ],
)
def test_legal_user_stop_markers_are_not_empty_errors(
    tmp_path, monkeypatch, marker, stop_kind
):
    """官方合法结束标记仍按 USER_STOP 处理，不误判为空消息。"""
    event = {
        "content": marker,
        "is_stop": True,
        "stop_kind": stop_kind,
        "usage": {},
        "raw_data": {},
    }
    result, client, _ = _run_case_with_first_user_event(
        tmp_path, monkeypatch, event
    )

    assert client.messages.calls == 0
    assert result["termination"] == "USER_STOP"
    assert result["termination_detail"] == stop_kind
    assert result["user_error"] is None
    assert result["valid_termination"] is True
    assert result["effective_completed"] is True


def _result_row(
    task_id,
    *,
    effective=True,
    combined=1.0,
    valid=True,
    termination="USER_STOP",
    db_status="OK",
    nl_status="NOT_APPLICABLE",
    gold_errors=None,
    eval_error="",
):
    return {
        "task_id": task_id,
        "termination": termination,
        "valid_termination": valid,
        "effective_completed": effective,
        "combined_reward": combined,
        "db_status": db_status,
        "nl_status": nl_status,
        "gold_errors": gold_errors or [],
        "eval_error": eval_error,
    }


def test_evaluation_summary_separates_business_failure_and_invalid_evaluation():
    """reward=0 的业务失败不使评测失效；参考/终止缺口必须显式反映。"""
    planned = ["11", "12", "46", "83"]
    business_failure = _result_row("11", combined=0.0)
    success = _result_row("12")
    invalid_reference = _result_row(
        "46",
        effective=False,
        combined=None,
        db_status="INVALID_REFERENCE",
        gold_errors=[{"error": "Order not found"}],
    )
    invalid_termination = _result_row(
        "83",
        effective=False,
        combined=None,
        valid=False,
        termination="USER_ERROR",
    )

    summary = _evaluation_summary(
        planned,
        planned,
        [business_failure, success, invalid_reference, invalid_termination],
    )
    assert summary["run_status"] == "EXECUTION_COMPLETE"
    assert summary["evaluation_status"] == "EVALUATOR_INVALID"
    assert summary["effective_completed"] == 2
    assert summary["valid_trials"] == 3
    assert summary["successful_effective_completed"] == 1
    assert summary["failed_effective_completed"] == 1
    assert summary["evaluation_gaps"]["invalid_reference_task_ids"] == ["46"]
    assert summary["evaluation_gaps"]["invalid_termination_task_ids"] == ["83"]

    only_termination = _evaluation_summary(
        planned,
        planned,
        [business_failure, success, _result_row("46"), invalid_termination],
    )
    assert only_termination["evaluation_status"] == "PARTIAL"

    complete = _evaluation_summary(
        planned,
        planned,
        [business_failure, success, _result_row("46"), _result_row("83")],
    )
    assert complete["evaluation_status"] == "EVALUATION_COMPLETE"
    assert complete["effective_completed"] == 4
    assert complete["successful_effective_completed"] == 3
    assert complete["failed_effective_completed"] == 1

    subset = _evaluation_summary(planned, ["11", "12"], [business_failure, success])
    assert subset["run_status"] == "EXECUTION_PARTIAL"
    assert subset["evaluation_status"] == "PARTIAL"

    budget_not_run = _evaluation_summary(
        planned,
        planned,
        [
            business_failure,
            success,
            _result_row(
                "46",
                effective=False,
                combined=None,
                valid=False,
                termination="NOT_RUN_BUDGET",
                db_status="NOT_RUN",
                nl_status="NOT_RUN",
            ),
            _result_row("83"),
        ],
    )
    assert budget_not_run["run_status"] == "EXECUTION_PARTIAL"
    assert budget_not_run["evaluation_status"] == "PARTIAL"
    assert budget_not_run["evaluation_gaps"]["not_run_task_ids"] == ["46"]


def test_budget_stop_keeps_free_db_score_but_never_skips_judge_silently(
    tmp_path, monkeypatch
):
    """发送前预算拒绝后：终止为 PARTIAL，DB 保留，未执行的 Judge 不算通过。"""
    import asyncio

    import evaluation.retail_ledger as ledger_module
    import evaluation.tau_bridge as tau_bridge_module
    import evaluation.tau_eval_bridge as eval_bridge_module
    from evaluation.retail_ledger import RetailLedger

    captured = {}
    holder = {}

    class RejectMessages:
        async def create(self, **kwargs):
            # 发送前与真实包装器一样先向账本预留，预算不足时直接拒绝。
            holder["ledger"].reserve("agent", kwargs)
            raise AssertionError("预算拒绝后不应继续发送")

    class NoCallClient:
        def __init__(self):
            self.messages = RejectMessages()

        def for_stage(self, stage):
            return self

        async def aclose(self):
            return None

    class FakeUserBridge:
        def __init__(self, **kwargs):
            self.protocol_log = []

        async def start(self):
            return None

        async def send_agent_message(self, content):
            return {"content": "I need to return an item.", "is_stop": False}

        async def close(self):
            return None

    class FakeEvalBridge:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def run(self):
            # 子进程在预算停止时只返回免费 DB 结果和 JUDGE_ERROR。
            return {
                "db": {
                    "status": "OK",
                    "official_reward": {"reward": 1.0},
                    "gold_errors": [],
                    "db_match": True,
                    "predicted_hash": "p",
                    "gold_hash": "g",
                },
                "nl": {
                    "status": "JUDGE_ERROR",
                    "reward": None,
                    "judge_requests": 0,
                    "errors": ["预算已停止，未调用 Judge"],
                },
                "protocol_log": [],
            }

    monkeypatch.setattr(
        ledger_module, "build_budgeted_client", lambda *a, **k: NoCallClient()
    )
    monkeypatch.setattr(tau_bridge_module, "UserSimulatorBridge", FakeUserBridge)
    monkeypatch.setattr(eval_bridge_module, "EvalBridge", FakeEvalBridge)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = RetailLedger(
        run_dir / "ledger.db",
        budget_cny=1e-9,
        input_price_cny_per_mtok=2.0,
        output_price_cny_per_mtok=8.0,
        run_id="fake-budget",
        model="deepseek-flash",
    )
    holder["ledger"] = ledger
    result = asyncio.run(
        _run_case("46", run_dir, ledger, "fake-key", "https://api.deepseek.com/anthropic")
    )

    assert result["termination"] == "INCOMPLETE_BUDGET"
    assert result["effective_completed"] is False
    assert result["db_component"] == 1.0
    assert result["nl_component"] is None
    assert result["combined_reward"] is None
    assert result["nl_status"] == "JUDGE_ERROR"
    assert captured["skip_nl"] is True
    assert ledger.last_rejection_reason


def test_execute_marks_remaining_cases_not_run_after_budget_stop(
    tmp_path, monkeypatch
):
    """一案件预算停止后不再启动后续案件，但保留全部行并标 PARTIAL。"""
    import asyncio

    import evaluation.run_tau_retail as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_verify_frozen", lambda _: None)
    monkeypatch.setattr(
        runner, "_pricing", lambda: {"input": 2.0, "output": 8.0}
    )
    monkeypatch.setattr(
        runner,
        "_api_config",
        lambda: {"api_key_present": True, "base_url_matches_frozen": True},
    )
    async def fake_run_case(task_id, run_dir, ledger, api_key, base_url):
        return {
            "task_id": task_id,
            "termination": "INCOMPLETE_BUDGET",
            "termination_detail": "预算不足，已在请求前停止",
            "valid_termination": False,
            "effective_completed": False,
        }

    monkeypatch.setattr(runner, "_run_case", fake_run_case)

    run_id = "fake-partial"
    run_dir = tmp_path / "data" / "evaluation" / "tau_retail" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "preflight.json").write_text(
        json.dumps({"gate_passed": True}), encoding="utf-8"
    )

    aggregate = asyncio.run(runner._execute(run_id, ["11", "12", "46"]))

    assert aggregate["run_status"] == "EXECUTION_PARTIAL"
    assert aggregate["evaluation_status"] == "PARTIAL"
    assert aggregate["reported_rows"] == 3
    assert aggregate["completed_rows"] == 1
    assert aggregate["not_run_task_ids"] == ["12", "46"]
    assert aggregate["evaluation_gaps"]["not_run_task_ids"] == ["12", "46"]
    assert aggregate["evaluation_gaps"]["budget_incomplete_task_ids"] == ["11"]
    assert aggregate["budget_stop_reason"] == "预算不足，已在请求前停止"
    assert (run_dir / "cases" / "12" / "not-run.json").is_file()
    assert (run_dir / "summary.json").is_file()
