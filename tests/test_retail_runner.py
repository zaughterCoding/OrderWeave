import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from core import intent_recognizer as intent_module
from evaluation import run_retail_baseline as runner
from evaluation.retail_budget import BudgetGuard, request_input_upper_tokens
from mcp import tool_manager as tool_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_offline_check_runs_without_creating_model_or_storage_clients(tmp_path):
    run_dir = tmp_path / "offline"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.run_retail_baseline",
            "--mode",
            "offline-check",
            "--run-dir",
            str(run_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    metrics = _read_json(run_dir / "metrics.json")
    run = _read_json(run_dir / "run.json")
    usage = [
        json.loads(line)
        for line in (run_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert metrics["status"] == runner.STATUS_PENDING_BUDGET
    assert metrics["offline_checks"]["status"] == "PASS"
    assert metrics["offline_checks"]["model_client_modules_loaded"] == []
    assert metrics["offline_checks"]["sendable_client_created"] is False
    assert metrics["intent"]["status"] == "NOT_RUN"
    assert metrics["retrieval"]["V"]["status"] == "NOT_RUN"
    assert run["data"]["unchanged"] is True
    assert run["data"]["validation_before"]["status"] == "PASS"
    assert usage[-1]["requests"] == 0
    assert usage[-1]["cost_cny"] == 0.0
    assert (run_dir / "intent_predictions.jsonl").read_text(encoding="utf-8") == ""
    assert (run_dir / "retrieval_predictions.jsonl").read_text(encoding="utf-8") == ""


def test_live_refuses_without_explicit_budget_before_any_request(tmp_path):
    run_dir = tmp_path / "live-refused"
    exit_code = runner.main(
        [
            "--mode",
            "live",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 2
    run = _read_json(run_dir / "run.json")
    usage = [
        json.loads(line)
        for line in (run_dir / "usage.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert run["status"] == runner.STATUS_REFUSED_BUDGET
    assert run["stages"]["live_intent"] == "REFUSED"
    assert run["stages"]["live_pipeline"] == "REFUSED"
    assert run["budget"]["requests"] == 0
    assert run["details"]["refusal"]["reason"].startswith(
        "live 模式需要维护者明确提供 --budget-cny"
    )
    assert usage[-1]["requests"] == 0


def test_live_refuses_zero_budget_even_with_explicit_pricing(tmp_path):
    run_dir = tmp_path / "live-zero"
    exit_code = runner.main(
        [
            "--mode",
            "live",
            "--budget-cny",
            "0",
            "--input-price-cny-per-mtok",
            "1",
            "--output-price-cny-per-mtok",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 2
    run = _read_json(run_dir / "run.json")
    assert run["status"] == runner.STATUS_REFUSED_BUDGET
    assert run["details"]["refusal"]["reason"].startswith("预算必须大于 0")
    assert run["budget"]["requests"] == 0


def test_live_refuses_when_conservative_upper_bound_exceeds_budget(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "live-preflight"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def unexpected_client(**kwargs):
        raise AssertionError("预检拒绝前不应创建模型客户端")

    monkeypatch.setattr(intent_module, "AsyncAnthropic", unexpected_client)
    monkeypatch.setattr(tool_module, "AsyncAnthropic", unexpected_client)
    exit_code = runner.main(
        [
            "--mode",
            "live",
            "--budget-cny",
            "0.000001",
            "--input-price-cny-per-mtok",
            "1",
            "--output-price-cny-per-mtok",
            "1",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 2
    run = _read_json(run_dir / "run.json")
    details = run["details"]["refusal"]
    preflight = details["preflight"]
    assert run["status"] == runner.STATUS_REFUSED_PREFLIGHT
    assert details["status"] == runner.STATUS_REFUSED_PREFLIGHT
    assert preflight["fits_budget"] is False
    assert preflight["total_upper_bound_cny"] > 0.000001
    assert preflight["input_count_basis"] == runner.INPUT_COUNT_BASIS
    assert preflight["format_overhead_tokens_per_request"] > 0
    assert run["budget"]["requests"] == 0


def test_explicit_id_filter_preserves_order_and_rejects_bad_ids():
    rows = [
        {"case_id": "A", "split": "dev"},
        {"case_id": "B", "split": "dev"},
    ]

    selected = runner._select_by_ids(rows, "case_id", "B, A", "意图")
    assert [row["case_id"] for row in selected] == ["B", "A"]

    for raw in ("A,A", "A,Z", " , "):
        with pytest.raises(runner.BaselineRefused) as caught:
            runner._select_by_ids(rows, "case_id", raw, "意图")
        assert caught.value.status == runner.STATUS_REFUSED_SELECTION


def test_smoke_complete_is_success_but_not_full_baseline():
    result = {"status": runner.STATUS_SMOKE_COMPLETE, "stages": {}}

    assert runner._exit_code("live", result, True, True) == 0
    assert runner.STATUS_SMOKE_COMPLETE != runner.STATUS_COMPLETE


def test_guard_classifies_only_known_stage_prompts():
    assert runner._classify_stage("你是客服意图分析专家。") == "intent_llm"
    assert runner._classify_stage("将以下用户查询改写") == "query_rewrite"
    assert (
        runner._classify_stage("根据用户查询，对以下检索结果排序")
        == "rerank"
    )


def test_local_fallback_cleanup_stays_inside_run_directory(tmp_path):
    inside = tmp_path / "run" / "chroma_local"
    inside.mkdir(parents=True)
    (inside / "data.bin").write_text("x", encoding="utf-8")
    outside = tmp_path / "shared-chroma"
    outside.mkdir()

    assert runner._cleanup_local_path(str(inside), tmp_path / "run") is True
    assert not inside.exists()
    assert runner._cleanup_local_path(str(outside), tmp_path / "run") is False
    assert outside.exists()


class _LoopMessages:
    def __init__(self, owner):
        self.owner = owner

    async def create(self, **kwargs):
        self.owner.create_loops.append(asyncio.get_running_loop())
        prompt = runner._request_prompt_text(kwargs)
        if "将以下用户查询改写" in prompt:
            text = '["子查询A","子查询B","子查询C"]'
        elif "根据用户查询，对以下检索结果" in prompt:
            text = "[0,1,2,3,4]"
        else:
            text = self.owner.intent_text
        usage = types.SimpleNamespace(
            input_tokens=self.owner.input_tokens,
            output_tokens=self.owner.output_tokens,
        )
        return types.SimpleNamespace(
            content=[{"type": "text", "text": text}],
            usage=usage,
        )


class _LoopClient:
    """记录每次模型调用和关闭所在事件循环的离线客户端。"""

    def __init__(
        self,
        intent_text='{"intent":"order_status","confidence":0.95,"reasoning":"ok"}',
        input_tokens=100,
        output_tokens=10,
    ):
        self.intent_text = intent_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.create_loops = []
        self.close_loops = []
        self.messages = _LoopMessages(self)

    async def close(self):
        self.close_loops.append(asyncio.get_running_loop())


async def _knowledge_handler(params, context):
    query = str(params.get("query", ""))
    return [
        {
            "id": f"{query}-{index}",
            "title": f"标题{index}",
            "content": "内容" * 20,
        }
        for index in range(5)
    ]


def _build_recognizer_and_manager(monkeypatch, intent_client, tool_client):
    monkeypatch.setattr(
        intent_module, "AsyncAnthropic", lambda **kwargs: intent_client
    )
    monkeypatch.setattr(
        tool_module, "AsyncAnthropic", lambda **kwargs: tool_client
    )
    recognizer = intent_module.IntentRecognizer(
        api_key="test-key", model="test-model"
    )
    manager = tool_module.MCPToolManager(
        api_key="test-key", model="test-model"
    )
    manager.register(
        tool_module.Tool(
            name="knowledge_search",
            description="测试检索",
            handler=_knowledge_handler,
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            cache_ttl=0.0,
            supports_rerank=True,
        )
    )
    return recognizer, manager


def _sample_rows():
    intents = [
        {
            "case_id": "I1",
            "split": "dev",
            "message": "订单到哪了",
            "history": [],
            "primary_intent": "order_status",
            "slots": [],
        },
        {
            "case_id": "I2",
            "split": "dev",
            "message": "我要申请退款",
            "history": [],
            "primary_intent": "refund",
            "slots": [],
        },
    ]
    queries = [
        {
            "query_id": "Q1",
            "split": "dev",
            "query": "退货政策",
            "answerability": "answerable",
            "qrels": {},
            "required_evidence_groups": [],
        },
        {
            "query_id": "Q2",
            "split": "dev",
            "query": "配送要多久",
            "answerability": "answerable",
            "qrels": {},
            "required_evidence_groups": [],
        },
    ]
    return intents, queries


def test_live_stages_share_one_event_loop_and_close_clients_there(monkeypatch):
    intent_client = _LoopClient()
    tool_client = _LoopClient()
    recognizer, manager = _build_recognizer_and_manager(
        monkeypatch, intent_client, tool_client
    )
    intents, queries = _sample_rows()
    guard = BudgetGuard(1.0, 1.0, 1.0)
    request_log = []
    lifecycle = runner._ClientLifecycle()
    guarded_intent = runner._GuardedClient(
        recognizer.client, guard, request_log
    )
    recognizer.client = guarded_intent
    guarded_tool = runner._GuardedClient(
        manager._client, guard, request_log
    )
    manager._client = guarded_tool
    lifecycle.track(intent_client)
    lifecycle.track(tool_client)

    outcome = asyncio.run(
        runner._run_live_stages(
            recognizer,
            manager,
            intents,
            queries,
            request_log,
            guarded_intent,
            lifecycle,
        )
    )

    assert outcome["pipeline_skipped"] is False
    assert len(outcome["intent"][0]) == 2
    assert len(outcome["pipeline"][0]) == 2
    assert len(intent_client.create_loops) == 2
    # 每条查询一次改写，并在候选超过 top_k 时一次重排。
    assert len(tool_client.create_loops) == 4
    all_loops = intent_client.create_loops + tool_client.create_loops
    assert len({id(loop) for loop in all_loops}) == 1
    assert intent_client.close_loops == [all_loops[0]]
    assert tool_client.close_loops == [all_loops[0]]
    assert lifecycle.closed_indices == [0, 1]
    assert lifecycle.errors == []


def test_live_stages_close_on_budget_stop_and_skip_pipeline(monkeypatch):
    intent_client = _LoopClient()
    recognizer, _ = _build_recognizer_and_manager(
        monkeypatch, intent_client, _LoopClient()
    )
    intents, queries = _sample_rows()
    prompt = intent_module.build_intent_prompt(intents[0]["message"], [])
    request = runner._preflight_request_kwargs(
        "test-model", prompt, runner.INTENT_MAX_TOKENS, 0.1
    )
    one_request = BudgetGuard(1.0, 1.0, 1.0).estimate(
        request_input_upper_tokens(request), runner.INTENT_MAX_TOKENS
    )
    guard = BudgetGuard(one_request + 1e-6, 1.0, 1.0)
    request_log = []
    lifecycle = runner._ClientLifecycle()
    guarded_intent = runner._GuardedClient(
        recognizer.client, guard, request_log
    )
    recognizer.client = guarded_intent
    lifecycle.track(intent_client)

    class _PipelineMustNotRun:
        called = False

        async def search_with_rewrite(self, *args, **kwargs):
            self.called = True
            raise AssertionError("预算停止后不应继续 P 链路")

    manager = _PipelineMustNotRun()
    outcome = asyncio.run(
        runner._run_live_stages(
            recognizer,
            manager,
            intents,
            queries,
            request_log,
            guarded_intent,
            lifecycle,
        )
    )

    assert outcome["pipeline_skipped"] is True
    assert manager.called is False
    assert len(intent_client.create_loops) == 1
    assert intent_client.close_loops == [intent_client.create_loops[0]]
    assert guarded_intent.stop_reason
    assert any(
        item["status"] == "refused_before_request" for item in request_log
    )


def test_live_stages_close_clients_on_unexpected_exception():
    class _FailingQueries:
        def __iter__(self):
            raise RuntimeError("stage failure")

    class _UnusedRecognizer:
        async def recognize(self, *args, **kwargs):
            raise AssertionError("空意图集不应调用识别")

    client = _LoopClient()
    lifecycle = runner._ClientLifecycle()
    lifecycle.track(client)
    guard_client = types.SimpleNamespace(stop_reason=None)
    loop_holder = []

    async def run():
        loop_holder.append(asyncio.get_running_loop())
        await runner._run_live_stages(
            _UnusedRecognizer(),
            object(),
            [],
            _FailingQueries(),
            [],
            guard_client,
            lifecycle,
        )

    with pytest.raises(RuntimeError, match="stage failure"):
        asyncio.run(run())

    # 异常穿越阶段循环时仍在同一事件循环释放客户端。
    assert client.close_loops == loop_holder
    assert lifecycle.closed_indices == [0]


def test_degraded_rerank_cannot_send_after_cost_bound_failure(monkeypatch):
    tool_client = _LoopClient(input_tokens=10_000_000, output_tokens=10)
    monkeypatch.setattr(
        tool_module, "AsyncAnthropic", lambda **kwargs: tool_client
    )
    manager = tool_module.MCPToolManager(
        api_key="test-key", model="test-model"
    )
    manager.register(
        tool_module.Tool(
            name="knowledge_search",
            description="测试检索",
            handler=_knowledge_handler,
            schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            cache_ttl=0.0,
            supports_rerank=True,
        )
    )
    guard = BudgetGuard(1.0, 1.0, 1.0)
    manager._client = runner._GuardedClient(tool_client, guard, [])

    result = asyncio.run(
        manager.search_with_rewrite(
            "knowledge_search", "退货政策", top_k=5
        )
    )

    # 改写响应暴露预留上界失效后，重排预留被永久拒绝，未再发送请求。
    assert len(tool_client.create_loops) == 1
    assert guard.cost_stop_reason
    assert result.reranked is False


def test_intent_retry_cannot_send_after_cost_bound_failure(monkeypatch):
    intent_client = _LoopClient(
        intent_text="不是 JSON", input_tokens=10_000_000, output_tokens=10
    )
    monkeypatch.setattr(
        intent_module, "AsyncAnthropic", lambda **kwargs: intent_client
    )
    recognizer = intent_module.IntentRecognizer(
        api_key="test-key", model="test-model"
    )
    guard = BudgetGuard(1.0, 1.0, 1.0)
    request_log = []
    recognizer.client = runner._GuardedClient(
        intent_client, guard, request_log
    )

    result = asyncio.run(recognizer.recognize("订单到哪了", history=[]))

    # 首次解析失败后的重试被费用停止状态拦下，未发出第二次请求。
    assert len(intent_client.create_loops) == 1
    assert guard.cost_stop_reason
    assert result.recognition_status == "request_error"
    assert any(
        item["status"] == "refused_before_request" for item in request_log
    )
