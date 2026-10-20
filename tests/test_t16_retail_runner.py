"""T16 多 Agent 零售适配与成对汇总的离线回归。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
    ResponseComposer,
)
from agents.retail_multi_agent import (
    RetailBillingAgent,
    RetailEscalationAgent,
    RetailGeneralAgent,
    RetailMultiAgent,
    RetailTechnicalAgent,
)
from agents.retail_runtime import RetailRuntimeConfig, build_retail_runtime
from agents.retail_tools import RETAIL_READ_ONLY_TOOLS, retail_tools
from agents.tools import make_tool
from core.intent_recognizer import IntentCategory, IntentRecognizer
from evaluation.retail_ledger import BudgetedClient, RetailLedger
from evaluation.run_t16_retail import (
    BUDGET_CNY,
    EXCLUDED_TASK_IDS,
    RUN_ORDER,
    TASK_IDS,
    USER_MAX_TOKENS,
    _cost_breakdown,
    _gaps,
    _pair_summary,
    _verify_task_46_offline,
)
from mcp.retail_store import RetailStore

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
POLICY = ROOT / "evaluation" / "data" / "retail_sandbox" / "policy.md"


def _usage_report(text: str, input_tokens: int = 120, output_tokens: int = 20):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake client 没有剩余响应")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def _role_text_response(text: str):
    return _usage_report(text)


def _intent_response(message: str = "refund"):
    return _usage_report(
        json.dumps(
            {
                "intent": message,
                "confidence": 0.95,
                "reasoning": "离线固定响应",
            }
        )
    )


def test_t16_contract_constants_are_frozen():
    assert TASK_IDS == ["11", "12", "48", "51", "83", "92", "108"]
    assert EXCLUDED_TASK_IDS == ["46"]
    assert RUN_ORDER == [
        "11A", "11B", "12B", "12A", "48A", "48B", "51B", "51A",
        "83A", "83B", "92B", "92A", "108A", "108B",
    ]
    assert BUDGET_CNY == 10.0
    assert USER_MAX_TOKENS == 1024


def _role_agent(cls, policy_text: str):
    return cls(
        client=None,
        model="fake-model",
        language="en",
        policy_text=policy_text,
    )


def test_retail_role_permissions_and_shared_guard_are_tool_wrapped():
    policy_text = POLICY.read_text(encoding="utf-8")
    billing = _role_agent(RetailBillingAgent, policy_text)
    technical = _role_agent(RetailTechnicalAgent, policy_text)
    store = RetailStore.__new__(RetailStore)
    tools = retail_tools(RetailStore.__new__(RetailStore), None, language="en")
    billing.set_shared_tools(tools)
    technical.set_shared_tools(tools)

    req = Request(
        message="退款", user_id="u", conv_id="c", turn=2, retail_primary="billing"
    )
    assert set(billing._tools_for_request(req)) == set(tools)
    assert set(technical._tools_for_request(req)) == set(RETAIL_READ_ONLY_TOOLS)

    state = {}
    assert billing._guard_tool_call(req, "prepare_return", {}, state) is None
    assert technical._guard_tool_call(req, "prepare_return", {}, state)

    # 准备后同一轮只能继续准备，不能提交；未知结果阻止一切后续工具。
    state["prepared"] = True
    assert billing._guard_tool_call(
        req, "return_delivered_order_items", {}, state
    )
    assert billing._guard_tool_call(req, "prepare_return", {}, state) is None
    state["unknown_result"] = True
    assert billing._guard_tool_call(req, "get_order_details", {}, state)
    assert technical._guard_tool_call(req, "get_product_details", {}, state)


def test_supporting_role_cannot_bind_or_prepare_even_if_model_asks():
    policy_text = POLICY.read_text(encoding="utf-8")
    technical = _role_agent(RetailTechnicalAgent, policy_text)
    technical.set_shared_tools(
        retail_tools(RetailStore.__new__(RetailStore), None, language="en")
    )
    req = Request(
        message="帮我退货", user_id="u", conv_id="c", turn=3, retail_primary="billing"
    )
    assert "prepare_return" not in technical._tools_for_request(req)
    assert "find_user_id_by_email" not in technical._tools_for_request(req)
    assert technical._guard_tool_call(
        req, "transfer_to_human_agents", {}, {}
    )


def test_escalation_agent_records_real_handoff_through_shared_tool():
    called = []

    def handoff(req, args):
        called.append((req.request_id, args))
        return {"success": True, "handoff_id": "HO-test"}

    spec = make_tool(
        "transfer_to_human_agents",
        "record simulated handoff",
        {
            "summary": {"type": "string"},
            "reason": {"type": "string"},
        },
        handoff,
        required=["summary", "reason"],
    )
    agent = RetailEscalationAgent(None, "fake-model", language="en")
    agent.set_shared_tools({"transfer_to_human_agents": spec})
    req = Request(
        message="I want a human agent",
        user_id="u",
        conv_id="c",
        turn=4,
        intent=IntentCategory.HUMAN_HANDOFF,
    )

    result = asyncio.run(agent.handle(req))

    assert result.success is True
    assert result.escalate is True
    assert result.tools_used == ["transfer_to_human_agents"]
    assert called and called[0][0] == req.request_id
    assert result.meta["handoff_id"] == "HO-test"


def test_composer_skips_rewrite_after_state_operation():
    client = _FakeClient([])
    composer = ResponseComposer(
        client, "fake-model", language="en", retail_mode=True
    )
    req = Request(
        message="confirm",
        user_id="u",
        conv_id="c",
        turn=5,
        retail_guard={"state_op_entered": True},
    )
    responses = [
        AgentResponse(AgentType.BILLING, "Primary details.", True),
        AgentResponse(AgentType.TECHNICAL, "Supporting note.", True),
    ]

    result = asyncio.run(composer.compose_result(req, responses))

    assert result.used_llm is False
    assert client.messages.calls == []
    assert result.content.startswith("Primary details.")
    assert "Supporting note." in result.content


def test_orchestrator_collaborates_with_distinct_stage_clients_and_budget():
    ledger = RetailLedger(
        tmp_path := Path(__file__).resolve().parent / "_t16_ledger_tmp.db",
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
        run_id="t16-offline",
        model="fake-model",
    )
    try:
        intent = _FakeClient([_intent_response("refund")])
        billing = _FakeClient([_role_text_response("Billing answer.")])
        technical = _FakeClient([_role_text_response("Technical answer.")])
        general = _FakeClient([_role_text_response("Fallback answer.")])
        composer = _FakeClient([_usage_report("Composed answer.")])

        def wrap(inner, stage):
            return BudgetedClient(inner, ledger, stage, count_turn=True)

        role_clients = {
            "general": wrap(general, "t16-roles-general"),
            "technical": wrap(technical, "t16-roles-technical"),
            "billing": wrap(billing, "t16-roles-billing"),
            "escalation": wrap(_FakeClient([]), "t16-roles-escalation"),
        }
        orchestrator = AgentOrchestrator(
            client=wrap(_FakeClient([]), "t16-roles-unused"),
            model="fake-model",
            intent_recognizer=IntentRecognizer(
                model="fake-model",
                client=wrap(intent, "t16-intent"),
                language="en",
            ),
            composer=ResponseComposer(
                wrap(composer, "t16-composer"),
                "fake-model",
                None,
                language="en",
                retail_mode=True,
            ),
            language="en",
            retail_mode=True,
            policy_text=POLICY.read_text(encoding="utf-8"),
            retail_role_classes={
                "general": RetailGeneralAgent,
                "technical": RetailTechnicalAgent,
                "billing": RetailBillingAgent,
                "escalation": RetailEscalationAgent,
            },
            retail_role_clients=role_clients,
        )
        multi = RetailMultiAgent(
            orchestrator, ledger=ledger, turn_label="t16", language="en"
        )
        req = Request(
            message=(
                "I need a refund for the damaged order, and the app shows "
                "an error when I try to return it."
            ),
            user_id="u",
            conv_id="c",
            turn=1,
        )

        result = asyncio.run(multi.handle(req))

        assert result.success is True
        assert result.meta["primary_agent"] == "billing"
        assert result.meta["supporting_agents"] == ["technical"]
        assert result.meta["composer_used"] is True
        stages = {
            entry["stage"]
            for entry in ledger.entries()
        }
        assert {
            "t16-intent",
            "t16-roles-billing",
            "t16-roles-technical",
            "t16-composer",
        } <= stages
        assert ledger.summary()["requests"] == 4
        assert ledger.summary()["spent_cny"] > 0
        assert req.retail_guard is not None
    finally:
        ledger.path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(ledger.path) + suffix).unlink(missing_ok=True)


def test_turn_request_limit_stops_cross_role_replay():
    ledger = RetailLedger(
        tmp_path := Path(__file__).resolve().parent / "_t16_limit_tmp.db",
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
        run_id="t16-limit",
        model="fake-model",
    )
    try:
        def wrap(inner, stage):
            return BudgetedClient(inner, ledger, stage, count_turn=True)

        orchestrator = AgentOrchestrator(
            client=wrap(_FakeClient([]), "t16-limit-roles-unused"),
            model="fake-model",
            intent_recognizer=IntentRecognizer(
                model="fake-model",
                client=wrap(_FakeClient([_intent_response()]), "t16-limit-intent"),
                language="en",
            ),
            composer=ResponseComposer(
                wrap(_FakeClient([]), "t16-limit-composer"),
                "fake-model",
                None,
                language="en",
                retail_mode=True,
            ),
            language="en",
            retail_mode=True,
            retail_role_classes={
                "general": RetailGeneralAgent,
                "technical": RetailTechnicalAgent,
                "billing": RetailBillingAgent,
                "escalation": RetailEscalationAgent,
            },
            retail_role_clients={
                "general": wrap(_FakeClient([_role_text_response("g")]), "t16-limit-g"),
                "technical": wrap(
                    _FakeClient([_role_text_response("t")]), "t16-limit-t"
                ),
                "billing": wrap(
                    _FakeClient([_role_text_response("b")]), "t16-limit-b"
                ),
                "escalation": wrap(
                    _FakeClient([]), "t16-limit-e"
                ),
            },
        )
        multi = RetailMultiAgent(
            orchestrator,
            ledger=ledger,
            turn_label="t16",
            max_requests_per_turn=2,
            language="en",
        )
        req = Request(
            message="Refund the damaged order and the app shows an error.",
            user_id="u",
            conv_id="c",
            turn=1,
        )
        from evaluation.retail_budget import TurnRequestLimitExceeded

        with pytest.raises(TurnRequestLimitExceeded):
            asyncio.run(multi.handle(req))
    finally:
        ledger.path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(ledger.path) + suffix).unlink(missing_ok=True)


def test_pair_summary_counts_only_common_effective_pairs():
    results = []
    for index, task_id in enumerate(TASK_IDS):
        a_ok = index in {0, 1}
        b_ok = index in {0, 2}
        for arm, ok in (("A", a_ok), ("B", b_ok)):
            results.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "effective_completed": index != 3,
                    "reward_success": ok,
                    "termination": "USER_STOP" if index != 3 else "AGENT_ERROR",
                    "db_status": "OK" if index != 3 else "OK",
                    "nl_status": "NOT_APPLICABLE",
                    "eval_error": "",
                    "valid_termination": index != 3,
                }
            )
    summary = _pair_summary(results)

    assert summary["valid_pairs"] == 6
    assert summary["valid_pair_fraction"] == "6/7"
    assert summary["counts_on_valid_pairs"] == {
        "both_success": 1,
        "only_a": 1,
        "only_b": 1,
        "both_fail": 3,
    }
    assert summary["a_success_on_valid_pairs"] == 2
    assert summary["b_success_on_valid_pairs"] == 2
    assert any(
        item["outcome"] == "unpaired" and item["task_id"] == TASK_IDS[3]
        for item in summary["pairs"]
    )


def test_gap_classification_and_cost_breakdown_are_separate():
    rows = [
        {
            "task_id": "11",
            "arm": "A",
            "termination": "USER_STOP",
            "db_status": "OK",
            "nl_status": "NOT_APPLICABLE",
            "eval_error": "",
            "valid_termination": True,
            "effective_completed": True,
        },
        {
            "task_id": "12",
            "arm": "B",
            "termination": "INCOMPLETE_BUDGET",
            "db_status": "OK",
            "nl_status": "NOT_APPLICABLE",
            "eval_error": "",
            "valid_termination": True,
            "effective_completed": False,
        },
        {
            "task_id": "108",
            "arm": "A",
            "termination": "USER_STOP",
            "db_status": "OK",
            "nl_status": "JUDGE_ERROR",
            "eval_error": "",
            "valid_termination": True,
            "effective_completed": False,
        },
    ]
    gaps = _gaps(rows)
    assert gaps["budget_incomplete"] == ["12B"]
    assert gaps["judge_error"] == ["108A"]
    # 预算停止单独归类，不重复计入无效终止。
    assert gaps["invalid_termination"] == []

    entries = [
        {"stage": "11A-agent", "actual_cny": 0.1, "input_tokens": 100,
         "output_tokens": 10},
        {"stage": "12B-user", "reserved_cny": 0.02, "input_tokens_upper": 20,
         "max_tokens": 5},
        {"stage": "108A-judge", "actual_cny": 0.03, "input_tokens": 30,
         "output_tokens": 3},
    ]
    cost = _cost_breakdown(entries)
    assert cost["system_cny"] == pytest.approx(0.1)
    assert cost["user_cny"] == pytest.approx(0.02)
    assert cost["judge_cny"] == pytest.approx(0.03)
    assert cost["total_cny"] == pytest.approx(0.15)


def test_multi_runtime_starts_and_closes_real_mcp_with_isolated_sqlite(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    config = RetailRuntimeConfig(
        db_path=tmp_path / "retail.db",
        seed_path=SEED,
        policy_path=POLICY,
        knowledge_path=(
            ROOT / "evaluation" / "data" / "retail_sandbox" / "knowledge.jsonl"
        ),
        knowledge_collection="t16_offline",
        ledger_path=tmp_path / "ledger.db",
        event_log_path=tmp_path / "events.jsonl",
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
        model="fake-model",
        base_url="",
        redis_url="redis://127.0.0.1:1/0",
        chroma_host="127.0.0.1",
        chroma_port=1,
        chroma_path=tmp_path / "chroma",
        case_id="t16-offline",
        run_id="t16-offline",
        profile_enabled=False,
        memory_key_prefix="t16:",
        memory_collection_suffix="_t16",
        max_tokens=2048,
        max_message_chars=4000,
        max_context_chars=12000,
        mcp_timeout_seconds=30.0,
        mcp_stderr_log=tmp_path / "mcp.log",
        agent_mode="multi",
        enable_memory=False,
        enable_knowledge=False,
    )

    def fake_build(api_key, base_url, ledger, stage):
        return BudgetedClient(_FakeClient([]), ledger, stage)

    monkeypatch.setattr(
        "agents.retail_runtime.build_budgeted_client", fake_build
    )
    runtime = build_retail_runtime(config)
    assert isinstance(runtime.agent, RetailMultiAgent)
    assert runtime.service.language == "zh"

    async def scenario():
        await runtime.start()
        assert set(runtime.agent.get_tools()) == set(
            retail_tools(runtime.store, None, language="zh")
        )
        await runtime.close()

    asyncio.run(scenario())
    assert runtime.mcp_client._closed is True
    assert runtime.mcp_client._stdio_cm is None
    assert (tmp_path / "retail.db").is_file()
    assert (tmp_path / "ledger.db").is_file()


class _HarnessStage:
    """只替代模型传输层；MCP、SQLite、User 与 Eval 子进程保持真实。"""

    def __init__(self, stage: str, state):
        self.stage = stage
        self.state = state

        class Messages:
            async def create(inner, **kwargs):
                if stage.endswith("-user"):
                    state["user_calls"] = state.get("user_calls", 0) + 1
                    text = (
                        "I need help with my order."
                        if state["user_calls"] == 1
                        else "###STOP###"
                    )
                    return _usage_report(text)
                if stage.endswith("-intent"):
                    return _intent_response("query")
                if stage.endswith("-judge"):
                    return _usage_report('{"reasoning": "ok", "passed": true}')
                return _usage_report("I can help with that.")

        self.messages = Messages()


def test_case_harness_uses_real_mcp_sqlite_user_and_eval_bridges(tmp_path):
    from evaluation import retail_ledger as ledger_module
    from evaluation.run_t16_retail import _run_case

    ledger = RetailLedger(
        tmp_path / "ledger.db",
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
        run_id="t16-harness",
        model="fake-model",
    )
    state: dict = {}

    class HarnessClient:
        def __init__(self, ledger, stage):
            self.ledger = ledger
            self.stage = stage

        def for_stage(self, stage, count_turn=None):
            return BudgetedClient(
                _HarnessStage(stage, state),
                self.ledger,
                stage,
                count_turn=bool(count_turn),
            )

        async def aclose(self):
            return None

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ledger_module,
        "build_budgeted_client",
        lambda api_key, base_url, ledger, stage: HarnessClient(ledger, stage),
    )
    try:
        for arm in ("A", "B"):
            state.clear()
            row = asyncio.run(
                _run_case(
                    "11", arm, tmp_path, ledger, "fake-key", None
                )
            )
            assert row["termination"] == "USER_STOP", row
            assert row["arm"] == arm
            assert row["ledger"]["requests"] >= 2
            assert (tmp_path / "arms" / arm / "cases" / "11" / "trial-1" / "result.json").is_file()
            assert (tmp_path / "arms" / arm / "cases" / "11" / "trial-1" / "retail.sqlite").is_file()
            assert (tmp_path / "arms" / arm / "cases" / "11" / "trial-1" / "mcp.log").is_file()
    finally:
        monkeypatch.undo()


def test_task_46_offline_verification_is_zero_cost_and_isolated(tmp_path):
    payload = _verify_task_46_offline(tmp_path)
    assert payload["model_requests"] == 0
    assert payload["cost_cny"] == 0.0
    assert payload["status"] == "EXCLUDED_INVALID_REFERENCE"
    assert payload["missing_gold_order_ids"] == ["#9502126", "#9502127"]
    assert (tmp_path / "excluded" / "46" / "reference.sqlite").is_file()
    assert (tmp_path / "excluded" / "46" / "offline-verification.json").is_file()
