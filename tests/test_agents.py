import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from mcp.tool_manager import ToolResult


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时这笔订单被重复扣款",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "amount": ["99 元"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != BillingAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_billing_fields" in BillingAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    billing_packet = BillingAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "verification_fields" in billing_packet
    assert general_packet != technical_packet != billing_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "人工升级" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.BILLING, "请提供两笔扣款的时间和金额。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "两笔扣款" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_billing_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.BILLING: [object()],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.BILLING]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    billing_tools = set(BillingAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"}
    assert billing_tools == {"check_billing_fields", "compare_amounts"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & technical_tools
    assert not technical_tools & billing_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "退款政策", "content": "7 天内可退款"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, billing, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"query": "退款", "secret": "nope"}, "不允许的工具参数"),
        ({"query": "退款", "top_k": True}, "参数 top_k 类型错误"),
        ({"query": "退款", "top_k": 0}, "参数 top_k 小于最小值 1"),
        ({"query": "退款", "top_k": 11}, "参数 top_k 大于最大值 10"),
    ],
)
def test_tool_input_validation_rejects_invalid_arguments(args, message):
    agent = GeneralAgent(FakeClient(), "test-model")
    agent.set_shared_tools(build_shared_rag_tools(None))
    spec = agent.get_tools()["search_knowledge_base"]

    with pytest.raises(ValueError, match=message):
        agent._validate_tool_input(spec, args)


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool_name"] == "lookup_error_code"
    assert response.tool_traces[0]["input"] == {"error_code": "401"}
    assert response.tool_traces[0]["success"] is True
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_concurrent_requests_keep_tool_traces_isolated():
    class ToolUseBlock:
        type = "tool_use"

        def __init__(self, code):
            self.id = f"toolu_{code}"
            self.name = "lookup_error_code"
            self.input = {"error_code": code}

    class TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class ConcurrentClient:
        class Messages:
            async def create(inner, **kwargs):
                messages = kwargs["messages"]
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    code = "500" if "500" in str(last_content) else "401"
                    await asyncio.sleep(0)
                    return type("Response", (), {"content": [TextBlock(f"{code} 排查完成")]})()
                code = "500" if "500" in str(last_content) else "401"
                await asyncio.sleep(0.01 if code == "401" else 0)
                return type("Response", (), {"content": [ToolUseBlock(code)]})()

        def __init__(self):
            self.messages = self.Messages()

    async def run_both():
        agent = TechnicalAgent(ConcurrentClient(), "test-model")
        return await asyncio.gather(
            agent.handle(make_request(message="登录报 401", entities={"error_code": ["401"]})),
            agent.handle(make_request(message="页面报 500", entities={"error_code": ["500"]})),
        )

    first, second = asyncio.run(run_both())
    assert first.tool_traces[0]["input"] == {"error_code": "401"}
    assert second.tool_traces[0]["input"] == {"error_code": "500"}


def test_monitor_penalty_falls_back_to_general_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    billing = BillingAgent(FakeClient(), "test-model")
    technical.stats.monitor_penalty = 0.5
    orchestrator._pool = {
        AgentType.GENERAL: [general],
        AgentType.TECHNICAL: [technical],
        AgentType.BILLING: [billing],
    }

    decision = orchestrator._route_decision(make_request(
        message="登录报 401",
        entities={"error_code": ["401"]},
    ))

    assert decision.primary_agent is AgentType.GENERAL
    assert decision.supporting_agents == []
    assert "monitor_fallback=[technical]" in decision.reason


def test_orchestrator_constructs_and_injects_shared_tools(monkeypatch):
    class FakeAnthropicClient:
        def __init__(self, **kwargs):
            class Messages:
                async def create(inner, **call_kwargs):
                    raise AssertionError("no model call expected")

            self.messages = Messages()

    monkeypatch.setattr(
        "agents.agent_orchestrator.AsyncAnthropic",
        lambda **kwargs: FakeAnthropicClient(**kwargs),
    )
    monkeypatch.setattr(
        "core.intent_recognizer.AsyncAnthropic",
        lambda **kwargs: FakeAnthropicClient(**kwargs),
    )

    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return SimpleNamespace(success=True, data=[], reranked=False)

    orchestrator = AgentOrchestrator(
        api_key="test-key",
        base_url="https://example.invalid",
        model="test-model",
        rag_tool_manager=RagManager(),
    )

    for agent_type in (AgentType.GENERAL, AgentType.TECHNICAL, AgentType.BILLING, AgentType.ESCALATION):
        assert "search_knowledge_base" in orchestrator._best_agent(agent_type).get_tools()
    assert set(orchestrator.get_stats()) == {"general_0", "technical_0", "billing_0", "escalation_0"}

    orchestrator.set_shared_tools({})
    assert "search_knowledge_base" not in orchestrator._best_agent(AgentType.GENERAL).get_tools()


def test_recognize_intent_is_exposed_independently():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

    class FakeRecognizer:
        async def recognize(self, message, history=None):
            return SimpleNamespace(message=message, history=history)

    orchestrator._intent_recognizer = FakeRecognizer()
    history = [{"role": "user", "content": "订单有问题"}]

    result = asyncio.run(orchestrator.recognize_intent("我要申请退款", history=history))

    assert result.message == "我要申请退款"
    assert result.history == history


class _BillingToolClient:
    """Call check_billing_fields once, then return a final text response."""

    def __init__(self):
        self.calls = []

        class Messages:
            def __init__(inner):
                self.responses = [
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                type="tool_use",
                                id="toolu_billing",
                                name="check_billing_fields",
                                input={},
                            )
                        ]
                    ),
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                type="text", text="已核对账单字段。"
                            )
                        ]
                    ),
                ]

            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                return self.responses.pop(0)

        self.messages = Messages()


class _FakeIntentRecognizer:
    def __init__(self, entities=None):
        self.entities = entities or {
            "order_id": ["AAAA1111"],
            "amount": ["50 USD"],
        }
        self.calls = 0

    async def recognize(self, message, history=None):
        self.calls += 1
        return SimpleNamespace(
            intent=IntentCategory.REFUND,
            intent_group="billing",
            urgency=UrgencyLevel.LOW,
            confidence=0.9,
            source_scores={},
            entities=dict(self.entities),
        )


def make_orchestrator_for_run(recognizer, client):
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._intent_recognizer = recognizer
    orchestrator._recent_tool_traces = deque(maxlen=20)
    orchestrator._skill_manager = None
    orchestrator._shared_tools = {}
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(client, "test-model")],
        AgentType.TECHNICAL: [TechnicalAgent(client, "test-model")],
        AgentType.BILLING: [BillingAgent(client, "test-model")],
        AgentType.ESCALATION: [EscalationAgent(client, "test-model")],
    }
    return orchestrator


def test_run_backfills_recognized_entities_for_billing_tools():
    client = _BillingToolClient()
    orchestrator = make_orchestrator_for_run(_FakeIntentRecognizer(), client)
    request = Request(
        message="订单 AAAA1111 扣了 50 USD，请处理退款",
        user_id="u1",
        conv_id="c1",
    )

    result = asyncio.run(orchestrator.run(request))

    assert result.agent_type is AgentType.BILLING
    assert request.entities["order_id"] == ["AAAA1111"]
    assert request.entities["amount"] == ["50 USD"]
    tool_result = json.loads(
        client.calls[1]["messages"][-1]["content"][0]["content"]
    )
    assert tool_result["fields"]["order_id"] is True
    assert tool_result["fields"]["amount"] is True


def test_run_merges_recognized_entities_without_clearing_caller_values():
    client = _BillingToolClient()
    orchestrator = make_orchestrator_for_run(_FakeIntentRecognizer(), client)
    request = Request(
        message="请处理退款",
        user_id="u1",
        conv_id="c1",
        entities={"order_id": ["CALLER-ORDER"]},
    )

    asyncio.run(orchestrator.run(request))

    assert request.entities["order_id"] == ["CALLER-ORDER"]
    assert request.entities["amount"] == ["50 USD"]


def test_prefilled_intent_skips_extra_recognition():
    client = _BillingToolClient()

    class ExplodingRecognizer:
        async def recognize(self, *args, **kwargs):
            raise AssertionError("prefilled intent must not be recognized again")

    orchestrator = make_orchestrator_for_run(ExplodingRecognizer(), client)
    request = Request(
        message="请处理退款",
        user_id="u1",
        conv_id="c1",
        intent=IntentCategory.REFUND,
        intent_group="billing",
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.9,
        entities={"order_id": ["KEEP-ME"]},
    )

    result = asyncio.run(orchestrator.run(request))

    assert result.agent_type is AgentType.BILLING
    assert request.entities == {"order_id": ["KEEP-ME"]}


class _KnowledgeToolClient:
    """Request the knowledge tool once, then answer with plain text."""

    def __init__(self, fail_final=False):
        self.calls = []
        self.fail_final = fail_final

        class Messages:
            def __init__(inner):
                self.responses = [
                    SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                type="tool_use",
                                id="toolu_kb",
                                name="search_knowledge_base",
                                input={"query": "退款政策"},
                            )
                        ]
                    ),
                    SimpleNamespace(
                        content=[SimpleNamespace(type="text", text="根据知识库答复。")]
                    ),
                ]

            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.fail_final and len(self.calls) >= 2:
                    raise RuntimeError("final reply failed")
                return self.responses.pop(0)

        self.messages = Messages()


class _PlainClient:
    """Answer without requesting any tool."""

    def __init__(self):
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="已处理。")]
                )

        self.messages = Messages()


class _RagManager:
    def __init__(self, result):
        self.result = result

    async def search_with_rewrite(self, tool_name, query, top_k=5):
        return self.result


def _knowledge_tools(result):
    return build_shared_rag_tools(_RagManager(result))


def _chat_orchestrator(result, client=None):
    orchestrator = make_orchestrator_for_run(
        _FakeIntentRecognizer(), client or _KnowledgeToolClient()
    )
    shared = _knowledge_tools(result)
    for agents in orchestrator._pool.values():
        for agent in agents:
            agent.set_shared_tools(shared)
    return orchestrator


class _FakeChatMemory:
    async def get_context(self, user_id, conv_id, query=None):
        return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")

    async def add_message(self, *args, **kwargs):
        return None

    async def update_profile(self, *args, **kwargs):
        return None


F6_CASES = [
    pytest.param(
        ToolResult(
            success=True,
            data=[{"title": "退款政策", "content": "7 天内可退款"}],
            tool_name="knowledge_search",
        ),
        True,
        id="valid-hit",
    ),
    pytest.param(
        ToolResult(
            success=False,
            data=[{"title": "知识库降级结果", "fallback": True}],
            tool_name="knowledge_search",
            error="chroma down",
            degraded=True,
        ),
        False,
        id="pure-degradation",
    ),
    pytest.param(
        ToolResult(
            success=True,
            data=[{"title": "退款政策", "content": "7 天内可退款"}],
            tool_name="knowledge_search",
            degraded=True,
        ),
        True,
        id="partial-degradation-with-hit",
    ),
    pytest.param(
        ToolResult(success=True, data=[], tool_name="knowledge_search"),
        False,
        id="empty-success",
    ),
    pytest.param(
        ToolResult(
            success=False, data=None, tool_name="knowledge_search", error="boom"
        ),
        False,
        id="failed-call",
    ),
]


@pytest.mark.parametrize(("rag_result", "expected"), F6_CASES)
def test_agent_counts_only_valid_knowledge_fragments(rag_result, expected):
    agent = GeneralAgent(_KnowledgeToolClient(), "test-model")
    agent.set_shared_tools(_knowledge_tools(rag_result))

    response = asyncio.run(
        agent.handle(
            make_request(
                message="请处理退款",
                intent=IntentCategory.REFUND,
                intent_group="billing",
            )
        )
    )

    assert response.success is True
    assert response.tools_used == ["search_knowledge_base"]
    assert response.tool_traces[0]["tool_name"] == "search_knowledge_base"
    assert (response.knowledge_hits > 0) is expected


def test_agent_without_knowledge_call_has_no_hits():
    agent = GeneralAgent(_PlainClient(), "test-model")
    agent.set_shared_tools(
        _knowledge_tools(
            ToolResult(
                success=True,
                data=[{"title": "退款政策", "content": "7 天内可退款"}],
                tool_name="knowledge_search",
            )
        )
    )

    response = asyncio.run(agent.handle(make_request()))

    assert response.tools_used == []
    assert response.knowledge_hits == 0


@pytest.mark.parametrize(("rag_result", "expected"), F6_CASES)
def test_chat_reports_knowledge_used_only_for_valid_hits(
    monkeypatch, rag_result, expected
):
    import api.main as api_main

    orchestrator = _chat_orchestrator(rag_result)
    monkeypatch.setattr(api_main, "_orchestrator", orchestrator)
    monkeypatch.setattr(api_main, "_memory", _FakeChatMemory())
    monkeypatch.setattr(api_main, "_spawn_background", lambda coro: coro.close())

    response = asyncio.run(
        api_main.chat(
            api_main.ChatRequest(
                message="请处理退款", user_id="u1", conv_id="c1"
            )
        )
    )

    assert response.knowledge_used is expected
    assert "search_knowledge_base" in response.tools_used


def test_failed_business_response_drops_hits_from_the_run_result():
    hit = ToolResult(
        success=True,
        data=[{"title": "退款政策", "content": "7 天内可退款"}],
        tool_name="knowledge_search",
    )
    orchestrator = make_orchestrator_for_run(
        _FakeIntentRecognizer(), _KnowledgeToolClient(fail_final=True)
    )
    for agents in orchestrator._pool.values():
        for agent in agents:
            agent.set_shared_tools(_knowledge_tools(hit))
    request = Request(
        message="请问系统有哪些政策说明",
        user_id="u1",
        conv_id="c1",
        intent=IntentCategory.QUERY,
        intent_group="general",
        urgency=UrgencyLevel.LOW,
        intent_confidence=0.9,
    )

    result = asyncio.run(orchestrator.run(request))

    assert "search_knowledge_base" in result.tools_used
    assert result.knowledge_hits == 0


def test_parallel_run_keeps_a_valid_hit_from_any_agent():
    hit = ToolResult(
        success=True,
        data=[{"title": "退款政策", "content": "7 天内可退款"}],
        tool_name="knowledge_search",
    )
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._recent_tool_traces = deque(maxlen=20)
    orchestrator._skill_manager = None
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(_PlainClient(), "test-model")],
        AgentType.TECHNICAL: [TechnicalAgent(_PlainClient(), "test-model")],
        AgentType.BILLING: [BillingAgent(_KnowledgeToolClient(), "test-model")],
        AgentType.ESCALATION: [EscalationAgent(_PlainClient(), "test-model")],
    }
    orchestrator._shared_tools = {}
    for agents in orchestrator._pool.values():
        for agent in agents:
            agent.set_shared_tools(_knowledge_tools(hit))

    class Composer:
        async def compose(self, req, responses):
            return " / ".join(response.content for response in responses)

    orchestrator._composer = Composer()
    decision = RoutingDecision(
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        reason="composite",
        confidence=0.9,
    )

    result = asyncio.run(
        orchestrator.run_parallel(make_request(message="请处理退款"), decision)
    )

    assert result.knowledge_hits == 1
    assert "search_knowledge_base" in result.tools_used
