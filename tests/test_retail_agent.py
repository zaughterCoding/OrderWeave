"""T13 单 Agent 工具范围、同轮保护和程序展示兜底。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentResponse, AgentType
from agents.retail_agent import RetailAgent, RetailService
from agents.retail_tools import retail_tool_scope, retail_tools
from mcp.retail_store import RetailStore

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
EMAIL = "chen.silva2698@example.com"
ORDER = "#W3069600"
ITEM = "5012998807"
PAYMENT = "credit_card_1565124"


@pytest.fixture()
def store(tmp_path):
    return RetailStore(tmp_path / "retail.db", SEED)


def test_retail_tool_scope_is_narrow_and_knowledge_is_top5(store):
    business = retail_tools(store)
    assert set(business) == set(retail_tool_scope(has_knowledge=False))
    assert len(business) == 9

    with_knowledge = retail_tools(store, tool_manager=object())
    assert len(with_knowledge) == 10
    assert set(with_knowledge) == set(retail_tool_scope())
    values = with_knowledge["search_knowledge_base"].input_schema["properties"][
        "top_k"
    ]
    assert values["minimum"] == 1
    assert values["maximum"] == 5

    # 旧客服工具不在零售白名单中。
    assert "search_knowledge_base" in with_knowledge
    assert "lookup_error_code" not in with_knowledge
    assert "check_billing_fields" not in with_knowledge


def test_retail_agent_limits_and_same_turn_guard():
    assert RetailAgent.agent_type is AgentType.RETAIL
    assert RetailAgent.max_tool_rounds == 10
    assert RetailAgent.max_tool_uses_per_response == 1
    assert RetailAgent.capture_tool_results is True
    assert RetailAgent.profile.max_tokens == 2048

    agent = RetailAgent.__new__(RetailAgent)
    req = SimpleNamespace(
        message="确认", user_id="client", conv_id="conv", turn=1
    )
    state = {}

    assert agent._guard_tool_call(req, "prepare_return", {}, state) is None
    agent._on_tool_result(
        req, "prepare_return", {}, {"success": True}, state
    )
    assert state["prepared"] is True
    assert agent._guard_tool_call(
        req, "return_delivered_order_items", {}, state
    )
    assert agent._guard_tool_call(req, "get_order_details", {}, state)
    # 同轮仍允许准备其他订单，便于多单逐笔展示。
    assert agent._guard_tool_call(req, "prepare_return", {}, state) is None


class PreparingAgent:
    """直接执行一次 prepare，用于验证服务层展示兜底。"""

    agent_type = AgentType.RETAIL

    def __init__(self, store, content="已整理退货详情。", success=True):
        self.store = store
        self.content = content
        self.success = success
        self.prepared = None

    async def handle(self, req):
        self.prepared = self.store.prepare_return(
            req.user_id, req.conv_id, ORDER, [ITEM], PAYMENT
        )
        assert self.prepared["success"], self.prepared
        return AgentResponse(
            agent_type=AgentType.RETAIL,
            content=self.content,
            success=self.success,
            latency_ms=1.0,
            tools_used=["prepare_return"],
            tool_traces=[
                {
                    "tool_name": "prepare_return",
                    "result_success": True,
                    "result": self.prepared,
                }
            ],
        )


def _service(store, agent, **kwargs):
    return RetailService(store=store, agent=agent, memory=None, **kwargs)


def test_service_programmatically_appends_and_marks_display(store):
    store.find_user_by_email("client", "conv", EMAIL)
    agent = PreparingAgent(store)
    service = _service(store, agent)

    result = __import__("asyncio").run(
        service.handle_turn("请帮我退货", "client", "conv")
    )

    assert "【退货申请详情（待确认）】" in result.response
    assert "Makeup Kit" in result.response
    assert result.display_appended == [agent.prepared["op_id"]]
    pending = store.active_pending("client", "conv")
    assert pending[0]["status"] == "displayed"
    assert pending[0]["display_text"] in result.response
    trace = service.get_tool_trace(result.request_id)
    assert trace is not None
    assert trace["tools_used"] == ["prepare_return"]


def test_failed_model_turn_still_displays_but_reports_failure(store):
    store.find_user_by_email("client", "conv", EMAIL)
    agent = PreparingAgent(store, content="", success=False)
    service = _service(store, agent)

    result = __import__("asyncio").run(
        service.handle_turn("请帮我退货", "client", "conv")
    )

    assert result.success is False
    assert "我已整理好待确认的退货详情" in result.response
    assert "【退货申请详情（待确认）】" in result.response
    assert store.active_pending("client", "conv")[0]["status"] == "displayed"


def test_service_rejects_empty_and_oversized_messages(store):
    store.find_user_by_email("client", "conv", EMAIL)
    service = _service(
        store, PreparingAgent(store), max_message_chars=10
    )

    async def scenario():
        with pytest.raises(ValueError, match="不能为空"):
            await service.handle_turn("   ", "client", "conv")
        with pytest.raises(ValueError, match="字符上限"):
            await service.handle_turn("x" * 11, "client", "conv")

    __import__("asyncio").run(scenario())
    assert store.current_turn("client", "conv") == 0


def test_english_agent_uses_english_prompt_and_tool_descriptions(tmp_path):
    """英文评测模式只替换模型可见文案，业务与中文默认路径不变。"""
    store = RetailStore(tmp_path / "retail.db", SEED, language="en")
    agent = RetailAgent(
        client=None,
        model="fake-model",
        store=store,
        tool_manager=None,
        max_tokens=2048,
    )

    assert agent.language == "en"
    assert "Retail order and returns service" in agent.profile.role
    assert "you are the retail customer service agent" in (
        agent.system_prompt.lower()
    )
    assert "退货" not in agent.system_prompt

    tools = agent._shared_tools
    assert "search_knowledge_base" not in tools
    assert len(tools) == 9
    assert "sandbox user" in tools["find_user_id_by_email"].description
    assert "pending-confirmation" in tools["prepare_return"].description.lower()
    assert "order list" in tools["get_user_details"].description.lower()
    assert "order ids" in tools["get_user_details"].description.lower()
    assert "get_user_details" in agent.system_prompt

    zh_store = RetailStore(tmp_path / "zh.db", SEED)
    zh_agent = RetailAgent(
        client=None,
        model="fake-model",
        store=zh_store,
        tool_manager=None,
        max_tokens=2048,
    )
    assert zh_agent.language == "zh"
    assert "退货" in zh_agent.system_prompt
    assert "沙箱用户" in zh_agent._shared_tools["find_user_id_by_email"].description
    assert "订单列表" in zh_agent._shared_tools["get_user_details"].description
    assert "get_user_details" in zh_agent.system_prompt


def test_get_user_details_description_matches_returned_orders(store):
    """工具说明提到订单列表，实际返回也确实包含本人订单号。"""
    store.find_user_by_email("client", "conv", EMAIL)
    details = store.get_user_details("client", "conv")
    assert details["success"] is True
    assert ORDER in details["user"]["orders"]
    assert "订单列表" in retail_tools(store)["get_user_details"].description
    english = retail_tools(store, language="en")["get_user_details"].description
    assert "order list" in english.lower()
