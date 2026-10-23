"""结构化 JSON 阶段的输出预算回归测试。

意图、改写、重排、画像、摘要合并和 Judge 都需要先推理再输出 JSON。
真实模型在 max_tokens 过小时会在 JSON 完成前被截断，调用方随后静默降级，
把故障伪装成“无结果”或“评分失败”。这里固定各阶段的最低输出预算。
"""
import asyncio
import json
from types import SimpleNamespace

from agents.agent_orchestrator import (
    AgentResponse,
    AgentType,
    Request,
    ResponseComposer,
)
from evaluation.evaluator import LLMJudge
from core.intent_recognizer import IntentRecognizer
from memory.conversation_memory import MemoryManager
from mcp.tool_manager import MCPToolManager

MIN_JSON_OUTPUT_TOKENS = 1024


class RecordingMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


class RecordingClient:
    def __init__(self, responses):
        self.messages = RecordingMessages(responses)


def assert_json_budget(call):
    assert call["max_tokens"] >= MIN_JSON_OUTPUT_TOKENS


def test_rerank_budget_survives_candidate_scoring():
    """重排候选较多时，模型会先逐条打分，输出预算不足会截断索引列表。"""
    async def scenario():
        client = RecordingClient(["[0, 1, 2]"])
        manager = MCPToolManager(api_key="test-key", model="test-model")
        manager._client = client
        items = [{"id": str(i), "content": f"文档{i}"} for i in range(3)]
        await manager._rerank("查询", items, top_k=1)
        assert client.messages.calls[0]["max_tokens"] >= 8192

    asyncio.run(scenario())


def test_query_rewrite_and_rerank_have_json_output_budget():
    async def scenario():
        client = RecordingClient(
            [
                json.dumps(["子查询一", "子查询二", "子查询三"]),
                "[0, 1, 2]",
            ]
        )
        manager = MCPToolManager(api_key="test-key", model="test-model")
        manager._client = client

        queries = await manager.rewrite_query("星桥计划怎么办理", n=3)
        assert queries[0] == "星桥计划怎么办理"

        items = [{"id": str(i), "content": f"文档{i}"} for i in range(3)]
        reranked, was_reranked = await manager._rerank("查询", items, top_k=1)

        assert was_reranked is True
        assert len(reranked) == 1
        for call in client.messages.calls:
            assert_json_budget(call)

    asyncio.run(scenario())


def test_query_rewrite_accepts_multiple_json_arrays():
    """模型偶尔返回多个并列数组，不能因此丢弃全部子查询。"""
    async def scenario():
        client = RecordingClient(
            ['["子查询一"] ["子查询二"] ["子查询三"]']
        )
        manager = MCPToolManager(api_key="test-key", model="test-model")
        manager._client = client

        queries = await manager.rewrite_query("原始查询", n=3)

        assert queries == ["原始查询", "子查询一", "子查询二", "子查询三"]

    asyncio.run(scenario())


def test_intent_json_has_output_budget():
    async def scenario():
        client = RecordingClient(
            [
                json.dumps(
                    {
                        "intent": "refund",
                        "confidence": 0.9,
                        "reasoning": "用户询问退款",
                    }
                )
            ]
        )
        recognizer = object.__new__(IntentRecognizer)
        recognizer.client = client
        recognizer.model = "test-model"

        result = await recognizer._llm_recognize("退款多久到账", None)

        assert result["intent"].value == "refund"
        assert_json_budget(client.messages.calls[0])

    asyncio.run(scenario())


class BlockClient:
    """按块返回响应，用于模拟只含 thinking 的截断输出。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._responses.pop(0))


def test_intent_retries_once_when_only_thinking_returned():
    """意图请求只返回 thinking 块时必须重试，不能直接降级为 OTHER。"""
    async def scenario():
        client = BlockClient(
            [
                [SimpleNamespace(type="thinking", thinking="推理未结束")],
                [
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(
                            {
                                "intent": "refund",
                                "confidence": 0.9,
                                "reasoning": "用户询问退款",
                            }
                        ),
                    )
                ],
            ]
        )
        recognizer = object.__new__(IntentRecognizer)
        recognizer.client = client
        recognizer.model = "test-model"

        result = await recognizer._llm_recognize("退款多久到账", None)

        assert result["intent"].value == "refund"
        assert len(client.calls) == 2
        assert client.calls[0]["max_tokens"] >= 2048

    asyncio.run(scenario())


def test_judge_json_has_output_budget():
    async def scenario():
        client = RecordingClient(
            [
                json.dumps(
                    {
                        "relevance": 0.9,
                        "accuracy": 0.8,
                        "completeness": 0.7,
                        "helpfulness": 0.85,
                    }
                )
            ]
        )
        scores = await LLMJudge(client, "test-model").judge("问题", "回答")

        assert scores.judge_failed is False
        assert scores.overall > 0
        assert_json_budget(client.messages.calls[0])

    asyncio.run(scenario())


def test_profile_json_has_output_budget():
    async def scenario():
        client = RecordingClient(
            [json.dumps({"preferences": ["夜间联系"], "entities": {}})]
        )
        manager = object.__new__(MemoryManager)
        manager._client = client
        manager._model = "test-model"

        async def fake_working_memory(user_id, conv_id):
            return [
                SimpleNamespace(
                    role=SimpleNamespace(value="user"),
                    content="请晚上联系我",
                )
            ]

        async def fake_profile(user_id):
            return {}

        class FakeProfile:
            def delete(self, ids):
                return None

            def add(self, **kwargs):
                return None

        manager._get_working_memory = fake_working_memory
        manager._get_profile = fake_profile
        manager._profile = FakeProfile()

        await manager.update_profile("t8_user", "t8_conv")

        assert_json_budget(client.messages.calls[0])
        assert client.messages.calls[0]["max_tokens"] >= 2048

    asyncio.run(scenario())


def test_composer_has_output_budget_for_long_agent_outputs():
    """两个 Agent 的长答复会先触发推理，预算不足时只会返回 thinking 块。"""
    async def scenario():
        client = RecordingClient(["合并后的回复"])
        composer = ResponseComposer(client, "test-model")
        responses = [
            AgentResponse(
                agent_type=AgentType.TECHNICAL,
                content="接口 401 排查步骤" * 40,
                success=True,
            ),
            AgentResponse(
                agent_type=AgentType.BILLING,
                content="账单核对字段" * 40,
                success=True,
            ),
        ]

        text = await composer.compose(
            Request(message="接口 401，同时扣款金额不符", user_id="u", conv_id="c"),
            responses,
        )

        assert text == "合并后的回复"
        assert client.messages.calls[0]["max_tokens"] >= 8192

    asyncio.run(scenario())


def test_summary_merge_has_output_budget():
    async def scenario():
        client = RecordingClient(["合并后的摘要"])
        manager = object.__new__(MemoryManager)
        manager._client = client
        manager._model = "test-model"

        merged = await manager._merge_summary("旧摘要", "新摘要")

        assert merged == "合并后的摘要"
        assert_json_budget(client.messages.calls[0])

    asyncio.run(scenario())
