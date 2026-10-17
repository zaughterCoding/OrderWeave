import asyncio
from types import SimpleNamespace

from agents.agent_orchestrator import Request
from agents.tools import build_shared_rag_tools
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import CircuitBreaker, MCPToolManager, Tool, ToolResult


def make_manager(monkeypatch, responses):
    class Messages:
        def __init__(self):
            self.responses = list(responses)
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if not self.responses:
                raise AssertionError("unexpected model call")
            text = self.responses.pop(0)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])

    messages = Messages()
    monkeypatch.setattr(
        "mcp.tool_manager.AsyncAnthropic",
        lambda **kwargs: SimpleNamespace(messages=messages),
    )
    return MCPToolManager(api_key="test-key", model="test-model"), messages


def knowledge_tool(handler, *, fallback=None, cache_ttl=0.0, supports_rerank=True):
    return Tool(
        name="knowledge_search",
        description="test knowledge search",
        handler=handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        cache_ttl=cache_ttl,
        supports_rerank=supports_rerank,
        fallback=fallback,
    )


def test_repeated_document_import_is_idempotent():
    class FakeCollection:
        def __init__(self):
            self.calls = []

        def upsert(self, **kwargs):
            self.calls.append(kwargs)

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection()
    documents = [{"title": "退款政策", "content": "7 天内可以申请退款。"}]

    assert kb.add_documents(documents) == 1
    assert kb.add_documents(documents) == 1
    assert len(kb._collection.calls) == 2
    assert kb._collection.calls[0]["ids"] == kb._collection.calls[1]["ids"]


def test_search_returns_document_id():
    class FakeCollection:
        def query(self, **kwargs):
            return {
                "ids": [["doc-1"]],
                "documents": [["7 天内可以申请退款。"]],
                "metadatas": [[{"title": "退款政策", "chunk_index": 0}]],
                "distances": [[0.1]],
            }

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection()

    result = kb.search("退款")

    assert result[0]["id"] == "doc-1"
    assert result[0]["title"] == "退款政策"
    assert result[0]["score"] == 0.9


def test_tool_result_is_not_degraded_by_default():
    result = ToolResult(success=True, data=[], tool_name="knowledge_search")

    assert result.degraded is False


def test_search_with_rewrite_merges_deduplicates_and_reranks(monkeypatch):
    manager, messages = make_manager(
        monkeypatch,
        [
            '["如何申请退款", "退款需要多久", "退款政策"]',
            "[4, 0, 1, 2, 3]",
        ],
    )
    queries = []

    async def handler(params, context):
        query = params["query"]
        queries.append(query)
        return [
            {"title": "退款政策", "content": "7 天内可退款"},
            {"title": f"文档-{query}", "content": query},
        ]

    manager.register(knowledge_tool(handler, cache_ttl=60.0))

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款流程", top_k=3)
    )

    assert result.success is True
    assert result.degraded is False
    assert result.reranked is True
    assert len(result.data) == 3
    assert result.data[0]["title"] == "文档-退款政策"
    assert len(queries) == 4
    assert len(messages.calls) == 2

    cached = asyncio.run(
        manager.call("knowledge_search", {"query": "退款流程", "top_k": 5})
    )

    assert cached.success is True
    assert cached.cached is True
    assert len(queries) == 4


def test_degraded_fallback_is_not_reported_as_success(monkeypatch):
    manager, _ = make_manager(monkeypatch, ['["备用查询"]'])

    async def handler(params, context):
        raise RuntimeError("chroma down")

    def fallback(params, context, error):
        return [
            {
                "title": "知识库降级结果",
                "content": "知识库暂时不可用",
                "fallback": True,
                "error": error,
            }
        ]

    manager.register(knowledge_tool(handler, fallback=fallback))

    direct = asyncio.run(manager.call("knowledge_search", {"query": "退款"}))
    assert direct.success is False
    assert direct.degraded is True
    assert direct.data[0]["fallback"] is True

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款", top_k=3)
    )

    assert result.success is False
    assert result.degraded is True
    assert result.data == []
    assert "降级" in result.error


def test_missing_fallback_still_marks_result_degraded(monkeypatch):
    manager, _ = make_manager(monkeypatch, [])

    async def handler(params, context):
        raise RuntimeError("chroma down")

    manager.register(knowledge_tool(handler))

    result = asyncio.run(manager.call("knowledge_search", {"query": "退款"}))

    assert result.success is False
    assert result.degraded is True
    assert result.data is None


def test_open_circuit_returns_degraded_without_calling_handler(monkeypatch):
    manager, _ = make_manager(monkeypatch, [])
    calls = []

    async def handler(params, context):
        calls.append(params)
        raise RuntimeError("chroma down")

    tool = knowledge_tool(handler)
    tool.breaker = CircuitBreaker(failure_threshold=1)
    manager.register(tool)

    first = asyncio.run(manager.call("knowledge_search", {"query": "退款"}))
    second = asyncio.run(manager.call("knowledge_search", {"query": "退款"}))

    assert first.success is False
    assert second.success is False
    assert second.degraded is True
    assert "熔断" in second.error
    assert len(calls) == 1
    assert manager.get_stats()["knowledge_search"]["circuit_state"] == "open"


def test_shared_rag_tool_does_not_report_pure_degradation_as_success():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return ToolResult(
                success=False,
                data=[{"title": "知识库降级结果", "fallback": True}],
                tool_name=tool_name,
                error="chroma down",
                degraded=True,
            )

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    request = Request(message="退款", user_id="user-1", conv_id="conv-1")

    result = asyncio.run(spec.handler(request, {"query": "退款"}))

    assert result["success"] is False
    assert result["degraded"] is True
    assert result["results"] == []


def test_shared_rag_tool_exposes_partial_degradation():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return ToolResult(
                success=True,
                data=[{"title": "退款政策", "content": "7 天内可退款"}],
                tool_name=tool_name,
                reranked=True,
                degraded=True,
            )

    spec = build_shared_rag_tools(RagManager())["search_knowledge_base"]
    request = Request(message="退款", user_id="user-1", conv_id="conv-1")

    result = asyncio.run(spec.handler(request, {"query": "退款"}))

    assert result["success"] is True
    assert result["degraded"] is True
    assert result["results"][0]["title"] == "退款政策"
