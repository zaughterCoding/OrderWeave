import asyncio
import json
from types import SimpleNamespace

import pytest

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


class _EvidenceCollection:
    """只实现证据装载和检索所需的最小集合接口。"""

    def __init__(self):
        self.documents = {}
        self.metadatas = {}
        self.upsert_calls = 0

    def get(self, ids, include=None):
        present = [evidence_id for evidence_id in ids if evidence_id in self.documents]
        return {
            "ids": present,
            "documents": [self.documents[evidence_id] for evidence_id in present],
            "metadatas": [self.metadatas[evidence_id] for evidence_id in present],
        }

    def upsert(self, ids, documents, metadatas):
        self.upsert_calls += 1
        for evidence_id, document, metadata in zip(ids, documents, metadatas):
            self.documents[evidence_id] = document
            self.metadatas[evidence_id] = metadata

    def query(self, query_texts, n_results):
        ids = list(self.documents)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.documents[evidence_id] for evidence_id in ids]],
            "metadatas": [[self.metadatas[evidence_id] for evidence_id in ids]],
            "distances": [[0.1] * len(ids)],
        }


def _evidence_passage(content="只有已送达订单才能申请退货。"):
    return {
        "evidence_id": "K01.P01",
        "doc_id": "K01",
        "title": "退货条件",
        "content": content,
        "language": "zh",
        "source_kind": "project_authored",
        "source_refs": [{"url": "https://example.com/policy", "section": "退货"}],
        "policy_release": "v1.0.1",
        "policy_commit": "fc0055dc",
        "scope": "project_zh",
        "category": "all",
    }


def test_evidence_upsert_is_idempotent_and_preserves_scalar_metadata():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = _EvidenceCollection()

    assert kb.upsert_evidence([_evidence_passage()]) == 1
    assert kb.upsert_evidence([_evidence_passage()]) == 0

    metadata = kb._collection.metadatas["K01.P01"]
    assert metadata["evidence_id"] == "K01.P01"
    assert metadata["policy_release"] == "v1.0.1"
    assert json.loads(metadata["source_refs_json"])[0]["section"] == "退货"
    assert kb._collection.upsert_calls == 1


def test_evidence_conflict_is_rejected_before_writing():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = _EvidenceCollection()
    kb.upsert_evidence([_evidence_passage()])

    with pytest.raises(ValueError, match="正文"):
        kb.upsert_evidence([_evidence_passage("另一段正文")])

    changed = _evidence_passage()
    changed["policy_release"] = "v1.0.2"
    with pytest.raises(ValueError, match="元数据"):
        kb.upsert_evidence([changed])

    assert kb._collection.upsert_calls == 1


def test_search_exposes_stable_evidence_identity_and_source():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = _EvidenceCollection()
    kb.upsert_evidence([_evidence_passage()])

    result = kb.search("退货条件")

    assert result[0]["id"] == "K01.P01"
    assert result[0]["evidence_id"] == "K01.P01"
    assert result[0]["doc_id"] == "K01"
    assert result[0]["policy_release"] == "v1.0.1"
    assert result[0]["source_refs"][0]["section"] == "退货"


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


def test_same_fragment_id_with_different_scores_is_merged_once(monkeypatch):
    manager, messages = make_manager(
        monkeypatch, ['["第二查询"]', "[0, 1]"]
    )

    async def handler(params, context):
        if params["query"] == "第二查询":
            return [
                {
                    "id": "doc-1",
                    "title": "退款政策",
                    "content": "7 天内可退款",
                    "score": 0.8,
                }
            ]
        return [
            {
                "id": "doc-1",
                "title": "退款政策",
                "content": "7 天内可退款",
                "score": 0.9,
            },
            {
                "id": "doc-2",
                "title": "发票说明",
                "content": "发票可在账户中心下载",
                "score": 0.7,
            },
        ]

    manager.register(knowledge_tool(handler))

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款", top_k=1)
    )

    assert result.success is True
    assert [item["id"] for item in result.data] == ["doc-1"]
    assert result.data[0]["score"] == 0.9
    rerank_prompt = messages.calls[1]["messages"][0]["content"]
    assert rerank_prompt.count('"id": "doc-1"') == 1


def test_same_title_with_different_fragment_ids_is_kept(monkeypatch):
    manager, _ = make_manager(monkeypatch, ['["第二查询"]'])

    async def handler(params, context):
        score = 0.9 if params["query"] == "退款" else 0.8
        return [
            {
                "id": "doc-a",
                "title": "退款政策",
                "content": "7 天内可退款",
                "score": score,
            },
            {
                "id": "doc-b",
                "title": "退款政策",
                "content": "7 天内可退款",
                "score": score,
            },
        ]

    manager.register(knowledge_tool(handler))

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款", top_k=5)
    )

    ids = [item["id"] for item in result.data]
    assert len(ids) == 2
    assert set(ids) == {"doc-a", "doc-b"}


def test_items_without_id_merge_by_stable_fragment_fields(monkeypatch):
    manager, _ = make_manager(monkeypatch, ['["第二查询"]'])

    async def handler(params, context):
        score = 0.9 if params["query"] == "退款" else 0.8
        return [
            {
                "title": "退款政策",
                "content": "7 天内可退款",
                "chunk": 0,
                "score": score,
            }
        ]

    manager.register(knowledge_tool(handler))

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款", top_k=5)
    )

    assert len(result.data) == 1
    assert result.data[0]["score"] == 0.9


def test_dedupe_keeps_top_k_slots_for_distinct_fragments(monkeypatch):
    manager, messages = make_manager(
        monkeypatch, ['["第二查询"]', "[0, 1, 2]"]
    )

    async def handler(params, context):
        second = params["query"] == "第二查询"
        if second:
            return [
                {"id": "doc-a", "title": "A", "content": "A", "score": 0.8},
                {"id": "doc-c", "title": "C", "content": "C", "score": 0.7},
                {"id": "doc-d", "title": "D", "content": "D", "score": 0.6},
            ]
        return [
            {"id": "doc-a", "title": "A", "content": "A", "score": 0.9},
            {"id": "doc-b", "title": "B", "content": "B", "score": 0.9},
        ]

    manager.register(knowledge_tool(handler))

    result = asyncio.run(
        manager.search_with_rewrite("knowledge_search", "退款", top_k=3)
    )

    ids = [item["id"] for item in result.data]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    rerank_prompt = messages.calls[1]["messages"][0]["content"]
    assert rerank_prompt.count('"id": "doc-a"') == 1
