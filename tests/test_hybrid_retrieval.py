"""混合召回的词法、排名、前置范围及缓存隔离回归。"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.retail_tools import retail_evidence_applicable
from agents.retail_tools import retail_knowledge_tool
from mcp.hybrid_retrieval import BM25Index, reciprocal_rank_fusion, tokenize
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool

ROOT = Path(__file__).resolve().parents[1]


def passages():
    return [json.loads(line) for name in ("retail_sandbox", "retail_support")
            for line in (ROOT / "evaluation/data" / name / "knowledge.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def active_packet():
    return {"enabled": True, "supported": True, "scope_state": "active",
            "product_id": "9924732112", "issue": "bluetooth_pairing"}


def test_tokenization_keeps_identifiers_and_shared_bilingual_aliases():
    assert tokenize("Wireless Earbuds BLUETOOTH") == tokenize("无线耳机 蓝牙")
    assert tokenize("ＰＡＩＲ-０１ 9924732112 #W1067251") == ["pair-01", "9924732112", "w1067251"]
    assert tokenize("退款去向") == ["退款", "款去", "去向"]
    assert tokenize("退") == ["退"]


def test_bm25_exact_id_length_normalization_zero_match_and_scope():
    docs = [{"evidence_id": "short", "content": "PAIR-01"},
            {"evidence_id": "long", "content": "PAIR-01 " + "unrelated " * 40}]
    index = BM25Index(docs)
    assert [r["evidence_id"] for r in index.search("PAIR-01", 2, set(index.items))] == ["short", "long"]
    assert index.search("nonexistent", 2, set(index.items)) == []
    assert index.search("PAIR-01", 1, {"long"})[0]["evidence_id"] == "long"
    with pytest.raises(ValueError, match="重复"):
        BM25Index([docs[0], docs[0]])


def test_rrf_merges_ranks_once_per_channel_with_stable_ties():
    a, b, c = [{"evidence_id": x} for x in "abc"]
    rows = reciprocal_rank_fusion([a, b], [b, c], 3)
    assert [r["evidence_id"] for r in rows] == ["b", "a", "c"]
    assert rows[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61)
    assert len(reciprocal_rank_fusion([a, a], [], 5)) == 1
    assert reciprocal_rank_fusion([], [], 5) == []


@pytest.mark.parametrize("field,value", [("product_id", "other"), ("issue", "battery"),
    ("support_version", "t17.0"), ("source_kind", "vendor_manual"), ("scope", "other"),
    ("policy_release", "old"), ("language", "en"), ("product_id", "")])
def test_support_metadata_mismatch_is_rejected(field, value):
    support = passages()[30]
    assert retail_evidence_applicable(support, active_packet(), "无线耳机配对")
    assert not retail_evidence_applicable({**support, field: value}, active_packet(), "无线耳机配对")
    assert not retail_evidence_applicable(support, None, "Wireless Earbuds 配对")


def test_category_scope_keeps_global_unknown_and_multi_category_policies():
    rows = {r["evidence_id"]: r for r in passages()}
    assert retail_evidence_applicable(rows["K01.P01"], None, "相机怎么退")
    assert not retail_evidence_applicable(rows["K08.P01"], None, "相机怎么退")
    assert retail_evidence_applicable(rows["K08.P01"], None, "准备清单怎么核对")
    assert retail_evidence_applicable(rows["K08.P01"], None, "相机和吸尘器一起退")
    assert retail_evidence_applicable(rows["K09.P01"], None, "T恤和吸尘器一起退")
    assert not retail_evidence_applicable({**rows["K01.P01"], "policy_commit": "old"}, None, "怎么退")


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_prefilter_happens_before_each_channel_top_k(monkeypatch, mode):
    docs = [{"evidence_id": f"id{i:02}", "content": "PAIR-01"} for i in range(25)]
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb.list_ids = lambda: [r["evidence_id"] for r in docs]
    kb.configure_retrieval(docs, mode)
    calls = []
    def dense(query, k, allowed):
        calls.append(allowed)
        return [{**docs[24], "score": 1}]
    monkeypatch.setattr(kb, "_dense_search", dense)
    found = kb.search("PAIR-01", top_k=1, item_filter=lambda r: r["evidence_id"] == "id24")
    assert found[0]["evidence_id"] == "id24"
    assert calls == ([] if mode == "bm25" else [{"id24"}])
    assert kb.search("PAIR-01", item_filter=lambda r: False) == []


def test_filter_context_is_forwarded_and_does_not_read_or_poison_cache():
    manager = MCPToolManager(api_key="test-key", client=SimpleNamespace(), model="fake")
    seen = []
    async def handler(params, context):
        predicate = (context or {}).get("knowledge_item_filter")
        seen.append(bool(predicate))
        return [r for r in [{"evidence_id": "a"}, {"evidence_id": "b"}] if predicate is None or predicate(r)]
    manager.register(Tool(name="knowledge_search", description="test", handler=handler, schema={}, cache_ttl=60))
    async def rewrite(*args, **kwargs): return ["same"]
    async def rerank(query, rows, k, diagnostics=None): return rows[:k], False
    manager.rewrite_query, manager._rerank = rewrite, rerank
    async def scenario():
        await manager.call("knowledge_search", {"query": "same", "top_k": 20})
        for key in ("a", "b"):
            result = await manager.search_with_rewrite("knowledge_search", "same", top_k=1,
                item_filter=lambda r: r["evidence_id"] == key)
            assert [r["evidence_id"] for r in result.data] == [key]
        result = await manager.call("knowledge_search", {"query": "same", "top_k": 20})
        assert len(result.data) == 2 and result.cached
    asyncio.run(scenario())
    assert seen == [False, True, True]


def test_prefilter_does_not_expand_ordinary_recall_budget():
    class Manager:
        async def search_with_rewrite(self, name, query, top_k, item_filter, expand_filtered_recall):
            assert expand_filtered_recall is False
            return SimpleNamespace(success=True, data=[], reranked=False)
    spec = retail_knowledge_tool(Manager(), support_enabled=False)
    asyncio.run(spec.handler(SimpleNamespace(message="退货政策"), {"query": "退货政策"}))


def test_model_query_cannot_widen_original_request_scope():
    rows = {r["evidence_id"]: r for r in passages()}
    class Manager:
        async def search_with_rewrite(self, name, query, top_k, item_filter):
            assert query == "Wireless Earbuds PAIR-01"
            assert not item_filter(rows["T17.E02"])
            assert not item_filter(rows["K07.P01"])
            assert item_filter(rows["K08.P01"])
            return SimpleNamespace(success=True, data=[], reranked=False)
    request = SimpleNamespace(message="我的烤箱需要寄回前清点什么", retail_support=None)
    spec = retail_knowledge_tool(Manager(), support_enabled=True)
    asyncio.run(spec.handler(request, {"query": "Wireless Earbuds PAIR-01"}))


def test_ablation_rejects_invalid_scope_gold_and_quality_regression():
    from copy import deepcopy
    from evaluation.run_hybrid_retrieval import select, validate_scope
    with pytest.raises(ValueError, match="gold"):
        validate_scope([{"query_id": "x", "allowed_ids": [], "qrels": {"id": 2},
                         "required_evidence_groups": [["id"]]}], [{"evidence_id": "id"}])
    dense = {"metrics": {split: {"all_evidence_hit@5": 0.9, "recall@5": 0.9, "protocol_clean": True}
                         for split in ("dev", "test", "support_smoke")},
             "ranking_stable": True, "warm_latency": {"p95_ms": 200}}
    hybrid = deepcopy(dense)
    hybrid["metrics"]["dev"]["all_evidence_hit@5"] = 1
    scope = {key: {"wrong_scope_hits": 0, "gold_filter_losses": 0, "ranking_stable": True,
                   "metrics": {"scope_smoke": {"protocol_clean": True}}} for key in ("dense", "bm25", "hybrid")}
    quality = {"dense": dense, "bm25": dense, "hybrid": hybrid}
    assert select(quality, scope)["recommended_default"] == "hybrid"
    hybrid["metrics"]["test"]["recall@5"] = 0.8
    assert select(quality, scope)["recommended_default"] == "dense"


def test_ablation_connection_failure_is_explicit_and_does_not_overwrite(tmp_path, monkeypatch):
    from evaluation import run_hybrid_retrieval as runner
    monkeypatch.setattr(runner.chromadb, "HttpClient", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="offline"):
        runner.run(output, "127.0.0.1", 1)
    assert json.loads((output / "run.json").read_text())["status"] == "RUN_ERROR"
    assert not (output / "summary.json").exists()
    with pytest.raises(FileExistsError):
        runner.run(output, "127.0.0.1", 1)


def test_retail_search_endpoint_uses_original_query_scope(monkeypatch):
    import api.main as api
    rows = {r["evidence_id"]: r for r in passages()}
    class Manager:
        async def search_with_rewrite(self, name, query, top_k, item_filter, expand_filtered_recall):
            assert not expand_filtered_recall
            assert not item_filter(rows["T17.E02"])
            assert item_filter(rows["K08.P01"])
            return SimpleNamespace(data=[], reranked=False)
    monkeypatch.setattr(api, "_retail_runtime", SimpleNamespace(config=SimpleNamespace(support_enabled=True)))
    monkeypatch.setattr(api, "_tool_manager", Manager())
    asyncio.run(api.search("烤箱寄回前准备什么", 5))
