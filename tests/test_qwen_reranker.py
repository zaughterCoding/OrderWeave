"""重排评分协议、完整排序与异常回退回归。"""
import asyncio
from types import SimpleNamespace

import pytest

from mcp.qwen_reranker import QwenReranker, validate_scores, passage_text
from mcp.tool_manager import MCPToolManager


@pytest.mark.parametrize("scores", [[0.1], [True, 0.2], [float("nan"), 0], [float("inf"), 0], [-1, 0], [1.1, 0], None])
def test_reject_invalid_scores(scores):
    with pytest.raises(ValueError):
        validate_scores(scores, 2)


def test_local_scores_preserve_metadata_and_ties():
    class Local:
        def score(self, query, items):
            return [0.1, 0.9, 0.9]

    manager = MCPToolManager("test", model="test", client=SimpleNamespace(), reranker=Local())
    items = [{"evidence_id": str(i), "content": str(i), "source_refs": [{"id": "source"}]} for i in range(3)]
    diag = {}
    result, applied = asyncio.run(manager._rerank("政策", items, 2, diag))
    assert applied and [i["evidence_id"] for i in result] == ["1", "2"]
    assert result[0]["source_refs"] == items[1]["source_refs"]
    assert "rerank_score" not in items[1]
    assert diag["rerank_complete"] and diag["rerank_order"] == [1, 2, 0]


def test_local_failure_never_calls_paid_model():
    class Local:
        def score(self, query, items):
            raise TimeoutError("test timeout")

    manager = MCPToolManager("test", model="test", client=SimpleNamespace(), reranker=Local())
    items = [{"content": str(i)} for i in range(3)]
    diag = {}
    result, applied = asyncio.run(manager._rerank("政策", items, 2, diag))
    assert result == items[:2] and not applied
    assert diag["rerank_status"] == "failed"


@pytest.mark.parametrize("order", [[0, 0, 1], [0, 1], [True, 1, 2], [0, 1, 4], [0, 1, 2, 3], [0, 1, "2"]])
def test_llm_partial_duplicate_and_invalid_orders_fall_back(order):
    import json

    class Messages:
        async def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(order))])

    manager = MCPToolManager("test", model="test", client=SimpleNamespace(messages=Messages()))
    items = [{"content": str(i)} for i in range(3)]
    diag = {}
    result, applied = asyncio.run(manager._rerank("政策", items, 2, diag))
    assert result == items[:2] and not applied
    assert diag["rerank_raw_order"] == order and not diag["rerank_complete"]


def test_complete_llm_order_and_empty_inputs():
    class Messages:
        async def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="[2, 1, 0]")])

    manager = MCPToolManager("test", model="test", client=SimpleNamespace(messages=Messages()))
    items = [{"content": str(i)} for i in range(3)]
    result, applied = asyncio.run(manager._rerank("政策", items, 2, {}))
    assert applied and result == [items[2], items[1]]
    local = QwenReranker()
    try:
        assert local.score("政策", []) == []
        with pytest.raises(ValueError):
            local.score("", items)
        with pytest.raises(ValueError):
            local.score("政策", [{"title": "没有正文"}])
        assert local._process is None
        assert passage_text({"title": "标题", "content": "正文"}) == "标题\n正文"
    finally:
        local.close()


def test_default_hybrid_and_explicit_retrieval_modes(monkeypatch):
    from dataclasses import replace
    from agents.retail_runtime import RetailRuntimeConfig, build_retail_runtime

    assert RetailRuntimeConfig.__dataclass_fields__["retrieval_mode"].default == "hybrid"
    assert RetailRuntimeConfig.__dataclass_fields__["reranker_model"].default == "qwen"
    for name, value in {"ANTHROPIC_API_KEY": "test", "LLM_MODEL": "test",
                        "RETAIL_INPUT_CNY_PER_MTOK": "1", "RETAIL_OUTPUT_CNY_PER_MTOK": "1",
                        "RETAIL_SUPPORT_ENABLED": "false"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("RETAIL_RETRIEVAL_MODE", raising=False)
    monkeypatch.delenv("RETAIL_RERANKER_MODEL", raising=False)
    config = RetailRuntimeConfig.from_env()
    assert config.retrieval_mode == "hybrid"
    assert config.reranker_model == "qwen"
    with pytest.raises(ValueError, match="RETAIL_RERANKER_MODEL"):
        build_retail_runtime(replace(config, reranker_model="invalid"))
    monkeypatch.setenv("RETAIL_RETRIEVAL_MODE", "dense")
    monkeypatch.setenv("RETAIL_RERANKER_MODEL", "llm")
    assert RetailRuntimeConfig.from_env().retrieval_mode == "dense"
    assert RetailRuntimeConfig.from_env().reranker_model == "llm"


def test_reranker_timeout_closes_worker():
    local = QwenReranker(timeout=0.001)
    process = SimpleNamespace(terminated=False,
                              stdin=SimpleNamespace(write=lambda x: None, flush=lambda: None, close=lambda: None),
                              stdout=SimpleNamespace(close=lambda: None), poll=lambda: None, wait=lambda timeout: None)
    process.terminate = lambda: setattr(process, "terminated", True)
    local._process = process
    with pytest.raises(TimeoutError):
        local.score("政策", [{"content": "正文"}])
    assert local._closed and process.terminated


def test_reranker_missing_model_never_downloads(tmp_path):
    from mcp.qwen_reranker import verify_model

    with pytest.raises(RuntimeError, match="缓存缺失"):
        verify_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reranker_missing_environment_fails_and_stays_closed(tmp_path):
    local = QwenReranker(python=tmp_path / "missing.exe")
    with pytest.raises(RuntimeError, match="独立推理环境"):
        local.score("政策", [{"content": "正文"}])
    assert local._closed
    with pytest.raises(RuntimeError, match="已关闭"):
        local.score("政策", [{"content": "正文"}])


def test_runtime_start_failure_closes_both_local_models():
    from agents.retail_runtime import RetailRuntime

    seen = []

    async def close_mcp():
        seen.append("mcp")

    async def close_client():
        seen.append("client")

    def broken_score(query, items):
        raise RuntimeError("模型启动失败")

    runtime = RetailRuntime.__new__(RetailRuntime)
    runtime._closed, runtime._started = False, False
    runtime.config = SimpleNamespace(enable_knowledge=True, embedding_model="qwen")
    runtime.knowledge_base = SimpleNamespace(_embedding_function=lambda texts: None,
                                            close=lambda: seen.append("embedding"))
    runtime.tool_manager = SimpleNamespace(_reranker=SimpleNamespace(score=broken_score, close=lambda: seen.append("reranker")))
    runtime.mcp_client = SimpleNamespace(close=close_mcp)
    runtime.memory = None
    runtime.client = SimpleNamespace(aclose=close_client)
    with pytest.raises(RuntimeError, match="模型启动失败"):
        asyncio.run(runtime.start())
    assert runtime._closed and seen == ["mcp", "embedding", "reranker", "client"]
