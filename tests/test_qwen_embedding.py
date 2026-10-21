"""Qwen 协议、查询编码和索引身份的离线回归。"""
from types import SimpleNamespace

import pytest

from evaluation import run_embedding_comparison as comparison
from mcp import knowledge_base as kb_module
from mcp.qwen_embedding import (
    IDENTITY, INSTRUCTION, MODEL_FILES, REVISION, QwenEmbeddingFunction,
    validate_vectors, verify_model,
)


def test_production_model_matches_compared_candidate():
    assert MODEL_FILES == comparison.MODEL_FILES
    assert REVISION == comparison.REVISION
    assert INSTRUCTION == comparison.INSTRUCTION


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[float("nan")] * 1024], [[0.0] * 1024]])
def test_invalid_worker_vectors_are_rejected(vectors):
    with pytest.raises(ValueError):
        validate_vectors(vectors, 1)


def test_missing_model_fails_without_download(tmp_path):
    with pytest.raises(RuntimeError, match="缓存缺失"):
        verify_model(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_worker_failure_closes_encoder_without_fallback(tmp_path):
    encoder = QwenEmbeddingFunction(python=tmp_path / "missing.exe")
    with pytest.raises(RuntimeError, match="独立环境缺失"):
        encoder(["文档"])
    assert encoder._closed
    with pytest.raises(RuntimeError, match="已关闭"):
        encoder.encode_query("问题")
    encoder.close()


def test_document_encoder_returns_chroma_numpy_vectors(monkeypatch):
    encoder = QwenEmbeddingFunction()
    monkeypatch.setattr(encoder, "_encode", lambda texts, query: [[1.0] + [0.0] * 1023])
    try:
        encoded = encoder(["文档"])
        assert encoded.shape == (1, 1024)
        assert encoded[0].tolist() == [1.0] + [0.0] * 1023
    finally:
        encoder.close()


def test_worker_timeout_stops_process(monkeypatch):
    encoder = QwenEmbeddingFunction(timeout=0.001)
    process = SimpleNamespace(terminated=False, stdin=SimpleNamespace(write=lambda x: None, flush=lambda: None, close=lambda: None),
                              stdout=SimpleNamespace(close=lambda: None), poll=lambda: None, wait=lambda timeout: None)
    process.terminate = lambda: setattr(process, "terminated", True)
    encoder._process = process
    with pytest.raises(RuntimeError, match="响应超时"):
        encoder.encode_query("问题")
    assert encoder._closed and process.terminated


def test_query_uses_query_encoder_and_close_owns_worker():
    seen = []
    collection = SimpleNamespace(query=lambda **kwargs: seen.append(kwargs) or {"ids": [[]]})
    encoder = SimpleNamespace(encode_query=lambda query: [[0.5]], close=lambda: seen.append("closed"))
    kb = kb_module.KnowledgeBase.__new__(kb_module.KnowledgeBase)
    kb._collection, kb._embedding_function = collection, encoder
    assert kb.search("问题", top_k=3) == []
    assert seen == [{"query_embeddings": [[0.5]], "n_results": 3}]
    kb.close()
    assert seen[-1] == "closed"


def test_qwen_collection_identity_is_checked_before_use(monkeypatch):
    calls = []
    client = SimpleNamespace(heartbeat=lambda: None)
    client.get_or_create_collection = lambda **kwargs: calls.append(kwargs) or SimpleNamespace(metadata={})
    monkeypatch.setattr(kb_module.chromadb, "HttpClient", lambda **kwargs: client)
    encoder = QwenEmbeddingFunction()
    try:
        with pytest.raises(RuntimeError, match="身份不符"):
            kb_module.KnowledgeBase(collection_name="legacy", embedding_function=encoder, load_demo_documents=False)
        assert calls[0]["name"] == "legacy" + encoder.collection_suffix
        assert calls[0]["metadata"]["embedding_identity"] == IDENTITY
    finally:
        encoder.close()
