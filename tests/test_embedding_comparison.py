"""U04 数据、编码配置和采用门槛的离线回归。"""
from __future__ import annotations

import argparse
import hashlib
import io
import json

import numpy as np
import pytest

from evaluation.run_embedding_comparison import (
    INSTRUCTION, ROUNDS, check_vectors, load_inputs, query_text, recommend,
    run, score_rows, verify_qwen, prepare_model,
)


def test_fixed_chinese_inputs_keep_original_splits_and_support_separate():
    passages, queries = load_inputs()
    assert len(passages) == 35 and len(queries) == 60
    assert {s: sum(q["split"] == s for q in queries) for s in ("dev", "test", "support_smoke")} == {
        "dev": 24, "test": 24, "support_smoke": 12,
    }
    assert sum(q["answerability"] == "answerable" for q in queries) == 52
    assert sum(q["answerability"] == "needs_tool" for q in queries) == 4
    assert all(q["language"] == "zh" for q in queries)


def test_query_instruction_is_fixed_and_document_text_is_not_rewritten():
    query = "Wireless Earbuds 连接不上，要怎么办？"
    assert query_text(query) == f"Instruct: {INSTRUCTION}\nQuery: {query}"
    assert query_text(query).count(query) == 1


def test_only_answerable_queries_enter_metrics_and_multi_group_needs_both():
    _, queries = load_inputs()
    ranking = [list(q["qrels"])[:1] if q["answerability"] == "answerable" else ["T17.E01"] for q in queries]
    result = score_rows(queries, ranking, {p["evidence_id"] for p in load_inputs()[0]})
    assert result["dev"]["answerable_total"] == 20
    assert result["dev"]["diagnostic_total"] == 4
    support = {r["query_id"]: r for r in result["support_smoke"]["per_query"]}
    assert support["U04-S11"]["all_evidence_hit@5"] == 0
    assert support["U04-S11"]["recall@5"] == 0.5
    assert all("score" not in q for q in result["dev"]["diagnostics"])
    with pytest.raises(ValueError, match="评分分母"):
        score_rows(queries, ranking[:-1], {p["evidence_id"] for p in load_inputs()[0]})


@pytest.mark.parametrize("problem", ["dimension", "nan", "zero_norm", "rounds"])
def test_invalid_vectors_are_rejected_before_indexing(problem):
    documents = np.array([[1.0, 0.0]], dtype=np.float32)
    queries = np.tile(documents, (ROUNDS, 1, 1))
    if problem == "dimension": documents = np.zeros((1, 3))
    if problem == "nan": documents[0, 0] = np.nan
    if problem == "zero_norm": documents[0] = 0
    if problem == "rounds": queries = queries[:1]
    with pytest.raises(ValueError):
        check_vectors(documents, queries, 1, 1, 2)


def _result(aeh, recall):
    return {"metrics": {s: {"all_evidence_hit@5": aeh, "recall@5": recall, "protocol_clean": True}
                        for s in ("dev", "test", "support_smoke")},
            "encoding": {"private_peak_mib": 2500}, "warm_latency": {"p95_ms": 400}}


def test_candidate_requires_quality_cost_and_compatibility_not_model_name():
    a, b = _result(0.5, 0.6), _result(0.7, 0.8)
    decision = recommend(a, b)
    assert decision["candidate_passed"]
    assert decision["runtime_default"] == "minilm"
    assert decision["deployment"] == "NOT_INTEGRATED_TOKENIZER_CONFLICT"
    b["warm_latency"]["p95_ms"] = 1001
    assert not recommend(a, b)["candidate_passed"]
    b["warm_latency"]["p95_ms"] = 100
    b["encoding"]["private_peak_mib"] = 3073
    assert not recommend(a, b)["candidate_passed"]
    b = _result(0.7, 0.8)
    b["metrics"]["test"]["all_evidence_hit@5"] = 0.4
    assert not recommend(a, b)["candidate_passed"]


def test_existing_output_is_not_overwritten_or_deleted(tmp_path):
    kept = tmp_path / "record.json"
    kept.write_text("prior evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="非空"):
        run(argparse.Namespace(output=str(tmp_path)))
    assert kept.read_text(encoding="utf-8") == "prior evidence"


def test_unavailable_chroma_records_error_and_never_claims_completion(tmp_path, monkeypatch):
    def unavailable(**kwargs):
        raise ConnectionError("offline chroma")

    monkeypatch.setattr("chromadb.HttpClient", unavailable)
    output = tmp_path / "new-run"
    with pytest.raises(ConnectionError):
        run(argparse.Namespace(output=str(output), chroma_host="127.0.0.1", chroma_port=8002))
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["status"] == "RUN_ERROR"
    assert not (output / "summary.json").exists()


def test_qwen_mismatched_identity_is_rejected_without_loading_weights(tmp_path):
    (tmp_path / "model-manifest.json").write_text(json.dumps({"model": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="身份"):
        verify_qwen(tmp_path)


def _small_model(monkeypatch):
    import evaluation.run_embedding_comparison as module

    bodies = {name: name.encode() for name in module.MODEL_FILES}
    monkeypatch.setattr(module, "MODEL_FILES", {name: (len(body), hashlib.sha256(body).hexdigest()) for name, body in bodies.items()})
    monkeypatch.setattr(module, "WEIGHT_SHA256", hashlib.sha256(bodies["model.safetensors"]).hexdigest())
    return bodies


def test_explicit_prepare_creates_verified_manifest_and_reuses_cache(tmp_path, monkeypatch):
    bodies = _small_model(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", lambda url, **kwargs: io.BytesIO(bodies[url.rsplit("/", 1)[-1]]))
    manifest = prepare_model(tmp_path)
    assert len(manifest["files"]) == 5

    def offline(*args, **kwargs):
        raise AssertionError("缓存完整时不能联网")

    monkeypatch.setattr("urllib.request.urlopen", offline)
    assert prepare_model(tmp_path) == manifest
    (tmp_path / "config.json").write_bytes(b"broken")
    with pytest.raises(ValueError, match="哈希"):
        prepare_model(tmp_path)
    assert (tmp_path / "config.json").read_bytes() == b"broken"


def test_bad_download_never_becomes_a_model_file(tmp_path, monkeypatch):
    _small_model(monkeypatch)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="校验失败"):
        prepare_model(tmp_path)
    assert list(tmp_path.iterdir()) == []
