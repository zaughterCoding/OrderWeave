"""U05 固定三路消融及合成范围干扰对照，只调用本地 Qwen。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
import traceback
import uuid

import chromadb

from agents.retail_tools import retail_evidence_applicable
from evaluation.retail_metrics import compute_retrieval_metrics, summarize_latency
from evaluation.run_embedding_comparison import INPUTS, load_inputs, sha256, write_json
from mcp.hybrid_retrieval import BM25_K1, BM25_B, CANDIDATE_K, RRF_CONSTANT
from mcp.knowledge_base import KnowledgeBase
from mcp.qwen_embedding import IDENTITY, QwenEmbeddingFunction

ROOT = Path(__file__).resolve().parents[1]
MODES = ("dense", "bm25", "hybrid")
FILES = (*INPUTS, "evaluation/data/hybrid_retrieval/scope_queries.jsonl",
         "evaluation/data/hybrid_retrieval/scope_decoys.jsonl", "mcp/hybrid_retrieval.py",
         "mcp/knowledge_base.py", "mcp/qwen_embedding.py", "mcp/tool_manager.py",
         "agents/retail_tools.py", "agents/retail_runtime.py", "evaluation/retail_metrics.py",
         "evaluation/run_hybrid_retrieval.py", "evaluation/embedding_requirements.lock")
SPEC = {"bm25_k1": BM25_K1, "bm25_b": BM25_B, "rrf_constant": RRF_CONSTANT,
        "candidate_k": CANDIDATE_K, "rounds": 3, "top_k": 5, "embedding_identity": IDENTITY,
        "acceptance": "all original slice AEH/Recall nondecreasing and one improves; filtered scope violations=0; P95<=Dense*1.25+10ms"}


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_scope(rows: list[dict], corpus: list[dict]) -> None:
    known = {row["evidence_id"] for row in corpus}
    if len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("范围题ID重复")
    for row in rows:
        allowed = set(row["allowed_ids"])
        groups = row["required_evidence_groups"]
        if (not allowed <= known or not groups or not row["qrels"]
                or not set(row["qrels"]) <= allowed
                or any(not group or not set(group) <= set(row["qrels"]) for group in groups)):
            raise ValueError("范围题gold或允许ID不完整")


def measure(kb: KnowledgeBase, queries: list[dict], filtered: bool, known: set[str]) -> dict:
    results = {mode: {"rankings": [], "latency_ms": []} for mode in MODES}
    for round_index in range(3):
        # 轮转执行顺序，三路各自保留独立计时，不复用Dense查询向量。
        for mode in MODES[round_index:] + MODES[:round_index]:
            kb.retrieval_mode = mode
            rankings, timings = [], []
            for query in queries:
                predicate = None
                if filtered:
                    predicate = lambda item, q=query: retail_evidence_applicable(item, q["support_packet"], q["query"])
                started = time.perf_counter()
                items = kb.search(query["query"], top_k=5, item_filter=predicate)
                timings.append((time.perf_counter() - started) * 1000)
                rankings.append([item["evidence_id"] for item in items])
            results[mode]["rankings"].append(rankings)
            results[mode]["latency_ms"].append(timings)
            print(f"{mode}: filtered={filtered} round={round_index + 1}/3 complete", flush=True)
    for mode, result in results.items():
        rankings = result["rankings"][0]
        result["ranking_stable"] = all(ranking == rankings for ranking in result["rankings"])
        result["metrics"] = {}
        for split in sorted({q["split"] for q in queries}):
            records = [{"query_id": q["query_id"], "answerability": q["answerability"],
                        "gold_qrels": q["qrels"], "required_groups": q["required_evidence_groups"],
                        "ranking": ranking} for q, ranking in zip(queries, rankings) if q["split"] == split]
            result["metrics"][split] = compute_retrieval_metrics(records, known_ids=known)
        result["first_round_latency"] = summarize_latency(result["latency_ms"][0])
        result["warm_latency"] = summarize_latency(result["latency_ms"][1] + result["latency_ms"][2])
        result["misses"] = [{"query_id": q["query_id"], "ranking": ranking,
                             "missing_groups": [group for group in q["required_evidence_groups"] if not set(group) & set(ranking)]}
                            for q, ranking in zip(queries, rankings) if q["answerability"] == "answerable"
                            and any(not set(group) & set(ranking) for group in q["required_evidence_groups"])]
        if "allowed_ids" in queries[0]:
            result["scope_violations"] = [{"query_id": q["query_id"],
                                           "wrong_ids": [key for key in ranking if key not in q["allowed_ids"]]}
                                          for q, ranking in zip(queries, rankings)
                                          if any(key not in q["allowed_ids"] for key in ranking)]
            result["wrong_scope_hits"] = sum(len(row["wrong_ids"]) for row in result["scope_violations"])
    return results


def select(quality: dict, filtered_scope: dict) -> dict:
    dense, hybrid = quality["dense"], quality["hybrid"]
    pairs = [(dense["metrics"][split][metric], hybrid["metrics"][split][metric])
             for split in ("dev", "test", "support_smoke") for metric in ("all_evidence_hit@5", "recall@5")]
    checks = {"original_slices_no_regression": all(b >= a - 1e-9 for a, b in pairs),
              "at_least_one_quality_gain": any(b > a + 1e-9 for a, b in pairs),
              "scope_violations_zero": all(result["wrong_scope_hits"] == 0 for result in filtered_scope.values()),
              "filtered_gold_preserved": filtered_scope["hybrid"]["gold_filter_losses"] == 0,
              "latency_within_dense_1_25_plus_10ms": hybrid["warm_latency"]["p95_ms"] <= dense["warm_latency"]["p95_ms"] * 1.25 + 10,
              "protocol_clean": all(result["ranking_stable"] and all(m["protocol_clean"] for m in result["metrics"].values())
                                    for result in (*quality.values(), *filtered_scope.values()))}
    return {"checks": checks, "recommended_default": "hybrid" if all(checks.values()) else "dense"}


def run(output: Path, host: str, port: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {name: sha256(ROOT / name) for name in FILES}
    write_json(output / "frozen.json", {"files": frozen, "spec": SPEC, "platform": platform.platform()})
    (output / "runner_frozen.py").write_bytes(Path(__file__).read_bytes())
    (output / "hybrid_frozen.py").write_bytes((ROOT / "mcp/hybrid_retrieval.py").read_bytes())
    (output / "filter_frozen.py").write_bytes((ROOT / "agents/retail_tools.py").read_bytes())
    write_json(output / "run.json", {"status": "RUNNING", "paid_api_calls": 0})
    encoder, kb, created, client = None, None, False, None
    try:
        passages, queries = load_inputs()
        scope_queries = read_rows("evaluation/data/hybrid_retrieval/scope_queries.jsonl")
        decoys = read_rows("evaluation/data/hybrid_retrieval/scope_decoys.jsonl")
        validate_scope(scope_queries, passages + decoys)
        for query in scope_queries:
            actual = {p["evidence_id"] for p in passages + decoys if retail_evidence_applicable(p, query["support_packet"], query["query"])}
            if actual != set(query["allowed_ids"]):
                raise RuntimeError(f"范围独立标注与实现不符：{query['query_id']}")
        client = chromadb.HttpClient(host=host, port=port, settings=chromadb.Settings(anonymized_telemetry=False))
        original = sorted(c.name for c in client.list_collections())
        encoder = QwenEmbeddingFunction()
        name = "u05_" + uuid.uuid4().hex[:12]
        kb = KnowledgeBase(chroma_host=host, chroma_port=port, collection_name=name,
                           load_demo_documents=False, embedding_function=encoder)
        created = kb.collection_name not in original
        if kb.backend != "server" or not created:
            raise RuntimeError("必须使用新的服务端独立集合")
        started = time.perf_counter()
        kb.upsert_evidence(passages)
        vector_index_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        kb.configure_retrieval(passages, "dense")
        lexical_index_ms = (time.perf_counter() - started) * 1000
        quality = measure(kb, queries, False, {p["evidence_id"] for p in passages})
        write_json(output / "quality.json", quality)
        kb.upsert_evidence(decoys)
        kb.configure_retrieval(passages + decoys, "dense")
        unfiltered = measure(kb, scope_queries, False, {p["evidence_id"] for p in passages + decoys})
        write_json(output / "scope_unfiltered.json", unfiltered)
        filtered = measure(kb, scope_queries, True, {p["evidence_id"] for p in passages + decoys})
        losses = []
        for query in scope_queries:
            actual = {p["evidence_id"] for p in passages + decoys if retail_evidence_applicable(p, query["support_packet"], query["query"])}
            if actual != set(query["allowed_ids"]):
                raise RuntimeError(f"范围独立标注与实现不符：{query['query_id']}")
            losses.extend([query["query_id"] for group in query["required_evidence_groups"] if not set(group) & actual])
        for result in filtered.values():
            result["gold_filter_losses"] = len(losses)
        write_json(output / "scope_filtered.json", filtered)
        selection = select(quality, filtered)
        if any(sha256(ROOT / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("运行期间代码或输入发生变化")
        summary = {"status": "EVALUATION_COMPLETE", "spec": SPEC, "quality_queries": len(queries),
                   "scope_queries": len(scope_queries), "core_passages": len(passages), "synthetic_decoys": len(decoys),
                   "vector_load_and_index_ms": vector_index_ms, "lexical_index_ms": lexical_index_ms,
                   "selection": selection, "paid_api_calls": 0, "cost_cny": 0}
    except Exception as exc:
        write_json(output / "run.json", {"status": "RUN_ERROR", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        try:
            if encoder is not None:
                encoder.close()
            if created:
                kb._client.delete_collection(kb.collection_name)
        except Exception as exc:
            write_json(output / "run.json", {"status": "RUN_ERROR", "error": f"清理失败：{exc}"})
            raise
    after = sorted(c.name for c in client.list_collections())
    write_json(output / "collections.json", {"before": original, "after": after})
    if original != after:
        write_json(output / "run.json", {"status": "RUN_ERROR", "error": "集合未恢复"})
        raise RuntimeError("集合未恢复")
    write_json(output / "summary.json", summary)
    write_json(output / "run.json", {"status": "EVALUATION_COMPLETE", "paid_api_calls": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8002)
    args = parser.parse_args()
    run(args.output.resolve(), args.chroma_host, args.chroma_port)
