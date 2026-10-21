"""U06 固定 Hybrid 候选池的真实重排对照，复用统一费用账本。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback
import urllib.request
import uuid

from evaluation.retail_metrics import compute_retrieval_metrics, summarize_latency
from evaluation.run_embedding_comparison import INPUTS, load_inputs, sha256, write_json
from mcp.qwen_reranker import MODEL_FILES, REVISION, SPEC, QwenReranker, verify_model
from mcp.tool_manager import build_rerank_prompt

ROOT = Path(__file__).resolve().parents[1]
FILES = (*INPUTS, "mcp/hybrid_retrieval.py", "mcp/knowledge_base.py", "mcp/tool_manager.py",
         "mcp/qwen_reranker.py", "mcp/qwen_embedding.py", "evaluation/retail_metrics.py",
         "evaluation/retail_ledger.py", "evaluation/run_reranker_comparison.py")


def prepare_model(path: Path) -> None:
    """仅显式准备时联网，未知文件与非空下载残片均不覆盖。"""
    path.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in MODEL_FILES.items():
        target = path / name
        if not target.exists():
            partial = path / (name + ".part")
            url = f"https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/resolve/{REVISION}/{name}"
            with partial.open("xb") as handle, urllib.request.urlopen(url, timeout=60) as response:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > size:
                        raise ValueError("模型下载超出固定大小")
                    handle.write(chunk)
            if partial.stat().st_size != size or sha256(partial) != digest:
                raise ValueError("模型下载校验失败，保留残片供检查")
            partial.rename(target)
    verify_model(path)
    write_json(path / "model-manifest.json", {"model": SPEC["model"], "revision": REVISION,
                                             "files": MODEL_FILES, "license": "Apache-2.0"})


def ndcg(ranking: list[str], qrels: dict, k: int = 5) -> float:
    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    ranking = list(dict.fromkeys(ranking))
    return dcg([qrels.get(key, 0) for key in ranking[:k]]) / ideal if ideal else 0.0


def coverage_audit(rows: list[dict], pools: list[dict]) -> dict:
    """按必需组审计前五保留及相对RRF原序的新增漏项。"""
    rankings = {row["query_id"]: row["ranking"] for row in rows}
    retained, total, misses, regressions = 0, 0, [], []
    for pool in pools:
        if pool["answerability"] != "answerable":
            continue
        groups = pool["required_evidence_groups"]
        top = set(rankings[pool["query_id"]][:5])
        missing = [group for group in groups if not top.intersection(group)]
        total += len(groups)
        retained += len(groups) - len(missing)
        if missing:
            misses.append({"query_id": pool["query_id"], "missing_groups": missing})
            original = {item["evidence_id"] for item in pool["candidates"][:5]}
            if all(original.intersection(group) for group in groups):
                regressions.append(pool["query_id"])
    return {"groups_retained": retained, "groups_total": total,
            "misses": misses, "regressions_vs_rrf": regressions}


def legacy_order(raw: str, count: int) -> tuple[list[int], dict]:
    """原生成式排序的过滤行为保持不变；另外记录完整排列是否合法。"""
    try:
        order = json.loads(raw[raw.find("["):raw.rfind("]") + 1])
        if not isinstance(order, list):
            raise ValueError("排序不是列表")
        valid = [i for i in order if isinstance(i, int) and 0 <= i < count]
        complete = all(type(i) is int for i in order) and len(order) == count and set(order) == set(range(count))
        return valid or list(range(count)), {"raw_order": order, "complete": complete,
                                             "applied": bool(valid), "status": "applied" if valid else "empty_output"}
    except Exception:
        return list(range(count)), {"complete": False, "applied": False, "status": "invalid_output"}


def summarize(rows: list[dict], pools: list[dict], known: set[str]) -> dict:
    by_id = {row["query_id"]: row for row in rows}
    summary = {}
    for split in ("all", "dev", "test", "support_smoke"):
        selected = [pool for pool in pools if split == "all" or pool["split"] == split]
        records = [dict(query_id=p["query_id"], split=p["split"], answerability=p["answerability"],
                        gold_qrels=p["qrels"], required_groups=p["required_evidence_groups"],
                        ranking=by_id[p["query_id"]]["ranking"]) for p in selected]
        metrics = compute_retrieval_metrics(records, ks=(5, 20), known_ids=known)
        answerable = [p for p in selected if p["answerability"] == "answerable"]
        metrics["ndcg@5"] = sum(ndcg(by_id[p["query_id"]]["ranking"], p["qrels"]) for p in answerable) / len(answerable)
        summary[split] = metrics
    summary["latency"] = summarize_latency(row["latency_ms"] for row in rows)
    summary["complete_count"] = sum(row.get("complete", False) for row in rows)
    summary["count"] = len(rows)
    summary["evidence_coverage"] = coverage_audit(rows, pools)
    return summary


def freeze_pool(output: Path, host: str, port: int) -> list[dict]:
    import chromadb
    from mcp.knowledge_base import KnowledgeBase
    from mcp.qwen_embedding import QwenEmbeddingFunction

    passages, queries = load_inputs()
    encoder = QwenEmbeddingFunction()
    client = chromadb.HttpClient(host=host, port=port, settings=chromadb.Settings(anonymized_telemetry=False))
    original = sorted(c.name for c in client.list_collections())
    kb = None
    try:
        kb = KnowledgeBase(chroma_host=host, chroma_port=port, collection_name="u06_" + uuid.uuid4().hex[:12],
                           load_demo_documents=False, embedding_function=encoder)
        if kb.backend != "server" or kb.collection_name in original:
            raise RuntimeError("候选冻结必须使用新的独立服务端集合")
        kb.upsert_evidence(passages)
        kb.configure_retrieval(passages, "hybrid")
        pools = []
        for query in queries:
            started = time.perf_counter()
            items = kb.search(query["query"], top_k=20)
            pools.append(dict(query, candidates=items, recall_ms=(time.perf_counter() - started) * 1000))
        write_json(output / "candidate-pools.json", pools)
    finally:
        encoder.close()
        if kb is not None and kb.collection_name not in original:
            client.delete_collection(kb.collection_name)
        after = sorted(c.name for c in client.list_collections())
        write_json(output / "collections.json", {"before": original, "after": after})
        if after != original:
            raise RuntimeError("候选冻结集合未恢复")
    return pools


async def evaluate(output: Path, pools: list[dict], ledger_path: Path) -> dict:
    from dotenv import load_dotenv
    from core.llm_utils import extract_text_content
    from evaluation.retail_budget import request_input_upper_tokens
    from evaluation.retail_ledger import RetailLedger, build_budgeted_client

    load_dotenv(ROOT / ".env")
    model = os.environ.get("LLM_MODEL", "")
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    if model != "deepseek-flash" or base_url.rstrip("/") != "https://api.deepseek.com/anthropic":
        raise ValueError("已核对价格仅适用于官方 deepseek-flash Anthropic 接口")
    ledger = RetailLedger(ledger_path, 5, 2, 8, run_id="u06", model=model)
    if ledger.summary()["requests"]:
        raise ValueError("账本已有请求，拒绝重复付费运行")
    client = build_budgeted_client(os.environ["ANTHROPIC_API_KEY"], base_url, ledger, "u06_rerank")
    llm = []
    requests = [dict(model=model, max_tokens=8192, temperature=0.0,
                     messages=[{"role": "user", "content": build_rerank_prompt(p["query"], p["candidates"])}]) for p in pools]
    upper = sum(request_input_upper_tokens(req) * 2 / 1e6 + 8192 * 8 / 1e6 for req in requests)
    write_json(output / "preflight.json", {"requests": len(requests), "upper_cost_cny": upper,
                                          "budget": ledger.summary(), "price_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"})
    if upper > ledger.summary()["remaining_cny"]:
        await client.aclose()
        raise ValueError("完整请求最坏上界超过剩余预算，未发送")
    try:
        for pool, request in zip(pools, requests):
            started = time.perf_counter()
            try:
                response = await client.messages.create(**request)
                raw = extract_text_content(response.content)
                order, diag = legacy_order(raw, len(pool["candidates"]))
                row = dict(query_id=pool["query_id"], raw=raw, response=response.model_dump(mode="json"),
                           request=request, ranking=[pool["candidates"][i]["evidence_id"] for i in order], **diag)
            except Exception as exc:
                row = dict(query_id=pool["query_id"], ranking=[i["evidence_id"] for i in pool["candidates"]],
                           complete=False, applied=False, status="failed", error=str(exc)[:300])
                llm.append(dict(row, latency_ms=(time.perf_counter() - started) * 1000))
                write_json(output / "llm.json", llm)
                raise
            llm.append(dict(row, latency_ms=(time.perf_counter() - started) * 1000))
            write_json(output / "llm.json", llm)
            print(f"LLM {len(llm)}/{len(pools)}, spent={ledger.summary()['spent_cny']:.6f}", flush=True)
            if ledger.summary()["cost_stop_reason"] or ledger.summary()["usage_missing"]:
                raise RuntimeError("费用账本完整性停止")
    finally:
        write_json(output / "budget.json", ledger.summary())
        write_json(output / "ledger-entries.json", ledger.entries())
        await client.aclose()

    reranker = QwenReranker()
    local = []
    try:
        started = time.perf_counter()
        reranker.score("模型预热", [{"content": "政策帮助"}])
        write_json(output / "local-startup.json", {"load_and_warm_ms": (time.perf_counter() - started) * 1000})
        for round_index in range(2):
            rows = []
            for pool in pools:
                started = time.perf_counter()
                scores = reranker.score(pool["query"], pool["candidates"])
                order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
                rows.append(dict(query_id=pool["query_id"], scores=scores, complete=True,
                                 ranking=[pool["candidates"][i]["evidence_id"] for i in order],
                                 latency_ms=(time.perf_counter() - started) * 1000, stats=reranker.last_stats))
                write_json(output / f"qwen-round-{round_index + 1}.json", rows)
                print(f"Qwen round={round_index + 1} {len(rows)}/{len(pools)}", flush=True)
            local.append(rows)
    finally:
        reranker.close()
        write_json(output / "local-process.json", {"pid": reranker._process.pid if reranker._process else None,
                                                  "closed": reranker._closed,
                                                  "exit_code": reranker._process.poll() if reranker._process else None})
    known = {p["evidence_id"] for p in load_inputs()[0]}
    rrf = [dict(query_id=p["query_id"], ranking=[i["evidence_id"] for i in p["candidates"]], latency_ms=0, complete=True) for p in pools]
    summaries = {"rrf": summarize(rrf, pools, known), "llm_legacy": summarize(llm, pools, known),
                 "qwen": summarize(local[0], pools, known)}
    summaries["qwen"]["repeat_latency"] = summarize_latency(row["latency_ms"] for row in local[1])
    summaries["qwen"]["ranking_stable"] = all(a["ranking"] == b["ranking"] for a, b in zip(*local))
    summaries["qwen"]["peak_private_mib"] = max(row["stats"].get("private_peak_mib", 0) for row in local[1])
    summaries["qwen"]["peak_rss_mib"] = max(row["stats"].get("rss_peak_mib", 0) for row in local[1])
    summaries["qwen"]["truncated_pairs"] = sum(row["stats"]["truncated_pairs"] for row in local[0])
    a, b = summaries["llm_legacy"]["all"], summaries["qwen"]["all"]
    quality_ok = all(b[m] >= a[m] - 1e-9 for m in ("all_evidence_hit@5", "mrr@5", "ndcg@5"))
    gain = any(b[m] > a[m] + 1e-9 for m in ("mrr@5", "ndcg@5"))
    latency_gain = summaries["qwen"]["latency"]["p95_ms"] < summaries["llm_legacy"]["latency"]["p95_ms"]
    summaries["selection"] = {"quality_non_decreasing": quality_ok, "ranking_gain": gain,
                              "latency_gain": latency_gain, "recommended": "qwen" if quality_ok and (gain or latency_gain) else "llm"}
    return summaries


def run(output: Path, ledger: Path, host: str, port: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("输出目录非空，拒绝覆盖")
    output.mkdir(parents=True, exist_ok=True)
    frozen = {name: sha256(ROOT / name) for name in FILES}
    write_json(output / "frozen.json", {"files": frozen, "spec": SPEC, "platform": platform.platform(),
                                       "rounds": {"llm": 1, "qwen": 2}, "candidate_k": 20, "top_k": 5})
    (output / "runner_frozen.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "run.json", {"status": "RUNNING"})
    try:
        pools = freeze_pool(output, host, port)
        write_json(output / "pool-identity.json", {"sha256": sha256(output / "candidate-pools.json")})
        summary = asyncio.run(evaluate(output, pools, ledger))
        if any(sha256(ROOT / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("运行期间输入或检索实现变化")
        write_json(output / "summary.json", summary)
        write_json(output / "run.json", {"status": "EVALUATION_COMPLETE"})
    except Exception as exc:
        write_json(output / "run.json", {"status": "RUN_ERROR", "error": str(exc), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--live", action="store_true", help="显式执行真实对照，本轮API上限¥5")
    parser.add_argument("--model-path", type=Path, default=ROOT / "data/models/qwen3-reranker-0.6b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ledger", type=Path, default=ROOT / "data/evaluation/u06/ledger.db")
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8002)
    args = parser.parse_args()
    if args.prepare_model:
        prepare_model(args.model_path)
    else:
        if not args.live:
            parser.error("真实对照需显式 --live，产生API费用，本轮上限¥5")
        if args.output is None:
            parser.error("需要 --output")
        run(args.output.resolve(), args.ledger.resolve(), args.chroma_host, args.chroma_port)
