"""零售识别与检索基线的可复现入口。

模式隔离是硬约束：offline-check 只导入标准库和纯离线模块，不创建任何
可发送模型请求的客户端；vector 只使用本地缓存的 Embedding 和独立 Chroma
集合；live 必须在显式预算与单价齐备后才导入并构造模型客户端。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from evaluation.retail_budget import (
    INPUT_COUNT_BASIS,
    REQUEST_FORMAT_OVERHEAD_TOKENS,
    BudgetExceeded,
    BudgetGuard,
    Reservation,
    request_input_upper_tokens,
)
from evaluation.retail_metrics import (
    RANK_KS,
    compute_retrieval_metrics,
    entities_to_slots,
    map_intent_label,
    run_offline_self_check,
    score_intents,
    score_slots,
    summarize_latency,
)
from evaluation.validate_retail_data import validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(__file__).resolve().parent / "data" / "retail"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "retail"

CONFIDENCE_THRESHOLD = 0.5
TOP_K = 5
RECALL_K = 5
REWRITE_N = 3
CACHE_TTL_S = 300.0
INTENT_MAX_TOKENS = 2048
REWRITE_MAX_TOKENS = 1024
RERANK_MAX_TOKENS = 8192

STATUS_PENDING_BUDGET = "OFFLINE_READY_REAL_BASELINE_PENDING_BUDGET"
STATUS_PARTIAL = "PARTIAL_RUN"
STATUS_SMOKE_COMPLETE = "SMOKE_COMPLETE"
STATUS_COMPONENT_COMPLETE = "COMPONENT_COMPLETE"
STATUS_COMPLETE = "BASELINE_COMPLETE"
STATUS_REFUSED_BUDGET = "REFUSED_NO_BUDGET"
STATUS_REFUSED_PREFLIGHT = "REFUSED_BUDGET_PREFLIGHT"
STATUS_REFUSED_SELECTION = "REFUSED_SELECTION"
STATUS_BLOCKED_CREDENTIALS = "BLOCKED_CREDENTIALS"
STATUS_BLOCKED_EMBEDDING = "BLOCKED_EMBEDDING_CACHE"
STATUS_BLOCKED_STORAGE = "BLOCKED_STORAGE"
STATUS_DATA_FAILED = "DATA_INTEGRITY_FAILED"
STATUS_ERROR = "RUN_ERROR"


class BaselineRefused(RuntimeError):
    """在发出任何远程请求之前停止，并保留原因供 run.json 记录。"""

    def __init__(
        self,
        status: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.details = details or {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id(mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{stamp}_{uuid.uuid4().hex[:6]}"


def _prepare_run_dir(run_id: str, requested: Optional[str]) -> Path:
    path = Path(requested) if requested else DEFAULT_OUTPUT_ROOT / run_id
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.exists() and any(path.iterdir()):
        raise BaselineRefused(
            STATUS_ERROR,
            f"输出目录非空，拒绝覆盖: {path}",
        )
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    path.write_text(text, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_dataset(root: Path) -> Dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = {}
    for item in manifest.get("artifacts", []):
        relative = item["path"]
        path = root / relative
        artifacts[relative] = (
            _sha256_file(path) if path.is_file() else "MISSING"
        )
    return {
        "version": manifest.get("version"),
        "manifest_sha256": _sha256_file(manifest_path),
        "artifacts": artifacts,
    }


def _retrieval_source_item(item: Any) -> Dict[str, Any]:
    """保留检索结果的稳定身份与来源字段，不记录正文。"""
    if not isinstance(item, dict):
        return {"evidence_id": str(item)}
    return {
        "evidence_id": str(item.get("evidence_id") or item.get("id") or ""),
        "id": item.get("id"),
        "doc_id": item.get("doc_id", ""),
        "title": item.get("title", ""),
        "score": item.get("score"),
        "source_kind": item.get("source_kind", ""),
        "source_refs": item.get("source_refs") or [],
        "policy_release": item.get("policy_release", ""),
        "policy_commit": item.get("policy_commit", ""),
        "scope": item.get("scope", ""),
        "category": item.get("category", ""),
    }


def _read_budget_ledger(path: Path, limit_cny: float) -> Dict[str, Any]:
    """读取跨阶段累计账本；不存在时按本次授权上限初始化。"""
    if not path.exists():
        return {"version": 1, "limit_cny": float(limit_cny), "entries": []}
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            f"预算账本不可读，无法建立可信累计额: {path}",
            {"error": f"{type(ex).__name__}: {str(ex)[:200]}"},
        ) from ex
    recorded = float(ledger.get("limit_cny", -1))
    if recorded != float(limit_cny):
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            f"预算账本上限 {recorded} 与本次授权 {limit_cny} 不一致",
            {"ledger": str(path)},
        )
    return ledger


def _ledger_entry_charge(entry: Dict[str, Any]) -> float:
    """未结算的已启动记录按计划上界占用，已完成记录按实际花费加在途占用。"""
    if entry.get("status") == "started":
        return float(entry.get("planned_upper_bound_cny") or 0.0)
    return float(entry.get("spent_cny") or 0.0) + float(
        entry.get("unsettled_cny") or 0.0
    )


def _ledger_committed(ledger: Dict[str, Any]) -> float:
    return sum(
        _ledger_entry_charge(entry) for entry in ledger.get("entries") or []
    )


def _write_budget_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    """原子写入账本，避免中断留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ledger_start(
    ledger: Dict[str, Any],
    run_id: str,
    stage_label: str,
    planned_upper_bound_cny: float,
    available_before_cny: float,
) -> None:
    ledger.setdefault("entries", []).append(
        {
            "run_id": run_id,
            "stage_label": stage_label,
            "status": "started",
            "started_at": _utc_now(),
            "planned_upper_bound_cny": float(planned_upper_bound_cny),
            "available_before_cny": float(available_before_cny),
        }
    )


def _ledger_finish(
    ledger: Dict[str, Any],
    run_id: str,
    fields: Dict[str, Any],
) -> None:
    for entry in ledger.get("entries") or []:
        if entry.get("run_id") == run_id:
            entry.update(fields)
            entry["finished_at"] = _utc_now()
            return


def _load_dataset(root: Path) -> Dict[str, Any]:
    return {
        "intents": _read_jsonl(root / "intent_slots.jsonl"),
        "queries": _read_jsonl(root / "retrieval.jsonl"),
        "knowledge": _read_jsonl(root / "knowledge.jsonl"),
        "manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
        "splits": json.loads((root / "splits.json").read_text(encoding="utf-8")),
        "labels": json.loads((root / "labels.json").read_text(encoding="utf-8")),
    }


def _selected(rows: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    if split == "all":
        return list(rows)
    return [row for row in rows if row.get("split") == split]


def _selected_splits(split: str) -> List[str]:
    return ["dev", "test"] if split == "all" else [split]


def _select_by_ids(
    rows: Sequence[Dict[str, Any]],
    key: str,
    raw_ids: Optional[str],
    label: str,
) -> List[Dict[str, Any]]:
    """按显式 ID 选取子集；未知或重复 ID 在付费前拒绝。"""
    if raw_ids is None:
        return list(rows)
    ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
    if not ids:
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            f"{label} ID 列表为空；未创建模型客户端",
        )
    if len(set(ids)) != len(ids):
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            f"{label} ID 列表含重复项；未创建模型客户端",
        )
    by_id = {str(row.get(key)): row for row in rows}
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            f"{label} ID 不在所选划分中: {missing}；未创建模型客户端",
        )
    return [by_id[item] for item in ids]


def _run_git(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _code_snapshot() -> Dict[str, Any]:
    head = _run_git(["rev-parse", "HEAD"])
    subject = _run_git(["log", "-1", "--format=%s"])
    status = _run_git(["status", "--porcelain", "-z", "--untracked-files=normal"])
    dirty: Dict[str, str] = {}
    entries = status.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        state = entry[:2]
        paths = [entry[3:]]
        if state[0] in {"R", "C"} and index < len(entries):
            paths.append(entries[index])
            index += 1
        for relative in paths:
            path = PROJECT_ROOT / relative
            if path.is_file():
                dirty[relative.replace("\\", "/")] = _sha256_file(path)
            else:
                dirty[relative.replace("\\", "/")] = "MISSING"
    return {
        "head": head,
        "head_subject": subject,
        "dirty_files": dict(sorted(dirty.items())),
    }


def _base_result(mode: str, split: str) -> Dict[str, Any]:
    return {
        "status": "NOT_RUN",
        "stages": {
            "offline_checks": "NOT_RUN",
            "storage_integration": "NOT_RUN",
            "vector_baseline": "NOT_RUN",
            "live_intent": "NOT_RUN",
            "live_pipeline": "NOT_RUN",
        },
        "blockers": [],
        "notes": [],
        "config": {"mode": mode, "split": split},
        "coverage": {},
        "intent": {"status": "NOT_RUN"},
        "slots": {"status": "NOT_RUN"},
        "retrieval": {
            "V": {"status": "NOT_RUN"},
            "P": {"status": "NOT_RUN"},
        },
        "latency": {},
        "cost": {
            "requests": 0,
            "cost_cny": 0.0,
            "basis": "no_remote_model_requests",
        },
        "errors": {},
        "offline_checks": None,
        "intent_predictions": [],
        "retrieval_predictions": [],
        "usage": [],
        "details": {},
    }


def _attempted_status(
    expected_ids: Sequence[str],
    attempted_ids: Sequence[str],
) -> str:
    return (
        "COMPLETE"
        if set(expected_ids) == set(attempted_ids)
        else "PARTIAL"
    )


def _coverage_for(
    expected_ids: Sequence[str],
    attempted_ids: Sequence[str],
) -> Dict[str, Any]:
    expected_set = set(expected_ids)
    attempted_set = set(attempted_ids)
    return {
        "expected": len(expected_set),
        "actual": len(attempted_set),
        "missing_ids": sorted(expected_set - attempted_set),
        "unexpected_ids": sorted(attempted_set - expected_set),
    }


def _per_split(
    records: Sequence[Dict[str, Any]],
    splits: Sequence[str],
    scorer,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for split in splits:
        subset = [row for row in records if row.get("split") == split]
        result = scorer(subset)
        result["status"] = "SCORED" if subset else "EMPTY"
        output[split] = result
    return output


def _latency_by_split(
    records: Sequence[Dict[str, Any]],
    splits: Sequence[str],
    key: str = "latency_ms",
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for split in splits:
        values = [
            row.get(key)
            for row in records
            if row.get("split") == split and not row.get("is_error")
        ]
        output[split] = summarize_latency(values)
    output["all"] = summarize_latency(
        [row.get(key) for row in records if not row.get("is_error")]
    )
    output["failed"] = summarize_latency(
        [row.get(key) for row in records if row.get("is_error")]
    )
    return output


def _split_expected_ids(
    rows: Sequence[Dict[str, Any]],
    id_key: str,
    split: str,
) -> List[str]:
    return [
        row[id_key]
        for row in rows
        if split == "all" or row.get("split") == split
    ]


def _variant_counts(
    rows: Sequence[Dict[str, Any]],
    key: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        cursor: Any = row
        for part in key.split("."):
            cursor = cursor.get(part) if isinstance(cursor, dict) else None
        if isinstance(cursor, str) and cursor:
            counts[cursor] = counts.get(cursor, 0) + 1
    return counts


def _intent_error_summary(
    score_records: Sequence[Dict[str, Any]],
    rows_by_id: Dict[str, Dict[str, Any]],
    artifacts: Sequence[Dict[str, Any]],
    limit: int = 12,
) -> Dict[str, Any]:
    artifacts_by_id = {row["case_id"]: row for row in artifacts}
    wrong: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for record in score_records:
        base = {
            "case_id": record.get("case_id"),
            "split": record.get("split"),
            "message": rows_by_id.get(record.get("case_id"), {}).get("message"),
            "gold": record.get("gold"),
            "prediction": record.get("prediction"),
            "raw_intent": artifacts_by_id.get(record.get("case_id"), {}).get(
                "raw_intent"
            ),
            "recognition_status": artifacts_by_id.get(
                record.get("case_id"), {}
            ).get("recognition_status"),
        }
        if record.get("is_error"):
            errors.append({**base, "error_kind": record.get("error_kind")})
        elif record.get("prediction") != record.get("gold"):
            wrong.append(base)
    return {
        "scored": len(score_records),
        "wrong_count": sum(
            1
            for record in score_records
            if not record.get("is_error")
            and record.get("prediction") != record.get("gold")
        ),
        "error_count": sum(1 for record in score_records if record.get("is_error")),
        "error_kinds": _variant_counts(score_records, "error_kind"),
        "wrong_samples": wrong[:limit],
        "error_samples": errors[:limit],
    }


def _slot_error_summary(
    metrics: Dict[str, Any],
    rows_by_id: Dict[str, Dict[str, Any]],
    limit: int = 12,
) -> Dict[str, Any]:
    rows = metrics.get("cases") or []
    mismatched = [row for row in rows if not row.get("exact_match")]
    return {
        "mismatch_count": len(mismatched),
        "samples": [
            {
                "case_id": row.get("case_id"),
                "split": row.get("split"),
                "message": rows_by_id.get(row.get("case_id"), {}).get("message"),
                "gold": row.get("gold"),
                "predicted": row.get("predicted"),
                "fp": row.get("fp"),
                "fn": row.get("fn"),
            }
            for row in mismatched[:limit]
        ],
    }


def _retrieval_error_summary(
    artifact_rows: Sequence[Dict[str, Any]],
    metrics_by_split: Dict[str, Any],
    mode: str,
) -> Dict[str, Any]:
    artifact_by_id = {row["query_id"]: row for row in artifact_rows}
    per_query = []
    for result in metrics_by_split.values():
        per_query.extend(result.get("per_query") or [])
    failed = [
        {
            "query_id": item.get("query_id"),
            "split": item.get("split"),
            "query": artifact_by_id.get(item.get("query_id"), {}).get("query"),
            "error": artifact_by_id.get(item.get("query_id"), {}).get("error"),
        }
        for item in per_query
        if item.get("is_error")
    ]
    zero_hit = [
        {
            "query_id": item.get("query_id"),
            "split": item.get("split"),
            "query": artifact_by_id.get(item.get("query_id"), {}).get("query"),
            "first_relevant_rank@5": item.get("first_relevant_rank@5"),
            "ranking": item.get("ranking"),
        }
        for item in per_query
        if item.get("first_relevant_rank@5") == 0
    ]
    protocol = [
        issue
        for result in metrics_by_split.values()
        for issue in (result.get("protocol_issues") or [])
    ]
    summary: Dict[str, Any] = {
        "mode": mode,
        "failed_count": len(failed),
        "zero_hit_at_5_count": len(zero_hit),
        "protocol_issues": protocol,
        "failed_samples": failed[:10],
        "zero_hit_samples": zero_hit[:10],
    }
    if mode == "P":
        summary["rewrite_status"] = _variant_counts(
            artifact_rows, "diagnostics.rewrite_status"
        )
        summary["rerank_status"] = _variant_counts(
            artifact_rows, "diagnostics.rerank_status"
        )
        summary["degraded_queries"] = sum(
            1 for row in artifact_rows if row.get("degraded")
        )
        summary["budget_stops"] = sum(
            1 for row in artifact_rows if row.get("budget_stop")
        )
    return summary


def _collection_name(run_id: str) -> str:
    compact = re.sub(r"[^a-z0-9_]", "_", run_id.lower())
    return f"t11_v1_{compact}"[:63].strip("_")


def _chroma_config(
    args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[str, int, str]:
    host = (
        args.chroma_host
        or os.getenv("CHROMA_HOST", "").strip()
        or "127.0.0.1"
    )
    raw_port = args.chroma_port or os.getenv("CHROMA_PORT", "").strip() or "8002"
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as ex:
        raise BaselineRefused(
            STATUS_BLOCKED_STORAGE,
            f"CHROMA_PORT 不是整数: {raw_port!r}",
        ) from ex
    path = (
        args.chroma_path
        or os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
        or str(run_dir / "chroma_local")
    )
    return host, port, path


def _collection_name_of(item: Any) -> str:
    return str(getattr(item, "name", item))


def _cleanup_local_path(fallback_path: str, run_dir: Path) -> bool:
    """只删除运行目录内的本地回退存储，避免误删共享持久化目录。"""
    path = Path(fallback_path).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        return False
    shutil.rmtree(path)
    return True


def _snapshot_collections(
    client: Any,
    exclude: Optional[str] = None,
) -> Dict[str, Any]:
    names: List[str] = []
    counts: Dict[str, Any] = {}
    for item in client.list_collections():
        name = _collection_name_of(item)
        names.append(name)
        if name == exclude:
            continue
        try:
            collection = (
                item
                if hasattr(item, "count")
                else client.get_collection(name)
            )
            counts[name] = collection.count()
        except Exception as ex:
            counts[name] = f"ERROR: {str(ex)[:200]}"
    return {"names": sorted(names), "counts": counts}


def _verify_evidence(
    kb: Any,
    knowledge: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    ids = [row["evidence_id"] for row in knowledge]
    issues: List[str] = []
    got = kb._collection.get(ids=ids, include=["documents", "metadatas"])
    got_ids = [str(value) for value in (got.get("ids") or [])]
    documents = got.get("documents") or []
    metadatas = got.get("metadatas") or []
    docs_by_id = dict(zip(got_ids, documents))
    metas_by_id = dict(zip(got_ids, metadatas))
    if set(got_ids) != set(ids):
        issues.append(
            "检索到的证据 ID 集不一致: "
            f"缺少 {sorted(set(ids) - set(got_ids))}, "
            f"多出 {sorted(set(got_ids) - set(ids))}"
        )
    expected_metadata = {
        row["evidence_id"]: kb._evidence_metadata(row) for row in knowledge
    }
    for row in knowledge:
        evidence_id = row["evidence_id"]
        if docs_by_id.get(evidence_id) != row["content"]:
            issues.append(f"{evidence_id} 正文不一致")
        metadata = metas_by_id.get(evidence_id) or {}
        expected = expected_metadata[evidence_id]
        for key, value in expected.items():
            if metadata.get(key) != value:
                issues.append(f"{evidence_id} 元数据不一致: {key}")
    count = kb.doc_count
    if count != len(ids):
        issues.append(f"集合数量 {count} 与预期 {len(ids)} 不一致")
    return {
        "passed": not issues,
        "issues": issues,
        "count": count,
        "ids": sorted(got_ids),
    }


def _empty_usage_row(mode: str, note: str) -> Dict[str, Any]:
    return {
        "type": "summary",
        "mode": mode,
        "requests": 0,
        "cost_cny": 0.0,
        "vendor_bill_verified": False,
        "note": note,
    }


def _retrieval_score_record(
    row: Dict[str, Any],
    ranking: Sequence[str],
    is_error: bool,
    note: str = "",
) -> Dict[str, Any]:
    return {
        "query_id": row["query_id"],
        "split": row["split"],
        "answerability": row.get("answerability"),
        "gold_qrels": row.get("qrels") or {},
        "required_groups": row.get("required_evidence_groups") or [],
        "ranking": list(ranking),
        "is_error": is_error,
        "note": note,
    }


def _run_vector_queries(
    kb: Any,
    queries: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    artifacts: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for row in queries:
        started = time.monotonic()
        try:
            items = kb.search(row["query"], top_k=TOP_K)
            latency_ms = (time.monotonic() - started) * 1000
            ranking = [
                str(item.get("evidence_id") or item.get("id"))
                for item in items
            ]
            artifact = {
                "query_id": row["query_id"],
                "split": row["split"],
                "mode": "V",
                "query": row["query"],
                "ranking": ranking,
                "scores": [
                    _retrieval_source_item(item) for item in items
                ],
                "is_error": False,
                "error_kind": None,
                "error": None,
                "latency_ms": latency_ms,
            }
            scores.append(
                _retrieval_score_record(row, ranking, is_error=False)
            )
        except Exception as ex:
            latency_ms = (time.monotonic() - started) * 1000
            artifact = {
                "query_id": row["query_id"],
                "split": row["split"],
                "mode": "V",
                "query": row["query"],
                "ranking": [],
                "scores": [],
                "is_error": True,
                "error_kind": "search_exception",
                "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "latency_ms": latency_ms,
            }
            scores.append(
                _retrieval_score_record(
                    row,
                    [],
                    is_error=True,
                    note="检索异常，按空排序计分",
                )
            )
        artifacts.append(artifact)
    return artifacts, scores


def _vector_metrics(
    scores: Sequence[Dict[str, Any]],
    splits: Sequence[str],
    known_ids: Set[str],
) -> Dict[str, Any]:
    return {
        split: compute_retrieval_metrics(
            [row for row in scores if row["split"] == split],
            known_ids=known_ids,
        )
        for split in splits
    }


def _fill_vector_result(
    result: Dict[str, Any],
    dataset: Dict[str, Any],
    splits: Sequence[str],
    split_arg: str,
    artifacts: Sequence[Dict[str, Any]],
    scores: Sequence[Dict[str, Any]],
    known_ids: Set[str],
    expected_ids: Optional[Sequence[str]] = None,
) -> None:
    # live 子集烟测按实际选中的查询计覆盖，避免把有意抽样写成缺失。
    if expected_ids is None:
        expected_ids = _split_expected_ids(
            dataset["queries"], "query_id", split_arg
        )
    else:
        expected_ids = list(expected_ids)
    attempted_ids = [row["query_id"] for row in artifacts]
    metrics = _vector_metrics(scores, splits, known_ids)
    protocol_clean = all(
        item.get("protocol_clean") for item in metrics.values()
    )
    result["intent"] = {"status": "NOT_RUN"}
    result["slots"] = {"status": "NOT_RUN"}
    result["retrieval"] = {
        "V": {
            "status": "SCORED" if artifacts else "EMPTY",
            "mode": "V",
            "top_k": TOP_K,
            "original_query_included": True,
            "per_split": metrics,
            "latency": _latency_by_split(artifacts, splits),
            "protocol_clean": protocol_clean,
        },
        "P": {"status": "NOT_RUN"},
    }
    result["latency"] = {
        "retrieval_V": _latency_by_split(artifacts, splits),
    }
    result["coverage"] = {
        "retrieval_V": _coverage_for(expected_ids, attempted_ids),
        "intent": _coverage_for(
            _split_expected_ids(dataset["intents"], "case_id", split_arg), []
        ),
        "retrieval_P": _coverage_for(expected_ids, []),
        "knowledge": {
            "expected": len(dataset["knowledge"]),
            "actual": None,
            "status": "CHECKED_SEPARATELY",
        },
    }
    result["errors"] = {
        "retrieval_V": _retrieval_error_summary(artifacts, metrics, "V"),
    }
    result["retrieval_predictions"] = list(artifacts)
    result["details"]["retrieval_V_complete"] = (
        _attempted_status(expected_ids, attempted_ids) == "COMPLETE"
    )
    result["details"]["vector_protocol_clean"] = protocol_clean


def run_offline_check(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """只校验资料和评分合同，不导入模型或存储客户端。"""
    checks = run_offline_self_check()
    passed = all(item["passed"] for item in checks)
    loaded = [
        name for name in ("anthropic", "chromadb") if name in sys.modules
    ]
    no_client = not loaded
    result = _base_result(args.mode, args.split)
    result["offline_checks"] = {
        "status": "PASS" if passed and no_client else "FAIL",
        "passed": sum(1 for item in checks if item["passed"]),
        "total": len(checks),
        "checks": checks,
        "model_client_modules_loaded": loaded,
        "sendable_client_created": False,
    }
    result["stages"]["offline_checks"] = (
        "PASS" if passed and no_client else "FAIL"
    )
    expected_intent = _split_expected_ids(
        dataset["intents"], "case_id", args.split
    )
    expected_queries = _split_expected_ids(
        dataset["queries"], "query_id", args.split
    )
    result["coverage"] = {
        "intent": _coverage_for(expected_intent, []),
        "retrieval_V": _coverage_for(expected_queries, []),
        "retrieval_P": _coverage_for(expected_queries, []),
        "knowledge": {
            "expected": len(dataset["knowledge"]),
            "actual": None,
            "status": "NOT_IMPORTED",
        },
    }
    result["blockers"] = [
        "PENDING_STORAGE",
        "PENDING_BUDGET: live_intent",
        "PENDING_BUDGET: live_pipeline",
    ]
    result["status"] = (
        STATUS_PENDING_BUDGET if passed and no_client else STATUS_ERROR
    )
    result["cost"] = {
        "requests": 0,
        "cost_cny": 0.0,
        "basis": "no_remote_model_requests",
    }
    result["usage"] = [
        _empty_usage_row(
            args.mode,
            "offline-check 未导入 anthropic 或 chromadb，未发送请求",
        )
    ]
    return result


def run_vector(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """使用本地 Embedding 缓存和独立真实 Chroma 集合运行 V。"""
    from mcp import knowledge_base as kb_module

    if not kb_module._embedding_cache_ready():
        raise BaselineRefused(
            STATUS_BLOCKED_EMBEDDING,
            "Embedding 本地缓存不完整，按约束不下载模型",
            {
                "cache_path": str(
                    kb_module._ProjectEmbeddingFunction.DOWNLOAD_PATH / "onnx"
                )
            },
        )

    host, port, fallback_path = _chroma_config(args, run_dir)
    collection_name = _collection_name(run_id)
    result = _base_result(args.mode, args.split)
    splits = _selected_splits(args.split)
    kb = None
    owns_collection = False
    collection_deleted = False
    snapshot_before: Dict[str, Any] = {}
    snapshot_after: Dict[str, Any] = {}
    import_ms = 0.0
    imported = 0
    second_import = 0
    try:
        kb = kb_module.KnowledgeBase(
            chroma_host=host,
            chroma_port=port,
            chroma_path=fallback_path,
            collection_name=collection_name,
            load_demo_documents=False,
        )
        client = kb._client
        backend = kb.backend
        snapshot_before = _snapshot_collections(
            client, exclude=collection_name
        )
        initial_count = kb.doc_count
        if initial_count:
            raise BaselineRefused(
                STATUS_BLOCKED_STORAGE,
                f"独立集合 {collection_name} 非空，拒绝复用未知内容",
                {"count": initial_count},
            )
        owns_collection = True

        import_started = time.monotonic()
        imported = kb.upsert_evidence(dataset["knowledge"])
        import_ms = (time.monotonic() - import_started) * 1000
        verification = _verify_evidence(kb, dataset["knowledge"])
        second_import = kb.upsert_evidence(dataset["knowledge"])
        count_after_import = kb.doc_count

        queries = _selected(dataset["queries"], args.split)
        artifacts, scores = _run_vector_queries(kb, queries)
        known_ids = {row["evidence_id"] for row in dataset["knowledge"]}
        _fill_vector_result(
            result, dataset, splits, args.split, artifacts, scores, known_ids
        )
        snapshot_after = _snapshot_collections(
            client, exclude=collection_name
        )
        old_unchanged = snapshot_before.get("counts") == snapshot_after.get(
            "counts"
        )
        result["coverage"]["knowledge"] = {
            "expected": len(dataset["knowledge"]),
            "actual": count_after_import,
            "imported_new": imported,
            "second_import_new": second_import,
            "verification": verification,
            "old_collections_unchanged": old_unchanged,
            "status": (
                "VERIFIED"
                if verification["passed"] and old_unchanged
                else "FAILED"
            ),
        }
        storage_ok = verification["passed"] and old_unchanged
        vector_complete = result["details"].get(
            "retrieval_V_complete", False
        )
        protocol_clean = result["details"].get(
            "vector_protocol_clean", False
        )
        result["stages"]["storage_integration"] = (
            "VERIFIED"
            if storage_ok and backend == "server"
            else "LOCAL_FALLBACK"
            if storage_ok
            else "FAILED"
        )
        if storage_ok and vector_complete and protocol_clean:
            result["stages"]["vector_baseline"] = "COMPLETE"
        elif storage_ok and vector_complete:
            result["stages"]["vector_baseline"] = "PROTOCOL_ISSUES"
        else:
            result["stages"]["vector_baseline"] = "FAILED"
        if backend != "server":
            result["blockers"].append("PENDING_STORAGE")
        result["blockers"].extend(
            ["PENDING_BUDGET: live_intent", "PENDING_BUDGET: live_pipeline"]
        )
        if result["stages"]["vector_baseline"] == "COMPLETE":
            result["status"] = STATUS_PENDING_BUDGET
        else:
            result["status"] = STATUS_PARTIAL
        result["cost"] = {
            "requests": 0,
            "cost_cny": 0.0,
            "basis": "no_remote_model_requests",
        }
        result["usage"] = [
            _empty_usage_row(
                args.mode,
                "vector 模式未创建模型客户端，未发送远程请求",
            )
        ]
        result["config"].update(
            {
                "chroma": {
                    "host": host,
                    "port": port,
                    "backend": backend,
                    "collection": collection_name,
                    "load_demo_documents": False,
                    "fallback_path": fallback_path,
                },
                "embedding": {
                    "backend": "onnx_all-MiniLM-L6-v2",
                    "cache_path": str(
                        kb_module._ProjectEmbeddingFunction.DOWNLOAD_PATH
                        / "onnx"
                    ),
                    "load_included_in_import_ms": True,
                },
                "retrieval": {
                    "top_k": TOP_K,
                    "recall_k": RECALL_K,
                    "rewrite_n": REWRITE_N,
                    "original_query_included": True,
                    "mode": "V",
                },
                "cache": "cold_fresh_knowledge_base",
                "concurrency": "serial",
            }
        )
        result["details"].update(
            {
                "backend": backend,
                "collection": collection_name,
                "collection_deleted": False,
                "import_ms": import_ms,
                "import_includes_embedding_load": True,
                "snapshot_before": snapshot_before,
                "snapshot_after_excluding_own": snapshot_after,
                "old_collections_unchanged": old_unchanged,
                "evidence_verification": verification,
                "initial_count": initial_count,
                "count_after_import": count_after_import,
                "second_import_new": second_import,
            }
        )
    finally:
        if kb is not None:
            if owns_collection:
                try:
                    kb._client.delete_collection(collection_name)
                    collection_deleted = True
                except Exception as ex:
                    result["notes"].append(
                        "清理独立集合失败: "
                        f"{type(ex).__name__}: {str(ex)[:200]}"
                    )
            result["details"]["collection_deleted"] = collection_deleted
            if kb.backend == "local":
                try:
                    cleaned = _cleanup_local_path(fallback_path, run_dir)
                    result["details"]["local_path_cleaned"] = cleaned
                    if not cleaned:
                        result["notes"].append(
                            "本地回退路径不在运行目录内，已保留: "
                            f"{fallback_path}"
                        )
                except Exception as ex:
                    result["details"]["local_path_cleaned"] = False
                    result["notes"].append(
                        "本地回退路径未删除，保留在运行目录内: "
                        f"{type(ex).__name__}: {str(ex)[:200]}"
                    )
    return result


class _GuardedMessages:
    """在真正发送前预留预算，并按 usage 或上界结算。"""

    def __init__(
        self,
        inner: Any,
        guard: BudgetGuard,
        request_log: List[Dict[str, Any]],
        guard_client: "_GuardedClient",
    ):
        self._inner = inner
        self._guard = guard
        self._request_log = request_log
        self._guard_client = guard_client

    async def create(self, **kwargs: Any) -> Any:
        prompt = _request_prompt_text(kwargs)
        stage = _classify_stage(prompt)
        max_tokens = int(kwargs.get("max_tokens") or 0)
        try:
            # 按实际发送的完整请求体计数，包含 system、model 与格式字段。
            input_tokens_upper = request_input_upper_tokens(kwargs)
        except Exception as ex:
            reason = (
                "无法建立完整请求 token 上界: "
                f"{type(ex).__name__}: {str(ex)[:200]}"
            )
            self._guard_client.stop_reason = reason
            self._request_log.append(
                {
                    "request_id": None,
                    "stage": stage,
                    "status": "refused_before_request",
                    "input_count_basis": INPUT_COUNT_BASIS,
                    "error": reason[:300],
                }
            )
            raise BudgetExceeded(reason) from ex
        try:
            reservation: Reservation = self._guard.reserve(
                stage, input_tokens_upper, max_tokens
            )
        except BudgetExceeded as ex:
            self._guard_client.stop_reason = str(ex)
            self._request_log.append(
                {
                    "request_id": None,
                    "stage": stage,
                    "status": "refused_before_request",
                    "input_tokens_upper": input_tokens_upper,
                    "input_count_basis": INPUT_COUNT_BASIS,
                    "error": str(ex)[:300],
                }
            )
            raise

        started = time.monotonic()
        try:
            response = await self._inner.create(**kwargs)
        except Exception as ex:
            latency_ms = (time.monotonic() - started) * 1000
            self._guard.settle_failure(
                reservation,
                f"{type(ex).__name__}: {ex}",
                latency_ms,
            )
            self._request_log.append(
                {
                    "request_id": reservation.request_id,
                    "stage": stage,
                    "status": "failed",
                    "input_tokens_upper": input_tokens_upper,
                    "input_count_basis": INPUT_COUNT_BASIS,
                    "latency_ms": latency_ms,
                }
            )
            raise

        latency_ms = (time.monotonic() - started) * 1000
        usage = getattr(response, "usage", None)
        input_tokens = _usage_value(usage, "input_tokens")
        output_tokens = _usage_value(usage, "output_tokens")
        self._guard.settle_success(
            reservation, input_tokens, output_tokens, latency_ms
        )
        usage_rows = self._guard.usage
        last_usage = usage_rows[-1] if usage_rows else {}
        if self._guard.cost_stop_reason and not self._guard_client.stop_reason:
            # 已发出的请求仍返回响应，但后续预留会被守卫永久拒绝。
            self._guard_client.stop_reason = self._guard.cost_stop_reason
        self._request_log.append(
            {
                "request_id": reservation.request_id,
                "stage": stage,
                "status": "success",
                "model_usage_present": (
                    input_tokens is not None and output_tokens is not None
                ),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_upper": reservation.input_tokens_upper,
                "input_count_basis": INPUT_COUNT_BASIS,
                "upper_bound_cny": reservation.upper_bound_cny,
                "charged_cny": last_usage.get("charged_cny"),
                "reservation_bound_violated": bool(
                    last_usage.get("reservation_bound_violated")
                ),
                "latency_ms": latency_ms,
            }
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GuardedClient:
    """只暴露预算受控的 messages；显式屏蔽未计费的远程 Embedding。"""

    def __init__(
        self,
        inner: Any,
        guard: BudgetGuard,
        request_log: List[Dict[str, Any]],
    ):
        self._inner = inner
        self._guard = guard
        self._request_log = request_log
        self.stop_reason: Optional[str] = None
        self._messages = _GuardedMessages(
            inner.messages, guard, request_log, self
        )

    @property
    def messages(self) -> _GuardedMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        if name == "embeddings":
            return None
        return getattr(self._inner, name)


def _usage_value(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return int(value) if value is not None else None


def _request_prompt_text(kwargs: Dict[str, Any]) -> str:
    parts: List[str] = []
    system = kwargs.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        parts.extend(_content_text(system))
    messages = kwargs.get("messages") or []
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict):
                parts.extend(_content_text(message.get("content")))
            else:
                parts.append(str(message))
    return "\n".join(parts)


def _content_text(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(
                block.get("text"), str
            ):
                parts.append(block["text"])
        return parts
    return [str(content)] if content is not None else []


def _classify_stage(prompt: str) -> str:
    if "你是客服意图分析专家" in prompt:
        return "intent_llm"
    if "将以下用户查询改写" in prompt:
        return "query_rewrite"
    if "根据用户查询，对以下检索结果" in prompt:
        return "rerank"
    return "unknown_model_call"


def _request_ids_since(
    request_log: Sequence[Dict[str, Any]],
    mark: int,
) -> List[str]:
    return [
        item["request_id"]
        for item in request_log[mark:]
        if item.get("request_id")
    ]


def _preflight_request_kwargs(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    """按生产请求的实际字段构造预检请求体。"""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }


def _preflight_upper_bound(
    guard: BudgetGuard,
    intents: Sequence[Dict[str, Any]],
    queries: Sequence[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    from core.intent_recognizer import build_intent_prompt
    from mcp.tool_manager import (
        build_rerank_prompt,
        build_rewrite_prompt,
        retrieval_config,
    )

    settings = retrieval_config(TOP_K)
    intent_upper = 0.0
    for row in intents:
        prompt = build_intent_prompt(
            row.get("message", ""), row.get("history") or []
        )
        request = _preflight_request_kwargs(
            model, prompt, INTENT_MAX_TOKENS, 0.1
        )
        intent_upper += 2 * guard.estimate(
            request_input_upper_tokens(request), INTENT_MAX_TOKENS
        )

    # 正常配置下最多 1 个原查询 + rewrite_n 个子查询，各召回 recall_k 条。
    candidate_upper = (settings["rewrite_n"] + 1) * settings["recall_k"]
    worst_items = [
        {
            "id": f"preflight-{index}",
            "title": "标" * 80,
            "content": "内" * 200,
        }
        for index in range(candidate_upper)
    ]
    retrieval_upper = 0.0
    for row in queries:
        rewrite_prompt = build_rewrite_prompt(
            str(row.get("query", "")), settings["rewrite_n"]
        )
        rewrite_request = _preflight_request_kwargs(
            model, rewrite_prompt, REWRITE_MAX_TOKENS, 0.3
        )
        retrieval_upper += guard.estimate(
            request_input_upper_tokens(rewrite_request),
            REWRITE_MAX_TOKENS,
        )
        rerank_prompt = build_rerank_prompt(
            str(row.get("query", "")), worst_items
        )
        rerank_request = _preflight_request_kwargs(
            model, rerank_prompt, RERANK_MAX_TOKENS, 0.0
        )
        retrieval_upper += guard.estimate(
            request_input_upper_tokens(rerank_request),
            RERANK_MAX_TOKENS,
        )
    total = intent_upper + retrieval_upper
    return {
        "input_count_basis": INPUT_COUNT_BASIS,
        "format_overhead_tokens_per_request": REQUEST_FORMAT_OVERHEAD_TOKENS,
        "intent_attempts_per_case": 2,
        "retrieval_config": settings,
        "rerank_candidate_upper_bound": candidate_upper,
        "intent_upper_bound_cny": intent_upper,
        "retrieval_upper_bound_cny": retrieval_upper,
        "total_upper_bound_cny": total,
        "budget_cny": guard.budget_cny,
        "fits_budget": total <= guard.budget_cny,
        "note": (
            "预检按配置路径构造完整请求体；每次请求前的预留按实际请求体"
            "重新计数，仍是权威上界。"
        ),
    }


def _live_credentials(
    args: argparse.Namespace,
) -> Tuple[str, Optional[str], str]:
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = (
        args.model
        or os.getenv("LLM_MODEL", "").strip()
        or os.getenv("ANTHROPIC_MODEL", "").strip()
    )
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    if not api_key:
        raise BaselineRefused(
            STATUS_BLOCKED_CREDENTIALS,
            "未设置 ANTHROPIC_API_KEY，未创建模型客户端",
        )
    if not model:
        raise BaselineRefused(
            STATUS_BLOCKED_CREDENTIALS,
            "未设置 LLM_MODEL 或 ANTHROPIC_MODEL，未创建模型客户端",
        )
    return api_key, base_url, model


class _ClientLifecycle:
    """记录 live 创建的异步客户端，并保证在与请求相同的事件循环中释放。"""

    def __init__(self) -> None:
        self.clients: List[Any] = []
        self.closed_indices: List[int] = []
        self.errors: List[str] = []
        self._attempted: Set[int] = set()

    def track(self, client: Any) -> None:
        self.clients.append(client)

    async def close_all(self) -> None:
        for index, client in enumerate(self.clients):
            key = id(client)
            if key in self._attempted:
                continue
            self._attempted.add(key)
            close = getattr(client, "close", None)
            if not callable(close):
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
                self.closed_indices.append(index)
            except Exception as ex:
                self.errors.append(
                    f"客户端 {index} 关闭失败: "
                    f"{type(ex).__name__}: {str(ex)[:200]}"
                )

    def close_pending_sync(self) -> None:
        """构造阶段异常时用独立事件循环关闭未使用客户端。"""
        if any(id(client) not in self._attempted for client in self.clients):
            asyncio.run(self.close_all())


async def _run_live_intent(
    recognizer: Any,
    rows: Sequence[Dict[str, Any]],
    request_log: List[Dict[str, Any]],
    guard_client: _GuardedClient,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    artifacts: List[Dict[str, Any]] = []
    intent_scores: List[Dict[str, Any]] = []
    slot_scores: List[Dict[str, Any]] = []
    for row in rows:
        mark = len(request_log)
        started = time.monotonic()
        slot_error = False
        predicted_slots: List[Dict[str, Any]] = []
        try:
            outcome = await recognizer.recognize(
                row["message"], history=row.get("history") or []
            )
            latency_ms = float(getattr(outcome, "latency_ms", 0.0) or 0.0)
            if not latency_ms:
                latency_ms = (time.monotonic() - started) * 1000
            status = str(getattr(outcome, "recognition_status", "ok"))
            is_error = status in {
                "unknown_label",
                "parse_error",
                "request_error",
            }
            if guard_client.stop_reason:
                is_error = True
                status = "budget_exceeded"
            mapped = (
                None
                if is_error
                else map_intent_label(getattr(outcome.intent, "value", None))
            )
            entities = dict(getattr(outcome, "entities", {}) or {})
            predicted_slots = entities_to_slots(entities)
            artifact = {
                "case_id": row["case_id"],
                "split": row["split"],
                "message": row["message"],
                "history_messages": len(row.get("history") or []),
                "raw_intent": getattr(outcome, "raw_intent", None),
                "predicted_intent": getattr(
                    outcome.intent, "value", str(outcome.intent)
                ),
                "mapped_intent": mapped,
                "entities": entities,
                "recognition_status": status,
                "diagnostics": getattr(outcome, "diagnostics", {}),
                "is_error": is_error,
                "error_kind": status if is_error else None,
                "latency_ms": latency_ms,
                "request_ids": _request_ids_since(request_log, mark),
                "slots": predicted_slots,
            }
        except Exception as ex:
            latency_ms = (time.monotonic() - started) * 1000
            slot_error = True
            status = (
                "budget_exceeded"
                if guard_client.stop_reason
                else "exception"
            )
            artifact = {
                "case_id": row["case_id"],
                "split": row["split"],
                "message": row["message"],
                "history_messages": len(row.get("history") or []),
                "raw_intent": None,
                "predicted_intent": None,
                "mapped_intent": None,
                "entities": {},
                "recognition_status": status,
                "diagnostics": {},
                "is_error": True,
                "error_kind": status,
                "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "latency_ms": latency_ms,
                "request_ids": _request_ids_since(request_log, mark),
                "slots": [],
            }
        artifacts.append(artifact)
        intent_scores.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "gold": row.get("primary_intent"),
                "prediction": artifact["mapped_intent"],
                "is_error": artifact["is_error"],
                "error_kind": artifact["error_kind"],
            }
        )
        slot_scores.append(
            {
                "case_id": row["case_id"],
                "split": row["split"],
                "gold_slots": row["slots"],
                "predicted_slots": predicted_slots,
                "is_error": slot_error,
            }
        )
        if guard_client.stop_reason:
            break
    return artifacts, intent_scores, slot_scores


async def _run_live_pipeline(
    manager: Any,
    rows: Sequence[Dict[str, Any]],
    request_log: List[Dict[str, Any]],
    guard_client: _GuardedClient,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    artifacts: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for row in rows:
        mark = len(request_log)
        diagnostics: Dict[str, Any] = {}
        started = time.monotonic()
        budget_stop = False
        try:
            result = await manager.search_with_rewrite(
                "knowledge_search",
                row["query"],
                top_k=TOP_K,
                diagnostics=diagnostics,
            )
            latency_ms = (time.monotonic() - started) * 1000
            budget_stop = bool(guard_client.stop_reason)
            success = bool(getattr(result, "success", False))
            ranking = []
            if success and isinstance(getattr(result, "data", None), list):
                for item in result.data:
                    if isinstance(item, dict):
                        ranking.append(
                            str(
                                item.get("evidence_id")
                                or item.get("id")
                                or ""
                            )
                        )
                    else:
                        ranking.append(str(item))
            ranking = [value for value in ranking if value]
            is_error = not success or budget_stop
            final_items = (
                [
                    _retrieval_source_item(item)
                    for item in result.data
                    if isinstance(item, dict)
                ]
                if success and isinstance(getattr(result, "data", None), list)
                else []
            )
            artifact = {
                "query_id": row["query_id"],
                "split": row["split"],
                "mode": "P",
                "query": row["query"],
                "ranking": ranking,
                "final_items": final_items,
                "final_ids": diagnostics.get("final_ids"),
                "pre_rerank_ids": diagnostics.get("pre_rerank_ids"),
                "reranked": bool(getattr(result, "reranked", False)),
                "degraded": bool(getattr(result, "degraded", False)),
                "is_error": is_error,
                "error_kind": (
                    "budget_exceeded"
                    if budget_stop
                    else "retrieval_failed"
                    if not success
                    else None
                ),
                "error": getattr(result, "error", None),
                "diagnostics": diagnostics,
                "budget_stop": budget_stop,
                "latency_ms": latency_ms,
                "request_ids": _request_ids_since(request_log, mark),
            }
        except Exception as ex:
            latency_ms = (time.monotonic() - started) * 1000
            budget_stop = bool(guard_client.stop_reason)
            artifact = {
                "query_id": row["query_id"],
                "split": row["split"],
                "mode": "P",
                "query": row["query"],
                "ranking": [],
                "final_items": [],
                "final_ids": diagnostics.get("final_ids"),
                "pre_rerank_ids": diagnostics.get("pre_rerank_ids"),
                "reranked": False,
                "degraded": False,
                "is_error": True,
                "error_kind": (
                    "budget_exceeded"
                    if budget_stop
                    else "pipeline_exception"
                ),
                "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "diagnostics": diagnostics,
                "budget_stop": budget_stop,
                "latency_ms": latency_ms,
                "request_ids": _request_ids_since(request_log, mark),
            }
        artifacts.append(artifact)
        scores.append(
            _retrieval_score_record(
                row,
                artifact["ranking"],
                is_error=artifact["is_error"],
                note=(
                    "预算耗尽，未完成付费链路"
                    if artifact["budget_stop"]
                    else artifact.get("error") or ""
                ),
            )
        )
        if guard_client.stop_reason:
            break
    return artifacts, scores


async def _run_live_stages(
    recognizer: Any,
    manager: Any,
    intents: Sequence[Dict[str, Any]],
    queries: Sequence[Dict[str, Any]],
    request_log: List[Dict[str, Any]],
    guard_client: _GuardedClient,
    lifecycle: _ClientLifecycle,
) -> Dict[str, Any]:
    """在同一事件循环内串行执行选中的组件，并在所有退出路径释放客户端。"""
    outcome: Dict[str, Any] = {
        "intent": ([], [], []),
        "pipeline": ([], []),
        "intent_skipped": recognizer is None,
        "pipeline_skipped": manager is None,
    }
    try:
        if recognizer is not None:
            outcome["intent"] = await _run_live_intent(
                recognizer, intents, request_log, guard_client
            )
        if manager is None or guard_client.stop_reason:
            outcome["pipeline_skipped"] = True
        else:
            outcome["pipeline"] = await _run_live_pipeline(
                manager, queries, request_log, guard_client
            )
        return outcome
    finally:
        await lifecycle.close_all()


def _load_vector_reference(
    source_dir: Path,
    queries: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """读取原方案已产生的真实 V 排序，作为未改检索层的共用参考。

    参考运行必须覆盖本轮全部查询；这里只复用排序与来源字段，不使用其
    gold 或分数，评分仍按当前冻结数据重算。
    """
    predictions = source_dir / "retrieval_predictions.jsonl"
    if not predictions.is_file():
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            f"V 参考缺少 retrieval_predictions.jsonl: {source_dir}",
        )
    rows = _read_jsonl(predictions)
    by_id = {
        str(row.get("query_id")): row
        for row in rows
        if row.get("mode") == "V"
    }
    expected = [row["query_id"] for row in queries]
    missing = [query_id for query_id in expected if query_id not in by_id]
    if missing:
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            "V 参考未覆盖本轮全部检索问题",
            {"missing_ids": missing, "source_dir": str(source_dir)},
        )
    artifacts: List[Dict[str, Any]] = []
    scores: List[Dict[str, Any]] = []
    for row in queries:
        source = by_id[row["query_id"]]
        artifact = dict(source)
        artifact["query"] = row["query"]
        artifact["split"] = row["split"]
        artifact["mode"] = "V"
        artifacts.append(artifact)
        scores.append(
            _retrieval_score_record(
                row,
                artifact.get("ranking") or [],
                is_error=bool(artifact.get("is_error")),
                note="复用原方案 V 参考排序",
            )
        )
    meta = {
        "source_dir": str(source_dir),
        "predictions_sha256": _sha256_file(predictions),
        "queries": len(artifacts),
        "shared_not_rerun": True,
    }
    return artifacts, scores, meta


def run_live(
    args: argparse.Namespace,
    dataset: Dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """在显式预算与单价齐备后运行意图、V 和 P；本任务不执行此模式。"""
    if args.budget_cny is None:
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            "live 模式需要维护者明确提供 --budget-cny；未创建模型客户端",
        )
    if (
        args.input_price_cny_per_mtok is None
        or args.output_price_cny_per_mtok is None
    ):
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            "live 模式需要显式提供当前官方输入／输出单价；未创建模型客户端",
        )
    if args.budget_cny <= 0:
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            "预算必须大于 0；未创建模型客户端",
        )
    if (
        args.input_price_cny_per_mtok <= 0
        or args.output_price_cny_per_mtok <= 0
    ):
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            "输入／输出单价必须大于 0；未创建模型客户端",
        )

    splits = _selected_splits(args.split)
    component = getattr(args, "component", "both")
    run_intent = component in ("both", "intent")
    run_retrieval = component in ("both", "retrieval")
    intents = _selected(dataset["intents"], args.split)
    queries = _selected(dataset["queries"], args.split)
    intent_filter = getattr(args, "intent_ids", None)
    query_filter = getattr(args, "query_ids", None)
    intents = _select_by_ids(intents, "case_id", intent_filter, "意图")
    queries = _select_by_ids(queries, "query_id", query_filter, "检索")
    if intent_filter is not None and not run_intent:
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            "component 未包含意图识别，不能指定意图子集",
        )
    if query_filter is not None and not run_retrieval:
        raise BaselineRefused(
            STATUS_REFUSED_SELECTION,
            "component 未包含检索，不能指定检索子集",
        )
    filtered = intent_filter is not None or query_filter is not None
    intents = intents if run_intent else []
    queries = queries if run_retrieval else []

    # 跨阶段账本：未结算的已启动记录按计划上界占用，避免重跑清零。
    ledger_path = (
        Path(getattr(args, "budget_ledger", None)).expanduser().resolve()
        if getattr(args, "budget_ledger", None)
        else (run_dir / "budget_ledger.json").resolve()
    )
    ledger = _read_budget_ledger(ledger_path, args.budget_cny)
    committed_before = _ledger_committed(ledger)
    available_cny = float(args.budget_cny) - committed_before
    if available_cny <= 0:
        raise BaselineRefused(
            STATUS_REFUSED_BUDGET,
            "累计账本已无剩余额度，未创建模型客户端",
            {
                "ledger": str(ledger_path),
                "limit_cny": float(args.budget_cny),
                "committed_cny": committed_before,
            },
        )
    guard = BudgetGuard(
        budget_cny=available_cny,
        input_price_cny_per_mtok=args.input_price_cny_per_mtok,
        output_price_cny_per_mtok=args.output_price_cny_per_mtok,
    )
    api_key, base_url, model = _live_credentials(args)
    from mcp import knowledge_base as kb_module
    from mcp import tool_manager as tool_manager_module
    from mcp.tool_manager import Tool, retrieval_config
    from core import intent_recognizer as intent_module

    # 预检与运行共用生产配置，避免只改评测常量或内部召回范围。
    retrieval_settings = retrieval_config(TOP_K)
    preflight = _preflight_upper_bound(guard, intents, queries, model)
    preflight.update(
        {
            "component": component,
            "authorized_limit_cny": float(args.budget_cny),
            "committed_before_cny": committed_before,
            "available_budget_cny": available_cny,
            "ledger": str(ledger_path),
        }
    )
    if not preflight["fits_budget"]:
        raise BaselineRefused(
            STATUS_REFUSED_PREFLIGHT,
            "保守费用上界超过授权预算，未创建模型客户端",
            {"preflight": preflight},
        )

    if run_retrieval and not kb_module._embedding_cache_ready():
        raise BaselineRefused(
            STATUS_BLOCKED_EMBEDDING,
            "Embedding 本地缓存不完整，按约束不下载模型",
            {"preflight": preflight},
        )

    host, port, fallback_path = (
        _chroma_config(args, run_dir)
        if run_retrieval
        else (None, None, None)
    )
    collection_name = _collection_name(run_id)
    result = _base_result(args.mode, args.split)
    result["details"].update(
        {
            "budget_preflight": preflight,
            "requested_counts": {
                "intent": len(intents),
                "retrieval": len(queries),
            },
            "selection": {
                "filtered": filtered,
                "split": args.split,
                "component": component,
                "intent_ids": [row["case_id"] for row in intents],
                "query_ids": [row["query_id"] for row in queries],
                "expected_intent": len(intents),
                "expected_queries": len(queries),
            },
            "experiment": {
                "ab_label": getattr(args, "ab_label", None),
                "config_label": getattr(args, "config_label", None),
                "stage_label": getattr(args, "stage_label", None),
                "component": component,
                "budget_ledger": str(ledger_path),
                "vector_from": (
                    str(Path(args.vector_from).expanduser().resolve())
                    if getattr(args, "vector_from", None)
                    else None
                ),
            },
            "budget_ledger": {
                "path": str(ledger_path),
                "limit_cny": float(args.budget_cny),
                "committed_before_cny": committed_before,
                "available_cny": available_cny,
            },
        }
    )
    if filtered:
        result["notes"].append(
            "本轮为指定子集烟测，不代表完整固定集基线。"
        )
    result["config"].update(
        {
            "model": model,
            "base_url_configured": bool(base_url),
            "component": component,
            "ab_label": getattr(args, "ab_label", None),
            "config_label": getattr(args, "config_label", None),
            "stage_label": getattr(args, "stage_label", None),
            "intent": {
                "enabled": run_intent,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
                "max_tokens_per_attempt": INTENT_MAX_TOKENS,
                "llm_retry_attempts": 2,
                "embedding_backend": "local_hash_char_ngram_256",
                "embedding_note": (
                    "AsyncAnthropic 0.40.0 无 embeddings 资源，"
                    "客户端被显式屏蔽远程 Embedding"
                ),
            },
            "retrieval": {
                "enabled": run_retrieval,
                "top_k": retrieval_settings["top_k"],
                "recall_k": retrieval_settings["recall_k"],
                "rewrite_n": retrieval_settings["rewrite_n"],
                "rewrite_max_tokens": REWRITE_MAX_TOKENS,
                "rerank_max_tokens": RERANK_MAX_TOKENS,
                "cache_ttl_s": CACHE_TTL_S,
                "original_query_included": retrieval_settings[
                    "original_query_included"
                ],
            },
            "chroma": (
                {
                    "host": host,
                    "port": port,
                    "collection": collection_name,
                    "load_demo_documents": False,
                    "fallback_path": fallback_path,
                }
                if run_retrieval
                else {"status": "NOT_RUN_COMPONENT"}
            ),
            "embedding": {
                "backend": "onnx_all-MiniLM-L6-v2",
                "cache_path": str(
                    kb_module._ProjectEmbeddingFunction.DOWNLOAD_PATH / "onnx"
                ),
                "used": run_retrieval,
            },
            "cache": "cold_fresh_recognizer_and_tool_manager",
            "concurrency": "serial",
            "input_accounting": {
                "basis": INPUT_COUNT_BASIS,
                "format_overhead_tokens": REQUEST_FORMAT_OVERHEAD_TOKENS,
            },
        }
    )

    request_log: List[Dict[str, Any]] = []
    lifecycle = _ClientLifecycle()
    guard_client: Optional[_GuardedClient] = None
    kb = None
    backend = "NOT_REQUIRED"
    imported = 0
    second_import = 0
    verification: Dict[str, Any] = {"status": "NOT_RUN_COMPONENT"}
    original_intent_factory = intent_module.AsyncAnthropic
    original_tool_factory = tool_manager_module.AsyncAnthropic
    owns_collection = False
    collection_deleted = False
    # 在发出任何付费请求前，把本轮可能占用的上界写入累计账本。
    _ledger_start(
        ledger,
        run_id,
        getattr(args, "stage_label", None) or component,
        preflight["total_upper_bound_cny"],
        available_cny,
    )
    _write_budget_ledger(ledger_path, ledger)
    ledger_status = "started"
    try:
        if run_retrieval:
            kb = kb_module.KnowledgeBase(
                chroma_host=host,
                chroma_port=port,
                chroma_path=fallback_path,
                collection_name=collection_name,
                load_demo_documents=False,
            )
            backend = kb.backend
            initial_count = kb.doc_count
            if initial_count:
                raise BaselineRefused(
                    STATUS_BLOCKED_STORAGE,
                    f"独立集合 {collection_name} 非空，拒绝复用未知内容",
                    {"count": initial_count},
                )
            owns_collection = True
            import_started = time.monotonic()
            imported = kb.upsert_evidence(dataset["knowledge"])
            import_ms = (time.monotonic() - import_started) * 1000
            verification = _verify_evidence(kb, dataset["knowledge"])
            second_import = kb.upsert_evidence(dataset["knowledge"])
            if not verification["passed"]:
                raise BaselineRefused(
                    STATUS_BLOCKED_STORAGE,
                    "知识证据导入校验失败",
                    {"verification": verification},
                )
            result["details"].update(
                {
                    "backend": backend,
                    "collection": collection_name,
                    "import_ms": import_ms,
                    "import_includes_embedding_load": True,
                    "imported_new": imported,
                    "second_import_new": second_import,
                    "evidence_verification": verification,
                }
            )

        guarded_clients: List[_GuardedClient] = []

        def guarded_factory(**kwargs: Any) -> _GuardedClient:
            inner = original_intent_factory(**kwargs)
            lifecycle.track(inner)
            wrapped = _GuardedClient(inner, guard, request_log)
            guarded_clients.append(wrapped)
            return wrapped

        def guarded_tool_factory(**kwargs: Any) -> _GuardedClient:
            inner = original_tool_factory(**kwargs)
            lifecycle.track(inner)
            wrapped = _GuardedClient(inner, guard, request_log)
            guarded_clients.append(wrapped)
            return wrapped

        # 两个模块各自持有 AsyncAnthropic 引用，只替换本轮实际运行的组件。
        recognizer = None
        tool_manager = None
        if run_intent:
            intent_module.AsyncAnthropic = guarded_factory
            recognizer = intent_module.IntentRecognizer(
                api_key=api_key,
                base_url=base_url,
                model=model,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
        if run_retrieval:
            tool_manager_module.AsyncAnthropic = guarded_tool_factory
            tool_manager = tool_manager_module.MCPToolManager(
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            tool_manager.register(
                Tool(
                    name="knowledge_search",
                    description="搜索零售知识库（ChromaDB 向量检索）",
                    handler=kb.search_handler,
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                    cache_ttl=CACHE_TTL_S,
                    timeout_s=30.0,
                    supports_rerank=True,
                )
            )
        guard_client = guarded_clients[0] if guarded_clients else None
        if guard_client is None:
            raise BaselineRefused(
                STATUS_REFUSED_SELECTION,
                "未创建任何组件客户端",
            )

        known_ids = {row["evidence_id"] for row in dataset["knowledge"]}
        if run_retrieval:
            vector_reference = None
            vector_from = getattr(args, "vector_from", None)
            if vector_from:
                # 未改检索层时复用原方案的真实 V 排序，不重算也不写成新运行。
                v_artifacts, v_scores, vector_reference = _load_vector_reference(
                    Path(vector_from).expanduser().resolve(),
                    queries,
                )
            else:
                # V 不产生模型费用，先跑完；随后 P 受同一预算守卫。
                v_artifacts, v_scores = _run_vector_queries(kb, queries)
            _fill_vector_result(
                result,
                dataset,
                splits,
                args.split,
                v_artifacts,
                v_scores,
                known_ids,
                expected_ids=[row["query_id"] for row in queries],
            )
            result["retrieval"]["V"]["source"] = (
                "reference_run" if vector_reference else "current_run"
            )
            if vector_reference:
                result["retrieval"]["V"]["reference"] = vector_reference
                result["details"]["vector_reference"] = vector_reference
            v_complete = result["details"].get("retrieval_V_complete", False)
            result["stages"]["vector_baseline"] = (
                "COMPLETE"
                if v_complete
                and result["details"].get("vector_protocol_clean")
                else "PROTOCOL_ISSUES"
                if v_complete
                else "FAILED"
            )
            result["stages"]["storage_integration"] = (
                "VERIFIED" if backend == "server" else "LOCAL_FALLBACK"
            )
        else:
            result["retrieval"]["V"] = {"status": "NOT_RUN_COMPONENT"}
            result["retrieval"]["P"] = {"status": "NOT_RUN_COMPONENT"}
            result["stages"]["vector_baseline"] = "NOT_RUN_COMPONENT"
            result["stages"]["storage_integration"] = "NOT_RUN_COMPONENT"
            result["coverage"]["retrieval_V"] = _coverage_for([], [])

        # 选中组件的全部模型调用、预算停止和客户端释放在同一事件循环。
        live_stages = asyncio.run(
            _run_live_stages(
                recognizer,
                tool_manager,
                intents,
                queries,
                request_log,
                guard_client,
                lifecycle,
            )
        )
        result["details"]["client_lifecycle"] = {
            "event_loop_scope": "single_asyncio_run",
            "clients_created": len(lifecycle.clients),
            "clients_closed": len(lifecycle.closed_indices),
            "close_errors": list(lifecycle.errors),
        }
        intent_artifacts: List[Dict[str, Any]] = []
        intent_scores: List[Dict[str, Any]] = []
        slot_scores: List[Dict[str, Any]] = []
        p_artifacts: List[Dict[str, Any]] = []
        p_scores: List[Dict[str, Any]] = []
        if run_intent:
            intent_artifacts, intent_scores, slot_scores = live_stages["intent"]
        if run_retrieval:
            p_artifacts, p_scores = live_stages["pipeline"]
            if live_stages["pipeline_skipped"]:
                result["notes"].append(
                    f"检索阶段预算停止: {guard_client.stop_reason}"
                )

        if run_intent:
            intent_metrics = _per_split(intent_scores, splits, score_intents)
            slot_metrics = _per_split(slot_scores, splits, score_slots)
            rows_by_id = {row["case_id"]: row for row in dataset["intents"]}
            expected_intent = [row["case_id"] for row in intents]
            attempted_intent = [row["case_id"] for row in intent_artifacts]
            result["intent"] = intent_metrics
            result["slots"] = slot_metrics
            result["coverage"]["intent"] = _coverage_for(
                expected_intent, attempted_intent
            )
            result["latency"]["intent"] = _latency_by_split(
                intent_artifacts, splits
            )
            result["errors"]["intent"] = _intent_error_summary(
                intent_scores, rows_by_id, intent_artifacts
            )
            result["errors"]["slots"] = _slot_error_summary(
                {
                    "cases": [
                        case
                        for split_metrics in slot_metrics.values()
                        for case in split_metrics.get("cases", [])
                    ]
                },
                rows_by_id,
            )
            result["intent_predictions"] = intent_artifacts
            result["stages"]["live_intent"] = (
                "COMPLETE"
                if _attempted_status(
                    expected_intent, attempted_intent
                )
                == "COMPLETE"
                else "PARTIAL"
            )
        else:
            result["stages"]["live_intent"] = "NOT_RUN_COMPONENT"
            result["intent"] = {"status": "NOT_RUN_COMPONENT"}
            result["slots"] = {"status": "NOT_RUN_COMPONENT"}
            result["coverage"]["intent"] = _coverage_for([], [])

        if run_retrieval:
            p_metrics = _per_split(
                p_scores,
                splits,
                lambda subset: compute_retrieval_metrics(
                    subset, known_ids=known_ids
                ),
            )
            result["retrieval"]["P"] = {
                "status": "SCORED" if p_artifacts else "NOT_RUN",
                "mode": "P",
                "top_k": retrieval_settings["top_k"],
                "rewrite_n": retrieval_settings["rewrite_n"],
                "recall_k": retrieval_settings["recall_k"],
                "per_split": p_metrics,
                "latency": _latency_by_split(p_artifacts, splits),
                "protocol_clean": all(
                    item.get("protocol_clean") for item in p_metrics.values()
                ),
            }
            result["latency"]["retrieval_P"] = _latency_by_split(
                p_artifacts, splits
            )
            expected_queries = [row["query_id"] for row in queries]
            attempted_queries = [row["query_id"] for row in p_artifacts]
            result["coverage"]["retrieval_P"] = _coverage_for(
                expected_queries, attempted_queries
            )
            result["retrieval_predictions"] = list(v_artifacts) + list(
                p_artifacts
            )
            result["errors"]["retrieval_P"] = _retrieval_error_summary(
                p_artifacts, p_metrics, "P"
            )
            result["stages"]["live_pipeline"] = (
                "COMPLETE"
                if _attempted_status(
                    expected_queries, attempted_queries
                )
                == "COMPLETE"
                else "PARTIAL"
            )
            result["coverage"]["knowledge"] = {
                "expected": len(dataset["knowledge"]),
                "actual": kb.doc_count,
                "imported_new": imported,
                "second_import_new": second_import,
                "verification": verification,
                "status": "VERIFIED" if verification["passed"] else "FAILED",
            }
        else:
            result["stages"]["live_pipeline"] = "NOT_RUN_COMPONENT"
            result["retrieval"]["P"] = {"status": "NOT_RUN_COMPONENT"}
            result["coverage"]["retrieval_P"] = _coverage_for([], [])
            result["coverage"]["knowledge"] = {
                "expected": len(dataset["knowledge"]),
                "actual": None,
                "status": "NOT_RUN_COMPONENT",
            }

        result["cost"] = {
            **guard.summary(),
            "basis": INPUT_COUNT_BASIS,
            "vendor_bill_verified": False,
            "authorized_limit_cny": float(args.budget_cny),
            "committed_before_cny": committed_before,
            "available_budget_cny": available_cny,
            "ledger_path": str(ledger_path),
            "component": component,
        }
        usage_rows = list(guard.usage)
        usage_rows.append(
            {
                "type": "summary",
                **guard.summary(),
                "vendor_bill_verified": False,
            }
        )
        result["usage"] = usage_rows
        if run_intent and run_retrieval:
            complete = (
                result["stages"]["live_intent"] == "COMPLETE"
                and result["stages"]["live_pipeline"] == "COMPLETE"
                and result["stages"]["vector_baseline"] == "COMPLETE"
                and result["stages"]["storage_integration"] == "VERIFIED"
                and not guard.cost_stop_reason
            )
        elif run_intent:
            complete = (
                result["stages"]["live_intent"] == "COMPLETE"
                and not guard.cost_stop_reason
            )
        else:
            complete = (
                result["stages"]["live_pipeline"] == "COMPLETE"
                and result["stages"]["vector_baseline"] == "COMPLETE"
                and result["stages"]["storage_integration"] == "VERIFIED"
                and not guard.cost_stop_reason
            )
        if complete and filtered:
            # 子集通过只说明烟测范围可用，不能签收完整固定集基线。
            result["status"] = STATUS_SMOKE_COMPLETE
        elif complete and component != "both":
            result["status"] = STATUS_COMPONENT_COMPLETE
        elif complete:
            result["status"] = STATUS_COMPLETE
        else:
            result["status"] = STATUS_PARTIAL
            if run_retrieval and backend != "server":
                result["blockers"].append("PENDING_STORAGE")
            if guard_client.stop_reason:
                result["blockers"].append("BUDGET_EXHAUSTED")
            if guard.cost_stop_reason:
                result["blockers"].append("COST_UPPER_BOUND_VIOLATED")
                result["notes"].append(
                    "实际费用超过预留上界，已永久停止后续请求: "
                    + guard.cost_stop_reason
                )
        ledger_status = (
            "complete"
            if result["status"] in (STATUS_COMPLETE, STATUS_COMPONENT_COMPLETE)
            else "partial"
            if result["status"] == STATUS_SMOKE_COMPLETE
            else "failed"
        )
    finally:
        intent_module.AsyncAnthropic = original_intent_factory
        tool_manager_module.AsyncAnthropic = original_tool_factory
        try:
            # 主循环已释放时为空操作；构造阶段异常时关闭未使用的客户端。
            lifecycle.close_pending_sync()
        except Exception as ex:
            result["notes"].append(
                "关闭模型客户端失败: "
                f"{type(ex).__name__}: {str(ex)[:200]}"
            )
        result["details"]["client_lifecycle"] = {
            "event_loop_scope": "single_asyncio_run",
            "clients_created": len(lifecycle.clients),
            "clients_closed": len(lifecycle.closed_indices),
            "close_errors": list(lifecycle.errors),
        }
        if kb is not None:
            if owns_collection:
                try:
                    kb._client.delete_collection(collection_name)
                    collection_deleted = True
                except Exception as ex:
                    result["notes"].append(
                        "清理独立集合失败: "
                        f"{type(ex).__name__}: {str(ex)[:200]}"
                    )
            result["details"]["collection_deleted"] = collection_deleted
            if kb.backend == "local":
                try:
                    cleaned = _cleanup_local_path(fallback_path, run_dir)
                    result["details"]["local_path_cleaned"] = cleaned
                    if not cleaned:
                        result["notes"].append(
                            "本地回退路径不在运行目录内，已保留: "
                            f"{fallback_path}"
                        )
                except Exception as ex:
                    result["details"]["local_path_cleaned"] = False
                    result["notes"].append(
                        "本地回退路径未删除，保留在运行目录内: "
                        f"{type(ex).__name__}: {str(ex)[:200]}"
                    )
        try:
            # 无论成功或异常都落账；未结算的在途预留保留为占用。
            summary = guard.summary()
            _ledger_finish(
                ledger,
                run_id,
                {
                    "status": (
                        ledger_status
                        if ledger_status != "started"
                        else "failed"
                    ),
                    "spent_cny": summary["spent_cny"],
                    "unsettled_cny": summary["reserved_cny"],
                    "requests": summary["requests"],
                    "cost_stop_reason": summary["cost_stop_reason"],
                    "result_status": result.get("status"),
                    "available_before_cny": available_cny,
                },
            )
            _write_budget_ledger(ledger_path, ledger)
        except Exception as ex:
            result["notes"].append(
                "累计账本写入失败，费用占用仍按本次内存结果记录: "
                f"{type(ex).__name__}: {str(ex)[:200]}"
            )
    return result


def _refused_result(
    args: argparse.Namespace,
    refused: BaselineRefused,
) -> Dict[str, Any]:
    result = _base_result(args.mode, args.split)
    result["status"] = refused.status
    result["details"]["refusal"] = {
        "status": refused.status,
        "reason": refused.reason,
        **refused.details,
    }
    result["blockers"].append(refused.reason)
    if args.mode == "live":
        component = getattr(args, "component", "both")
        result["stages"]["live_intent"] = (
            "REFUSED"
            if component in ("both", "intent")
            else "NOT_RUN_COMPONENT"
        )
        result["stages"]["live_pipeline"] = (
            "REFUSED"
            if component in ("both", "retrieval")
            else "NOT_RUN_COMPONENT"
        )
    result["usage"] = [
        _empty_usage_row(
            args.mode,
            "请求前已拒绝：" + refused.reason,
        )
    ]
    return result


def _error_result(
    args: argparse.Namespace,
    error: BaseException,
) -> Dict[str, Any]:
    result = _base_result(args.mode, args.split)
    result["status"] = STATUS_ERROR
    result["details"]["error"] = {
        "type": type(error).__name__,
        "message": str(error)[:1000],
        "traceback": traceback.format_exc()[-4000:],
    }
    result["usage"] = [
        _empty_usage_row(
            args.mode,
            "运行异常终止；见 run.json 的 details.error",
        )
    ]
    return result


def _build_coverage(
    dataset: Dict[str, Any],
    args: argparse.Namespace,
    result: Dict[str, Any],
) -> None:
    """补齐模式未显式填写的覆盖项，避免把缺失误写成通过。"""
    expected_intent = _split_expected_ids(
        dataset["intents"], "case_id", args.split
    )
    expected_queries = _split_expected_ids(
        dataset["queries"], "query_id", args.split
    )
    coverage = result.setdefault("coverage", {})
    coverage.setdefault(
        "intent", _coverage_for(expected_intent, [])
    )
    coverage.setdefault(
        "retrieval_V", _coverage_for(expected_queries, [])
    )
    coverage.setdefault(
        "retrieval_P", _coverage_for(expected_queries, [])
    )
    coverage.setdefault(
        "knowledge",
        {
            "expected": len(dataset["knowledge"]),
            "actual": None,
            "status": "NOT_IMPORTED",
        },
    )


def _write_artifacts(
    run_dir: Path,
    run_meta: Dict[str, Any],
    metrics: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    _write_json(run_dir / "run.json", run_meta)
    _write_json(run_dir / "metrics.json", metrics)
    _write_jsonl(
        run_dir / "intent_predictions.jsonl",
        result.get("intent_predictions") or [],
    )
    _write_jsonl(
        run_dir / "retrieval_predictions.jsonl",
        result.get("retrieval_predictions") or [],
    )
    _write_jsonl(
        run_dir / "usage.jsonl",
        result.get("usage") or [],
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.run_retail_baseline",
        description=(
            "零售意图／槽位与知识检索组件基线。默认只跑零费用的 "
            "offline-check；vector 使用本地 Embedding 缓存和独立 Chroma "
            "集合；live 必须在显式预算与当前单价齐备后才会创建模型客户端。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  uv run --locked --offline python -m evaluation.run_retail_baseline "
            "--mode offline-check\n"
            "  uv run --locked --offline python -m evaluation.run_retail_baseline "
            "--mode vector --split dev\n"
            "  # live 需维护者另行授权预算，且必须显式提供当前官方单价：\n"
            "  uv run --locked --offline python -m evaluation.run_retail_baseline "
            "--mode live --split dev --budget-cny <上限> \\\n"
            "    --input-price-cny-per-mtok <输入单价> "
            "--output-price-cny-per-mtok <输出单价>\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("offline-check", "vector", "live"),
        default="offline-check",
        help="offline-check 不建模型／存储客户端；vector 只跑 V；live 需预算",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "all"),
        default="all",
        help="评测划分；默认 all（dev＋test）",
    )
    parser.add_argument(
        "--component",
        choices=("both", "intent", "retrieval"),
        default="both",
        help=(
            "live 模式运行的组件；默认 both 保持原行为，"
            "intent／retrieval 只运行单一组件"
        ),
    )
    parser.add_argument(
        "--budget-ledger",
        default=None,
        help=(
            "live 模式跨阶段累计账本路径；默认运行目录内 budget_ledger.json，"
            "分阶段实验须显式传同一路径"
        ),
    )
    parser.add_argument(
        "--stage-label",
        default=None,
        help="本轮阶段标签，如 A-dev、B-test",
    )
    parser.add_argument(
        "--ab-label",
        default=None,
        help="方案标签，如 A 或 B",
    )
    parser.add_argument(
        "--config-label",
        default=None,
        help="冻结配置标签，用于核对 A/B 实际参数",
    )
    parser.add_argument(
        "--vector-from",
        default=None,
        help=(
            "复用指定运行目录的真实 V 排序；仅检索组件使用，"
            "不重跑 V 也不写成独立运行"
        ),
    )
    parser.add_argument(
        "--intent-ids",
        default=None,
        help="live 模式可选：逗号分隔的意图样本 ID，必须属于所选划分",
    )
    parser.add_argument(
        "--query-ids",
        default=None,
        help="live 模式可选：逗号分隔的检索问题 ID，必须属于所选划分",
    )
    parser.add_argument(
        "--budget-cny",
        type=float,
        default=None,
        help="live 模式维护者明确授权的总预算（人民币）",
    )
    parser.add_argument(
        "--input-price-cny-per-mtok",
        type=float,
        default=None,
        help="live 模式当前官方输入单价（元/百万 token），不得沿用旧价",
    )
    parser.add_argument(
        "--output-price-cny-per-mtok",
        type=float,
        default=None,
        help="live 模式当前官方输出单价（元/百万 token），不得沿用旧价",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="覆盖 LLM_MODEL；仅 live 模式使用",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="输出目录；默认 data/evaluation/retail/<run_id>",
    )
    parser.add_argument(
        "--chroma-host",
        default=None,
        help="Chroma 服务地址；默认 CHROMA_HOST 或 127.0.0.1",
    )
    parser.add_argument(
        "--chroma-port",
        type=int,
        default=None,
        help="Chroma 服务端口；默认 CHROMA_PORT 或 8002",
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="HTTP 不可用时的本地回退路径；默认运行目录内隔离路径",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_id = _new_run_id(args.mode)
    try:
        run_dir = _prepare_run_dir(run_id, args.run_dir)
    except BaselineRefused as refused:
        print(f"输出目录不可用: {refused.reason}", file=sys.stderr)
        return 2

    started_at = _utc_now()
    code = _code_snapshot()
    data_before = _hash_dataset(DATA_ROOT)
    result = _base_result(args.mode, args.split)
    dataset: Optional[Dict[str, Any]] = None
    validation_before: Optional[Dict[str, Any]] = None
    validation_after: Optional[Dict[str, Any]] = None

    try:
        validation_before = validate(DATA_ROOT)
        dataset = _load_dataset(DATA_ROOT)
        if args.mode == "offline-check":
            result = run_offline_check(args, dataset)
        elif args.mode == "vector":
            result = run_vector(args, dataset, run_id, run_dir)
        else:
            result = run_live(args, dataset, run_id, run_dir)
    except BaselineRefused as refused:
        result = _refused_result(args, refused)
    except Exception as ex:
        result = _error_result(args, ex)
    finally:
        data_after = _hash_dataset(DATA_ROOT)
        try:
            validation_after = validate(DATA_ROOT)
        except Exception as ex:
            validation_after = {
                "status": "FAIL",
                "error": f"{type(ex).__name__}: {str(ex)[:500]}",
            }

    if dataset is not None:
        _build_coverage(dataset, args, result)

    data_unchanged = data_before == data_after
    validation_ok = (
        (validation_before or {}).get("status") == "PASS"
        and (validation_after or {}).get("status") == "PASS"
    )
    if not data_unchanged or not validation_ok:
        result["status"] = STATUS_DATA_FAILED
        result["blockers"].append("DATA_INTEGRITY_FAILED")
        if validation_before is not None and validation_before.get(
            "status"
        ) != "PASS":
            result["notes"].append(
                "运行前资料校验未通过: "
                + json.dumps(validation_before, ensure_ascii=False)[:500]
            )

    exit_code = _exit_code(args.mode, result, data_unchanged, validation_ok)
    finished_at = _utc_now()
    selection_meta = result.get("details", {}).get("selection") or {}
    run_meta = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "mode": args.mode,
        "split": args.split,
        "component": getattr(args, "component", "both"),
        "status": result["status"],
        "exit_code": exit_code,
        "stages": result["stages"],
        "blockers": result["blockers"],
        "notes": result["notes"],
        "code": code,
        "data": {
            "version": data_before.get("version"),
            "before": data_before,
            "after": data_after,
            "unchanged": data_unchanged,
            "validation_before": validation_before,
            "validation_after": validation_after,
        },
        "expected": {
            "intent": (
                selection_meta["expected_intent"]
                if "expected_intent" in selection_meta
                else len(
                    _selected(dataset["intents"], args.split)
                    if dataset
                    else []
                )
            ),
            "retrieval": (
                selection_meta["expected_queries"]
                if "expected_queries" in selection_meta
                else len(
                    _selected(dataset["queries"], args.split)
                    if dataset
                    else []
                )
            ),
            "knowledge": len(dataset["knowledge"]) if dataset else None,
        },
        "selection": result.get("details", {}).get("selection"),
        "config": result.get("config", {}),
        "budget": result.get("cost", {}),
        "coverage": result.get("coverage", {}),
        "details": result.get("details", {}),
        "artifacts": {
            "run.json": "runtime metadata and hashes",
            "metrics.json": "per-split metrics and per-case analysis",
            "intent_predictions.jsonl": "intent and slot predictions",
            "retrieval_predictions.jsonl": "V/P rankings and diagnostics",
            "usage.jsonl": "request usage or requests=0 summary",
        },
    }
    metrics = {
        "run_id": run_id,
        "dataset_version": data_before.get("version"),
        "mode": args.mode,
        "split": args.split,
        "component": getattr(args, "component", "both"),
        "status": result["status"],
        "stages": result["stages"],
        "blockers": result["blockers"],
        "data": run_meta["data"],
        "coverage": result.get("coverage", {}),
        "offline_checks": result.get("offline_checks"),
        "intent": result.get("intent"),
        "slots": result.get("slots"),
        "retrieval": result.get("retrieval"),
        "latency": result.get("latency"),
        "cost": result.get("cost"),
        "errors": result.get("errors"),
        "details": result.get("details", {}),
        "notes": result["notes"],
    }
    _write_artifacts(run_dir, run_meta, metrics, result)

    print(f"run_id: {run_id}")
    print(f"mode: {args.mode}  split: {args.split}")
    print(f"status: {result['status']}")
    print(f"output: {run_dir}")
    for split in _selected_splits(args.split):
        retrieval_v = (result.get("retrieval") or {}).get("V") or {}
        per_split = retrieval_v.get("per_split") or {}
        if split in per_split:
            item = per_split[split]
            print(
                f"V/{split}: recall@5={item.get('recall@5'):.4f} "
                f"mrr@5={item.get('mrr@5'):.4f} "
                f"hit@5={item.get('hit@5'):.4f} "
                f"n={item.get('answerable_total')}"
            )
        intent = (result.get("intent") or {}).get(split)
        if isinstance(intent, dict) and "accuracy" in intent:
            print(
                f"intent/{split}: accuracy={intent['accuracy']:.4f} "
                f"macro_f1={intent['macro_f1']:.4f} "
                f"n={intent.get('total')}"
            )
    if result["blockers"]:
        print("blockers: " + "; ".join(result["blockers"]))
    return exit_code


def _exit_code(
    mode: str,
    result: Dict[str, Any],
    data_unchanged: bool,
    validation_ok: bool,
) -> int:
    if not data_unchanged or not validation_ok:
        return 2
    if mode == "offline-check":
        return 0 if result["stages"]["offline_checks"] == "PASS" else 2
    if mode == "vector":
        return (
            0
            if result["stages"]["vector_baseline"] == "COMPLETE"
            else 2
        )
    return (
        0
        if result["status"]
        in (STATUS_COMPLETE, STATUS_SMOKE_COMPLETE, STATUS_COMPONENT_COMPLETE)
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
