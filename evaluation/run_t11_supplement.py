"""T11 补充烟测：历史消歧与检索来源追溯。

只覆盖两项缺口：两条带非空历史的意图用例，以及一条已有 dev 问题的
生产 P 链路来源字段追溯。复用零售基线的预算守卫、客户端生命周期和
独立 Chroma 集合约束，不改变检索算法或知识正文。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from evaluation import run_retail_baseline as runner
from evaluation.retail_budget import BudgetGuard
from evaluation.validate_retail_data import validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIOR_RUN_ID = "live_20261018T161500Z_d89d09"
PRIOR_SPENT_CNY = 0.035512
CUMULATIVE_CAP_CNY = 5.0

# 历史只含普通用户可见对话；不包含认证、订单状态或工具结果。
HISTORY_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "H-DISAMB-REFUND-01",
        "split": "smoke",
        "history": [
            {
                "role": "user",
                "content": "我买的耳机戴着不合适，想退货。",
            },
            {
                "role": "assistant",
                "content": "可以，我可以继续协助您办理耳机退货。您希望我继续为您处理退货吗？",
            },
        ],
        "message": "是的，请继续。",
        "expected_intent": "refund",
    },
    {
        "case_id": "H-DISAMB-QUERY-01",
        "split": "smoke",
        "history": [
            {
                "role": "user",
                "content": "我想先了解耳机退货前需要做哪些准备。",
            },
            {
                "role": "assistant",
                "content": "可以，我可以继续为您说明退货前需要做的准备。您希望我继续说明吗？",
            },
        ],
        "message": "是的，请继续。",
        "expected_intent": "query",
    },
]

RETRIEVAL_QUERY_ID = "R-D07-01"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_view(item: Any) -> Dict[str, Any]:
    """只保留来源追溯需要的字段，正文用哈希和长度代替。"""
    if not isinstance(item, dict):
        return {"raw": str(item)[:200]}
    content = item.get("content")
    return {
        "id": item.get("id"),
        "evidence_id": item.get("evidence_id"),
        "doc_id": item.get("doc_id"),
        "title": item.get("title"),
        "source_kind": item.get("source_kind"),
        "source_refs": item.get("source_refs"),
        "policy_release": item.get("policy_release"),
        "policy_commit": item.get("policy_commit"),
        "scope": item.get("scope"),
        "category": item.get("category"),
        "chunk": item.get("chunk"),
        "score": item.get("score"),
        "content_len": len(content) if isinstance(content, str) else None,
        "content_sha256": _sha256_text(content) if isinstance(content, str) else None,
    }


def _verify_stage(
    stage: str,
    views: Sequence[Dict[str, Any]],
    knowledge_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """核对阶段结果中的证据身份与来源是否仍与知识资产一致。"""
    missing_ids: List[str] = []
    id_field_mismatches: List[str] = []
    source_mismatches: List[Dict[str, Any]] = []
    content_mismatches: List[str] = []
    for view in views:
        evidence_id = str(view.get("evidence_id") or "")
        if not evidence_id or evidence_id not in knowledge_by_id:
            missing_ids.append(evidence_id or "<empty>")
            continue
        row = knowledge_by_id[evidence_id]
        if view.get("id") != evidence_id:
            id_field_mismatches.append(evidence_id)
        for field in ("doc_id", "source_kind", "policy_release", "policy_commit"):
            actual = view.get(field)
            expected = row.get(field, "")
            if actual != expected:
                source_mismatches.append(
                    {
                        "evidence_id": evidence_id,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
        if view.get("source_refs") != row.get("source_refs"):
            source_mismatches.append(
                {
                    "evidence_id": evidence_id,
                    "field": "source_refs",
                    "expected": row.get("source_refs"),
                    "actual": view.get("source_refs"),
                }
            )
        if view.get("content_sha256") != _sha256_text(row.get("content", "")):
            content_mismatches.append(evidence_id)
    return {
        "stage": stage,
        "items": len(views),
        "missing_evidence_ids": missing_ids,
        "id_field_mismatches": id_field_mismatches,
        "source_mismatches": source_mismatches,
        "content_mismatches": content_mismatches,
        "passed": not (
            missing_ids
            or id_field_mismatches
            or source_mismatches
            or content_mismatches
        ),
    }


class _CaptureMessages:
    """记录真实发送给模型的提示词，再原样转交内层客户端。"""

    def __init__(self, inner: Any, records: List[Dict[str, Any]]):
        self._inner = inner
        self._records = records

    async def create(self, **kwargs: Any) -> Any:
        prompt = runner._request_prompt_text(kwargs)
        self._records.append(
            {
                "stage": runner._classify_stage(prompt),
                "prompt": prompt,
                "prompt_sha256": _sha256_text(prompt),
                "model": kwargs.get("model"),
                "max_tokens": kwargs.get("max_tokens"),
                "temperature": kwargs.get("temperature"),
            }
        )
        return await self._inner.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CaptureClient:
    """在预算守卫内层记录实际请求体。"""

    def __init__(self, inner: Any, records: List[Dict[str, Any]]):
        self._inner = inner
        self._records = records
        self._messages = _CaptureMessages(inner.messages, records)

    @property
    def messages(self) -> _CaptureMessages:
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _prompt_contains(records: Sequence[Dict[str, Any]], text: str) -> bool:
    return any(text in str(record.get("prompt") or "") for record in records)


def _usage_delta(usage: Sequence[Dict[str, Any]], mark: int) -> Dict[str, Any]:
    rows = list(usage[mark:])
    return {
        "requests": len(rows),
        "charged_cny": round(
            sum(float(row.get("charged_cny") or 0.0) for row in rows), 9
        ),
        "failures": sum(1 for row in rows if row.get("status") == "failed"),
        "usage_missing": sum(
            1
            for row in rows
            if row.get("status") == "success"
            and not row.get("model_usage_present")
        ),
    }


async def _run_live(
    *,
    history_cases: Sequence[Dict[str, Any]],
    query_row: Dict[str, Any],
    knowledge: Sequence[Dict[str, Any]],
    kb: Any,
    api_key: str,
    base_url: Optional[str],
    model: str,
    guard: BudgetGuard,
    request_log: List[Dict[str, Any]],
    lifecycle: Any,
    outcome: Dict[str, Any],
) -> Dict[str, Any]:
    """在唯一事件循环内串行执行两条意图用例和一条检索链路。"""
    from core import intent_recognizer as intent_module
    from mcp import tool_manager as tool_manager_module
    from mcp.tool_manager import Tool

    original_intent_factory = intent_module.AsyncAnthropic
    original_tool_factory = tool_manager_module.AsyncAnthropic
    capture_records: List[Dict[str, Any]] = []
    result: Dict[str, Any] = outcome
    result.setdefault("history_cases", [])
    result.setdefault("retrieval", {})

    def tracked_intent_factory(**kwargs: Any) -> Any:
        inner = original_intent_factory(**kwargs)
        lifecycle.track(inner)
        return inner

    def tracked_tool_factory(**kwargs: Any) -> Any:
        inner = original_tool_factory(**kwargs)
        lifecycle.track(inner)
        return inner

    recognizer = None
    manager = None
    try:
        intent_module.AsyncAnthropic = tracked_intent_factory
        tool_manager_module.AsyncAnthropic = tracked_tool_factory
        recognizer = intent_module.IntentRecognizer(
            api_key=api_key,
            base_url=base_url,
            model=model,
            confidence_threshold=runner.CONFIDENCE_THRESHOLD,
        )
        manager = tool_manager_module.MCPToolManager(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        # 守卫在最外层，记录层紧贴真实客户端；远程 Embedding 仍被屏蔽。
        recognizer.client = runner._GuardedClient(
            _CaptureClient(recognizer.client, capture_records),
            guard,
            request_log,
        )
        manager._client = runner._GuardedClient(
            _CaptureClient(manager._client, capture_records),
            guard,
            request_log,
        )

        for case in history_cases:
            capture_mark = len(capture_records)
            usage_mark = len(guard.usage)
            request_mark = len(request_log)
            started = time.monotonic()
            outcome = await recognizer.recognize(
                case["message"], history=case["history"]
            )
            latency_ms = float(getattr(outcome, "latency_ms", 0.0) or 0.0)
            if not latency_ms:
                latency_ms = (time.monotonic() - started) * 1000
            status = str(getattr(outcome, "recognition_status", "ok"))
            is_error = status in {"unknown_label", "parse_error", "request_error"}
            predicted = getattr(outcome.intent, "value", str(outcome.intent))
            mapped = None if is_error else runner.map_intent_label(predicted)
            prompts = list(capture_records[capture_mark:])
            intent_prompts = [
                record for record in prompts if record.get("stage") == "intent_llm"
            ]
            entities = dict(getattr(outcome, "entities", {}) or {})
            result["history_cases"].append(
                {
                    "case_id": case["case_id"],
                    "message": case["message"],
                    "history": list(case["history"]),
                    "expected_intent": case["expected_intent"],
                    "actual_prompts": intent_prompts,
                    "history_in_prompt": {
                        message["content"]: _prompt_contains(
                            intent_prompts, message["content"]
                        )
                        for message in case["history"]
                    },
                    "raw_intent": getattr(outcome, "raw_intent", None),
                    "predicted_intent": predicted,
                    "mapped_intent": mapped,
                    "confidence": getattr(outcome, "confidence", None),
                    "entities": entities,
                    "recognition_status": status,
                    "diagnostics": getattr(outcome, "diagnostics", {}),
                    "is_error": is_error,
                    "error_kind": status if is_error else None,
                    "latency_ms": latency_ms,
                    "request_ids": runner._request_ids_since(request_log, request_mark),
                    "usage": _usage_delta(guard.usage, usage_mark),
                    "expectation_met": (
                        not is_error and mapped == case["expected_intent"]
                    ),
                }
            )
            if guard.cost_stop_reason:
                result["retrieval"]["skipped"] = "intent_budget_stop"
                return result

        knowledge_by_id = {row["evidence_id"]: row for row in knowledge}
        raw_records: List[Dict[str, Any]] = []
        rerank_records: List[Dict[str, Any]] = []

        async def traced_handler(params: Dict[str, Any], context: Any) -> Any:
            data = await kb.search_handler(params, context)
            raw_records.append(
                {
                    "query": params.get("query"),
                    "top_k": params.get("top_k"),
                    "items": [_source_view(item) for item in data]
                    if isinstance(data, list)
                    else [],
                }
            )
            return data

        manager.register(
            Tool(
                name="knowledge_search",
                description="搜索零售知识库（ChromaDB 向量检索）",
                handler=traced_handler,
                schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                cache_ttl=runner.CACHE_TTL_S,
                timeout_s=30.0,
                supports_rerank=True,
            )
        )

        original_rerank = manager._rerank

        async def traced_rerank(
            query: str,
            items: List[Any],
            top_k: int,
            diagnostics: Optional[Dict[str, Any]] = None,
        ) -> Any:
            merged_views = [_source_view(item) for item in items]
            output, applied = await original_rerank(
                query, items, top_k, diagnostics=diagnostics
            )
            rerank_records.append(
                {
                    "query": query,
                    "input": merged_views,
                    "output": [_source_view(item) for item in output],
                    "applied": bool(applied),
                }
            )
            return output, applied

        manager._rerank = traced_rerank

        diagnostics: Dict[str, Any] = {}
        retrieval_usage_mark = len(guard.usage)
        retrieval_request_mark = len(request_log)
        retrieval_started = time.monotonic()
        tool_result = await manager.search_with_rewrite(
            "knowledge_search",
            query_row["query"],
            top_k=runner.TOP_K,
            diagnostics=diagnostics,
        )
        retrieval_latency_ms = (time.monotonic() - retrieval_started) * 1000
        final_views = (
            [_source_view(item) for item in tool_result.data]
            if isinstance(tool_result.data, list)
            else []
        )

        # 以真实 ToolResult 调用共享 RAG 适配器，不再次发送模型请求。
        original_search = manager.search_with_rewrite

        async def _replay_search(*args: Any, **kwargs: Any) -> Any:
            return tool_result

        manager.search_with_rewrite = _replay_search  # type: ignore[method-assign]
        try:
            from agents.tools import build_shared_rag_tools

            shared_tool = build_shared_rag_tools(manager)["search_knowledge_base"]
            shared_output = shared_tool.handler(
                types.SimpleNamespace(message=query_row["query"]),
                {"query": query_row["query"], "top_k": runner.TOP_K},
            )
            if inspect.isawaitable(shared_output):
                shared_output = await shared_output
        finally:
            manager.search_with_rewrite = original_search  # type: ignore[method-assign]

        shared_views = [
            _source_view(item) for item in (shared_output or {}).get("results", [])
        ]

        stages = {
            "raw_recall": [
                {
                    "query": record["query"],
                    "top_k": record["top_k"],
                    "items": record["items"],
                }
                for record in raw_records
            ],
            "merged_pre_rerank": (
                rerank_records[0]["input"]
                if rerank_records
                else [
                    item
                    for record in raw_records
                    for item in record["items"]
                ]
            ),
            "rerank_output": rerank_records[0]["output"] if rerank_records else [],
            "tool_result": final_views,
            "shared_tool_return": shared_views,
        }
        verification = {
            "raw_recall": [
                _verify_stage(
                    f"raw_recall:{record['query']}",
                    record["items"],
                    knowledge_by_id,
                )
                for record in raw_records
            ],
            "merged_pre_rerank": _verify_stage(
                "merged_pre_rerank",
                stages["merged_pre_rerank"],
                knowledge_by_id,
            ),
            "rerank_output": _verify_stage(
                "rerank_output", stages["rerank_output"], knowledge_by_id
            ),
            "tool_result": _verify_stage(
                "tool_result", stages["tool_result"], knowledge_by_id
            ),
            "shared_tool_return": _verify_stage(
                "shared_tool_return", stages["shared_tool_return"], knowledge_by_id
            ),
        }

        result["retrieval"] = {
            "query_id": query_row["query_id"],
            "query": query_row["query"],
            "success": bool(tool_result.success),
            "error": tool_result.error,
            "reranked": bool(tool_result.reranked),
            "degraded": bool(tool_result.degraded),
            "diagnostics": diagnostics,
            "rewrite": {
                "sub_queries": diagnostics.get("sub_queries"),
                "status": diagnostics.get("rewrite_status"),
            },
            "stages": stages,
            "source_verification": verification,
            "shared_tool_return_meta": {
                "success": bool((shared_output or {}).get("success")),
                "reranked": bool((shared_output or {}).get("reranked")),
                "degraded": bool((shared_output or {}).get("degraded")),
                "reused_real_tool_result": True,
                "additional_model_requests": 0,
            },
            "latency_ms": retrieval_latency_ms,
            "request_ids": runner._request_ids_since(
                request_log, retrieval_request_mark
            ),
            "usage": _usage_delta(guard.usage, retrieval_usage_mark),
        }
        return result
    finally:
        intent_module.AsyncAnthropic = original_intent_factory
        tool_manager_module.AsyncAnthropic = original_tool_factory
        await lifecycle.close_all()


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T11 补充烟测：历史消歧与检索来源追溯"
    )
    parser.add_argument("--budget-cny", type=float, default=None)
    parser.add_argument(
        "--prior-spent-cny",
        type=float,
        default=PRIOR_SPENT_CNY,
        help="本轮之前的累计已花费用；默认取上一轮烟测报告值",
    )
    parser.add_argument("--input-price-cny-per-mtok", type=float, default=None)
    parser.add_argument("--output-price-cny-per-mtok", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--chroma-host", default=None)
    parser.add_argument("--chroma-port", type=int, default=None)
    parser.add_argument("--chroma-path", default=None)
    return parser.parse_args(argv)


def _prior_usage() -> Dict[str, Any]:
    path = (
        runner.DEFAULT_OUTPUT_ROOT / PRIOR_RUN_ID / "usage.jsonl"
    )
    rows = _read_jsonl(path)
    summary = next(
        (row for row in reversed(rows) if row.get("type") == "summary"),
        None,
    )
    if summary is None:
        raise runner.BaselineRefused(
            runner.STATUS_BLOCKED_STORAGE,
            f"缺少上一轮用量汇总: {path}",
        )
    spent = float(summary.get("spent_cny") or 0.0)
    if abs(spent - PRIOR_SPENT_CNY) > 1e-9:
        raise runner.BaselineRefused(
            runner.STATUS_ERROR,
            f"上一轮累计费用 {spent:.9f} 与报告 {PRIOR_SPENT_CNY:.9f} 不一致",
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.budget_cny or args.budget_cny <= 0:
        print("必须显式提供大于 0 的 --budget-cny", file=sys.stderr)
        return 2
    if args.input_price_cny_per_mtok is None or args.output_price_cny_per_mtok is None:
        print("必须提供当前官方输入／输出单价", file=sys.stderr)
        return 2

    run_id = f"supplement_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    run_dir = runner._prepare_run_dir(run_id, args.run_dir)
    started_at = _utc_now()
    result: Dict[str, Any] = {
        "run_id": run_id,
        "mode": "t11-supplement",
        "status": "RUNNING",
        "started_at": started_at,
        "scope": {
            "history_case_ids": [case["case_id"] for case in HISTORY_CASES],
            "retrieval_query_id": RETRIEVAL_QUERY_ID,
            "not_full_baseline": True,
        },
        "errors": [],
    }
    request_log: List[Dict[str, Any]] = []
    lifecycle = runner._ClientLifecycle()
    kb = None
    owns_collection = False
    collection_name = runner._collection_name(run_id)
    snapshot_before: Dict[str, Any] = {}
    snapshot_after: Dict[str, Any] = {}
    collection_deleted = False
    guard: Optional[BudgetGuard] = None
    data_root = runner.DATA_ROOT
    live_outcome: Dict[str, Any] = {"history_cases": [], "retrieval": {}}
    try:
        data_before = {
            "validation": validate(data_root),
            "hash": runner._hash_dataset(data_root),
        }
        result["data"] = {"before": data_before}
        prior = _prior_usage()
        prior_spent = float(args.prior_spent_cny or 0.0)
        result["budget"] = {
            "cumulative_cap_cny": CUMULATIVE_CAP_CNY,
            "prior_run_summary": prior,
            "prior_spent_cny": prior_spent,
        }

        guard = BudgetGuard(
            args.budget_cny,
            args.input_price_cny_per_mtok,
            args.output_price_cny_per_mtok,
        )
        api_key, base_url, model = runner._live_credentials(args)
        from mcp import knowledge_base as kb_module

        if not kb_module._embedding_cache_ready():
            raise runner.BaselineRefused(
                runner.STATUS_BLOCKED_EMBEDDING,
                "Embedding 本地缓存不完整，按约束不下载模型",
            )
        dataset = runner._load_dataset(data_root)
        query_row = next(
            row
            for row in dataset["queries"]
            if row["query_id"] == RETRIEVAL_QUERY_ID
        )
        preflight = runner._preflight_upper_bound(
            guard,
            list(HISTORY_CASES),
            [query_row],
            model,
        )
        result["budget"]["preflight"] = preflight
        if not preflight["fits_budget"]:
            raise runner.BaselineRefused(
                runner.STATUS_REFUSED_PREFLIGHT,
                "保守费用上界超过本轮剩余预算，未创建模型客户端",
                {"preflight": preflight},
            )
        if prior_spent + preflight["total_upper_bound_cny"] > CUMULATIVE_CAP_CNY:
            raise runner.BaselineRefused(
                runner.STATUS_REFUSED_PREFLIGHT,
                "累计保守上界超过 5 元总上限，未创建模型客户端",
                {"preflight": preflight},
            )

        host, port, fallback_path = runner._chroma_config(args, run_dir)
        result["config"] = {
            "model": model,
            "base_url_configured": bool(base_url),
            "chroma": {"host": host, "port": port, "collection": collection_name},
            "top_k": runner.TOP_K,
            "rewrite_n": runner.REWRITE_N,
            "input_accounting_basis": runner.INPUT_COUNT_BASIS,
        }

        kb = kb_module.KnowledgeBase(
            chroma_host=host,
            chroma_port=port,
            chroma_path=fallback_path,
            collection_name=collection_name,
            load_demo_documents=False,
        )
        if kb.doc_count:
            raise runner.BaselineRefused(
                runner.STATUS_BLOCKED_STORAGE,
                f"独立集合 {collection_name} 非空，拒绝复用未知内容",
                {"count": kb.doc_count},
            )
        owns_collection = True
        snapshot_before = runner._snapshot_collections(
            kb._client, exclude=collection_name
        )
        imported = kb.upsert_evidence(dataset["knowledge"])
        evidence_check = runner._verify_evidence(kb, dataset["knowledge"])
        if not evidence_check["passed"]:
            raise runner.BaselineRefused(
                runner.STATUS_BLOCKED_STORAGE,
                "知识证据导入校验失败",
                {"verification": evidence_check},
            )
        result["storage"] = {
            "backend": kb.backend,
            "collection": collection_name,
            "imported_new": imported,
            "evidence_verification": evidence_check,
            "collections_before": snapshot_before,
        }

        live_outcome: Dict[str, Any] = {"history_cases": [], "retrieval": {}}
        outcome = asyncio.run(
            _run_live(
                history_cases=HISTORY_CASES,
                query_row=query_row,
                knowledge=dataset["knowledge"],
                kb=kb,
                api_key=api_key,
                base_url=base_url,
                model=model,
                guard=guard,
                request_log=request_log,
                lifecycle=lifecycle,
                outcome=live_outcome,
            )
        )
        result["history"] = {
            "cases": outcome.get("history_cases", []),
        }
        history_cases = outcome.get("history_cases", [])
        result["history"]["summary"] = {
            "total": len(history_cases),
            "passed": sum(1 for case in history_cases if case["expectation_met"]),
            "failed": [
                case["case_id"]
                for case in history_cases
                if not case["expectation_met"]
            ],
        }
        retrieval = outcome.get("retrieval", {})
        result["retrieval"] = retrieval

        prior_predictions_path = (
            runner.DEFAULT_OUTPUT_ROOT / PRIOR_RUN_ID / "retrieval_predictions.jsonl"
        )
        prior_ranking: Optional[List[str]] = None
        if prior_predictions_path.is_file():
            for row in _read_jsonl(prior_predictions_path):
                if row.get("query_id") == RETRIEVAL_QUERY_ID and row.get("mode") == "P":
                    prior_ranking = list(row.get("ranking") or [])
                    break
        current_ranking = [
            str(item.get("evidence_id") or item.get("id") or "")
            for item in retrieval.get("stages", {}).get("tool_result", [])
        ]
        result["retrieval"]["prior_ranking"] = prior_ranking
        result["retrieval"]["matches_prior_ranking"] = (
            prior_ranking == current_ranking if prior_ranking is not None else None
        )

        verification_rows = list(retrieval.get("source_verification", {}).values())
        flat_verification: List[Dict[str, Any]] = []
        for value in verification_rows:
            if isinstance(value, list):
                flat_verification.extend(value)
            elif isinstance(value, dict):
                flat_verification.append(value)
        trace_passed = bool(flat_verification) and all(
            row.get("passed") for row in flat_verification
        )
        history_passed = (
            result["history"]["summary"]["failed"] == []
            and result["history"]["summary"]["total"] == len(HISTORY_CASES)
        )
        pipeline_ok = bool(retrieval.get("success")) and not retrieval.get("degraded")
        if history_passed and trace_passed and pipeline_ok:
            result["status"] = "SUPPLEMENT_COMPLETE"
        else:
            result["status"] = "SUPPLEMENT_PARTIAL"
            if not history_passed:
                result["errors"].append("历史消歧用例未全部达到预期")
            if not trace_passed:
                result["errors"].append("检索来源字段追溯未通过")
            if not pipeline_ok:
                result["errors"].append("生产 P 链路未成功或发生降级")
    except runner.BaselineRefused as ex:
        result["status"] = ex.status
        result["errors"].append(ex.reason)
        result["refusal"] = {"status": ex.status, "reason": ex.reason, "details": ex.details}
    except Exception as ex:
        result["status"] = runner.STATUS_ERROR
        result["errors"].append(f"{type(ex).__name__}: {str(ex)[:500]}")
    finally:
        # 异常中断时也保留已经发生的付费结果，便于如实核对。
        if "history" not in result:
            partial_cases = list(live_outcome.get("history_cases") or [])
            result["history"] = {
                "cases": partial_cases,
                "summary": {
                    "total": len(partial_cases),
                    "passed": sum(
                        1 for case in partial_cases if case["expectation_met"]
                    ),
                    "failed": [
                        case["case_id"]
                        for case in partial_cases
                        if not case["expectation_met"]
                    ],
                },
            }
        if "retrieval" not in result:
            result["retrieval"] = live_outcome.get("retrieval") or {}
        result["client_lifecycle"] = {
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
                    result["errors"].append(
                        f"cleanup_delete_collection: {type(ex).__name__}: {ex}"
                    )
            try:
                snapshot_after = runner._snapshot_collections(kb._client)
            except Exception as ex:
                result["errors"].append(f"cleanup_snapshot: {type(ex).__name__}: {ex}")
        result["cleanup"] = {
            "collection": collection_name,
            "collection_deleted": collection_deleted,
            "collections_before": snapshot_before,
            "collections_after": snapshot_after,
        }
        try:
            data_after = {
                "validation": validate(data_root),
                "hash": runner._hash_dataset(data_root),
            }
            result["data"]["after"] = data_after
            result["data"]["unchanged"] = data_after == result["data"].get("before")
        except Exception as ex:
            result["errors"].append(f"data_after: {type(ex).__name__}: {ex}")
        if guard is not None:
            this_round = guard.summary()
            cumulative_spent = prior_spent + float(this_round["spent_cny"])
            result["budget"]["this_round"] = this_round
            result["budget"]["cumulative"] = {
                "cap_cny": CUMULATIVE_CAP_CNY,
                "prior_spent_cny": prior_spent,
                "this_round_spent_cny": this_round["spent_cny"],
                "spent_cny": round(cumulative_spent, 9),
                "remaining_cny": round(CUMULATIVE_CAP_CNY - cumulative_spent, 9),
                "requests": int(this_round["requests"]),
                "within_cap": cumulative_spent <= CUMULATIVE_CAP_CNY + 1e-12,
            }
        result["request_log"] = request_log
        result["finished_at"] = _utc_now()
        runner._write_json(run_dir / "results.json", result)

    print(json.dumps(
        {
            "run_id": run_id,
            "status": result["status"],
            "run_dir": str(run_dir),
            "history": result.get("history", {}).get("summary"),
            "retrieval_success": result.get("retrieval", {}).get("success"),
            "budget": result.get("budget", {}).get("cumulative"),
        },
        ensure_ascii=False,
    ))
    return 0 if result["status"] in {
        "SUPPLEMENT_COMPLETE",
        runner.STATUS_SMOKE_COMPLETE,
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
