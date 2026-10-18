"""零售组件基线的纯离线指标。

本模块不导入模型、ChromaDB 或网络客户端，只把已记录的预测转换成固定
指标。所有分母约定在此集中实现，运行入口不得另写一套评分逻辑。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# 冻结的六类主意图及别名映射，与 evaluation/data/retail/labels.json 一致。
PRIMARY_INTENTS: Tuple[str, ...] = (
    "order_status",
    "refund",
    "query",
    "human_handoff",
    "greeting",
    "other",
)
INTENT_ALIASES: Dict[str, str] = {
    "logistics": "order_status",
    "escalation": "human_handoff",
}
KNOWN_CORE_LABELS: Set[str] = {
    "query",
    "complaint",
    "request",
    "greeting",
    "escalation",
    "technical",
    "billing",
    "account",
    "feedback",
    "order_status",
    "logistics",
    "refund",
    "invoice",
    "payment_issue",
    "account_security",
    "technical_login",
    "technical_crash",
    "human_handoff",
    "other",
}

SLOT_TYPES: Tuple[str, ...] = (
    "order_id",
    "item_id",
    "product",
    "payment_method_id",
    "quantity",
    "amount",
)

RANK_KS: Tuple[int, ...] = (1, 3, 5)


def map_intent_label(label: Any) -> Optional[str]:
    """把生产标签映射到六类；未知标签返回 None，不能当作 other。"""
    if label is None:
        return None
    value = str(label).strip().lower()
    if not value:
        return None
    if value in PRIMARY_INTENTS:
        return value
    if value in INTENT_ALIASES:
        return INTENT_ALIASES[value]
    if value in KNOWN_CORE_LABELS:
        return "other"
    return None


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _prf(tp: int, fp: int, fn: int) -> Dict[str, Any]:
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # 支持数为 0 时仍按固定 0 约定给分，另用状态标明不具备解释力。
        "support_status": "N/A" if tp + fn == 0 else "scored",
    }


def score_intents(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按六类固定口径计算意图指标。

    每条记录需含 gold、prediction、is_error。is_error 为真时预测一律视为
    无效，即使兜底输出了合法标签也不进入主表。
    """
    per_class: Dict[str, Dict[str, Any]] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    error_kinds: Dict[str, int] = {}
    invalid_count = 0
    error_count = 0
    correct = 0

    rows: List[Dict[str, Any]] = []
    for record in records:
        gold = map_intent_label(record.get("gold"))
        prediction = None if record.get("is_error") else map_intent_label(record.get("prediction"))
        is_error = bool(record.get("is_error"))
        if is_error:
            error_count += 1
            kind = str(record.get("error_kind") or "error")
            error_kinds[kind] = error_kinds.get(kind, 0) + 1
        elif prediction is None:
            invalid_count += 1
            error_kinds["invalid_label"] = error_kinds.get("invalid_label", 0) + 1

        if gold is None:
            raise ValueError(f"gold 标签不在固定六类内: {record.get('gold')!r}")
        if not is_error and prediction == gold:
            correct += 1

        confusion.setdefault(gold, {})
        key = prediction if prediction is not None else "__invalid__"
        confusion[gold][key] = confusion[gold].get(key, 0) + 1
        rows.append(
            {
                "case_id": record.get("case_id"),
                "split": record.get("split"),
                "gold": gold,
                "prediction": prediction,
                "is_error": is_error,
                "correct": (not is_error) and prediction == gold,
            }
        )

    for label in PRIMARY_INTENTS:
        tp = sum(
            1
            for row in rows
            if row["gold"] == label and row["prediction"] == label and not row["is_error"]
        )
        fp = sum(
            1
            for row in rows
            if row["prediction"] == label and row["gold"] != label
        )
        fn = sum(
            1
            for row in rows
            if row["gold"] == label and (row["prediction"] != label or row["is_error"])
        )
        metrics = _prf(tp, fp, fn)
        metrics.update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": sum(1 for row in rows if row["gold"] == label),
                "predicted": sum(1 for row in rows if row["prediction"] == label),
            }
        )
        per_class[label] = metrics

    total = len(rows)
    macro_f1 = (
        sum(per_class[label]["f1"] for label in PRIMARY_INTENTS) / len(PRIMARY_INTENTS)
        if total
        else 0.0
    )
    valid_outputs = total - error_count - invalid_count
    return {
        "total": total,
        "correct": correct,
        "accuracy": _ratio(correct, total),
        "macro_f1": macro_f1,
        "macro_f1_labels": list(PRIMARY_INTENTS),
        "per_class": per_class,
        "confusion": confusion,
        "error_count": error_count,
        "error_kinds": error_kinds,
        "invalid_count": invalid_count,
        "valid_output_rate": _ratio(valid_outputs, total),
        "cases": rows,
    }


def normalize_slot_value(slot_type: str, value: Any) -> str:
    """只做标注规范允许的规范化，保留商品名、数量等表面形式。"""
    text = str(value).strip()
    if slot_type == "order_id":
        # 仅规范已形如 W+数字 的订单号，允许可选 # 前缀。
        if text.startswith("#"):
            text = text[1:].strip()
        if text[:1].lower() == "w" and text[1:].isdigit():
            return f"#W{text[1:]}"
    return text


def slot_pairs(slots: Iterable[Any]) -> Set[Tuple[str, str]]:
    """把槽位列表转换为去重的 (类型, 规范化文本) 集合。"""
    pairs: Set[Tuple[str, str]] = set()
    for slot in slots or []:
        if isinstance(slot, dict):
            slot_type = slot.get("type")
            text = slot.get("text", slot.get("value"))
        else:
            slot_type, text = slot
        if slot_type not in SLOT_TYPES or text is None:
            continue
        normalized = normalize_slot_value(slot_type, text)
        if normalized:
            pairs.add((slot_type, normalized))
    return pairs


def entities_to_slots(entities: Dict[str, Any]) -> List[Dict[str, str]]:
    """把 IntentResult.entities 转成评分用的槽位列表，只保留六目标类。"""
    slots: List[Dict[str, str]] = []
    for slot_type in SLOT_TYPES:
        values = entities.get(slot_type) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            slots.append({"type": slot_type, "text": value})
    return slots


def score_slots(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按类型＋规范化文本集合计算 micro／分类型指标与整条完全匹配率。"""
    per_type: Dict[str, Dict[str, Any]] = {}
    total_gold = total_pred = total_tp = 0
    exact = 0
    errors = 0
    rows: List[Dict[str, Any]] = []

    for record in records:
        gold = slot_pairs(record.get("gold_slots") or [])
        is_error = bool(record.get("is_error"))
        predicted = set() if is_error else slot_pairs(record.get("predicted_slots") or [])
        if is_error:
            errors += 1
        tp = len(gold & predicted)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        total_gold += len(gold)
        total_pred += len(predicted)
        total_tp += tp
        exact += int(gold == predicted)
        rows.append(
            {
                "case_id": record.get("case_id"),
                "split": record.get("split"),
                "gold": sorted(gold),
                "predicted": sorted(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "exact_match": gold == predicted,
            }
        )

    for slot_type in SLOT_TYPES:
        type_gold = sum(1 for row in rows for t, _ in row["gold"] if t == slot_type)
        type_pred = sum(1 for row in rows for t, _ in row["predicted"] if t == slot_type)
        type_tp = sum(
            1
            for row in rows
            for pair in row["gold"]
            if pair[0] == slot_type and pair in row["predicted"]
        )
        type_fp = type_pred - type_tp
        type_fn = type_gold - type_tp
        metrics = _prf(type_tp, type_fp, type_fn)
        metrics.update(
            {
                "tp": type_tp,
                "fp": type_fp,
                "fn": type_fn,
                "support": type_gold,
                "predicted": type_pred,
            }
        )
        per_type[slot_type] = metrics

    return {
        "total": len(rows),
        "exact_match_count": exact,
        "exact_match_rate": _ratio(exact, len(rows)),
        "gold_total": total_gold,
        "predicted_total": total_pred,
        "tp": total_tp,
        "fp": total_pred - total_tp,
        "fn": total_gold - total_tp,
        "micro_precision": _ratio(total_tp, total_pred),
        "micro_recall": _ratio(total_tp, total_gold),
        "micro_f1": _ratio(2 * total_tp, total_pred + total_gold),
        "per_type": per_type,
        "error_count": errors,
        "cases": rows,
    }


def dedupe_ranking(ranking: Sequence[Any]) -> Tuple[List[str], int]:
    """按首次出现位置去重，返回新排序和被去掉的重复数。"""
    seen: Set[str] = set()
    ordered: List[str] = []
    duplicates = 0
    for item in ranking or []:
        value = str(item)
        if value in seen:
            duplicates += 1
            continue
        seen.add(value)
        ordered.append(value)
    return ordered, duplicates


def compute_retrieval_metrics(
    records: Sequence[Dict[str, Any]],
    ks: Sequence[int] = RANK_KS,
    known_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """计算可答问题的 Recall／MRR／Hit／AllEvidenceHit。

    records 中的非 answerable 样本不进入常规分母，只在覆盖统计中单列。
    未知证据 ID 保留原位置且不计命中；关联失败按空排序计分。
    """
    answerable = [r for r in records if r.get("answerability") == "answerable"]
    diagnostics = [r for r in records if r.get("answerability") != "answerable"]
    per_query: List[Dict[str, Any]] = []
    protocol_issues: List[Dict[str, Any]] = []
    aggregate: Dict[int, Dict[str, float]] = {
        k: {"recall": 0.0, "mrr": 0.0, "hit": 0.0, "all_evidence_hit": 0.0}
        for k in ks
    }

    for record in answerable:
        query_id = record.get("query_id")
        qrels = {
            str(evidence_id): int(score)
            for evidence_id, score in (record.get("gold_qrels") or {}).items()
        }
        relevant = {evidence_id for evidence_id, score in qrels.items() if score >= 2}
        if not relevant:
            raise ValueError(f"可答问题缺少 qrel>=2 的相关证据: {query_id}")
        groups = [list(group) for group in (record.get("required_groups") or [])]
        is_error = bool(record.get("is_error"))
        raw_ranking = [] if is_error else list(record.get("ranking") or [])
        ranking, duplicates = dedupe_ranking(raw_ranking)
        invalid = (
            [evidence_id for evidence_id in ranking if evidence_id not in known_ids]
            if known_ids is not None
            else []
        )
        if duplicates:
            protocol_issues.append(
                {"query_id": query_id, "type": "duplicate_evidence", "count": duplicates}
            )
        if invalid:
            protocol_issues.append(
                {"query_id": query_id, "type": "unknown_evidence_id", "ids": invalid}
            )
        if not ranking and not is_error:
            protocol_issues.append(
                {"query_id": query_id, "type": "empty_ranking", "count": 1}
            )

        metrics: Dict[str, Any] = {
            "query_id": query_id,
            "split": record.get("split"),
            "relevant_count": len(relevant),
            "ranking": ranking,
            "invalid_ids": invalid,
            "duplicates_removed": duplicates,
            "is_error": is_error,
        }
        for k in ks:
            top = ranking[:k]
            top_set = set(top)
            hits = len(relevant & top_set)
            recall = _ratio(hits, len(relevant))
            first_rank = next(
                (index + 1 for index, evidence_id in enumerate(top) if evidence_id in relevant),
                0,
            )
            mrr = 1.0 / first_rank if first_rank else 0.0
            hit = 1.0 if hits else 0.0
            all_hit = 1.0 if groups and all(any(e in top_set for e in group) for group in groups) else 0.0
            metrics[f"recall@{k}"] = recall
            metrics[f"mrr@{k}"] = mrr
            metrics[f"hit@{k}"] = hit
            metrics[f"all_evidence_hit@{k}"] = all_hit
            metrics[f"first_relevant_rank@{k}"] = first_rank
            aggregate[k]["recall"] += recall
            aggregate[k]["mrr"] += mrr
            aggregate[k]["hit"] += hit
            aggregate[k]["all_evidence_hit"] += all_hit
        per_query.append(metrics)

    denominator = len(answerable)
    summary: Dict[str, Any] = {
        "answerable_total": denominator,
        "diagnostic_total": len(diagnostics),
        "ks": list(ks),
        "per_query": per_query,
        "protocol_issues": protocol_issues,
        "protocol_clean": not protocol_issues,
    }
    for k in ks:
        summary[f"recall@{k}"] = _ratio(aggregate[k]["recall"], denominator)
        summary[f"mrr@{k}"] = _ratio(aggregate[k]["mrr"], denominator)
        summary[f"hit@{k}"] = _ratio(aggregate[k]["hit"], denominator)
        summary[f"all_evidence_hit@{k}"] = _ratio(
            aggregate[k]["all_evidence_hit"], denominator
        )
    summary["diagnostics"] = [
        {
            "query_id": record.get("query_id"),
            "split": record.get("split"),
            "answerability": record.get("answerability"),
            "ranking": list(record.get("ranking") or []),
            "is_error": bool(record.get("is_error")),
            "note": record.get("note", ""),
        }
        for record in diagnostics
    ]
    return summary


def summarize_latency(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """用 nearest-rank 计算本次样本的 P50／P95，不宣称生产分位。"""
    samples = sorted(float(value) for value in values if value is not None)
    if not samples:
        return {
            "n": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "percentile_method": "nearest-rank",
        }

    def nearest_rank(percent: float) -> float:
        index = max(0, math.ceil(percent * len(samples)) - 1)
        return samples[index]

    return {
        "n": len(samples),
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": nearest_rank(0.50),
        "p95_ms": nearest_rank(0.95),
        "percentile_method": "nearest-rank",
    }


def _close(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-9)
    return actual == expected


def run_offline_self_check() -> List[Dict[str, Any]]:
    """用手算例验证评分合同；返回逐项检查结果。"""
    checks: List[Dict[str, Any]] = []

    def check(name: str, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "passed": _close(actual, expected),
            }
        )

    check("alias_logistics", map_intent_label("logistics"), "order_status")
    check("alias_escalation", map_intent_label("escalation"), "human_handoff")
    check("known_core_to_other", map_intent_label("billing"), "other")
    check("unknown_label_is_invalid", map_intent_label("not_a_label"), None)

    intent_records = [
        {"gold": "order_status", "prediction": "order_status"},
        {"gold": "refund", "prediction": "refund"},
        {"gold": "query", "prediction": "query"},
        {"gold": "human_handoff", "prediction": "escalation"},
        {"gold": "greeting", "prediction": "greeting"},
        {"gold": "other", "prediction": "other", "is_error": True, "error_kind": "unknown_label"},
        {"gold": "other", "prediction": "other", "is_error": True, "error_kind": "request_error"},
    ]
    intent_metrics = score_intents(intent_records)
    # 前五条 correct；两条错误即使兜底为 other 也不能计对。
    check("intent_accuracy", intent_metrics["accuracy"], 5 / 7)
    check("intent_error_count", intent_metrics["error_count"], 2)
    check("intent_other_f1", intent_metrics["per_class"]["other"]["f1"], 0.0)
    check("intent_macro_labels", len(intent_metrics["macro_f1_labels"]), 6)

    slot_records = [
        {
            "gold_slots": [
                {"type": "order_id", "text": "#W9000101"},
                {"type": "product", "text": "耳机"},
                {"type": "quantity", "text": "两件"},
            ],
            "predicted_slots": [
                {"type": "order_id", "text": "W9000101"},
                {"type": "product", "text": "耳机"},
                {"type": "quantity", "text": "两件"},
            ],
        },
        {"gold_slots": [], "predicted_slots": []},
        {
            "gold_slots": [{"type": "order_id", "text": "#W9000102"}],
            "predicted_slots": [
                {"type": "order_id", "text": "#W9000102"},
                {"type": "order_id", "text": "#W9000102"},
            ],
        },
        {
            "gold_slots": [{"type": "amount", "text": "120元"}],
            "predicted_slots": [],
        },
    ]
    slot_metrics = score_slots(slot_records)
    check("slot_micro_tp", slot_metrics["tp"], 4)
    check("slot_micro_precision", slot_metrics["micro_precision"], 1.0)
    check("slot_micro_recall", slot_metrics["micro_recall"], 4 / 5)
    check("slot_exact_rate", slot_metrics["exact_match_rate"], 3 / 4)
    check("slot_item_support_status", slot_metrics["per_type"]["item_id"]["support_status"], "N/A")

    ranking_record = {
        "query_id": "q-ranking",
        "answerability": "answerable",
        "gold_qrels": {"A": 2, "B": 2},
        "required_groups": [["A"], ["B"]],
        "ranking": ["X", "A", "B"],
    }
    ranking_metrics = compute_retrieval_metrics(
        [ranking_record], known_ids={"X", "A", "B"}
    )
    check("ranking_recall@1", ranking_metrics["recall@1"], 0.0)
    check("ranking_recall@3", ranking_metrics["recall@3"], 1.0)
    check("ranking_mrr@3", ranking_metrics["mrr@3"], 0.5)
    check("ranking_hit@3", ranking_metrics["hit@3"], 1.0)
    check("ranking_all_evidence@1", ranking_metrics["all_evidence_hit@1"], 0.0)
    check("ranking_all_evidence@3", ranking_metrics["all_evidence_hit@3"], 1.0)

    group_record = {
        "query_id": "q-group",
        "answerability": "answerable",
        "gold_qrels": {"A": 2, "B": 2, "C": 2},
        "required_groups": [["A", "B"], ["C"]],
        "ranking": ["A", "C"],
    }
    group_metrics = compute_retrieval_metrics(
        [group_record], known_ids={"A", "B", "C"}
    )
    check("group_all_evidence@2", group_metrics["all_evidence_hit@3"], 1.0)
    check("group_recall@3", group_metrics["recall@3"], 2 / 3)

    dedupe_record = dict(ranking_record, query_id="q-dedupe", ranking=["A", "A", "B"])
    dedupe_metrics = compute_retrieval_metrics(
        [dedupe_record], known_ids={"X", "A", "B"}
    )
    check("dedupe_recall@1", dedupe_metrics["recall@1"], 0.5)
    check("dedupe_recall@3", dedupe_metrics["recall@3"], 1.0)

    failure_record = dict(ranking_record, query_id="q-failure", ranking=["A"], is_error=True)
    failure_metrics = compute_retrieval_metrics(
        [ranking_record, failure_record], known_ids={"X", "A", "B"}
    )
    check("failure_keeps_denominator", failure_metrics["answerable_total"], 2)
    check("failure_recall@3", failure_metrics["recall@3"], 0.5)

    unknown_record = {
        "query_id": "q-unknown",
        "answerability": "answerable",
        "gold_qrels": {"A": 2},
        "required_groups": [["A"]],
        "ranking": ["Z", "A"],
    }
    unknown_metrics = compute_retrieval_metrics(
        [unknown_record], known_ids={"A", "B"}
    )
    check("unknown_id_keeps_position", unknown_metrics["mrr@3"], 0.5)
    check("unknown_id_protocol_clean", unknown_metrics["protocol_clean"], False)

    mixed_records = [
        ranking_record,
        {
            "query_id": "q-needs-tool",
            "answerability": "needs_tool",
            "gold_qrels": {},
            "required_groups": [],
            "ranking": ["A"],
        },
        {
            "query_id": "q-unanswerable",
            "answerability": "unanswerable",
            "gold_qrels": {},
            "required_groups": [],
            "ranking": ["A"],
        },
    ]
    mixed_metrics = compute_retrieval_metrics(mixed_records, known_ids={"X", "A", "B"})
    check("non_answerable_excluded", mixed_metrics["answerable_total"], 1)
    check("diagnostic_count", mixed_metrics["diagnostic_total"], 2)

    latency = summarize_latency([10.0, 20.0, 30.0, 40.0])
    check("latency_p50_nearest_rank", latency["p50_ms"], 20.0)
    check("latency_p95_nearest_rank", latency["p95_ms"], 40.0)

    return checks
