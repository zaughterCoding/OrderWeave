"""零售评分与预算边界的离线回归；不调用模型或存储。"""
import json
import math

import pytest

from evaluation.retail_budget import (
    REQUEST_FORMAT_OVERHEAD_TOKENS,
    BudgetExceeded,
    BudgetGuard,
    request_input_upper_tokens,
)
from evaluation.retail_metrics import (
    compute_retrieval_metrics,
    entities_to_slots,
    map_intent_label,
    run_offline_self_check,
    score_intents,
    score_slots,
    summarize_latency,
)


def test_offline_self_check_examples_all_pass():
    checks = run_offline_self_check()

    assert checks
    assert [check["name"] for check in checks if not check["passed"]] == []


def test_failed_or_unknown_intent_cannot_be_scored_as_correct_other():
    metrics = score_intents(
        [
            {"gold": "other", "prediction": "other", "is_error": True, "error_kind": "unknown_label"},
            {"gold": "other", "prediction": None},
            {"gold": "refund", "prediction": "refund"},
        ]
    )

    assert metrics["correct"] == 1
    assert metrics["accuracy"] == pytest.approx(1 / 3)
    assert metrics["per_class"]["other"]["f1"] == 0.0
    assert metrics["error_count"] == 1
    assert metrics["invalid_count"] == 1
    assert metrics["valid_output_rate"] == pytest.approx(1 / 3)


def test_macro_f1_always_uses_the_fixed_six_labels():
    metrics = score_intents([{"gold": "refund", "prediction": "refund"}])

    assert metrics["macro_f1_labels"] == [
        "order_status",
        "refund",
        "query",
        "human_handoff",
        "greeting",
        "other",
    ]
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["per_class"]["order_status"]["support_status"] == "N/A"


def test_slot_scoring_uses_typed_sets_and_empty_samples():
    metrics = score_slots(
        [
            {
                "gold_slots": [
                    {"type": "order_id", "text": "#W9000101"},
                    {"type": "product", "text": "耳机"},
                ],
                "predicted_slots": [
                    {"type": "order_id", "text": "W9000101"},
                    {"type": "product", "text": "耳机"},
                    {"type": "product", "text": "耳机"},
                ],
            },
            {"gold_slots": [], "predicted_slots": []},
        ]
    )

    assert metrics["exact_match_count"] == 2
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["per_type"]["item_id"]["support_status"] == "N/A"


def test_entities_to_slots_keeps_only_six_target_types():
    slots = entities_to_slots(
        {
            "order_id": ["#W9000101"],
            "date": ["明天"],
            "error_code": ["500"],
        }
    )

    assert slots == [{"type": "order_id", "text": "#W9000101"}]


def test_unknown_evidence_id_keeps_its_ranking_position():
    metrics = compute_retrieval_metrics(
        [
            {
                "query_id": "q1",
                "answerability": "answerable",
                "gold_qrels": {"A": 2},
                "required_groups": [["A"]],
                "ranking": ["Z", "A"],
            }
        ],
        known_ids={"A"},
    )

    assert metrics["mrr@3"] == pytest.approx(0.5)
    assert metrics["protocol_clean"] is False
    assert metrics["per_query"][0]["invalid_ids"] == ["Z"]


def test_unanswerable_queries_are_excluded_from_regular_denominator():
    metrics = compute_retrieval_metrics(
        [
            {
                "query_id": "q1",
                "answerability": "answerable",
                "gold_qrels": {"A": 2},
                "required_groups": [["A"]],
                "ranking": ["A"],
            },
            {
                "query_id": "q2",
                "answerability": "unanswerable",
                "gold_qrels": {},
                "required_groups": [],
                "ranking": ["A"],
            },
        ],
        known_ids={"A"},
    )

    assert metrics["answerable_total"] == 1
    assert metrics["diagnostic_total"] == 1
    assert metrics["recall@1"] == 1.0


def test_latency_uses_nearest_rank():
    summary = summarize_latency([40.0, 10.0, 30.0, 20.0])

    assert summary["p50_ms"] == 20.0
    assert summary["p95_ms"] == 40.0
    assert summary["n"] == 4


def test_request_input_upper_bound_covers_multibyte_complete_request():
    content = "退款申请😀"
    request = {
        "model": "deepseek-flash",
        "max_tokens": 1024,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": content}],
    }

    upper = request_input_upper_tokens(request)
    serialized = json.dumps(
        request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert upper == len(serialized) + REQUEST_FORMAT_OVERHEAD_TOKENS
    assert upper >= len(content.encode("utf-8")) + REQUEST_FORMAT_OVERHEAD_TOKENS
    # 多字节文本的 UTF-8 字节数大于按字符数的经验估计。
    assert upper > int(len(content) * 1.25)


def test_request_input_upper_bound_includes_system_model_and_format_fields():
    base = {"messages": [{"role": "user", "content": "退货政策"}]}
    full = {
        **base,
        "system": "只返回 JSON",
        "model": "deepseek-flash",
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    assert request_input_upper_tokens(full) > request_input_upper_tokens(base)
    expected = len(
        json.dumps(
            full,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ) + REQUEST_FORMAT_OVERHEAD_TOKENS
    assert request_input_upper_tokens(full) == expected


def test_budget_refuses_before_the_request_when_upper_bound_exceeds_budget():
    guard = BudgetGuard(
        budget_cny=0.001,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
    )

    with pytest.raises(BudgetExceeded):
        guard.reserve("intent", input_tokens=5000, max_tokens=2048)
    assert guard.summary()["requests"] == 0


def test_budget_charges_full_upper_bound_when_usage_is_missing():
    guard = BudgetGuard(
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
    )
    reservation = guard.reserve("intent", input_tokens=1000, max_tokens=1000)

    charged = guard.settle_success(reservation, None, None, latency_ms=5.0)

    assert charged == pytest.approx(reservation.upper_bound_cny)
    assert guard.summary()["usage_missing"] == 1
    assert guard.summary()["cost_stop_reason"] is None
    assert math.isclose(guard.summary()["spent_cny"], charged)


def test_budget_failure_still_consumes_the_reservation():
    guard = BudgetGuard(
        budget_cny=1.0,
        input_price_cny_per_mtok=1.0,
        output_price_cny_per_mtok=1.0,
    )
    reservation = guard.reserve("rerank", input_tokens=1000, max_tokens=500)

    guard.settle_failure(reservation, "timeout", latency_ms=3.0)

    summary = guard.summary()
    assert summary["failures"] == 1
    assert summary["spent_cny"] == pytest.approx(reservation.upper_bound_cny)
    assert summary["reserved_cny"] == 0.0
    assert summary["cost_stop_reason"] is None


def test_budget_stops_forever_after_actual_usage_exceeds_reservation():
    guard = BudgetGuard(
        budget_cny=100.0,
        input_price_cny_per_mtok=10.0,
        output_price_cny_per_mtok=10.0,
    )
    reservation = guard.reserve("intent", input_tokens=1000, max_tokens=100)
    actual_charge = guard.settle_success(
        reservation,
        input_tokens=2000,
        output_tokens=100,
        latency_ms=1.0,
    )

    summary = guard.summary()
    # 真实费用不能压回预算，剩余额度即使充足也必须停止。
    assert actual_charge > reservation.upper_bound_cny
    assert summary["spent_cny"] == pytest.approx(actual_charge)
    assert summary["remaining_cny"] > 0
    assert summary["reservation_bound_violations"] == 1
    assert summary["cost_stop_reason"]
    row = guard.usage[-1]
    assert row["reservation_bound_violated"] is True
    assert row["bound_violation"]["input_tokens"]["actual"] == 2000
    assert row["charged_cny"] == pytest.approx(actual_charge)

    with pytest.raises(BudgetExceeded, match="费用上界此前已失效"):
        guard.reserve("intent", input_tokens=1, max_tokens=1)
    assert summary["requests"] == 1
