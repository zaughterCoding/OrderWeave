"""对照评分与原排序解析的独立小例子。"""
import math

import pytest

from evaluation.run_reranker_comparison import ndcg, legacy_order, coverage_audit


def test_ndcg_graded_relevance_and_rank_discount():
    qrels = {"a": 2, "b": 1}
    assert ndcg(["a", "b"], qrels) == pytest.approx(1)
    assert ndcg(["b", "a"], qrels) == pytest.approx((1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert ndcg(["x"], qrels) == 0
    assert ndcg(["a", "a", "b"], qrels) == pytest.approx(1)


def test_legacy_duplicate_and_partial_output_are_not_repaired():
    order, diag = legacy_order("[1, 1]", 3)
    assert order == [1, 1] and not diag["complete"] and diag["applied"]
    order, diag = legacy_order("[2,0,1]", 3)
    assert order == [2, 0, 1] and diag["complete"]
    order, diag = legacy_order("not JSON", 3)
    assert order == [0, 1, 2] and not diag["applied"]


def test_coverage_counts_groups_and_new_regressions():
    pools = [{"query_id": "q", "answerability": "answerable",
              "required_evidence_groups": [["a"], ["b", "c"]],
              "candidates": [{"evidence_id": "a"}, {"evidence_id": "b"}]}]
    result = coverage_audit([{"query_id": "q", "ranking": ["a", "x"]}], pools)
    assert result == {"groups_retained": 1, "groups_total": 2,
                      "misses": [{"query_id": "q", "missing_groups": [["b", "c"]]}],
                      "regressions_vs_rrf": ["q"]}
