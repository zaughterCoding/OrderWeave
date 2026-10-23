"""Judge 的完整性、有效分母与请求前预算，全部使用 Fake。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from evaluation.evaluator import LLMJudge
from evaluation.retail_budget import BudgetStopExceeded
from evaluation.retail_ledger import BudgetedClient, RetailLedger


VALID = dict(relevance=0.8, accuracy=0.8, completeness=0.8, helpfulness=0.8)


class Messages:
    def __init__(self, text=None, stop_reason="end_turn"):
        self.text = json.dumps(VALID) if text is None else text
        self.stop_reason = stop_reason
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=10, output_tokens=100),
        )


@pytest.mark.parametrize("text,stop,reason", [
    (json.dumps(VALID), "max_tokens", "incomplete_response"),
    (json.dumps(VALID), None, "incomplete_response"),
    ("", "end_turn", "empty_response"),
    ("not json", "end_turn", "invalid_json"),
    ('{"relevance":0.8}', "end_turn", "missing_field"),
    (json.dumps({**VALID, "accuracy": "0.9"}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": True}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": None}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": float("nan")}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": float("inf")}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": -0.1}), "end_turn", "invalid_score"),
    (json.dumps({**VALID, "accuracy": 1.1}), "end_turn", "invalid_score"),
    (json.dumps(VALID) + " trailing", "end_turn", "invalid_json"),
])
def test_invalid_scores_fail_once_without_a_score(text, stop, reason):
    messages = Messages(text, stop)
    value = asyncio.run(LLMJudge(SimpleNamespace(messages=messages), "fake").judge("q", "a"))
    assert value.judge_failed and value.failure_reason == reason
    assert value.overall is None
    assert value.accuracy is None
    assert len(messages.calls) == 1


def test_valid_subset_summary_names_its_denominator():
    from evaluation.evaluator import summarize_judgements
    async def run():
        good = await LLMJudge(SimpleNamespace(messages=Messages()), "fake").judge("q", "a")
        bad = await LLMJudge(SimpleNamespace(messages=Messages("", "max_tokens")), "fake").judge("q", "a")
        return summarize_judgements([good, bad], planned=3)

    summary = asyncio.run(run())
    assert summary["planned"] == 3 and summary["processed"] == 2
    assert summary["valid"] == summary["failed"] == summary["mean_denominator"] == 1
    assert summary["failure_reasons"] == {"incomplete_response": 1}
    assert summary["coverage"] == pytest.approx(1 / 3)
    assert summary["means"]["overall"] == pytest.approx(0.8)
    empty = summarize_judgements([], planned=3)
    assert empty["means"]["overall"] is None and empty["coverage"] == 0


def test_configured_budget_is_reserved_before_the_fake_request(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_JUDGE_MAX_TOKENS", "4096")
    messages = Messages()
    ledger = RetailLedger(tmp_path / "ledger.sqlite", 1, 2, 8, run_id="fake", model="fake")
    judge = LLMJudge(BudgetedClient(SimpleNamespace(messages=messages), ledger, "judge"), "fake")
    value = asyncio.run(judge.judge("q", "a"))
    assert not value.judge_failed
    assert messages.calls[0]["max_tokens"] == ledger.entries()[0]["max_tokens"] == 4096
    entry = ledger.entries()[0]
    assert entry["reserved_cny"] == pytest.approx((entry["input_tokens_upper"] * 2 + 4096 * 8) / 1_000_000)
    assert value.stop_reason == "end_turn" and value.output_tokens == 100

    ledger = RetailLedger(tmp_path / "small.sqlite", 0.02, 2, 8, run_id="small", model="fake")
    messages = Messages()
    judge = LLMJudge(BudgetedClient(SimpleNamespace(messages=messages), ledger, "judge"), "fake")
    with pytest.raises(BudgetStopExceeded):
        asyncio.run(judge.judge("q", "a"))
    assert not messages.calls


def test_default_and_explicit_output_limit(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_MAX_TOKENS", raising=False)
    client = SimpleNamespace(messages=Messages())
    assert LLMJudge(client, "fake").max_tokens == 4096
    monkeypatch.setenv("EVAL_JUDGE_MAX_TOKENS", "2048")
    assert LLMJudge(client, "fake", max_tokens=3072).max_tokens == 3072
    monkeypatch.setenv("EVAL_JUDGE_MAX_TOKENS", "0")
    with pytest.raises(ValueError):
        LLMJudge(client, "fake")
