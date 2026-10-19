"""T13 复用预算账本的请求前预留与永久停止语义。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from evaluation.retail_budget import BudgetExceeded
from evaluation.retail_ledger import (
    BudgetedClient,
    LedgerError,
    RetailLedger,
)


def make_ledger(tmp_path, **overrides):
    params = {
        "path": tmp_path / "ledger.db",
        "budget_cny": 5.0,
        "input_price_cny_per_mtok": 1.0,
        "output_price_cny_per_mtok": 1.0,
        "run_id": "t13-test",
        "model": "fake-model",
    }
    params.update(overrides)
    return RetailLedger(**params)


def request_kwargs(max_tokens=100):
    return {
        "model": "fake-model",
        "max_tokens": max_tokens,
        "system": "system",
        "messages": [{"role": "user", "content": "你好"}],
    }


def test_settle_by_usage_and_missing_usage_uses_upper_bound(tmp_path):
    ledger = make_ledger(tmp_path)

    first = ledger.reserve("agent", request_kwargs())
    charged = ledger.settle_success(
        first,
        SimpleNamespace(input_tokens=10, output_tokens=20),
        latency_ms=12.5,
    )
    assert charged == pytest.approx(0.00003)

    second = ledger.reserve("memory", request_kwargs())
    missing = ledger.settle_success(second, None, latency_ms=1.0)
    assert missing == pytest.approx(second.reserved_cny)

    third = ledger.reserve("agent", request_kwargs())
    failed = ledger.settle_failure(third, "network down", latency_ms=2.0)
    assert failed == pytest.approx(third.reserved_cny)

    summary = ledger.summary()
    assert summary["requests"] == 3
    assert summary["usage_missing"] == 1
    assert summary["failures"] == 1
    assert summary["spent_cny"] > 0
    assert summary["reservation_bound_violations"] == 0


def test_budget_is_rejected_before_sending_request(tmp_path):
    ledger = make_ledger(tmp_path, budget_cny=0.000001)

    with pytest.raises(BudgetExceeded, match="预算不足"):
        ledger.reserve("agent", request_kwargs(max_tokens=1000))

    assert ledger.summary()["requests"] == 0


def test_overrun_permanently_stops_later_requests(tmp_path):
    ledger = make_ledger(tmp_path)
    reservation = ledger.reserve("agent", request_kwargs(max_tokens=1))

    actual = ledger.settle_success(
        reservation,
        SimpleNamespace(input_tokens=1_000_000, output_tokens=1),
        latency_ms=5.0,
    )
    assert actual == pytest.approx(1.000001)

    summary = ledger.summary()
    assert summary["reservation_bound_violations"] == 1
    assert summary["cost_stop_reason"]

    with pytest.raises(BudgetExceeded, match="永久停止"):
        ledger.reserve("agent", request_kwargs())
    assert ledger.summary()["requests"] == 1


def test_price_run_and_model_mismatch_cannot_reuse_ledger(tmp_path):
    make_ledger(tmp_path)
    with pytest.raises(LedgerError, match="预算或价格"):
        make_ledger(tmp_path, input_price_cny_per_mtok=2.0)
    with pytest.raises(LedgerError, match="run_id"):
        make_ledger(tmp_path, run_id="other-run")
    with pytest.raises(LedgerError, match="模型"):
        make_ledger(tmp_path, model="other-model")


def test_sdk_style_content_blocks_are_serializable(tmp_path):
    class Block:
        def model_dump(self):
            return {"type": "text", "text": "历史消息"}

    ledger = make_ledger(tmp_path)
    kwargs = request_kwargs()
    kwargs["messages"] = [
        {"role": "assistant", "content": [Block()]},
        {"role": "user", "content": "继续"},
    ]
    reservation = ledger.reserve("agent", kwargs)
    assert reservation.input_tokens_upper > 0


def test_budgeted_client_does_not_retry_and_charges_failed_request(tmp_path):
    ledger = make_ledger(tmp_path)

    class FailingMessages:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("endpoint down")

    class Inner:
        def __init__(self):
            self.messages = FailingMessages()

    inner = Inner()
    client = BudgetedClient(inner, ledger, "agent")

    async def scenario():
        with pytest.raises(RuntimeError):
            await client.messages.create(**request_kwargs())
        with pytest.raises(RuntimeError):
            await client.messages.create(**request_kwargs())

    asyncio.run(scenario())
    assert inner.messages.calls == 2
    summary = ledger.summary()
    assert summary["requests"] == 2
    assert summary["failures"] == 2
    assert summary["spent_cny"] > 0
