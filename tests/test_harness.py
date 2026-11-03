"""评测 harness 的不变量：pass^k 定义、写入安全判定、轨迹映射与统一思考设置。"""
import asyncio

import pytest

from evaluation.harness.report import bootstrap_ci, pass_hat_k
from evaluation.harness.session import classify, safety_checks, trajectory_messages
from evaluation.retail_ledger import BudgetedClient


def test_pass_hat_k_matches_tau_definition():
    # 3 次中 2 次成功：pass^1=2/3，pass^2=C(2,2)/C(3,2)=1/3，pass^3=0。
    assert pass_hat_k({"a": 2}, 3, 1) == pytest.approx(2 / 3)
    assert pass_hat_k({"a": 2}, 3, 2) == pytest.approx(1 / 3)
    assert pass_hat_k({"a": 2, "b": 3}, 3, 3) == pytest.approx(0.5)


def test_bootstrap_interval_contains_mean():
    low, high = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0])
    assert low <= 0.6 <= high


def test_safety_flags_write_without_display_and_duplicates():
    events = [{"event_type": "identity_bound", "op_id": None},
              {"event_type": "return_displayed", "op_id": "a"},
              {"event_type": "return_requested", "op_id": "a"},
              {"event_type": "return_requested", "op_id": "a"},
              {"event_type": "return_requested", "op_id": "b"}]
    result = safety_checks(events)
    assert result["unconfirmed_writes"] == ["b"]
    assert result["duplicate_writes"] == ["a"]
    assert result["identity_bound"] is True


def test_trajectory_keeps_tool_call_result_pairs_in_order():
    turns = [{"user": "hi", "assistant": "done", "tool_traces": [
        {"tool_name": "get_user_details", "tool_use_id": "t1", "input": {"user_id": "u"}, "result": {"ok": 1}}]}]
    messages = trajectory_messages(turns, "###STOP###")
    roles = [m["role"] for m in messages]
    assert roles == ["assistant", "user", "assistant", "tool", "assistant", "user"]
    assert messages[2]["tool_calls"][0]["id"] == messages[3]["id"] == "t1"


class _Inner:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"usage": None})()


class _Ledger:
    def consume_turn_request(self, stage):
        pass

    def reserve(self, stage, kwargs):
        return "r"

    def settle_success(self, *args):
        pass


def test_budgeted_client_applies_thinking_without_overriding_explicit():
    inner = _Inner()
    client = BudgetedClient(type("C", (), {"messages": inner})(), _Ledger(), "agent",
                            thinking={"type": "disabled"}).for_stage("user")
    asyncio.run(client.messages.create(model="m", messages=[]))
    asyncio.run(client.messages.create(model="m", messages=[], extra_body={"thinking": {"type": "enabled"}}))
    assert inner.calls[0]["extra_body"]["thinking"] == {"type": "disabled"}
    assert inner.calls[1]["extra_body"]["thinking"] == {"type": "enabled"}


def test_classify_separates_agent_failures_from_infrastructure():
    assert classify("USER_STOP", 1.0, "") == (True, True)
    assert classify("AGENT_ERROR", 0.0, "") == (True, False)
    assert classify("USER_ERROR", 0.0, "") == (False, False)
    assert classify("USER_STOP", None, "") == (False, False)
