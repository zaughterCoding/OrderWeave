"""两阶段写操作的不变量与各类写操作主路径；需要 τ² 公开种子，缺失时跳过。"""
import json

import pytest

from agents.retail_tools import is_execution_attempt
from evaluation.harness import config as cfg
from mcp_tools.retail_store import RetailStore

pytestmark = pytest.mark.skipif(not cfg.SEED_PATH.exists(), reason="缺少 τ² retail 公开种子")
YES = "Yes, please proceed."


@pytest.fixture()
def seed():
    return json.loads(cfg.SEED_PATH.read_text(encoding="utf-8"))


def _store(tmp_path):
    return RetailStore(tmp_path / "s.db", cfg.SEED_PATH, language="en")


def _bind(store, seed, user_id, conv="c"):
    store.begin_turn("u", conv)
    assert store.find_user_by_email("u", conv, seed["users"][user_id]["email"])["success"]


def _order(seed, status, predicate=lambda o, u: True):
    for order in seed["orders"].values():
        user = seed["users"][order["user_id"]]
        if order["status"] == status and predicate(order, user):
            return order, user
    raise AssertionError("种子中没有符合条件的订单")


def _prepare_and_display(store, tool, args, conv="c"):
    turn = store.current_turn("u", conv)
    prepared = store.write_action("u", conv, tool, args, "")
    assert prepared["success"] and prepared["action_status"] == "pending_confirmation", prepared
    assert store.mark_pending_displayed("u", conv, prepared["op_id"], turn, False)
    return prepared


def test_prepare_changes_nothing_and_same_turn_cannot_execute(tmp_path, seed):
    store = _store(tmp_path)
    order, user = _order(seed, "delivered")
    _bind(store, seed, user["user_id"])
    args = {"order_id": order["order_id"], "item_ids": [order["items"][0]["item_id"]],
            "payment_method_id": order["payment_history"][0]["payment_method_id"]}
    before = store.business_snapshot()
    store.write_action("u", "c", "return_delivered_order_items", args, "")
    # 未展示的详情不能执行；同一轮再次调用只会重新生成详情。
    again = store.write_action("u", "c", "return_delivered_order_items", args, YES)
    assert again["action_status"] == "pending_confirmation"
    assert store.business_snapshot() == before


def test_confirmed_execution_happens_once(tmp_path, seed):
    store = _store(tmp_path)
    order, user = _order(seed, "delivered")
    _bind(store, seed, user["user_id"])
    args = {"order_id": order["order_id"], "item_ids": [order["items"][0]["item_id"]],
            "payment_method_id": order["payment_history"][0]["payment_method_id"]}
    _prepare_and_display(store, "return_delivered_order_items", args)
    store.begin_turn("u", "c")
    done = store.write_action("u", "c", "return_delivered_order_items", args, YES)
    assert done["success"] and done["status"] == "return requested"
    repeat = store.write_action("u", "c", "return_delivered_order_items", args, YES)
    assert repeat.get("action_status") != "executed" and not repeat.get("success")


def test_rejected_decision_blocks_execution(tmp_path, seed):
    store = _store(tmp_path)
    order, user = _order(seed, "pending")
    _bind(store, seed, user["user_id"])
    args = {"order_id": order["order_id"], "reason": "no longer needed"}
    prepared = _prepare_and_display(store, "cancel_pending_order", args)
    turn = store.begin_turn("u", "c")
    store.record_confirmation_decisions("u", "c", turn, [{"op_id": prepared["op_id"], "approved": False}], "test")
    result = store.write_action("u", "c", "cancel_pending_order", args, YES)
    assert result["error_code"] == "confirmation_required"
    assert store.business_snapshot()["orders"][order["order_id"]]["status"] == "pending"


def test_state_change_after_display_requires_new_confirmation(tmp_path, seed):
    store = _store(tmp_path)
    order, user = _order(seed, "pending", lambda o, u: True)
    _bind(store, seed, user["user_id"])
    address = {"address1": "1 Test St", "address2": "", "city": "Austin", "state": "TX", "country": "USA", "zip": "78701"}
    _prepare_and_display(store, "modify_pending_order_address", {"order_id": order["order_id"], **address})
    cancel = {"order_id": order["order_id"], "reason": "ordered by mistake"}
    store.begin_turn("u", "c")
    store.write_action("u", "c", "cancel_pending_order", cancel, "")
    # 取消详情替换了同订单的地址详情，旧详情不能再执行。
    store.begin_turn("u", "c")
    result = store.write_action("u", "c", "modify_pending_order_address", {"order_id": order["order_id"], **address}, YES)
    assert result.get("action_status") == "pending_confirmation"


def test_other_users_order_is_rejected(tmp_path, seed):
    store = _store(tmp_path)
    order, user = _order(seed, "pending")
    other = next(u for u in seed["users"].values() if u["user_id"] != user["user_id"])
    _bind(store, seed, other["user_id"])
    result = store.write_action("u", "c", "cancel_pending_order",
                                {"order_id": order["order_id"], "reason": "no longer needed"}, "")
    assert not result["success"]


def _variant(seed, item):
    variants = seed["products"][item["product_id"]]["variants"]
    return next((v for v in variants.values() if v["available"] and v["item_id"] != item["item_id"]), None)


@pytest.mark.parametrize("tool", ["exchange_delivered_order_items", "cancel_pending_order", "modify_pending_order_items",
                                  "modify_pending_order_address", "modify_pending_order_payment", "modify_user_address"])
def test_each_write_kind_main_path(tmp_path, seed, tool):
    store = _store(tmp_path)
    status = "delivered" if tool == "exchange_delivered_order_items" else "pending"
    if tool in {"exchange_delivered_order_items", "modify_pending_order_items"}:
        order, user = _order(seed, status, lambda o, u: _variant(seed, o["items"][0]) is not None)
        new = _variant(seed, order["items"][0])
        args = {"order_id": order["order_id"], "item_ids": [order["items"][0]["item_id"]],
                "new_item_ids": [new["item_id"]], "payment_method_id": order["payment_history"][0]["payment_method_id"]}
    elif tool == "modify_pending_order_payment":
        order, user = _order(seed, status, lambda o, u: any(
            p["id"] != o["payment_history"][0]["payment_method_id"] and p["source"] != "gift_card"
            for p in u["payment_methods"].values()))
        target = next(p["id"] for p in user["payment_methods"].values()
                      if p["id"] != order["payment_history"][0]["payment_method_id"] and p["source"] != "gift_card")
        args = {"order_id": order["order_id"], "payment_method_id": target}
    elif tool == "cancel_pending_order":
        order, user = _order(seed, status)
        args = {"order_id": order["order_id"], "reason": "ordered by mistake"}
    else:
        order, user = _order(seed, status)
        args = {"address1": "9 New Rd", "address2": "Apt 2", "city": "Boston", "state": "MA", "country": "USA",
                "zip": "02108"}
        args = {"user_id": user["user_id"], **args} if tool == "modify_user_address" else {"order_id": order["order_id"], **args}
    _bind(store, seed, user["user_id"])
    _prepare_and_display(store, tool, args)
    pending = store.active_pending("u", "c")
    assert is_execution_attempt(tool, args, pending)
    store.begin_turn("u", "c")
    done = store.write_action("u", "c", tool, args, YES)
    assert done["success"] and done["action_status"] == "executed", done
    assert any(e["event_type"].endswith("_executed") for e in store.list_events("u", "c"))
