"""T13 零售 SQLite 存储的离线业务回归。

覆盖 B01-B20 中可在无模型条件下验证的查询、身份、金额、确认、
事务与未知结果恢复语义；不调用任何模型端点。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from mcp.retail_store import (
    EXPECTED_SEED_SHA256,
    RetailStore,
    RetailStoreError,
    normalize_order_id,
    safe_calculate,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
POLICY = ROOT / "evaluation" / "data" / "retail_sandbox" / "policy.md"
KNOWLEDGE = ROOT / "evaluation" / "data" / "retail_sandbox" / "knowledge.jsonl"

CHEN_EMAIL = "chen.silva2698@example.com"
HARPER_EMAIL = "harper.brown3965@example.com"
ANYA_EMAIL = "anya.garcia2061@example.com"
JAMES_EMAIL = "james.li4495@example.com"

CHEN_CREDIT = "credit_card_1565124"
CHEN_GIFT = "gift_card_7250692"
CHEN_ORDER = "#W3069600"
CHEN_OTHER_ORDER = "#W8171054"
CHEN_GIFT_ORDER = "#W2598834"
HARPER_ORDER = "#W1840144"
JAMES_ORDER = "#W4435622"


@pytest.fixture()
def store(tmp_path: Path) -> RetailStore:
    return RetailStore(tmp_path / "retail.db", SEED)


def bind(
    store: RetailStore,
    email: str = CHEN_EMAIL,
    client: str = "client-a",
    conv: str = "conv-a",
) -> None:
    result = store.find_user_by_email(client, conv, email)
    assert result["success"], result


def prepare(
    store: RetailStore,
    order_id: str,
    item_ids,
    payment_method_id: str,
    *,
    client: str = "client-a",
    conv: str = "conv-a",
):
    result = store.prepare_return(
        client, conv, order_id, item_ids, payment_method_id
    )
    assert result["success"], result
    return result


def display(
    store: RetailStore,
    op_id: str,
    *,
    client: str = "client-a",
    conv: str = "conv-a",
    turn: int | None = None,
) -> None:
    current = store.current_turn(client, conv) if turn is None else turn
    assert store.mark_pending_displayed(client, conv, op_id, current, False)


def submit(
    store: RetailStore,
    order_id: str,
    item_ids,
    payment_method_id: str,
    message: str,
    *,
    client: str = "client-a",
    conv: str = "conv-a",
):
    return store.submit_return(
        client, conv, order_id, item_ids, payment_method_id, message
    )


def mutate_order(store: RetailStore, order_id: str, **changes) -> None:
    """模拟展示后订单被其他流程修改。"""
    with closing(sqlite3.connect(store.db_path)) as connection:
        row = connection.execute(
            "SELECT payload, version FROM orders WHERE order_id=?",
            (order_id,),
        ).fetchone()
        payload = json.loads(row[0])
        payload.update(changes)
        connection.execute(
            "UPDATE orders SET payload=?, version=version+1 WHERE order_id=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True), order_id),
        )
        connection.commit()


def test_seed_import_counts_and_restart_does_not_reset(store, tmp_path):
    counts = store.stats()["counts"]
    assert {key: counts[key] for key in ("users", "orders", "products")} == {
        "users": 500,
        "orders": 1000,
        "products": 50,
    }

    bind(store, HARPER_EMAIL)
    before = store.current_turn("client-a", "conv-a")

    reopened = RetailStore(tmp_path / "retail.db", SEED)
    assert reopened.bound_user_id("client-a", "conv-a") == "harper_brown_7363"
    assert reopened.current_turn("client-a", "conv-a") == before
    assert reopened.stats()["counts"]["users"] == 500


def test_seed_conflict_is_rejected_without_overwrite(tmp_path):
    db_path = tmp_path / "retail.db"
    RetailStore(db_path, SEED)

    with pytest.raises(RetailStoreError, match="公开种子哈希不一致"):
        RetailStore(db_path, SEED, expected_seed_sha256="0" * 64)

    other_seed = tmp_path / "other.json"
    other_seed.write_text(
        SEED.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    other_sha = hashlib.sha256(other_seed.read_bytes()).hexdigest()
    with pytest.raises(RetailStoreError, match="其他种子"):
        RetailStore(db_path, other_seed, expected_seed_sha256=other_sha)

    damaged = tmp_path / "damaged.db"
    source = sqlite3.connect(db_path)
    target = sqlite3.connect(damaged)
    try:
        source.backup(target)
        target.execute("DELETE FROM meta")
        target.commit()
    finally:
        target.close()
        source.close()
    with pytest.raises(RetailStoreError, match="缺少元数据"):
        RetailStore(damaged, SEED)

    assert hashlib.sha256(SEED.read_bytes()).hexdigest() == EXPECTED_SEED_SHA256


def test_identity_binding_and_cross_user_isolation(store):
    assert store.get_user_details("client-a", "conv-a")["success"] is False
    assert (
        store.get_order_details("client-a", "conv-a", CHEN_ORDER)["success"]
        is False
    )

    bind(store, CHEN_EMAIL)
    assert store.bound_user_id("client-a", "conv-a") == "chen_silva_7485"
    product_id = store.get_order_details(
        "client-a", "conv-a", CHEN_ORDER
    )["order"]["items"][0]["product_id"]

    # 同一会话不能切换到其他用户。
    switch = store.find_user_by_email("client-a", "conv-a", JAMES_EMAIL)
    assert switch["success"] is False
    assert "不能切换" in switch["error"]

    # 另一客户端/会话绑定 James 后，查不到 Chen 的订单，也不返回订单内容。
    bind(store, JAMES_EMAIL, client="client-b", conv="conv-b")
    foreign = store.get_order_details("client-b", "conv-b", CHEN_ORDER)
    assert foreign["success"] is False
    assert "order" not in foreign
    assert "Skateboard" not in json.dumps(foreign, ensure_ascii=False)

    own = store.get_order_details("client-b", "conv-b", JAMES_ORDER)
    assert own["success"] is True
    assert own["order"]["user_id"] == "james_li_5688"

    # 商品目录是公开资料，但订单仍按绑定身份限制。
    assert store.get_product_details(
        "client-b", "conv-b", product_id
    )["success"] is True


def test_read_only_queries_do_not_mutate_business_tables(store):
    bind(store)
    before = store.business_snapshot()

    user = store.get_user_details("client-a", "conv-a")
    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)
    product = store.get_product_details(
        "client-a", "conv-a", order["order"]["items"][0]["product_id"]
    )

    assert user["success"] and order["success"] and product["success"]
    assert store.business_snapshot() == before

    raw = json.loads(SEED.read_text(encoding="utf-8"))["orders"][CHEN_ORDER]
    # 查询结果只来自种子记录，不额外编造承运商、预计送达等字段。
    assert order["order"] == raw
    assert set(order["order"]) == set(raw)
    assert set(order["order"]["fulfillments"][0]) == {"tracking_id", "item_ids"}


def test_ambiguous_reference_requires_explicit_order(store):
    bind(store)
    # “Tablet” 在 Chen 的两笔已送达订单中都存在，不能自动选一笔。
    assert store.get_order_details("client-a", "conv-a", "Tablet")["success"] is False
    assert (
        store.prepare_return(
            "client-a", "conv-a", "Tablet", ["8551474201"], CHEN_CREDIT
        )["success"]
        is False
    )
    assert store.active_pending("client-a", "conv-a") == []

    first = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    second = store.get_order_details(
        "client-a", "conv-a", "#W9571698"
    )["order"]
    assert first["order_id"] != second["order_id"]
    assert {item["name"] for item in first["items"]} & {
        item["name"] for item in second["items"]
    } == {"Tablet"}


def test_prepare_requires_delivered_order(store):
    bind(store, ANYA_EMAIL)
    store.begin_turn("client-a", "conv-a")
    before = store.business_snapshot()

    cases = {
        "#W4140680": "cancelled",
        "#W6310710": "processed",
        "#W6436609": "pending",
    }
    for order_id, status in cases.items():
        result = store.prepare_return(
            "client-a",
            "conv-a",
            order_id,
            [str(store.get_order_details("client-a", "conv-a", order_id)["order"]["items"][0]["item_id"])],
            "gift_card_4374071",
        )
        assert result["success"] is False
        assert status in result["error"]

    assert store.active_pending("client-a", "conv-a") == []
    assert store.business_snapshot() == before


def test_prepare_validates_item_ownership_quantity_and_amount(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    before = store.business_snapshot()

    missing = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, ["not-an-item"], CHEN_CREDIT
    )
    assert missing["success"] is False
    assert "不属于该订单" in missing["error"]

    empty = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, [], CHEN_CREDIT
    )
    assert empty["success"] is False
    assert "不能为空" in empty["error"]

    not_list = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, "4545791457", CHEN_CREDIT
    )
    assert not_list["success"] is False

    too_many = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, ["4545791457"] * 2, CHEN_CREDIT
    )
    assert too_many["success"] is False
    assert "超过订单中的 1 件" in too_many["error"]

    result = prepare(
        store,
        CHEN_ORDER,
        ["4545791457", "5012998807"],
        CHEN_CREDIT,
    )
    assert result["amount_cents"] == 18606 + 25871
    assert result["estimated_amount"] == "444.77"
    assert "Skateboard" in result["display_text"]
    assert "Makeup Kit" in result["display_text"]
    assert "× 1" in result["display_text"]
    assert "¥444.77" in result["display_text"]
    assert "mastercard" in result["display_text"]
    assert "不代表已到账" in result["display_text"]
    assert "订单号：#W3069600" in result["display_text"]

    # 准备只写待确认记录，不改订单、用户或商品。
    assert store.business_snapshot() == before


def test_duplicate_item_id_means_multiple_units(store):
    bind(store, HARPER_EMAIL)
    store.begin_turn("client-a", "conv-a")

    one = prepare(store, HARPER_ORDER, ["8590708195"], "paypal_2306935")
    two = prepare(
        store, HARPER_ORDER, ["8590708195", "8590708195"], "paypal_2306935"
    )

    assert one["amount_cents"] == 15761
    assert len(one["items"]) == 1 and one["items"][0]["quantity"] == 1
    assert two["amount_cents"] == 31522
    assert len(two["items"]) == 1 and two["items"][0]["quantity"] == 2
    assert "退货件数：2 件" in two["display_text"]

    # 同一订单重新准备会作废旧详情，只保留新的待确认记录。
    pending = store.active_pending("client-a", "conv-a")
    assert [item["op_id"] for item in pending] == [two["op_id"]]

    too_many = store.prepare_return(
        "client-a",
        "conv-a",
        HARPER_ORDER,
        ["8590708195", "8590708195", "8590708195"],
        "paypal_2306935",
    )
    assert too_many["success"] is False
    assert "超过订单中的 2 件" in too_many["error"]


def test_refund_destination_limited_to_original_or_own_gift_card(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")

    # 原支付方式合法。
    original = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    assert "原支付方式" in original["display_text"]

    # 本人已有礼品卡也合法。
    gift = prepare(store, CHEN_ORDER, ["9494281769"], CHEN_GIFT)
    assert "本人已有礼品卡" in gift["display_text"]

    # 他人支付方式、不存在的礼品卡、本人但非原支付方式且非礼品卡均拒绝。
    other_user = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, ["8551474201"], "gift_card_4374071"
    )
    assert other_user["success"] is False
    assert "不属于当前用户" in other_user["error"]

    missing = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, ["8551474201"], "gift_card_0000000"
    )
    assert missing["success"] is False

    # #W2598834 原支付方式是礼品卡；改成本人信用卡仍不合法。
    wrong_card = store.prepare_return(
        "client-a", "conv-a", CHEN_GIFT_ORDER, ["6245746168"], CHEN_CREDIT
    )
    assert wrong_card["success"] is False
    assert "原支付方式或当前用户已有的礼品卡" in wrong_card["error"]

    assert prepare(store, CHEN_GIFT_ORDER, ["6245746168"], CHEN_GIFT)[
        "amount_cents"
    ] == 4600


def test_confirmation_requires_prior_display(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)

    # 尚未标记展示，不能提交。
    not_displayed = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert not_displayed["success"] is False
    assert "尚未展示" in not_displayed["error"]

    # 标记展示后，同一轮仍不能提交。
    display(store, prepared["op_id"], turn=1)
    same_turn = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert same_turn["success"] is False
    assert "同一轮不能提交" in same_turn["error"]

    assert (
        store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"][
            "status"
        ]
        == "delivered"
    )


@pytest.mark.parametrize(
    "message",
    [
        "先别退，我再想想",
        "不要提交",
        "确认吗？",
        "如果退款到礼品卡就确认",
        "确认，但是把数量改成两件",
        "好的，等等",
    ],
)
def test_negation_question_condition_and_change_do_not_confirm(store, message):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    result = submit(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, message)

    assert result["success"] is False
    assert "明确确认" in result["error"]
    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "delivered"
    assert store.active_pending("client-a", "conv-a")[0]["status"] == "displayed"
    assert any(
        event["event_type"] == "return_rejected"
        for event in store.list_events("client-a", "conv-a")
    )

    # 拒绝不消费确认；用户在之后同一轮明确确认仍可正常提交。
    approved = submit(
        store,
        CHEN_ORDER,
        ["4545791457"],
        CHEN_CREDIT,
        "确认，就按这些申请退货",
    )
    assert approved["success"] is True


def test_submit_arguments_must_match_displayed_details(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    mismatch = submit(
        store, CHEN_ORDER, ["9494281769"], CHEN_CREDIT, "确认"
    )
    assert mismatch["success"] is False
    assert "与已展示的详情不一致" in mismatch["error"]

    wrong_payment = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_GIFT, "确认"
    )
    assert wrong_payment["success"] is False
    assert "不一致" in wrong_payment["error"]

    changed_order = submit(
        store, CHEN_OTHER_ORDER, ["4545791457"], CHEN_CREDIT, "确认"
    )
    assert changed_order["success"] is False


def test_cross_session_confirmation_cannot_authorize(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)

    bind(store, CHEN_EMAIL, client="client-a", conv="conv-new")
    store.begin_turn("client-a", "conv-new")
    result = submit(
        store,
        CHEN_ORDER,
        ["4545791457"],
        CHEN_CREDIT,
        "是的",
        conv="conv-new",
    )
    assert result["success"] is False
    assert "没有已展示" in result["error"]
    assert (
        store.get_order_details("client-a", "conv-new", CHEN_ORDER)["order"][
            "status"
        ]
        == "delivered"
    )


def test_confirmation_is_consumed_once_and_write_is_transactional(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    first = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert first["success"] is True
    assert first["status"] == "return requested"
    assert first["estimated_amount"] == "186.06"

    second = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert second["success"] is False
    assert "没有已展示" in second["error"]

    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "return requested"
    assert order["return_items"] == ["4545791457"]
    assert order["return_payment_method_id"] == CHEN_CREDIT
    assert store.stats()["counts"]["return_operations"] == 1

    # 已申请退货的订单不能再次准备或追加。
    again = store.prepare_return(
        "client-a", "conv-a", CHEN_ORDER, ["9494281769"], CHEN_CREDIT
    )
    assert again["success"] is False
    assert "只有 delivered" in again["error"]


def test_changing_details_requires_new_display_and_confirmation(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    old = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, old["op_id"], turn=1)

    # 下一轮改变商品，旧确认应作废，旧参数不能再提交。
    store.begin_turn("client-a", "conv-a")
    new = prepare(store, CHEN_ORDER, ["9494281769"], CHEN_CREDIT)
    display(store, new["op_id"], turn=2)
    old_submit = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认"
    )
    assert old_submit["success"] is False
    assert "与已展示的详情不一致" in old_submit["error"]

    store.begin_turn("client-a", "conv-a")
    new_submit = submit(
        store, CHEN_ORDER, ["9494281769"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert new_submit["success"] is True
    assert new_submit["return_items"] == ["9494281769"]


def test_multi_order_confirmation_scope(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    first = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    second = prepare(store, CHEN_OTHER_ORDER, ["9747045638"], CHEN_CREDIT)
    display(store, first["op_id"], turn=1)
    display(store, second["op_id"], turn=1)

    # 两笔待确认详情时，简短“好的”不足以确定范围。
    store.begin_turn("client-a", "conv-a")
    vague = submit(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "好的")
    assert vague["success"] is False
    assert "未明确确认范围" in vague["error"]

    # 点名订单只确认该订单；确认 A 不批准 B。
    only_a = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认 #W3069600"
    )
    assert only_a["success"] is True
    not_b = submit(
        store,
        CHEN_OTHER_ORDER,
        ["9747045638"],
        CHEN_CREDIT,
        "确认 #W3069600",
    )
    assert not_b["success"] is False

    # 明确“都”覆盖全部已展示详情时，B 可继续按展示内容提交。
    explicit_b = submit(
        store,
        CHEN_OTHER_ORDER,
        ["9747045638"],
        CHEN_CREDIT,
        "都确认",
    )
    assert explicit_b["success"] is True
    assert (
        store.get_order_details("client-a", "conv-a", CHEN_OTHER_ORDER)["order"][
            "status"
        ]
        == "return requested"
    )


def test_earlier_success_is_not_rolled_back_by_later_failure(store):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    first = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    second = prepare(store, CHEN_OTHER_ORDER, ["9747045638"], CHEN_CREDIT)
    display(store, first["op_id"], turn=1)
    display(store, second["op_id"], turn=1)

    # 展示后另一流程修改了 B，导致 B 提交失败。
    mutate_order(store, CHEN_OTHER_ORDER, status="cancelled")

    store.begin_turn("client-a", "conv-a")
    first_result = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "都确认"
    )
    assert first_result["success"] is True

    second_result = submit(
        store, CHEN_OTHER_ORDER, ["9747045638"], CHEN_CREDIT, "都确认"
    )
    assert second_result["success"] is False
    assert "展示后已发生变化" in second_result["error"]

    # 第一笔成功不被回滚；第二笔保持未写入。
    assert (
        store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"][
            "status"
        ]
        == "return requested"
    )
    assert (
        store.get_order_details("client-a", "conv-a", CHEN_OTHER_ORDER)["order"][
            "status"
        ]
        == "cancelled"
    )
    assert store.stats()["counts"]["return_operations"] == 1


def test_unknown_result_after_commit_can_be_checked_without_blind_retry(
    store, monkeypatch
):
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    original_tx = store._tx
    state = {"raised": False}

    @contextmanager
    def commit_then_lose_response():
        with original_tx() as connection:
            yield connection
        if not state["raised"]:
            state["raised"] = True
            raise RuntimeError("模拟写入成功但响应丢失")

    monkeypatch.setattr(store, "_tx", commit_then_lose_response)

    unknown = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert unknown["success"] is False
    assert "退货写入失败" in unknown["error"]

    # 先查询恢复：写入实际已生效。
    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "return requested"
    assert store.stats()["counts"]["return_operations"] == 1

    # 不盲重试；再次提交被拒绝且不产生追加记录。
    retry = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert retry["success"] is False
    assert store.stats()["counts"]["return_operations"] == 1


def test_success_only_changes_return_fields_and_uses_order_price(store):
    bind(store)
    before = store.business_snapshot()
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(
        store, CHEN_ORDER, ["4545791457", "5012998807"], CHEN_CREDIT
    )
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")
    result = submit(
        store,
        CHEN_ORDER,
        ["4545791457", "5012998807"],
        CHEN_CREDIT,
        "确认，就按这些申请退货",
    )
    assert result["success"] is True
    assert result["estimated_amount"] == "444.77"
    assert "不表示退款已到账" in result["note"]

    after = store.business_snapshot()
    # 用户与商品完全不变，避免修改支付余额或商品行。
    assert after["users"] == before["users"]
    assert after["products"] == before["products"]

    original = before["orders"][CHEN_ORDER]
    updated = after["orders"][CHEN_ORDER]
    for key in original:
        if key in {"status", "return_items", "return_payment_method_id"}:
            continue
        assert updated[key] == original[key]
    assert updated["status"] == "return requested"
    assert updated["return_items"] == ["4545791457", "5012998807"]
    assert updated["return_payment_method_id"] == CHEN_CREDIT
    assert updated["payment_history"] == original["payment_history"]

    # 其他订单不受影响。
    for order_id, payload in before["orders"].items():
        if order_id != CHEN_ORDER:
            assert after["orders"][order_id] == payload


def test_handoff_is_recorded_as_mock_event(store):
    bind(store)
    before = store.business_snapshot()
    result = store.record_handoff(
        "client-a", "conv-a", "用户要求人工处理退货", "超出沙箱范围"
    )
    assert result["success"] is True
    assert result["status"] == "recorded_mock"
    assert "未连接真实坐席" in result["note"]
    assert store.business_snapshot() == before
    events = store.list_events("client-a", "conv-a")
    assert events[-1]["event_type"] == "handoff_requested"


def test_calculate_is_restricted_and_not_order_authority():
    assert safe_calculate("(186.06 + 258.71) * 2") == {
        "success": True,
        "expression": "(186.06 + 258.71) * 2",
        "result": "889.54",
        "note": "仅用于金额计算；订单金额仍以 get_order_details 为准。",
    }
    for expression in (
        "__import__('os').system('echo x')",
        "open('x')",
        "(1).__class__",
        "1 / 0",
        "",
    ):
        result = safe_calculate(expression)
        assert result["success"] is False, expression

    # 订单金额只来自订单行价，不能通过计算器改写。
    assert normalize_order_id("w3069600") == CHEN_ORDER


def test_frozen_policy_and_knowledge_assets_are_complete_and_ordered():
    policy = POLICY.read_text(encoding="utf-8")
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == (
        "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467"
    )
    assert hashlib.sha256(KNOWLEDGE.read_bytes()).hexdigest() == (
        "e7360f9d22531ed836da12fe34d90c971a9300a3ed943031d75f8f6dc3b734c0"
    )
    rows = [
        json.loads(line)
        for line in KNOWLEDGE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 29
    assert len({row["evidence_id"] for row in rows}) == 29
    for row in rows:
        assert row["evidence_id"].startswith("K")
        assert row["content"].strip()
        assert row["source_refs"]
        assert row["policy_commit"] == (
            "fc0055dc4e0a316c3f83133267fbd6faaa770992"
        )
    joined = json.dumps(rows, ensure_ascii=False)
    assert "旧演示" not in joined
    assert "7 天" not in joined or "未规定" in joined
    assert "one tool call at a time" in policy
    assert "explicit user confirmation" in policy


def test_english_prepare_display_confirm_submit(tmp_path):
    """英文业务链：待确认详情、明确确认、提交与真实业务快照。"""
    store = RetailStore(tmp_path / "retail.db", SEED, language="en")
    bind(store)
    store.begin_turn("client-a", "conv-a")

    prepared = prepare(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
    )
    text = prepared["display_text"]
    assert "RETURN REQUEST DETAILS (PENDING CONFIRMATION)" in text
    assert "Estimated refund amount: $1,124.98" in text
    assert "Refund destination:" in text
    assert "退货" not in text
    assert prepared["note"].startswith("The details have not been shown")

    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    rejected = submit(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
        "No, cancel that.",
    )
    assert rejected["success"] is False
    assert "explicit confirmation" in rejected["error"]
    assert (
        store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"][
            "status"
        ]
        == "delivered"
    )

    approved = submit(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
        "Yes, please submit this return.",
    )
    assert approved["success"] is True
    assert approved["status"] == "return requested"
    assert approved["return_items"] == ["4545791457", "8551474201"]
    assert approved["estimated_amount"] == "$1,124.98"

    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "return requested"
    assert order["return_items"] == ["4545791457", "8551474201"]


def test_english_acceptance_of_displayed_amount_can_complete_submission(tmp_path):
    """明确提交指令加接受已展示金额，应走同一生产提交路径。"""
    store = RetailStore(tmp_path / "retail.db", SEED, language="en")
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
    )
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    approved = submit(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
        "Yes, please submit this return. The estimated refund of $1,124.98 "
        "to my original credit card payment method works for me.",
    )
    assert approved["success"] is True
    assert approved["status"] == "return requested"
    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "return requested"
    assert order["return_items"] == ["4545791457", "8551474201"]


@pytest.mark.parametrize(
    "message",
    [
        "The estimated refund of $1,124.98 to my credit card works for me.",
        "Yes, please submit this return. The estimated refund of $1,000.00 works for me.",
        "Yes, please submit this return if the estimated refund of $1,124.98 works for me.",
        "Yes, please submit this return. The estimated refund of $1,124.98 to my bank account works for me.",
        "Yes, please submit this return with an extra item. The estimated refund of $1,124.98 works for me.",
        "My friend said yes. The estimated refund of $1,124.98 works for me.",
        "Yes, please submit this return. The estimated refund of $1,124.98 does not work for me.",
    ],
)
def test_english_amount_acceptance_keeps_confirmation_guards(tmp_path, message):
    """仅有金额接受、金额不符、条件、改去向、改商品、引用他人或否定仍不得写入。"""
    store = RetailStore(tmp_path / "retail.db", SEED, language="en")
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
    )
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    rejected = submit(
        store,
        CHEN_ORDER,
        ["8551474201", "4545791457"],
        CHEN_CREDIT,
        message,
    )
    assert rejected["success"] is False
    order = store.get_order_details("client-a", "conv-a", CHEN_ORDER)["order"]
    assert order["status"] == "delivered"
    assert store.active_pending("client-a", "conv-a")[0]["status"] == "displayed"


def test_chinese_confirmation_rules_remain_strict(store):
    """英文金额接受规则不改变中文否定规则和明确确认。"""
    bind(store)
    store.begin_turn("client-a", "conv-a")
    prepared = prepare(store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT)
    display(store, prepared["op_id"], turn=1)
    store.begin_turn("client-a", "conv-a")

    rejected = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，但是把数量改成两件"
    )
    assert rejected["success"] is False
    approved = submit(
        store, CHEN_ORDER, ["4545791457"], CHEN_CREDIT, "确认，就按这些申请退货"
    )
    assert approved["success"] is True


def test_handoff_summary_uses_server_identity_state(store):
    """交接摘要不得保留与已绑定身份矛盾的模型陈述。"""
    bind(store)
    before = store.business_snapshot()
    result = store.record_handoff(
        "client-a",
        "conv-a",
        "用户（尚未验证身份）没有订单号，希望人工查看订单。",
        "用户明确要求人工",
    )
    assert result["success"] is True
    assert store.business_snapshot() == before

    event = store.list_events("client-a", "conv-a")[-1]
    payload = event["payload"]
    assert event["event_type"] == "handoff_requested"
    assert payload["identity"] == {
        "source": "server_session",
        "bound": True,
        "user_id": "chen_silva_7485",
    }
    assert "服务端会话已绑定用户 chen_silva_7485" in payload["summary"]
    assert "尚未验证身份" not in payload["summary"]
    assert "没有订单号" in payload["summary"]
    assert "希望人工查看订单" in payload["summary"]
    assert payload["reason"] == "用户明确要求人工"

    reopened = RetailStore(store.db_path, SEED)
    persisted = reopened.list_events("client-a", "conv-a")[-1]["payload"]
    assert persisted["summary"] == payload["summary"]
    assert persisted["identity"] == payload["identity"]


def test_handoff_summary_reports_unbound_server_state(store):
    """未绑定会话也不得沿用模型摘要中的已验证陈述。"""
    store.record_handoff(
        "client-unbound",
        "conv-unbound",
        "用户身份已验证。希望人工处理订单问题。",
        "超出范围",
    )
    event = store.list_events("client-unbound", "conv-unbound")[-1]
    payload = event["payload"]
    assert payload["identity"] == {
        "source": "server_session",
        "bound": False,
        "user_id": None,
    }
    assert "服务端会话尚未绑定用户" in payload["summary"]
    assert "身份已验证" not in payload["summary"]
    assert "希望人工处理订单问题" in payload["summary"]


def test_english_rejects_invalid_language(tmp_path):
    with pytest.raises(RetailStoreError):
        RetailStore(tmp_path / "retail.db", SEED, language="fr")
