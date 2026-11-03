"""事实账本不变量：引用须逐字来自本轮消息、改口保留取代关系、无效操作被拒。"""
from memory.task_facts import apply_ops, quoted, state_view

MSG1 = "I want to return the blue kettle and refund it to my PayPal."
MSG2 = "Actually, send the refund to my gift card instead."


def test_quote_must_come_from_message():
    assert quoted("refund it to my paypal", MSG1)
    assert quoted("I’d", "I'd like")
    assert not quoted("refund to my visa", MSG1)


def test_change_of_mind_supersedes_and_keeps_history():
    first = apply_ops([], {"add": [
        {"kind": "item", "subject": "item", "value": "blue kettle", "quote": "the blue kettle"},
        {"kind": "payment", "subject": "refund destination", "value": "PayPal", "quote": "refund it to my PayPal"},
        {"kind": "payment", "subject": "x", "value": "Visa", "quote": "my Visa please"},
    ]}, MSG1, 1)
    assert len(first["facts"]) == 2 and first["rejected"][0]["reason"] == "quote_not_in_message"
    paypal = next(f for f in first["facts"] if f["value"] == "PayPal")
    second = apply_ops(first["facts"], {"supersede": [
        {"id": paypal["id"], "value": "gift card", "quote": "send the refund to my gift card"},
        {"id": "Fmissing", "value": "cash", "quote": "send the refund"},
    ]}, MSG2, 2)
    assert [r["reason"] for r in second["rejected"]] == ["unknown_or_inactive_fact"]
    view = state_view(second["facts"])
    assert {f["value"] for f in view["current"]} == {"blue kettle", "gift card"}
    assert view["changed_mind"] == [{"turn": 2, "kind": "payment", "subject": "refund destination",
                                     "old": "PayPal", "new": "gift card"}]
    # 已取代的事实不能再被取代。
    third = apply_ops(second["facts"], {"supersede": [{"id": paypal["id"], "value": "Visa", "quote": "gift card"}]},
                      MSG2, 3)
    assert third["rejected"][0]["reason"] == "unknown_or_inactive_fact"
