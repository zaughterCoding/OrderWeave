"""请求清单不变量：引用与订单号须来自本轮对话原文、进度只由业务事件推出、转人工只提醒一次。"""
from types import SimpleNamespace

from agents.retail_hooks import OpenRequestsHandoffHook
from agents.retail_plan import apply_ops, empty_plan, open_goals, refresh, reminder

MSG = "Please change the address on order W5270061 to my home, and swap the backpack to grey."
PREV = "I found order #W5270061 and order #W5782623."


def _executed(event_id, kind, order_id, op_id):
    return {"event_id": event_id, "event_type": f"{kind}_executed", "order_id": order_id, "op_id": op_id,
            "payload": {"kind": kind}}


def test_quote_kind_and_order_must_come_from_dialogue():
    out = apply_ops(empty_plan(), {"add": [
        {"kind": "modify_address", "order_id": "W5270061", "summary": "home address", "quote": "change the address"},
        {"kind": "modify_items", "order_id": None, "summary": "grey backpack", "quote": "swap the backpack to grey"},
        {"kind": "modify_items", "order_id": "#W9999999", "summary": "x", "quote": "swap the backpack"},
        {"kind": "refund_now", "summary": "x", "quote": "Please change"},
        {"kind": "cancel", "summary": "x", "quote": "cancel everything"},
    ]}, MSG, PREV, turn=1, after_event_id=0)
    assert [r["reason"] for r in out["rejected"]] == ["order_not_in_dialogue", "bad_kind", "quote_not_in_message"]
    goals = out["plan"]["goals"]
    assert [(g["kind"], g["order_id"]) for g in goals] == [("modify_address", "#W5270061"), ("modify_items", None)]
    # 订单号可以来自上一条 Agent 回复；未出现过的订单号不能关联。
    items = goals[1]["id"]
    linked = apply_ops(out["plan"], {"link": [{"id": items, "order_id": "#W5782623"},
                                              {"id": items, "order_id": "#W0000001"}]},
                       "the second one", PREV, turn=2, after_event_id=0)
    assert [r["reason"] for r in linked["rejected"]] == ["order_not_in_dialogue"]
    assert linked["plan"]["goals"][1]["order_id"] == "#W5782623"
    # 重复新增同类同单请求被拒。
    dup = apply_ops(out["plan"], {"add": [{"kind": "modify_address", "order_id": "W5270061", "summary": "again",
                                           "quote": "change the address"}]}, MSG, PREV, turn=2, after_event_id=0)
    assert dup["rejected"][0]["reason"] == "duplicate"


def test_status_only_from_events_and_each_event_claims_one_goal():
    plan = apply_ops(empty_plan(), {"add": [
        {"kind": "modify_address", "order_id": "W5270061", "summary": "a", "quote": "change the address"},
        {"kind": "modify_items", "order_id": None, "summary": "b", "quote": "swap the backpack"},
        {"kind": "modify_items", "order_id": "W5270061", "summary": "c", "quote": "to grey"},
    ]}, MSG, PREV, turn=1, after_event_id=5)["plan"]
    # 提出之前的事件不能认领；待确认记录只推出 awaiting_confirmation。
    events = [_executed(4, "modify_address", "#W5270061", "OP-old"), _executed(7, "modify_items", "#W5782623", "OP-1")]
    pending = [{"kind": "modify_address", "order_id": "#W5270061", "status": "displayed", "prepared_turn": 2}]
    plan = refresh(plan, events, pending)
    statuses = [(g["status"], g["order_id"]) for g in plan["goals"]]
    assert statuses == [("awaiting_confirmation", "#W5270061"), ("done", "#W5782623"), ("open", "#W5270061")]
    # 重复刷新不会让同一事件认领第二项。
    assert refresh(plan, events, pending) == plan
    note = reminder(plan, True, turn=3, pending=pending)
    assert "Order item change for order #W5270061: not prepared yet" in note
    assert "Order address change for order #W5270061: details shown earlier" in note
    # 提醒只由类型与订单号确定性生成：已完成项不列出，也不含模型写的摘要。
    assert len(note.splitlines()) == 4 and "OP-1" not in note


def test_dropped_goal_leaves_checklist_and_reminder():
    plan = apply_ops(empty_plan(), {"add": [{"kind": "cancel", "order_id": "W5270061", "summary": "x",
                                             "quote": "change the address"}]}, MSG, PREV, 1, 0)["plan"]
    goal_id = plan["goals"][0]["id"]
    dropped = apply_ops(plan, {"drop": [{"id": goal_id, "quote": "never mind"}]},
                        "Never mind, keep it.", "", 2, 0)["plan"]
    assert dropped["goals"][0]["status"] == "dropped"
    assert open_goals(dropped) == [] and reminder(dropped, True, 2, []) == ""


def test_handoff_hook_reminds_once_and_skips_nudged_goals():
    hook = OpenRequestsHandoffHook()
    goal = {"id": "R1", "kind": "return", "order_id": "#W1", "summary": "s", "status": "open", "nudged": False}
    req = SimpleNamespace(retail_plan=[goal])
    ctx = SimpleNamespace(req=req, state={}, agent=SimpleNamespace(language="en"))
    assert hook.advisory and hook.pre_tool(ctx, "get_order_details", {}) is None
    assert "transfer was not made" in hook.pre_tool(ctx, "transfer_to_human_agents", {})
    assert req.plan_handoff_nudged == ["R1"]
    # 同一请求内第二次调用放行；已提醒过的请求在后续轮次也放行。
    assert hook.pre_tool(ctx, "transfer_to_human_agents", {}) is None
    later = SimpleNamespace(req=SimpleNamespace(retail_plan=[{**goal, "nudged": True}]), state={},
                            agent=SimpleNamespace(language="en"))
    assert hook.pre_tool(later, "transfer_to_human_agents", {}) is None
