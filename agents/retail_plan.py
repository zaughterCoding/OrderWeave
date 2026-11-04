"""会话请求清单：记录用户要求办理的写操作，并由业务记录推出每项进度。

多动作会话的主要失败是漏做：Agent 专注于一项时遗忘其余请求，或在仍有可办事项时转人工。
模型每轮从用户最新消息提出清单增量（新增、放弃、关联订单），确定性代码只校验不变量：
引用必须逐字出现在本轮用户消息中、类型限于七类写操作、关联的订单号必须出现在本轮对话原文里。
每项状态（未准备、待确认、已完成）只由真实业务事件与待确认记录推出，模型不能标记完成。
清单只用于提醒与编排，不构成身份或交易授权。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from memory.task_facts import quoted
from mcp_tools.retail_domain import WRITE_KINDS
from mcp_tools.retail_store import normalize_order_id

logger = logging.getLogger(__name__)

KINDS = tuple(WRITE_KINDS.values())
ACTIVE = ("open", "awaiting_confirmation")
MAX_OPS = 10
MAX_ACTIVE = 12

_LABELS = {
    "return": ("Return", "退货"), "exchange": ("Exchange", "换货"), "cancel": ("Order cancellation", "取消订单"),
    "modify_items": ("Order item change", "修改订单商品"), "modify_address": ("Order address change", "修改订单地址"),
    "modify_payment": ("Order payment change", "修改订单支付方式"),
    "modify_user_address": ("Account address change", "修改账户地址"),
}

_SYSTEM = """You keep the checklist of changes a customer has asked for in a retail support conversation.
Given the current checklist, the agent's previous message and the customer's latest message, propose changes caused by the latest message only.

- add: a change the customer asks for: return, exchange, cancel (a pending order), modify_items (a pending order), modify_address (a pending order's shipping address), modify_payment (a pending order), modify_user_address (the account's default address). Use one entry per order and change type. Add it even if details are still missing. Do not add an entry that is already on the checklist.
- drop: the customer no longer wants an earlier entry (they withdraw it, or accept that it cannot be done). When they switch to a different change (for example an exchange instead of a return), drop the old entry and add the new one.
- link: it has become clear which order an entry refers to; give the entry id and the order id exactly as written in the messages.
Each add and drop needs "quote": an exact, verbatim substring of the customer's latest message that supports it. Questions, requests for information, and confirmations of details already shown add nothing. If nothing changes, return empty lists.

Return only JSON: {"add": [{"kind": "...", "order_id": "#W... or null", "summary": "short description", "quote": "..."}],
"drop": [{"id": "...", "quote": "..."}], "link": [{"id": "...", "order_id": "#W..."}]}"""


def _goal_id(turn: int, index: int, kind: str) -> str:
    return "R" + hashlib.sha256(f"{turn}:{index}:{kind}".encode("utf-8")).hexdigest()[:8]


def _order_in(order_id: str, texts: List[str]) -> bool:
    """订单号必须逐字出现在本轮对话原文中，防止模型编造或串用订单。"""
    bare = order_id.lstrip("#").upper()
    return bool(bare) and any(re.search(rf"(?<![A-Z0-9]){re.escape(bare)}(?![0-9])", str(t or "").upper())
                              for t in texts)


def empty_plan() -> Dict[str, Any]:
    return {"goals": [], "nudged": []}


def apply_ops(plan: Dict[str, Any], ops: Dict[str, Any], message: str, previous_reply: str,
              turn: int, after_event_id: int) -> Dict[str, Any]:
    """在副本上应用增量，返回 {plan, accepted, rejected}；只校验不变量，不做语义判断。"""
    goals = [dict(g) for g in plan.get("goals") or []]
    by_id = {g["id"]: g for g in goals}
    accepted, rejected = [], []
    texts = [message, previous_reply]
    items = [("add", op) for op in ops.get("add") or []] + [("drop", op) for op in ops.get("drop") or []] \
        + [("link", op) for op in ops.get("link") or []]
    for index, (action, op) in enumerate(items[:MAX_OPS]):
        reason = ""
        order_id = normalize_order_id(op.get("order_id")) if isinstance(op, dict) and op.get("order_id") else ""
        if not isinstance(op, dict):
            reason = "malformed"
        elif action in {"add", "drop"} and not quoted(op.get("quote", ""), message):
            reason = "quote_not_in_message"
        elif action == "add" and op.get("kind") not in KINDS:
            reason = "bad_kind"
        elif action in {"drop", "link"} and by_id.get(op.get("id"), {}).get("status") not in ACTIVE:
            reason = "unknown_or_inactive_goal"
        elif order_id and not _order_in(order_id, texts):
            reason = "order_not_in_dialogue"
        elif action == "link" and not order_id:
            reason = "missing_order"
        elif action == "add" and any(g["status"] in ACTIVE and g["kind"] == op["kind"]
                                     and (g.get("order_id") or "") == order_id for g in goals):
            reason = "duplicate"
        if reason:
            rejected.append({"action": action, "op": op, "reason": reason})
            continue
        if action == "add":
            goal = {"id": _goal_id(turn, index, op["kind"]), "turn": turn, "kind": op["kind"],
                    "order_id": order_id or None, "summary": str(op.get("summary") or "")[:160],
                    "quote": str(op["quote"])[:240], "status": "open", "after_event_id": int(after_event_id)}
            goals.append(goal)
            by_id[goal["id"]] = goal
        elif action == "drop":
            by_id[op["id"]].update(status="dropped", ended_turn=turn, end_quote=str(op["quote"])[:240])
        else:
            by_id[op["id"]]["order_id"] = order_id
        accepted.append({"action": action, "op": op})
    active = [g for g in goals if g["status"] in ACTIVE]
    for goal in active[:max(0, len(active) - MAX_ACTIVE)]:
        # 超出容量时最早的未完成项转为过期，避免清单无界增长。
        goal.update(status="expired", ended_turn=turn)
    return {"plan": {**plan, "goals": goals}, "accepted": accepted, "rejected": rejected}


def _event_kind(event: Dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    if event_type == "return_requested":
        return "return"
    if event_type.endswith("_executed"):
        return str((event.get("payload") or {}).get("kind") or event_type[:-len("_executed")])
    return ""


def refresh(plan: Dict[str, Any], events: List[Dict[str, Any]], pending: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按业务事件与待确认记录推出每项状态；每个已执行事件最多认领一项，且须晚于该项提出。"""
    goals = [dict(g) for g in plan.get("goals") or []]
    claimed = {g.get("done_op_id") for g in goals if g.get("done_op_id")}
    executed = [e for e in events if _event_kind(e) and e.get("op_id") not in claimed]
    active = [g for g in goals if g["status"] in ACTIVE]
    # 先认领订单号一致的项，再认领尚未关联订单的同类项。
    for exact in (True, False):
        for goal in active:
            if goal["status"] == "done" or bool(goal.get("order_id")) != exact:
                continue
            event = next((e for e in executed if _event_kind(e) == goal["kind"]
                          and int(e["event_id"]) > int(goal.get("after_event_id") or 0)
                          and (not exact or normalize_order_id(e.get("order_id")) == goal["order_id"])), None)
            if event is not None:
                executed.remove(event)
                goal.update(status="done", done_op_id=event.get("op_id"),
                            order_id=goal.get("order_id") or (normalize_order_id(event.get("order_id")) or None))
    for goal in active:
        if goal["status"] == "done":
            continue
        waiting = any(p.get("kind") == goal["kind"] and p.get("status") in ("prepared", "displayed")
                      and (not goal.get("order_id") or normalize_order_id(p.get("order_id")) == goal["order_id"])
                      for p in pending)
        goal["status"] = "awaiting_confirmation" if waiting else "open"
    return {**plan, "goals": goals}


def open_goals(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    nudged = set(plan.get("nudged") or [])
    return [{**g, "nudged": g["id"] in nudged} for g in plan.get("goals") or [] if g["status"] in ACTIVE]


def plan_view(plan: Dict[str, Any]) -> Dict[str, Any]:
    """状态快照中的清单视图：进度来自业务记录，不由模型声明。"""
    rows = [{k: g.get(k) for k in ("id", "kind", "order_id", "summary", "status")}
            for g in plan.get("goals") or [] if g["status"] in ACTIVE + ("done",)]
    return {"note": "Changes the customer asked for in this conversation. Status comes from the business records "
                    "(done = carried out, awaiting_confirmation = details shown, open = not prepared yet). Work "
                    "through open items one at a time and do not end or transfer while items you can handle remain "
                    "open. This is not authorization.",
            "items": rows}


def _label(goal: Dict[str, Any], english: bool) -> str:
    label = _LABELS.get(goal["kind"], (goal["kind"], goal["kind"]))[0 if english else 1]
    if goal.get("order_id"):
        label += f" for order {goal['order_id']}" if english else f"（订单 {goal['order_id']}）"
    return label


def reminder(plan: Dict[str, Any], english: bool, turn: int, pending: List[Dict[str, Any]]) -> str:
    """执行结果被固定话术替换时，用确定性文字列出仍未完成的请求；不使用模型文本。"""
    shown_now = {(p.get("kind"), normalize_order_id(p.get("order_id"))) for p in pending
                 if int(p.get("prepared_turn") or 0) == turn}
    lines = []
    for goal in plan.get("goals") or []:
        if goal["status"] == "open":
            state = "not prepared yet" if english else "尚未准备"
        elif goal["status"] == "awaiting_confirmation" and (goal["kind"], goal.get("order_id") or "") not in shown_now:
            state = "details shown earlier, awaiting your confirmation" if english else "此前已展示详情，等待您确认"
        else:
            continue
        lines.append(f"- {_label(goal, english)}: {state}")
    if not lines:
        return ""
    head = ("Still open from your earlier requests:" if english else "您此前提出、尚未完成的请求：")
    tail = ("Let me know if you would like to continue with them." if english else "如需继续办理，请告诉我。")
    return "\n".join([head, *lines, tail])


def handoff_feedback(goals: List[Dict[str, Any]], english: bool) -> str:
    rows = "\n".join(f"- {g['id']} {g['kind']} {g.get('order_id') or ''} ({g['status']}): {g.get('summary', '')}"
                     for g in goals)
    if english:
        return ("[Checklist check by the system, not the customer] The transfer was not made. These requested "
                f"changes are still open:\n{rows}\nTransfer only when a request cannot be handled with your tools "
                "and policy. Tell the customer what you can still do for each open item (or why it cannot be "
                "done) and ask how they want to proceed. If they still want a human after that, you may transfer.")
    return ("[系统清单检查，不是用户消息] 本次未转人工。以下请求仍未完成：\n" + rows +
            "\n只有在工具与政策确实无法处理时才转人工。请向用户说明每项仍可如何办理（或为何不能办理）并询问如何继续；"
            "说明后用户仍要求人工，可以再转接。")


async def extract(client: Any, model: str, plan: Dict[str, Any], previous_reply: str, message: str,
                  max_tokens: int = 500) -> Optional[Dict[str, Any]]:
    """调用模型提出清单增量；失败返回 None（本轮不更新清单，不阻断对话）。"""
    from core.llm_utils import extract_text_content

    checklist = [{k: g.get(k) for k in ("id", "kind", "order_id", "summary", "status")}
                 for g in plan.get("goals") or [] if g["status"] in ACTIVE]
    payload = {"checklist": checklist, "previous_agent_message": str(previous_reply or "")[-1500:],
               "latest_customer_message": str(message or "")[:3000]}
    try:
        response = await client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0.0, system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        if getattr(response, "stop_reason", "end_turn") != "end_turn":
            return None
        text = extract_text_content(response.content)
        match = re.search(r"\{.*\}", text, re.S)
        ops = json.loads(match.group(0) if match else text)
        return ops if isinstance(ops, dict) else None
    except Exception as ex:
        from agents.agent_orchestrator import _stop_exception

        if _stop_exception(ex):
            raise
        logger.warning("请求清单抽取失败，本轮不更新: %s", ex)
        return None


__all__ = ["KINDS", "empty_plan", "apply_ops", "refresh", "open_goals", "plan_view", "reminder",
           "handoff_feedback", "extract"]
