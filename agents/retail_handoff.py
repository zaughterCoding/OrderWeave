"""角色交接的有界引用视图；不能提供身份或交易授权。"""
from __future__ import annotations

from typing import Any, TypedDict

from agents.retail_support import support_handoff_view

ROLES = {"general", "technical", "billing", "escalation"}


class RoleOutcome(TypedDict):
    role: str
    turn: int
    success: bool
    note: str
    source: str
    tool_refs: list[dict[str, Any]]
    evidence_ids: list[str]


class RoleHandoff(TypedDict):
    version: int
    from_role: str
    to_role: str
    source_turn: int
    target_turn: int
    task: dict[str, str]
    identity: dict[str, Any]
    user_report: dict[str, Any]
    pending_refs: list[dict[str, Any]]
    event_refs: list[dict[str, Any]]
    support_refs: dict[str, Any]
    previous_role: dict[str, Any]


def bounded_outcome(value, turn):
    """旧 checkpoint 只保留已知字段；角色回复仅作非权威信息。"""
    if not isinstance(value, dict) or value.get("role") not in ROLES:
        return {}
    source_turn = value.get("turn")
    if type(source_turn) is not int or not 0 < source_turn <= turn:
        return {}
    refs = value.get("tool_refs")
    refs = refs if isinstance(refs, list) else []
    evidence = value.get("evidence_ids")
    evidence = evidence if isinstance(evidence, list) else []
    return {
        "role": value["role"], "turn": source_turn, "success": value.get("success") is True,
        "note": str(value.get("note") or "")[:300], "source": "role_response_informational_only",
        "tool_refs": [{"tool": str(r.get("tool") or "")[:60],
                       "tool_use_id": str(r.get("tool_use_id") or "")[:80],
                       "result_success": r.get("result_success") is True,
                       "source": "tool_trace_reference"}
                      for r in refs[:8] if isinstance(r, dict)],
        "evidence_ids": [str(e)[:80] for e in evidence[:6]],
    }


def role_outcome(req, response) -> RoleOutcome:
    evidence = []
    refs = []
    for trace in list(response.tool_traces or []):
        refs.append({"tool": trace.get("tool_name"), "tool_use_id": trace.get("tool_use_id"),
                     "result_success": trace.get("result_success")})
        result = trace.get("result")
        if trace.get("tool_name") == "search_knowledge_base" and isinstance(result, dict) and result.get("success") is True:
            evidence.extend(str(item.get("evidence_id") or item.get("id") or "")
                            for item in result.get("results", []) if isinstance(item, dict))
    return bounded_outcome({"role": response.agent_type.value, "turn": req.turn,
                            "success": response.success, "note": response.content,
                            "tool_refs": refs, "evidence_ids": list(dict.fromkeys(evidence))}, req.turn)


def build_role_handoff(store, req, previous, target_role) -> RoleHandoff:
    """每次派发都重读业务引用，不从上一角色回复继承权威状态。"""
    previous = bounded_outcome(previous, req.turn)
    pending = store.active_pending(req.user_id, req.conv_id)
    events = store.list_events(req.user_id, req.conv_id)
    bound = store.bound_user_id(req.user_id, req.conv_id)
    packet = req.retail_support or {}
    view = support_handoff_view(packet, pending)
    work = packet.get("work_record") or {}
    flow = view["return_flow"]
    demand = packet.get("current_demand") or {}
    steps = list(work.get("taken_steps") or []) + list(packet.get("taken_steps") or [])
    retrieved = list(work.get("retrieved_evidence") or [])
    return {
        "version": 1, "from_role": previous.get("role", ""), "to_role": target_role,
        "source_turn": previous.get("turn", 0), "target_turn": req.turn,
        "task": {"goal": "return" if pending or flow.get("status") in {"active", "awaiting_confirmation"}
                 else str(demand.get("kind") or "retail")[:40],
                 "phase": str(view["phase"])[:40]},
        "identity": {"source": "server_session", "bound": bool(bound), "user_id": bound},
        "user_report": {"source": "user_message", "turn": req.turn, "text": req.message[:200]},
        "pending_refs": [{"op_id": p["op_id"], "order_id": p["order_id"], "status": p["status"],
                          "source": "business_sqlite", "prepared_turn": p["prepared_turn"],
                          "displayed_turn": p.get("displayed_turn")} for p in pending[:6]],
        "event_refs": [{"event_id": e["event_id"], "kind": e["event_type"],
                        "order_id": e.get("order_id"), "op_id": e.get("op_id"),
                        "source": "business_sqlite"} for e in events[-6:]],
        "support_refs": {"source": "host_support_record", "record_turn": work.get("last_turn", 0), "view_turn": req.turn,
                         "product_id": str(packet.get("product_id") or "")[:40],
                         "issue": str(packet.get("issue") or "")[:60],
                         "user_reported_steps": list(dict.fromkeys(str(s.get("step_id") or "")[:40]
                                                    for s in steps if isinstance(s, dict)))[:6],
                         "retrieved_evidence_ids": list(dict.fromkeys(str(e.get("evidence_id") or "")[:80]
                                                     for e in retrieved if isinstance(e, dict)))[:6],
                         "unresolved": [str(u)[:120] for u in view["unresolved"][-4:]]},
        "previous_role": previous,
    }


def handoff_note(packet, language):
    """模拟人工记录使用来源明确的交接引用，不把角色说明当业务事实。"""
    if not isinstance(packet, dict) or packet.get("version") != 1:
        return ""
    previous = bounded_outcome(packet.get("previous_role"), packet.get("target_turn", 0))
    refs = [str(event.get("event_id")) for event in packet.get("event_refs", [])[:6]]
    support = packet.get("support_refs") or {}
    if language == "en":
        return (f"\nRole handoff: {packet.get('from_role')} -> {packet.get('to_role')}; "
                f"source turn: {packet.get('source_turn')}; business event references: {', '.join(refs)}. "
                f"User-reported steps: {support.get('user_reported_steps', [])}. "
                f"Prior role note (unverified): {previous.get('note', '')}")
    return (f"\n角色交接：{packet.get('from_role')}→{packet.get('to_role')}；"
            f"来源轮次：{packet.get('source_turn')}；业务事件引用：{', '.join(refs)}。"
            f"用户报告已尝试步骤：{support.get('user_reported_steps', [])}。"
            f"上一角色说明（未核验）：{previous.get('note', '')}")
