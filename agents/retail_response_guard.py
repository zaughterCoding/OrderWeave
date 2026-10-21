"""以本轮业务事实约束有限高风险表述，不执行工具或重试交易。"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


_ACTIONS = {
    "prepare_return": "return_prepared",
    "return_delivered_order_items": "return_requested",
    "transfer_to_human_agents": "handoff_requested",
}
_QUESTIONS = re.compile(r"[？?]|(?:是否|有没有|能否|吗)(?:[，。！；\s]|$)|^(?:can|could|has|have|is|are|did)\b", re.I)
_UNOBSERVED = re.compile(
    r"退款.{0,8}(?:已到账|未到账|没有到账|尚未到账|已经退回)|"
    r"仓库.{0,8}(?:已收货|未收货|没有收到|尚未收到)|"
    r"邮件.{0,5}(?:已发送|未发送)|"
    r"(?:已|已经).{0,4}创建.{0,4}真实工单|"
    r"(?:refund.{0,20}(?:has been issued|has not been issued|has arrived)|"
    r"warehouse.{0,20}(?:has received|has not received)|email.{0,12}(?:was sent|has been sent))", re.I,
)
_SUBMITTED = re.compile(
    r"(?:已|已经|成功).{0,6}(?:提交|申请|办理).{0,5}退货|退货申请.{0,5}(?:已提交|成功)|"
    r"(?:return request.{0,15}(?:submitted|successful)|submitted.{0,15}return)", re.I,
)
_HANDOFF = re.compile(
    r"(?:已|已经).{0,6}(?:转接|连接|接入|记录).{0,8}(?:人工|坐席|交接)|"
    r"(?:live agent.{0,12}(?:connected|taken over)|handoff.{0,12}(?:recorded|completed))", re.I,
)
_IDENTITY = re.compile(r"身份.{0,5}(?:已核实|已验证|核对成功)|已.{0,4}(?:核实|验证).{0,4}身份|identity.{0,10}(?:verified|confirmed)", re.I)
_EXCHANGE = re.compile(r"(?:可以|可为您|建议|支持|帮您|为您办理).{0,12}换货|后续如需.{0,15}换货.{0,20}(?:交易流程|原有流程)|(?:can|recommend|offer|arrange).{0,20}(?:exchange|replacement)", re.I)
_GIFT_REFUSAL = re.compile(r"礼品卡.{0,18}(?:余额不足|没有余额|余额为\s*0|余额是\s*0).{0,18}(?:不能|无法|不支持|不可).{0,10}(?:退款|退货)|礼品卡.{0,100}余额.{0,30}低于.{0,20}无法覆盖|gift card.{0,20}(?:insufficient|zero|no balance).{0,25}(?:cannot|can't|unable)", re.I)
_CHANGE_DESTINATION = re.compile(r"(?:之后|后续|提交后|稍后|随时).{0,12}(?:更改|修改|更换|调整).{0,8}(?:退款去向|支付方式)|(?:can|may).{0,15}change.{0,15}(?:refund destination|payment method).{0,15}(?:later|after)", re.I)
_ORDER_ID = re.compile(r"#?W\d+", re.I)
_STATUS = re.compile(r"(?:订单状态|order status)\s*(?:[:：]|为|是)\s*([^\n，。.,;；]+)", re.I)
_AMOUNT = re.compile(r"(?:预计申请金额|预计退款金额|退款金额|estimated refund amount)\s*[:：为是]?\s*[¥￥$]?\s*(\d[\d,]*(?:\.\d+)?)", re.I)


@dataclass
class GuardedResponse:
    text: str
    record: dict[str, Any]


def _ok(trace: dict[str, Any]) -> bool:
    result = trace.get("result")
    return (
        not trace.get("skipped") and not trace.get("blocked_by_guard")
        and trace.get("success") is not False
        and trace.get("result_success") is True
        and isinstance(result, dict) and result.get("success") is True
    )


def _money(cents: int, english: bool) -> str:
    return f"{'$' if english else '¥'}{Decimal(cents) / 100:.2f}"


def _items(item_ids: list[str]) -> str:
    return ", ".join(f"{key} × {count}" for key, count in sorted(Counter(item_ids).items()))


def _order_id(value: Any) -> str:
    return "#" + str(value).strip().lstrip("#").upper() if value else ""


def constrain_response(
    text: str, *, traces: list[dict[str, Any]], events: list[dict[str, Any]],
    pending_before: list[dict[str, Any]], pending_after: list[dict[str, Any]],
    turn: int, bound_user_id: str | None, language: str = "zh",
) -> GuardedResponse:
    """输入仅限服务层选出的同会话本轮事实；开放语义仍需独立复核。"""
    started = time.perf_counter()
    candidate = text
    english = language == "en"
    findings: list[str] = []
    facts: list[dict[str, Any]] = []
    blocks: list[str] = []
    prepared = [
        p for p in pending_after
        if p.get("status") == "prepared" and p.get("prepared_turn") == turn
    ]
    submitted = [e for e in events if e.get("event_type") == "return_requested"]
    handoffs = [e for e in events if e.get("event_type") == "handoff_requested"]
    already_submitted = any(
        t.get("tool_name") == "get_order_details" and _ok(t)
        and (t["result"].get("order") or {}).get("status") == "return requested"
        for t in traces
    )
    for event in submitted:
        payload = event["payload"]
        order_id, op_id = event["order_id"], event["op_id"]
        facts.append({"kind": "submitted", "order_id": order_id, "op_id": op_id,
                      "source": "event", "event_id": event["event_id"],
                      "amount_cents": payload["amount_cents"],
                      "item_ids": payload["item_ids"],
                      "payment_method_id": payload["payment_method_id"]})
        # 金额、商品与去向直接取事务事件，不使用模型措辞或其他订单的结果。
        detail = (
            f"Order: {order_id}\nOperation ID: {op_id}\n"
            f"Items: {_items(payload['item_ids'])}\n"
            f"Estimated refund amount: {_money(payload['amount_cents'], english)} "
            f"(based on the order price)\nRefund destination: {payload['payment_method_id']}"
            if english else
            f"订单号：{order_id}\n操作编号：{op_id}\n"
            f"退货商品（item_id × 数量）：{_items(payload['item_ids'])}\n"
            f"预计申请金额：{_money(payload['amount_cents'], english)}（按订单成交价计算）\n"
            f"退款去向：{payload['payment_method_id']}"
        )
        blocks.append(
            ("The return request was submitted. Order status: return requested.\n" if english
             else "退货申请已提交，订单状态为 return requested（已申请退货）。\n")
            + detail + ("\nRefund arrival, warehouse receipt and email delivery have not been verified. "
                        "This sandbox does not support changing the destination after submission."
                        if english else "\n提交成功不代表退款已到账；退款到账、仓库收货及邮件发送状态尚未核实。"
                        "本沙箱不支持提交后更改退款去向。")
        )
    for event in handoffs:
        facts.append({"kind": "handoff_recorded", "op_id": event["op_id"],
                      "source": "event", "event_id": event["event_id"], "status": "recorded_mock"})
        blocks.append(
            f"A simulated handoff was recorded ({event['op_id']}); no live agent was connected "
            "and no real support ticket was created."
            if english else f"已记录模拟人工交接（{event['op_id']}）；未连接真实坐席，未创建真实工单。"
        )
    for item in pending_after:
        facts.append({"kind": "prepared" if item in prepared else "awaiting_confirmation",
                      "order_id": item["order_id"], "op_id": item["op_id"],
                      "source": "pending", "turn": item.get("prepared_turn"),
                      "amount_cents": item.get("amount_cents"),
                      "payment_method_id": item.get("payment_method_id")})
    if prepared:
        blocks.append(
            "I have prepared the return details for your confirmation. No return was submitted "
            "for these details; please review them and confirm in a later reply."
            if english else "我已整理好待确认的退货详情，请核对以下内容；这些详情尚未提交，"
            "如确认无误，请在下一轮明确回复“确认，就按这些申请退货”。"
        )

    for index, trace in enumerate(traces):
        name = trace.get("tool_name")
        if name not in _ACTIONS:
            continue
        result = trace.get("result")
        result = result if isinstance(result, dict) else {}
        args = trace.get("input") or {}
        order_id = _order_id(args.get("order_id") or result.get("order_id"))
        op_id = result.get("op_id") or result.get("handoff_id")
        blocked = bool(trace.get("skipped") or trace.get("blocked_by_guard"))
        matching = [e for e in events if not blocked and (order_id or op_id) and e.get("event_type") == _ACTIONS[name]
                    and (not order_id or e.get("order_id") == order_id)
                    and (not op_id or e.get("op_id") == op_id)]
        # 旧 pending 可被本轮明确拒绝；仅当前 prepare 或当前事件构成新成功。
        if matching:
            if not _ok(trace):
                findings.append("event_overrides_tool_error")
            if _ok(trace) and name == "return_delivered_order_items":
                payload = matching[-1]["payload"]
                if (result.get("status") != "return requested"
                    or Counter(result.get("return_items") or []) != Counter(payload["item_ids"])
                    or result.get("return_payment_method_id") != payload["payment_method_id"]):
                    findings.append("tool_event_conflict")
            continue
        if not blocked and name == "prepare_return" and any(p["op_id"] == op_id for p in prepared):
            continue
        known_failure = blocked or (
            trace.get("success") is not False and result.get("success") is False
        )
        status = "failed" if known_failure else "unknown"
        facts.append({"kind": status, "tool": name, "order_id": order_id,
                      "op_id": op_id, "source": "tool_trace", "trace_index": index})
        label = ({"prepare_return": "Return preparation" if english else "准备退货",
                  "return_delivered_order_items": "Return submission" if english else "提交退货",
                  "transfer_to_human_agents": "Simulated handoff" if english else "模拟人工交接"}[name]
                 + (f" ({order_id})" if order_id else ""))
        if known_failure:
            reason = str(result.get("error") or "").strip()
            blocks.append(
                f"{label}: success was not confirmed. {reason} Please check the details or authorization."
                if english else f"{label}：工具未确认操作成功。{reason} 请核对办理信息或确认授权。"
            )
        else:
            blocks.append(
                f"{label}: the result cannot currently be confirmed; do not assume success or no write. "
                "No operation was replayed to correct this reply."
                if english else f"{label}：目前无法确认结果，不能判定成功或未写入。"
                "不会为了修正回复而重放业务操作。"
            )
            findings.append("unknown_action_result")

    # 保护服务端原文，避免把确认说明中的否定例子误当成模型断言。
    for item in pending_before + pending_after:
        display = item.get("display_text") or ""
        if display:
            text = text.replace(display, "")
    kept: list[str] = []
    for clause in re.split(r"(?<=[。！？!?；;\n])", text):
        if _QUESTIONS.search(clause):
            kept.append(clause)
            continue
        issue = ""
        replacement = ""
        if _GIFT_REFUSAL.search(clause):
            issue = "gift_card_balance_refusal"
            replacement = (
                "An existing gift card owned by you may be a refund destination; "
                "eligibility must be checked with the business tools."
                if english else "本人已有礼品卡可作为退款去向；是否可办理须以业务工具校验为准。"
            )
        elif _EXCHANGE.search(clause) and not re.search(r"不(?:能|支持|可)|cannot|can't|not support", clause, re.I):
            issue = "unsupported_exchange_advice"
            replacement = ("This sandbox does not support exchanges." if english
                           else "本沙箱不支持换货办理。")
        elif _CHANGE_DESTINATION.search(clause) and (submitted or already_submitted or "提交后" in clause) and not re.search(r"不支持|不能|cannot|not support", clause, re.I):
            issue = "post_submission_destination_change"
            replacement = ("This sandbox does not support changing the destination after submission."
                           if english else "本沙箱不支持提交后更改退款去向。")
        elif any(_UNOBSERVED.search(part) and not _QUESTIONS.search(part) and not re.search(
            r"不代表|不表示|尚未核实|无法确认|未知|does not mean|not verified|unknown|cannot confirm", part, re.I
        ) for part in re.split(r"[，,]", clause)):
            issue = "unverified_external_status"
            replacement = ("Refund arrival, warehouse receipt and email delivery have not been verified."
                           if english else "退款到账、仓库收货及邮件发送状态尚未核实。")
        elif _SUBMITTED.search(clause) and not submitted and not re.search(
            r"尚未|没有|未提交|不能|无法|确认后|提交后|will|not |cannot", clause, re.I
        ):
            issue = "unsupported_submission_claim"
            replacement = ("No new submitted return is verified in this turn." if english
                           else "本轮没有可核验的新退货提交结果。")
        elif _HANDOFF.search(clause) and not handoffs and not re.search(
            r"尚未|没有|未连接|未接入|不代表|not |cannot", clause, re.I
        ):
            issue = "unsupported_handoff_claim"
            replacement = ("No new simulated handoff is verified in this turn." if english
                           else "本轮没有可核验的新模拟交接记录。")
        elif _IDENTITY.search(clause) and not bound_user_id and not re.search(
            r"尚未|没有|未验证|未核实|not |cannot", clause, re.I
        ):
            issue = "unsupported_identity_claim"
            replacement = ("Please verify your identity with your email or full name and zip code."
                           if english else "请提供本人邮箱，或姓名加邮编以核对身份。")
        if issue:
            findings.append(issue)
            kept.append(replacement)
        else:
            kept.append(clause)
    text = "".join(kept).strip()

    # 只核对明确带标签的金额／状态和订单标识，不凭数字出现就认定退款事实。
    orders: dict[str, dict[str, Any]] = {}
    for index, trace in enumerate(traces):
        if trace.get("tool_name") != "get_order_details" or not _ok(trace):
            continue
        order = trace["result"].get("order") or {}
        if not bound_user_id or order.get("user_id") != bound_user_id or not order.get("order_id"):
            continue
        requested = (trace.get("input") or {}).get("order_id")
        if requested and _order_id(requested) != order["order_id"]:
            findings.append("tool_order_mismatch")
            continue
        orders[str(order["order_id"])] = order
        facts.append({"kind": "order_read", "order_id": order["order_id"],
                      "source": "tool_trace", "trace_index": index, "status": order["status"]})
    known_ids = set(orders) | {p["order_id"] for p in pending_after + pending_before}
    known_ids |= {e["order_id"] for e in submitted}
    assertions = "".join(c for c in re.split(r"(?<=[。！？!?；;\n])", text) if not _QUESTIONS.search(c))
    invalid_ids = {"#" + m.group().lstrip("#").upper() for m in _ORDER_ID.finditer(assertions)} - known_ids
    amount_matches = list(_AMOUNT.finditer(assertions))
    amounts = {p.get("amount_cents") for p in pending_after}
    bad_amount = bool(amount_matches) and (
        len(pending_after) != 1 or any(
            Decimal(m.group(1).replace(",", "")) * 100 not in amounts for m in amount_matches
        )
    )
    if invalid_ids or (bad_amount and not blocks) or (_STATUS.search(assertions) and not orders and not blocks):
        findings.append("unverified_labeled_order_fact")
        text = ("Please read the order with the business tools before confirming its details."
                if english else "请先通过业务工具核对订单，再确认具体状态和金额。")
    if orders and _STATUS.search(assertions) and not blocks:
        def checked_status(match: re.Match) -> str:
            # 只替换对应状态字段，不能覆盖后续退款去向等待补询问。
            ids = list(_ORDER_ID.finditer(text[:match.start()]))
            order = orders.get(_order_id(ids[-1].group())) if ids else (
                next(iter(orders.values())) if len(orders) == 1 else None
            )
            if _QUESTIONS.search(match.group()):
                return match.group()
            if order is None:
                findings.append("ambiguous_order_status")
                return "Order status: unverified" if english else "订单状态：尚未核实"
            status = str(order["status"]).lower()
            allowed = {status}
            translated = {"delivered": "已送达", "return requested": "已申请退货"}.get(status)
            if translated:
                allowed.update({translated, f"{status}（{translated}）", f"{translated}（{status}）",
                                f"{status} ({translated})", f"{translated} ({status})"})
            if match.group(1).strip().strip("*` ").lower() in allowed:
                return match.group()
            findings.append("inconsistent_order_status")
            return match.group()[:match.start(1) - match.start()] + str(order["status"])

        text = _STATUS.sub(checked_status, text)

    if blocks:
        text = "\n\n".join(blocks)
    for item in pending_after:
        display = item.get("display_text") or ""
        if display and (item in prepared or display in candidate):
            text = (text + "\n\n" + display).strip()
    return GuardedResponse(text=text, record={
        "version": "u09.1", "changed": text != candidate,
        "mode": "controlled" if blocks else "checked",
        "findings": sorted(set(findings)), "facts": facts,
        "candidate_text": candidate,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "model_calls": 0,
    })
