"""会话内的时序事实账本：模型提议用户陈述的事实，系统校验原文引用后记账。

每轮由模型从用户最新消息中提出增量（新增、取代、撤回、待澄清），确定性代码只校验
不变量：引用必须逐字出现在本轮用户消息中、被取代或撤回的事实必须存在且仍有效、
数量与长度受限。用户改口时旧事实标记为已取代并保留取代关系，状态快照同时给出
当前事实与最近改口，供 Agent 使用最新决定。事实只来自用户陈述，不提供身份或交易授权。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KINDS = ("goal", "order", "item", "variant", "payment", "address", "reason", "preference", "background")
MAX_OPS = 12
MAX_ACTIVE = 40

_SYSTEM = """You maintain a ledger of facts the customer has stated in a retail support conversation.
Given the current ledger, the agent's previous message and the customer's latest message, propose changes caused by the latest message only.

- add: a new fact the customer states or decides (what they want done, which order or item, desired variant or options, payment or refund method, address, reason, preferences, or background details they mention).
- supersede: the customer changes an earlier fact (a different item, method, option, address, decision); give the id of the old fact and the new value.
- retract: the customer withdraws an earlier fact without replacing it.
- unresolved: the customer's statement leaves a needed detail ambiguous (for example two possible items); describe the ambiguity.
Each change needs "quote": an exact, verbatim substring of the customer's latest message that supports it. Do not record the agent's statements, guesses, or facts only found in tools. Merely answering yes to a confirmation adds nothing. If nothing changes, return empty lists.

kind is one of: goal, order, item, variant, payment, address, reason, preference, background.
Return only JSON: {"add": [{"kind": "...", "subject": "short label", "value": "...", "quote": "..."}],
"supersede": [{"id": "...", "value": "...", "quote": "..."}], "retract": [{"id": "...", "quote": "..."}],
"unresolved": [{"kind": "...", "subject": "...", "value": "what is ambiguous", "quote": "..."}]}"""


def _norm(text: str) -> str:
    text = str(text or "").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def quoted(quote: str, message: str) -> bool:
    """引用逐字校验：只规范空白、大小写与引号／连字符的排版差异。"""
    q = _norm(quote)
    return len(q) >= 2 and q in _norm(message)


def _fact_id(turn: int, index: int, value: str) -> str:
    return "F" + hashlib.sha256(f"{turn}:{index}:{value}".encode("utf-8")).hexdigest()[:8]


def apply_ops(facts: List[Dict[str, Any]], ops: Dict[str, Any], message: str, turn: int) -> Dict[str, Any]:
    """在副本上应用增量，返回 {facts, accepted, rejected}；只校验不变量，不做语义判断。"""
    facts = [dict(f) for f in facts]
    by_id = {f["id"]: f for f in facts}
    accepted, rejected = [], []
    items = [("add", op) for op in ops.get("add") or []] + [("unresolved", op) for op in ops.get("unresolved") or []] \
        + [("supersede", op) for op in ops.get("supersede") or []] + [("retract", op) for op in ops.get("retract") or []]
    for index, (action, op) in enumerate(items[:MAX_OPS]):
        reason = ""
        if not isinstance(op, dict):
            reason = "malformed"
        elif not quoted(op.get("quote", ""), message):
            reason = "quote_not_in_message"
        elif action in {"supersede", "retract"} and by_id.get(op.get("id"), {}).get("status") not in {"active", "unresolved"}:
            reason = "unknown_or_inactive_fact"
        elif action in {"add", "unresolved"} and op.get("kind") not in KINDS:
            reason = "bad_kind"
        elif action != "retract" and not (0 < len(str(op.get("value") or "")) <= 240):
            reason = "bad_value"
        if reason:
            rejected.append({"action": action, "op": op, "reason": reason})
            continue
        quote = str(op["quote"])[:240]
        if action == "retract":
            by_id[op["id"]].update(status="retracted", ended_turn=turn, end_quote=quote)
        else:
            old = by_id.get(op.get("id")) if action == "supersede" else None
            fact = {"id": _fact_id(turn, index, str(op["value"])), "turn": turn,
                    "kind": old["kind"] if old else op["kind"],
                    "subject": str((old or op).get("subject") or "")[:80],
                    "value": str(op["value"])[:240], "quote": quote,
                    "status": "unresolved" if action == "unresolved" else "active", "supersedes": old["id"] if old else None}
            if old:
                old.update(status="superseded", ended_turn=turn, superseded_by=fact["id"])
            facts.append(fact)
            by_id[fact["id"]] = fact
        accepted.append({"action": action, "op": op})
    if sum(1 for f in facts if f["status"] in {"active", "unresolved"}) > MAX_ACTIVE:
        # 超出容量时最早的有效事实转为过期，避免状态无界增长。
        for fact in facts:
            if fact["status"] in {"active", "unresolved"}:
                fact.update(status="expired", ended_turn=turn)
                if sum(1 for f in facts if f["status"] in {"active", "unresolved"}) <= MAX_ACTIVE:
                    break
    return {"facts": facts, "accepted": accepted, "rejected": rejected}


def state_view(facts: List[Dict[str, Any]], recent_changes: int = 6) -> Dict[str, Any]:
    """状态快照中的事实视图：当前有效／待澄清事实，以及最近的改口与撤回。"""
    current = [{k: f[k] for k in ("id", "turn", "kind", "subject", "value", "status")}
               for f in facts if f["status"] in {"active", "unresolved"}]
    by_id = {f["id"]: f for f in facts}
    changes = []
    for f in facts:
        if f["status"] == "superseded":
            new = by_id.get(f.get("superseded_by"), {})
            changes.append({"turn": f.get("ended_turn"), "kind": f["kind"], "subject": f["subject"],
                            "old": f["value"], "new": new.get("value")})
        elif f["status"] == "retracted":
            changes.append({"turn": f.get("ended_turn"), "kind": f["kind"], "subject": f["subject"],
                            "old": f["value"], "new": None})
    changes.sort(key=lambda c: c["turn"] or 0)
    return {"note": "Facts the customer stated, each checked against their own words. When they changed their "
                    "mind, follow the newest value; ask about unresolved items. This is not authorization.",
            "current": current, "changed_mind": changes[-recent_changes:]}


async def extract(client: Any, model: str, facts: List[Dict[str, Any]], previous_reply: str, message: str,
                  max_tokens: int = 700) -> Optional[Dict[str, Any]]:
    """调用模型提出增量；失败返回 None（本轮不更新账本，不阻断对话）。"""
    from core.llm_utils import extract_text_content

    ledger = [{k: f[k] for k in ("id", "kind", "subject", "value", "status")}
              for f in facts if f["status"] in {"active", "unresolved"}]
    payload = {"ledger": ledger, "previous_agent_message": str(previous_reply or "")[-1500:],
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
        logger.warning("事实抽取失败，本轮不更新账本: %s", ex)
        return None
