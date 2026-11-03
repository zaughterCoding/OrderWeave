"""待确认动作的确认判定：由用户本轮真实消息得出结构化结论。

判定在 Agent 运行前完成并写入业务库；执行时业务库只认同一轮、同一操作的判定，
模型的工具参数或回复文字无法替代。分类失败时不写判定，由业务库回退到规则判断。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

_SYSTEM = """You decide whether a retail customer's latest message explicitly approves carrying out pending actions exactly as they were shown.

Approve an action only when the latest message clearly tells the agent to go ahead now with those shown details (for example "yes", "yes please proceed", "confirm", "go ahead with the cancellation").
Do not approve when the message asks a question, adds a condition, requests any change to the details, hesitates or asks to wait, refers to a different order or action, or only acknowledges information.
When several actions are pending, approve only the ones the message clearly covers; a plain "yes" approves all of them only if the previous agent message asked to confirm all of them together.
Words that merely name the action (cancel, change, exchange, modify, return) inside an approval are part of the approval, not a refusal.

Return only JSON: {"decisions": [{"op_id": "...", "approved": true or false, "reason": "short reason"}]} with one entry per pending action."""


async def classify_confirmations(client: Any, model: str, message: str, previous_reply: str,
                                 pending: List[Dict[str, Any]], max_tokens: int = 400) -> Optional[List[Dict[str, Any]]]:
    """返回每个待确认动作的判定；调用或解析失败返回 None。"""
    if not pending or client is None:
        return None
    payload = {"pending_actions": [{"op_id": p["op_id"], "details": p.get("display_text", "")} for p in pending],
               "previous_agent_message": str(previous_reply or "")[-2000:],
               "latest_customer_message": str(message or "")[:2000]}
    try:
        response = await client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0.0, system=_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
        if getattr(response, "stop_reason", "end_turn") != "end_turn":
            return None
        text = extract_text_content(response.content)
        match = re.search(r"\{.*\}", text, re.S)
        decisions = json.loads(match.group(0) if match else text)["decisions"]
    except Exception as ex:
        from agents.agent_orchestrator import _stop_exception

        if _stop_exception(ex):
            raise
        logger.warning("确认判定失败，回退规则判断: %s", ex)
        return None
    known = {p["op_id"] for p in pending}
    result = []
    for item in decisions if isinstance(decisions, list) else []:
        if isinstance(item, dict) and item.get("op_id") in known:
            result.append({"op_id": item["op_id"], "approved": item.get("approved") is True,
                           "reason": str(item.get("reason") or "")[:300]})
    # 漏判的动作按未确认处理，不能默认批准。
    covered = {r["op_id"] for r in result}
    result += [{"op_id": op, "approved": False, "reason": "not covered by the classifier"} for op in known - covered]
    return result
