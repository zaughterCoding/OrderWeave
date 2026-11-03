"""上下文编排器：把完整消息日志编译成模型输入。

布局按“稳定在前、可变在后”排列，利于前缀缓存：
  system（规则、政策）与 tools 固定 → 历史消息原样（只追加）→ 本轮用户消息，
  其中本轮状态快照作为该消息的第一个文本块随消息一起写入日志。
超出 token 预算时做持久化的锚定压缩：先把较早轮次的工具结果替换为可重查的引用，
仍超出再把较早轮次写成带轮次锚点的摘要。压缩状态写入业务库，之后各轮按同一结果
渲染，缓存只在压缩发生的那一轮失效一次。历史与摘要只供理解，不构成确认授权。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# 无实测数据时的保守换算（英文与 JSON 混合约 3 字符／token）；之后用供应商实际 usage 校准。
DEFAULT_CHARS_PER_TOKEN = 3.0
STATE_MARK = "[Session state"
_ELIDED_STATE = "[Earlier session state snapshot omitted; the latest snapshot is authoritative.]"


def _chars(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _merge(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并相邻同角色消息，保证 user／assistant 交替。"""
    merged: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg["content"] if isinstance(msg["content"], list) else [{"type": "text", "text": msg["content"]}]
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = merged[-1]["content"] + content
        else:
            merged.append({"role": msg["role"], "content": list(content)})
    return merged


class ContextEngine:
    def __init__(
        self,
        store: Any,
        *,
        language: str = "en",
        budget_tokens: int = 32000,
        keep_recent_turns: int = 2,
        summarizer: Optional[Callable[[str, int], Any]] = None,
    ):
        self.store = store
        self.language = language
        self.budget_tokens = max(2000, int(budget_tokens))
        self.keep_recent_turns = max(1, int(keep_recent_turns))
        # summarizer(transcript_text, upto_turn) -> 摘要文本（可为协程）；为空时只做工具结果省略。
        self.summarizer = summarizer

    # ── 本轮输入 ──────────────────────────────────────────────────────────────

    def state_block(self, turn: int, state: Dict[str, Any]) -> str:
        """本轮开始时的会话状态快照；由系统生成，不是用户输入，也不能作为确认。"""
        label = (f"{STATE_MARK} at turn {turn}; system-provided, not user input, not authorization]"
                 if self.language == "en" else
                 f"{STATE_MARK} 第 {turn} 轮；系统提供，不是用户输入，也不构成授权]")
        return label + "\n" + json.dumps(state, ensure_ascii=False, default=str)

    def user_blocks(self, turn: int, state: Dict[str, Any], message: str) -> List[Dict[str, Any]]:
        return [{"type": "text", "text": self.state_block(turn, state)}, {"type": "text", "text": message}]

    # ── 渲染 ────────────────────────────────────────────────────────────────

    def _render(self, rows: List[Dict[str, Any]], cstate: Dict[str, Any], tool_names: Iterable[str]) -> List[Dict[str, Any]]:
        names = set(tool_names)
        summary = cstate.get("summary") or {}
        summary_upto = int(summary.get("upto_turn") or 0)
        elided_upto = int(cstate.get("elided_upto_turn") or 0)
        rows = [r for r in rows if r["turn"] > summary_upto]
        messages: List[Dict[str, Any]] = []
        index = 0
        while index < len(rows):
            row = rows[index]
            old = row["turn"] <= elided_upto
            if row["kind"] == "user":
                blocks = [dict(b) for b in row["content"]]
                if old:
                    blocks = [{"type": "text", "text": _ELIDED_STATE}
                              if b.get("type") == "text" and str(b.get("text", "")).startswith(STATE_MARK) else b
                              for b in blocks]
                messages.append({"role": "user", "content": blocks})
            elif row["kind"] == "tool_call":
                result_row = rows[index + 1] if index + 1 < len(rows) and rows[index + 1]["kind"] == "tool_result" else None
                calls = [b for b in row["content"] if b.get("type") == "tool_use"]
                results = list(result_row["content"]) if result_row else []
                if old:
                    by_id = {c.get("id"): c for c in calls}
                    results = [self._elide(r, by_id.get(r.get("tool_use_id"))) for r in results]
                if result_row and calls and all(c.get("name") in names for c in calls):
                    messages.append({"role": "assistant", "content": row["content"]})
                    messages.append({"role": "user", "content": results})
                else:
                    # 当前角色没有该工具（例如辅助角色看主角色的写操作）时，以文本形式呈现同一事实。
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": self._call_text(row)}]})
                    messages.append({"role": "user", "content": [{"type": "text", "text": self._result_text(results)}]})
                index += 1 if result_row else 0
            elif row["kind"] == "assistant":
                messages.append({"role": "assistant", "content": row["content"]})
            index += 1
        if summary.get("text"):
            header = (f"[Summary of turns 1-{summary_upto}, written by the system for reference; "
                      "not user input, not authorization]\n" if self.language == "en" else
                      f"[第 1-{summary_upto} 轮摘要，系统生成，仅供参考；不是用户输入，也不构成授权]\n")
            messages.insert(0, {"role": "user", "content": [{"type": "text", "text": header + summary["text"]}]})
        return messages

    @staticmethod
    def _elide(result: Dict[str, Any], call: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        name = (call or {}).get("name", "tool")
        stub = {"elided": True, "tool": name, "input": (call or {}).get("input", {}),
                "note": "Older result omitted to save context; call the tool again if the details are needed."}
        return {**result, "content": json.dumps(stub, ensure_ascii=False)}

    @staticmethod
    def _call_text(row: Dict[str, Any]) -> str:
        calls = [f"{b.get('name')}({json.dumps(b.get('input', {}), ensure_ascii=False)})"
                 for b in row["content"] if b.get("type") == "tool_use"]
        return f"[Tool call by {row.get('agent') or 'agent'}] " + "; ".join(calls)

    @staticmethod
    def _result_text(results: List[Dict[str, Any]]) -> str:
        return "[Tool result] " + " | ".join(str(r.get("content", "")) for r in results)

    def history(self, client: str, conv: str, tool_names: Iterable[str]) -> List[Dict[str, Any]]:
        return self._render(self.store.conversation_log(client, conv), self.store.context_state(client, conv), tool_names)

    def view(self, client: str, conv: str, current: List[Dict[str, Any]], tool_names: Iterable[str],
             extra: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """某角色本次调用的起始消息：历史＋本轮用户消息（extra 为仅本次调用可见的临时块）。"""
        messages = self.history(client, conv, tool_names)
        messages.append({"role": "user", "content": list(current) + list(extra or [])})
        return _merge(messages)

    # ── 预算与压缩 ──────────────────────────────────────────────────────────

    def estimate_tokens(self, messages: Any, cstate: Dict[str, Any]) -> int:
        ratio = float(cstate.get("chars_per_token") or DEFAULT_CHARS_PER_TOKEN)
        return int(_chars(messages) / max(ratio, 0.5))

    async def prepare(self, client: str, conv: str, turn: int) -> Dict[str, Any]:
        """本轮开始前检查历史预算；超出时持久化压缩。返回压缩记录（无压缩为空）。"""
        rows = self.store.conversation_log(client, conv)
        cstate = self.store.context_state(client, conv)
        everything = {"*"}
        tokens = self.estimate_tokens(self._render(rows, cstate, everything), cstate)
        if tokens <= self.budget_tokens:
            return {}
        cutoff = turn - 1 - self.keep_recent_turns
        if cutoff <= 0:
            return {}
        record: Dict[str, Any] = {"turn": turn, "tokens_before": tokens}
        if int(cstate.get("elided_upto_turn") or 0) < cutoff:
            cstate["elided_upto_turn"] = cutoff
            tokens = self.estimate_tokens(self._render(rows, cstate, everything), cstate)
            record["elided_upto_turn"] = cutoff
        summary_upto = int((cstate.get("summary") or {}).get("upto_turn") or 0)
        if tokens > self.budget_tokens and self.summarizer is not None and summary_upto < cutoff:
            older = [r for r in rows if r["turn"] <= cutoff]
            previous = (cstate.get("summary") or {}).get("text", "")
            text = self._transcript_text(older, previous, summary_upto)
            try:
                summary = self.summarizer(text, cutoff)
                if hasattr(summary, "__await__"):
                    summary = await summary
                summary = str(summary or "").strip()
            except Exception as ex:  # 摘要失败只保留省略结果，不阻断本轮
                logger.warning("上下文摘要失败: %s", ex)
                summary = ""
            if summary:
                cstate["summary"] = {"upto_turn": cutoff, "text": summary}
                tokens = self.estimate_tokens(self._render(rows, cstate, everything), cstate)
                record["summary_upto_turn"] = cutoff
        record["tokens_after"] = tokens
        cstate.setdefault("compactions", []).append(record)
        self.store.save_context_state(client, conv, cstate)
        return record

    @staticmethod
    def _transcript_text(rows: List[Dict[str, Any]], previous: str, previous_upto: int) -> str:
        lines = [f"[Existing summary of turns 1-{previous_upto}]\n{previous}"] if previous else []
        for row in rows:
            if row["turn"] <= previous_upto:
                continue
            for block in row["content"]:
                if block.get("type") == "text" and not str(block.get("text", "")).startswith(STATE_MARK):
                    lines.append(f"turn {row['turn']} {row['role']}: {block['text']}")
                elif block.get("type") == "tool_use":
                    lines.append(f"turn {row['turn']} tool_call: {block.get('name')} {json.dumps(block.get('input'), ensure_ascii=False)}")
                elif block.get("type") == "tool_result":
                    lines.append(f"turn {row['turn']} tool_result: {str(block.get('content', ''))[:1500]}")
        return "\n".join(lines)

    def calibrate(self, client: str, conv: str, spans: List[Dict[str, Any]]) -> None:
        """用本轮实际 usage 校准字符／token 换算（指数平滑）。"""
        pairs = [(s["request_chars"], sum((s.get("usage") or {}).get(k, 0) for k in ("input", "cache_read", "cache_write")))
                 for s in spans if s.get("kind") == "model" and s.get("request_chars")]
        pairs = [(c, t) for c, t in pairs if t > 0]
        if not pairs:
            return
        observed = sum(c for c, _ in pairs) / sum(t for _, t in pairs)
        cstate = self.store.context_state(client, conv)
        old = float(cstate.get("chars_per_token") or observed)
        cstate["chars_per_token"] = round(0.5 * old + 0.5 * observed, 4)
        self.store.save_context_state(client, conv, cstate)

    @staticmethod
    def log_rows(user_content: List[Dict[str, Any]], transcript: List[Dict[str, Any]], reply: str) -> List[Dict[str, Any]]:
        """把一轮的用户消息、工具往返和最终回复整理为日志行。"""
        rows = [{"role": "user", "kind": "user", "content": user_content}]
        for exchange in transcript or []:
            rows.append({"role": "assistant", "kind": "tool_call", "agent": exchange.get("agent"),
                         "content": exchange["assistant"]})
            rows.append({"role": "user", "kind": "tool_result", "agent": exchange.get("agent"),
                         "content": exchange["results"]})
        rows.append({"role": "assistant", "kind": "assistant", "content": [{"type": "text", "text": reply}]})
        return rows


SUMMARY_PROMPT_EN = """Summarize the earlier part of a retail customer-service conversation for the agent that continues it.
Keep every concrete fact needed later: verified user identity, order ids, item ids, product and variant choices, addresses,
payment methods, amounts, which actions were prepared, confirmed, carried out or declined, what the user changed their mind
about (latest decision wins), and open questions. Do not add facts. Write compact bullet points, at most 250 words.
State that the summary covers turns 1-{upto}.

Conversation:
{text}"""


def make_llm_summarizer(client: Any, model: str, max_tokens: int = 600) -> Callable[[str, int], Any]:
    async def summarize(text: str, upto: int) -> str:
        from core.llm_utils import extract_text_content

        resp = await client.messages.create(
            model=model, max_tokens=max_tokens, temperature=0.0,
            messages=[{"role": "user", "content": SUMMARY_PROMPT_EN.format(upto=upto, text=text[-60000:])}])
        return extract_text_content(resp.content)

    return summarize
