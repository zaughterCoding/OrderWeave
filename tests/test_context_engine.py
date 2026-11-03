"""上下文编排与统一工具循环的不变量：前缀只追加、压缩持久稳定、钩子拒绝可追踪。"""
import asyncio
import json
from types import SimpleNamespace

from agents.agent_loop import LoopHook, run_tool_loop
import pytest

from agents.tools import AgentToolSpec
from core.context_engine import ContextEngine
from evaluation.harness import config as cfg
from mcp_tools.retail_store import RetailStore

needs_seed = pytest.mark.skipif(not cfg.SEED_PATH.exists(), reason="缺少 τ² retail 公开种子")


def _store(tmp_path):
    return RetailStore(tmp_path / "s.db", cfg.SEED_PATH, language="en")


def _exchange(call_id, name, payload):
    return {"agent": "general",
            "assistant": [{"type": "tool_use", "id": call_id, "name": name, "input": {"order_id": "#W1"}}],
            "results": [{"type": "tool_result", "tool_use_id": call_id, "content": json.dumps(payload)}]}


def _turn(engine, store, turn, message, exchanges, reply, tools=("get_order_details",)):
    current = engine.user_blocks(turn, {"turn": turn}, message)
    view = engine.view("u", "c", current, tools)
    store.save_turn_messages("u", "c", turn, engine.log_rows(current, exchanges, reply))
    return view


@needs_seed
def test_history_is_append_only_prefix(tmp_path):
    store = _store(tmp_path)
    engine = ContextEngine(store)
    first = _turn(engine, store, 1, "Where is order #W1?", [_exchange("t1", "get_order_details", {"status": "pending"})],
                  "It is pending.")
    second = _turn(engine, store, 2, "Cancel it.", [], "Here are the details.")
    third = _turn(engine, store, 3, "Yes.", [], "Done.")
    # 上一轮的完整输入是下一轮输入的前缀：缓存可以命中全部历史。
    assert second[:len(first)] == first
    assert third[:len(second)] == second
    roles = [m["role"] for m in third]
    assert all(a != b for a, b in zip(roles, roles[1:]))
    assert any(b.get("type") == "tool_use" for m in third for b in m["content"])


@needs_seed
def test_missing_tool_renders_as_text(tmp_path):
    store = _store(tmp_path)
    engine = ContextEngine(store)
    _turn(engine, store, 1, "hi", [_exchange("t1", "cancel_pending_order", {"ok": True})], "ok")
    view = engine.view("u", "c", engine.user_blocks(2, {}, "next"), {"get_order_details"})
    assert not any(b.get("type") in {"tool_use", "tool_result"} for m in view for b in m["content"])
    assert "cancel_pending_order" in json.dumps(view)


@needs_seed
def test_compaction_is_persisted_and_stable(tmp_path):
    store = _store(tmp_path)
    summaries = []

    def summarize(text, upto):
        summaries.append(upto)
        return f"user verified; order #W1 discussed (turns 1-{upto})"

    engine = ContextEngine(store, budget_tokens=2000, keep_recent_turns=1, summarizer=summarize)
    big = {"blob": "x" * 9000}
    for turn in range(1, 6):
        asyncio.run(engine.prepare("u", "c", turn))
        _turn(engine, store, turn, f"message {turn}", [_exchange(f"t{turn}", "get_order_details", big)], f"reply {turn}")
    state = store.context_state("u", "c")
    assert state["elided_upto_turn"] >= 3 and state["compactions"]
    first = engine.view("u", "c", engine.user_blocks(6, {}, "again"), ("get_order_details",))
    second = engine.view("u", "c", engine.user_blocks(6, {}, "again"), ("get_order_details",))
    assert first == second
    text = json.dumps(first)
    # 先省略旧工具结果，仍超预算再摘要；被摘要覆盖的轮次不再原文出现，保留轮次保持原文。
    assert summaries and "Summary of turns 1-3" in text and "message 1" not in text
    assert text.count("x" * 9000) == 2


class _Block(SimpleNamespace):
    pass


class _FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=self.responses.pop(0), stop_reason="end_turn",
                               usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=90,
                                                     cache_creation_input_tokens=0))


class _Deny(LoopHook):
    name = "deny_writes"

    def pre_tool(self, ctx, name, args):
        return "blocked" if name == "write" else None


def test_loop_hooks_block_and_trace():
    calls = []
    spec = AgentToolSpec(name="write", description="w", input_schema={"type": "object", "properties": {}},
                         handler=lambda req, args: calls.append(args) or {"success": True})
    client = _FakeClient([
        [_Block(type="tool_use", id="a", name="write", input={})],
        [_Block(type="text", text="I could not do that.")],
    ])
    agent = SimpleNamespace(agent_type=SimpleNamespace(value="general"), language="en", _client=client,
                            _model="m", profile=SimpleNamespace(max_tokens=100, temperature=0),
                            max_tool_rounds=3, capture_tool_results=True,
                            _knowledge_hit_count=lambda name, value: 0)
    req = SimpleNamespace(retail_guard={}, transcript=[], spans=[])
    out = asyncio.run(run_tool_loop(agent, req, [{"role": "user", "content": "go"}], {"write": spec}, [_Deny()],
                                    system="s", stop_check=lambda r, e: False))
    assert calls == [] and out.content == "I could not do that."
    assert out.tool_traces[0]["blocked_by"] == "deny_writes"
    assert [s["kind"] for s in req.spans] == ["model", "tool", "model"]
    assert req.spans[0]["usage"]["cache_read"] == 90
    assert len(req.transcript) == 1
