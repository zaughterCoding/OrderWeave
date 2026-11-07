"""Web 控制台机制检查：事件父子关系、快照差异与一轮 SSE 主路径；模型为脚本 Fake。"""
import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agents.agent_loop import run_tool_loop
from agents.agent_orchestrator import AgentResponse, AgentType
from agents.retail_agent import RetailService
from agents.tools import AgentToolSpec
from api import web_console
from evaluation.retail_ledger import BudgetedClient, RetailLedger
from mcp_tools.retail_store import RetailStore
from monitor import live_events

EMAIL = "sofia.li7352@example.com"


def test_spans_nest_and_parallel_children_share_parent():
    events = []

    async def child(name):
        with live_events.span("step", name):
            await asyncio.sleep(0)
            live_events.emit("tool", name + "-tool")

    async def scenario():
        with live_events.capture(events.append):
            with live_events.span("node", "root"):
                await asyncio.gather(child("a"), child("b"))

    asyncio.run(scenario())
    root = events[0]["id"]
    starts = {e["name"]: e for e in events if e["type"] == "start"}
    assert starts["a"]["parent"] == root and starts["b"]["parent"] == root
    tools = {e["name"]: e["parent"] for e in events if e["type"] == "point"}
    assert tools == {"a-tool": starts["a"]["id"], "b-tool": starts["b"]["id"]}
    assert events[-1]["type"] == "end" and events[-1]["id"] == root


def test_no_sink_is_noop_and_sink_errors_never_propagate():
    with live_events.span("node", "x") as handle:
        assert handle.id is None

    def broken(_event):
        raise RuntimeError("boom")

    with live_events.capture(broken):
        live_events.emit("tool", "t")
        with live_events.span("node", "y"):
            pass


def test_diff_compares_new_business_rows_with_seed_and_tracks_state_rows():
    seed = {"orders": {"#W1": {"order_id": "#W1", "status": "pending", "items": [{"price": 1}]}}}
    previous = {"business": {"orders": {}}, "state": {"pending_actions": {}}}
    current = {"business": {"orders": {"#W1": {"order_id": "#W1", "status": "cancelled", "items": [{"price": 1}]}}},
               "state": {"pending_actions": {"op1": {"status": "displayed", "updated_at": "t1"}}}}
    changes = web_console.diff_snapshots(previous, current, lambda t, k: seed.get(t, {}).get(k))
    order = next(c for c in changes if c["table"] == "orders")
    assert order["fields"] == [{"path": "status", "before": "pending", "after": "cancelled"}]
    assert {"group": "state", "table": "pending_actions", "key": "op1", "op": "added"} in changes
    later = {**current, "state": {"pending_actions": {"op1": {"status": "consumed", "updated_at": "t2"}}}}
    fields = web_console.diff_snapshots(current, later, lambda t, k: None)[0]["fields"]
    assert fields == [{"path": "status", "before": "displayed", "after": "consumed"}]


def test_graph_spec_matches_compiled_workflow_nodes(tmp_path):
    from agents.retail_workflow import RetailWorkflow

    store = RetailStore(tmp_path / "db.sqlite", _seed())
    workflow = RetailWorkflow(SimpleNamespace(store=store, agent=SimpleNamespace()), tmp_path / "cp.sqlite")

    async def compiled_nodes():
        await workflow.start()
        try:
            return set(workflow._graph.get_graph().nodes)
        finally:
            await workflow.close()

    nodes = asyncio.run(compiled_nodes())
    for mode in ("single", "multi"):
        assert set(web_console.graph_spec(mode)["nodes"]) <= nodes


def _seed():
    from evaluation.harness import config as cfg

    return cfg.SEED_PATH


class _Inner:
    """脚本模型：先调用查找用户工具，再给出文本回复。"""

    def __init__(self):
        self.calls = 0
        self.messages = self

    async def create(self, **_kwargs):
        self.calls += 1
        usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        if self.calls == 1:
            content = [SimpleNamespace(type="tool_use", id="tu1", name="find_user_id_by_email",
                                       input={"email": EMAIL})]
            return SimpleNamespace(content=content, usage=usage, stop_reason="tool_use")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Found your account.")],
                               usage=usage, stop_reason="end_turn")


class _Agent:
    agent_type = AgentType.RETAIL
    language = "en"
    max_tool_rounds = 3
    _model = "fake"
    profile = SimpleNamespace(max_tokens=256, temperature=0)

    def __init__(self, store, client):
        self.store, self._client = store, client

    @staticmethod
    def _knowledge_hit_count(_name, _value):
        return 0

    async def handle(self, req):
        tool = AgentToolSpec("find_user_id_by_email", "find user", {
            "type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]},
            lambda r, a: self.store.find_user_by_email(r.user_id, r.conv_id, a["email"]))
        out = await run_tool_loop(self, req, [{"role": "user", "content": req.message}],
                                  {tool.name: tool}, [], system="policy", stop_check=lambda r, e: False)
        return AgentResponse(agent_type=AgentType.RETAIL, content=out.content, success=True, latency_ms=0,
                             tools_used=out.tools_used, tool_traces=out.tool_traces)


def test_turn_streams_nodes_model_tool_and_db_change(tmp_path):
    async def factory(folder, mode, language):
        ledger = RetailLedger(path=folder / "ledger.sqlite", budget_cny=1.0, input_price_cny_per_mtok=2.0,
                              output_price_cny_per_mtok=8.0, run_id="test", model="fake")
        store = RetailStore(folder / "retail.sqlite", _seed(), language=language)
        agent = _Agent(store, BudgetedClient(_Inner(), ledger, "agent"))
        service = RetailService(store, agent, workflow_mode="langgraph", language=language,
                                checkpoint_path=folder / "cp.sqlite")

        async def close():
            await service.workflow.close()

        return SimpleNamespace(service=service, ledger=ledger, close=close)

    manager = web_console.SessionManager(tmp_path / "sessions", runtime_factory=factory)
    with TestClient(web_console.create_app(manager)) as client:
        created = client.post("/api/web/sessions", json={"mode": "single", "language": "en"}).json()
        assert created["snapshot"]["bound_user"] is None
        with client.stream("POST", f"/api/web/sessions/{created['id']}/turns",
                           json={"message": f"My email is {EMAIL}"}) as response:
            events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
        kinds = [(e["type"], e.get("kind"), e.get("name")) for e in events]
        assert kinds[0][0] == "turn_start" and kinds[-1][0] == "turn_end"
        assert ("start", "node", "prepare_state") in kinds and ("end", "node", "update_result") in kinds
        assert ("start", "model", "agent") in kinds and ("point", "tool", "find_user_id_by_email") in kinds
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs)
        db = next(e for e in events if e["type"] == "db")
        assert db["trigger"]["tool"] == "find_user_id_by_email"
        assert db["snapshot"]["bound_user"] == "sofia_li_9219"
        assert "#W4689314" in db["snapshot"]["business"]["orders"]
        assert events[-1]["response"] == "Found your account."
        assert events[-1]["cost_cny"] > 0
        # 回放文件与实时事件一致，可离线重放。
        replay = client.get(f"/api/web/replays/recorded/{created['id']}").text.splitlines()
        assert json.loads(replay[0])["type"] == "session" and len(replay) == len(events) + 1
        assert client.get(f"/api/web/sessions/{created['id']}/snapshots").json()["turns"] == [0, 1]
