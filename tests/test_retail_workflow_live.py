"""付费前验证冻结、脚本分支和证据写入；不发送真实模型请求。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentResponse, AgentType
from agents.retail_agent import RetailService
from evaluation import run_retail_workflow_live as live
from mcp.retail_store import RetailStore


def test_preflight_uses_fresh_ledger_and_refuses_overwrite(tmp_path):
    directory = tmp_path / "new-run"
    live.preflight(directory)
    record = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    assert record["ledger"]["requests"] == 0
    assert record["ledger"]["budget_cny"] == 5
    assert record["planned_cny"] < 5
    with pytest.raises(ValueError, match="覆盖"):
        live.preflight(directory)


def test_changed_code_or_plan_cannot_start_paid_run(tmp_path, monkeypatch):
    directory = tmp_path / "changed-run"
    live.preflight(directory)
    changed = live.frozen_plan()
    changed["initial"] = "不同输入"
    monkeypatch.setattr(live, "frozen_plan", lambda: changed)
    with pytest.raises(RuntimeError, match="冻结"):
        asyncio.run(live.live(directory))
    assert not (directory / "started.json").exists()
    assert live.ledger(directory).summary()["requests"] == 0


class ScriptAgent:
    def __init__(self, store):
        self.store = store

    async def handle(self, req):
        pending = self.store.active_pending(req.user_id, req.conv_id)
        if "转人工" in req.message:
            tool = "transfer_to_human_agents"
            result = self.store.record_handoff(req.user_id, req.conv_id, "用户要求人工", "待确认退货已退出")
        elif req.message.startswith("确认"):
            tool = "return_delivered_order_items"
            item = pending[0]
            result = self.store.submit_return(req.user_id, req.conv_id, live.ORDER,
                item["items"]["item_ids"], item["payment_method_id"], req.message)
        else:
            tool = "prepare_return"
            self.store.find_user_by_email(req.user_id, req.conv_id, "chen.silva2698@example.com")
            modified = "礼品卡" in req.message
            result = self.store.prepare_return(req.user_id, req.conv_id, live.ORDER,
                [live.ITEM, live.TABLET] if modified else [live.ITEM], live.GIFT if modified else live.CARD)
        return AgentResponse(agent_type=AgentType.RETAIL, content="处理结果", success=True,
                             tools_used=[tool], tool_traces=[{"tool_name": tool, "result": result,
                             "result_success": result["success"], "success": True}])


@pytest.mark.parametrize("case", ["restart_confirm", "modify", "cancel", "human"])
def test_driver_exercises_real_checkpointer_and_all_script_paths(tmp_path, monkeypatch, case):
    def build(config):
        store = RetailStore(config.db_path, config.seed_path)
        service = RetailService(store, ScriptAgent(store), workflow_mode="langgraph")
        async def start():
            await service.workflow.start()
        async def close():
            await service.workflow.close()
        runtime = SimpleNamespace(config=config, store=store, service=service,
            ledger=live.ledger(tmp_path), start=start, close=close,
            client=SimpleNamespace(_inner=SimpleNamespace(messages=SimpleNamespace(create=None))))
        return runtime

    monkeypatch.setattr(live, "build_retail_runtime", build)
    monkeypatch.setattr(live, "_apply_runtime_adaptations", lambda *args: None)
    record = asyncio.run(live.run_trial(tmp_path, live.frozen_plan(), case, "single"))
    assert record["status"] == "PASS", record
    assert all(record["checks"].values())
    assert record["turns"][0]["result"]["workflow"]["next"] == ["wait_user"]
    assert (tmp_path / f"{case}-single" / "result.json").is_file()
    if case == "restart_confirm":
        assert len(record["runtime_cycles"]) == 2
        assert record["restart_checkpoint_found"]
