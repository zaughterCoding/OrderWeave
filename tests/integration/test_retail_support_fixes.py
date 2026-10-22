"""真实 API／MCP 的交接修复回归，生成仅用 Fake。"""
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from tests.integration.test_retail_support import EMAIL, ORDER, SupportFakeAnthropic, _cleanup, _configure, _patch_fake
from tests.test_retail_role_workflow import packets


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("engine", ["legacy", "langgraph"])
def test_item_routing_restart_and_current_human_handoff(monkeypatch, tmp_path, mode, engine):
    name = "u02f" + uuid.uuid4().hex[:8]
    support, legacy = _configure(monkeypatch, tmp_path, name, mode)
    monkeypatch.setenv("RETAIL_WORKFLOW_MODE", engine)
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    user, conv = "user-" + name, "conv-" + name
    def chat(client, text):
        response = client.post("/chat", json={"message": text, "user_id": user, "conv_id": conv})
        assert response.status_code == 200, response.text
        payload = response.json()
        trace = client.get("/trace/tool/" + payload["request_id"]).json()["trace"]
        return payload, trace
    try:
        with TestClient(api_main.app) as client:
            prepared, _ = chat(client, f"我要退货。我的邮箱是 {EMAIL}，订单 {ORDER}，蓝色无线耳机（6452271382）一件，退回原支付方式。")
            assert "【退货申请详情（待确认）】" in prepared["response"]
            if mode == "multi":
                assert prepared["primary_agent"] == "billing"
        fake = SupportFakeAnthropic()
        _patch_fake(monkeypatch, fake)
        with TestClient(api_main.app) as client:
            result, trace = chat(client, "这笔退货先不要提交，请转人工")
            runtime = api_main._retail_runtime
            assert not runtime.store.active_pending(user, conv)
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            events = runtime.store.list_events(user, conv)
            assert sum(e["event_type"] == "handoff_requested" for e in events) == 1
            if mode == "multi":
                handoff = next(t for t in trace["tool_traces"] if t["tool_name"] == "transfer_to_human_agents")
                assert "当前阶段：handoff_requested" in handoff["input"]["summary"]
                assert "退货流程：handed_off；等待：无" in handoff["input"]["summary"]
                current_lines = handoff["input"]["summary"].split("上一角色说明", 1)[0]
                assert "等待用户明确确认" not in current_lines
            else:
                packet = list(packets(fake.messages))[-1]["support"]
                assert packet["current_state"]["phase"] == "handoff_requested"
                assert packet["return_flow"]["awaiting"] == ""
            if engine == "langgraph":
                assert trace["workflow"]["resumed"] and trace["workflow"]["next"] == []
                if mode == "multi":
                    assert trace["workflow"]["role_handoff"]["task"]["phase"] == "handoff_requested"
                    config = runtime.service.workflow.config(user, conv)
                    state = client.portal.call(runtime.service.workflow._graph.aget_state, config)
                    assert state.values["handoff"]["task"]["phase"] == "handoff_requested"
            assert runtime.ledger.summary()["spent_cny"] == 0
        log = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log.count("call tool=prepare_return ") == 1
        assert log.count("call tool=transfer_to_human_agents ") == 1
    finally:
        _cleanup(support, legacy)
