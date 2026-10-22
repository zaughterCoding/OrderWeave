"""实际 API／MCP／支持记录的角色图交接，生成全部使用 Fake。"""
import uuid

from fastapi.testclient import TestClient

import api.main as api_main
from tests.integration.test_retail_support import (
    EMAIL, ORDER, SupportFakeAnthropic, _cleanup, _configure, _patch_fake,
)
from tests.test_retail_role_workflow import packets


def test_api_role_handoff_survives_restart_and_reaches_human_record(monkeypatch, tmp_path):
    name = "u02r" + uuid.uuid4().hex[:8]
    support, legacy = _configure(monkeypatch, tmp_path, name, "multi")
    monkeypatch.setenv("RETAIL_WORKFLOW_MODE", "langgraph")
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    user, conv = "client-" + name, "conv-" + name
    def chat(client, text):
        response = client.post("/chat", json={"message": text, "user_id": user, "conv_id": conv})
        assert response.status_code == 200, response.text
        value = response.json()
        trace = client.get("/trace/tool/" + value["request_id"]).json()["trace"]
        return value, trace["workflow"], trace
    try:
        with TestClient(api_main.app) as client:
            _, workflow, _ = chat(client, "我的无线耳机蓝牙搜不到，我已经重新配对过了，还有什么办法？")
            assert "technical" in workflow["node_path"] and "run_agent" not in workflow["node_path"]
        fake = SupportFakeAnthropic()
        _patch_fake(monkeypatch, fake)
        with TestClient(api_main.app) as client:
            value, workflow, _ = chat(client, f"还是不行，我要退货。我的邮箱是 {EMAIL}，订单 {ORDER}，退无线耳机。")
            assert "【退货申请详情（待确认）】" in value["response"]
            assert workflow["next"] == ["wait_user"]
            packet = next(p["role_handoff"] for p in packets(fake.messages) if p["agent_type"] == "billing")
            assert packet["from_role"] == "technical" and packet["source_turn"] == 1
            assert packet["support_refs"]["source"] == "host_support_record"
            assert "PAIR-03" in packet["support_refs"]["user_reported_steps"]
            assert packet["previous_role"]["tool_refs"][0]["tool"] == "search_knowledge_base"
            _, workflow, trace = chat(client, "转人工")
            assert workflow["resumed"] and "leave_return" in workflow["node_path"]
            assert "escalation" in workflow["node_path"] and workflow["next"] == []
            handoff = next(t for t in trace["tool_traces"] if t["tool_name"] == "transfer_to_human_agents")
            assert "billing→escalation" in handoff["input"]["summary"]
            assert api_main._retail_runtime.store.stats()["counts"]["return_operations"] == 0
            assert api_main._retail_runtime.ledger.summary()["spent_cny"] == 0
        log = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log.count("call tool=prepare_return ") == 1
        assert log.count("call tool=transfer_to_human_agents ") == 1
    finally:
        _cleanup(support, legacy)
