"""实际 API／MCP 共享入口的图等待与 legacy 回退。"""
import json
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from tests.integration.test_retail_support import (
    EMAIL, ORDER, SupportFakeAnthropic, _cleanup, _configure, _patch_fake,
)


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_api_restart_from_displayed_details(monkeypatch, tmp_path, mode, engine):
    name = "u02" + uuid.uuid4().hex[:8]
    support, legacy = _configure(monkeypatch, tmp_path, name, mode)
    if engine == "legacy":
        monkeypatch.setenv("RETAIL_WORKFLOW_MODE", "legacy")
    else:
        monkeypatch.delenv("RETAIL_WORKFLOW_MODE", raising=False)
    _patch_fake(monkeypatch, SupportFakeAnthropic())
    user, conv = "client-" + name, "conv-" + name

    def chat(client, message):
        response = client.post("/chat", json={"message": message, "user_id": user, "conv_id": conv})
        assert response.status_code == 200, response.text
        payload = response.json()
        trace = client.get("/trace/tool/" + payload["request_id"]).json()["trace"]
        return payload, trace

    try:
        with TestClient(api_main.app) as client:
            runtime = api_main._retail_runtime
            assert runtime.config.workflow_mode == engine
            prepared, trace = chat(client, f"我的邮箱是 {EMAIL}，订单 {ORDER}，申请退货，退无线耳机。")
            assert "【退货申请详情（待确认）】" in prepared["response"]
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if engine == "langgraph":
                assert trace["workflow"]["next"] == ["wait_user"]
                assert trace["workflow"]["model_calls"] > 0
            else:
                assert trace["workflow"] == {}
        _patch_fake(monkeypatch, SupportFakeAnthropic())
        with TestClient(api_main.app) as client:
            confirmed, trace = chat(client, "确认，就按这些申请退货")
            assert "return requested" in confirmed["response"]
            runtime = api_main._retail_runtime
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            assert runtime.ledger.summary()["spent_cny"] == 0
            if engine == "langgraph":
                assert trace["workflow"]["resumed"]
                assert trace["workflow"]["next"] == []
                assert trace["workflow"]["tool_calls"] == 1
        rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 2
        log = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log.count("call tool=prepare_return ") == 1
        assert log.count("call tool=return_delivered_order_items ") == 1
    finally:
        _cleanup(support, legacy)
