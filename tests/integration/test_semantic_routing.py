"""选择性路由开启后的最小业务闭环；真实MCP／存储，模型全部Fake。"""
import uuid

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from core import retail_semantic_router as semantic_module
from tests.integration.test_retail_support import (
    EMAIL, ORDER, _cleanup, _configure, _patch_fake, _return_flow_anthropic,
)
from tests.test_semantic_routing import FakeEncoder, fake_model


@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_support_return_info_restart_and_exits_keep_transaction_guards(monkeypatch, tmp_path, engine):
    name = "u03" + uuid.uuid4().hex[:8]
    support, legacy = _configure(monkeypatch, tmp_path, name, "multi")
    monkeypatch.setenv("RETAIL_WORKFLOW_MODE", engine)
    monkeypatch.setenv("RETAIL_SEMANTIC_ROUTING", "true")
    monkeypatch.setenv("RETAIL_SEMANTIC_MODEL_PATH", str(fake_model(tmp_path)))
    monkeypatch.setattr(semantic_module, "QwenEmbeddingFunction", lambda **kwargs: FakeEncoder())
    fake = _return_flow_anthropic()
    _patch_fake(monkeypatch, fake)
    user = "u-" + name

    def chat(client, conv, message):
        response = client.post("/chat", json={"message": message, "user_id": user, "conv_id": conv})
        assert response.status_code == 200, response.text
        payload = response.json()
        trace = client.get("/trace/tool/" + payload["request_id"]).json()["trace"]
        return payload, trace

    def prepare(client, conv):
        chosen, trace = chat(client, conv, f"还是不行，我要退货。我的邮箱是 {EMAIL}，订单 {ORDER}，退无线耳机。")
        assert "退款去向" in chosen["response"]
        assert trace["semantic_routing"]["path"] == "rule"
        prepared, trace = chat(client, conv, "原支付方式")
        assert "【退货申请详情（待确认）】" in prepared["response"]
        assert trace["semantic_routing"]["encoder_calls"] == 0
        assert trace["primary_agent"] == "billing"
        if engine == "langgraph":
            assert trace["workflow"]["next"] == ["wait_user"]

    try:
        with TestClient(api_main.app) as client:
            for text in ("取消退货", "转人工"):
                conv = name + text
                prepare(client, conv)
                _, trace = chat(client, conv, text)
                runtime = api_main._retail_runtime
                assert not runtime.store.active_pending(user, conv)
                assert runtime.store.stats()["counts"]["return_operations"] == 0
                if engine == "langgraph":
                    assert trace["workflow"]["next"] == []
            conv = name + "return"
            _, trace = chat(client, conv, "我的无线耳机蓝牙搜不到，我已经重新配对过了，还有什么办法？")
            assert trace["primary_agent"] == "technical" and trace["semantic_routing"]["path"] == "rule"
            prepare(client, conv)
            assert api_main._retail_runtime.store.stats()["counts"]["return_operations"] == 0
        _patch_fake(monkeypatch, _return_flow_anthropic())
        with TestClient(api_main.app) as client:
            result, trace = chat(client, conv, "确认，就按这些申请退货")
            assert "return requested" in result["response"]
            assert trace["semantic_routing"]["reason"] == "pending"
            if engine == "langgraph":
                assert trace["workflow"]["resumed"]
            assert api_main._retail_runtime.store.stats()["counts"]["return_operations"] == 1
            chat(client, conv, "确认，就按这些申请退货")
            assert api_main._retail_runtime.store.stats()["counts"]["return_operations"] == 1
            assert api_main._retail_runtime.ledger.summary()["spent_cny"] == 0
    finally:
        _cleanup(support, legacy)
