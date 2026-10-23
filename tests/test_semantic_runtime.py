"""开启语义开关后的真实API／CLI与工作流入口；编码和生成均为Fake。"""
import asyncio

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import agents.retail_runtime as runtime_module
from core import retail_semantic_router as semantic_module
from tests.integration.test_retail_support import SupportFakeAnthropic, _configure, _patch_fake
from tests.test_semantic_routing import FakeEncoder, fake_model


@pytest.mark.parametrize("entry", ["api", "cli"])
@pytest.mark.parametrize("engine", ["langgraph", "legacy"])
def test_optin_api_cli_and_legacy_share_selective_boundary(monkeypatch, tmp_path, capsys, entry, engine):
    _configure(monkeypatch, tmp_path, "u03entry", "multi")
    for key in ("RETAIL_ENABLE_MEMORY", "RETAIL_ENABLE_KNOWLEDGE", "RETAIL_SUPPORT_ENABLED"):
        monkeypatch.setenv(key, "false")
    monkeypatch.setenv("RETAIL_SEMANTIC_ROUTING", "true")
    monkeypatch.setenv("RETAIL_SEMANTIC_MODEL_PATH", str(fake_model(tmp_path)))
    monkeypatch.setenv("RETAIL_WORKFLOW_MODE", engine)
    encoder = FakeEncoder()
    monkeypatch.setattr(semantic_module, "QwenEmbeddingFunction", lambda **kwargs: encoder)
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    build = runtime_module.build_retail_runtime
    runtimes = []

    def capture():
        runtime = build()
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(runtime_module, "build_retail_runtime", capture)
    if entry == "api":
        with TestClient(api_main.app) as client:
            value = client.post("/chat", json={"message": "嗯嗯", "user_id": "u", "conv_id": "c"}).json()
            trace = client.get("/trace/tool/" + value["request_id"]).json()["trace"]
            assert trace["semantic_routing"]["path"] == "local"
            assert value["primary_agent"] == "general"
    else:
        inputs = iter(["嗯嗯", "退出"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
        asyncio.run(api_main._cli())
        assert "Agent 模式 multi" in capsys.readouterr().out
        trace = runtimes[0].service.get_recent_tool_traces()[0]
        assert trace["semantic_routing"]["path"] == "local"
    assert len(runtimes) == 1 and runtimes[0]._closed and encoder.closed
    assert not any("意图分析专家" in str(call) for call in fake.messages.calls)


def test_default_enabled_and_single_does_not_build_classifier(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "u03off", "single")
    for key in ("RETAIL_ENABLE_MEMORY", "RETAIL_ENABLE_KNOWLEDGE", "RETAIL_SUPPORT_ENABLED"):
        monkeypatch.setenv(key, "false")
    monkeypatch.delenv("RETAIL_SEMANTIC_ROUTING", raising=False)
    assert runtime_module.RetailRuntimeConfig.from_env().semantic_routing
    assert runtime_module.RetailRuntimeConfig.__dataclass_fields__["semantic_routing"].default
    monkeypatch.setenv("RETAIL_SEMANTIC_ROUTING", "true")
    _patch_fake(monkeypatch, SupportFakeAnthropic())
    runtime = runtime_module.build_retail_runtime()
    try:
        assert not hasattr(runtime.agent, "orchestrator")
    finally:
        asyncio.run(runtime.close())


def test_explicit_disable_preserves_original_multi_path(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "u03disabled", "multi")
    for key in ("RETAIL_ENABLE_MEMORY", "RETAIL_ENABLE_KNOWLEDGE", "RETAIL_SUPPORT_ENABLED"):
        monkeypatch.setenv(key, "false")
    monkeypatch.setenv("RETAIL_SEMANTIC_ROUTING", "false")
    _patch_fake(monkeypatch, SupportFakeAnthropic())
    runtime = runtime_module.build_retail_runtime()
    try:
        assert runtime.agent.orchestrator._retail_semantic_router is None
    finally:
        asyncio.run(runtime.close())
