"""默认零售装配及入口检查；生成使用 Fake，不启用检索或记忆服务。"""
import asyncio
from dataclasses import asdict

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import agents.retail_runtime as runtime_module
from agents.agent_orchestrator import AgentType
from agents.retail_agent import RetailAgent
from agents.retail_multi_agent import RetailMultiAgent
from tests.integration.test_retail_support import (
    SupportFakeAnthropic, _configure, _patch_fake,
)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, "defaults", "multi")
    for key in ("RETAIL_AGENT_MODE", "RETAIL_WORKFLOW_MODE"):
        monkeypatch.delenv(key, raising=False)
    for key in ("RETAIL_SUPPORT_ENABLED", "RETAIL_ENABLE_KNOWLEDGE", "RETAIL_ENABLE_MEMORY"):
        monkeypatch.setenv(key, "false")
    _patch_fake(monkeypatch, SupportFakeAnthropic())


@pytest.mark.parametrize("mode", [None, "", "single", "multi"])
@pytest.mark.parametrize("engine", [None, "legacy"])
def test_runtime_defaults_and_explicit_fallbacks(configured, monkeypatch, mode, engine):
    if mode is not None:
        monkeypatch.setenv("RETAIL_AGENT_MODE", mode)
    if engine is not None:
        monkeypatch.setenv("RETAIL_WORKFLOW_MODE", engine)
    runtime = runtime_module.build_retail_runtime()
    try:
        expected = mode or "multi"
        assert runtime.config.agent_mode == expected
        assert runtime.config.workflow_mode == (engine or "langgraph")
        assert isinstance(runtime.agent, RetailMultiAgent if expected == "multi" else RetailAgent)
        assert runtime.service.summary()["agent_mode"] == expected
        assert (runtime.service.workflow is not None) == (engine is None)
        if expected == "multi":
            assert set(runtime.agent.orchestrator._pool) == {
                AgentType.GENERAL, AgentType.TECHNICAL, AgentType.BILLING, AgentType.ESCALATION,
            }
    finally:
        asyncio.run(runtime.close())


def test_config_constructor_defaults_and_retail_entry_gate(configured, monkeypatch):
    values = asdict(runtime_module.RetailRuntimeConfig.from_env())
    for key in ("agent_mode", "workflow_mode"):
        values.pop(key)
    config = runtime_module.RetailRuntimeConfig(**values)
    assert (config.agent_mode, config.workflow_mode) == ("multi", "langgraph")
    monkeypatch.delenv("RETAIL_MODE", raising=False)
    assert not runtime_module.retail_mode_enabled() and not api_main._retail_mode()
    monkeypatch.setenv("RETAIL_MODE", "true")
    assert runtime_module.retail_mode_enabled() and api_main._retail_mode()


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_api_cli_build_default_multi(configured, monkeypatch, capsys, entry):
    built = []
    build = runtime_module.build_retail_runtime

    def capture():
        runtime = build()
        built.append(runtime)
        return runtime

    monkeypatch.setattr(runtime_module, "build_retail_runtime", capture)
    if entry == "api":
        with TestClient(api_main.app) as client:
            assert client.get("/health").json()["agent_mode"] == "multi"
    else:
        monkeypatch.setattr("builtins.input", lambda prompt="": "退出")
        asyncio.run(api_main._cli())
        assert "Agent 模式 multi" in capsys.readouterr().out
    assert len(built) == 1
    assert isinstance(built[0].agent, RetailMultiAgent)
    assert built[0].config.workflow_mode == "langgraph"
    assert built[0]._closed
