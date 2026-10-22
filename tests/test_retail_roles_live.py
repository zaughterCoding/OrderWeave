"""新付费运行器的冻结和分支检查，生成全部注入 Fake。"""
import asyncio
import json
from dataclasses import replace

import pytest

from evaluation import run_retail_roles_live as live


def test_preflight_fresh_budget_and_frozen_profile(tmp_path):
    directory = tmp_path / "new"
    live.preflight(directory)
    row = json.loads((directory / "preflight.json").read_text(encoding="utf-8"))
    assert row["ledger"]["budget_cny"] == 5 and row["ledger"]["requests"] == 0
    assert row["planned_cny"] < 5
    config = live.config_for(directory, "restart", "langgraph")
    assert (config.embedding_model, config.retrieval_mode, config.reranker_model) == ("qwen", "hybrid", "qwen")
    assert config.agent_mode == "multi" and config.enable_knowledge and config.support_enabled
    assert not config.enable_memory
    with pytest.raises(ValueError, match="覆盖"):
        live.preflight(directory)


def test_changed_frozen_plan_cannot_send_requests(tmp_path, monkeypatch):
    directory = tmp_path / "changed"
    live.preflight(directory)
    changed = live.frozen_plan()
    changed["confirm"] = "不同输入"
    monkeypatch.setattr(live, "frozen_plan", lambda: changed)
    with pytest.raises(RuntimeError, match="冻结"):
        asyncio.run(live.live(directory))
    assert not (directory / "started.json").exists()
    assert live.ledger(directory).summary()["requests"] == 0


@pytest.mark.parametrize("case,engine", [("restart", "langgraph"), ("restart", "legacy"),
    ("modify", "langgraph"), ("human", "legacy"), ("human", "langgraph"), ("cancel", "langgraph")])
def test_fake_driver_real_mcp_and_checkpoint(monkeypatch, tmp_path, case, engine):
    from tests.integration.test_retail_support import (
        SupportFakeAnthropic, SupportFakeMessages, _last_tool_result, _current_user_message,
        _patch_fake, _cleanup, _text, _tool_use,
    )
    class Messages(SupportFakeMessages):
        def _agent_response(self, messages):
            text = _current_user_message(messages)
            if _last_tool_result(messages) is None:
                if text == live.frozen_plan()["modify"]:
                    return _tool_use("prepare_return", {"order_id": live.ORDER, "item_ids": [live.ITEM], "payment_method_id": live.GIFT}, self._next_id())
                if text in {live.frozen_plan()["negative"], live.frozen_plan()["question"]}:
                    return _text("本轮仅说明预计申请金额 258.84，等待后续真实确认。")
            return super()._agent_response(messages)
    fake = SupportFakeAnthropic()
    fake.messages = Messages()
    fake.max_retries = 0
    _patch_fake(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    original = live.config_for
    # 驱动检查复用已缓存轻量检索；正式配置仍由上面的冻结检查约束。
    monkeypatch.setattr(live, "config_for", lambda *args: replace(original(*args), embedding_model="minilm", retrieval_mode="dense", reranker_model="llm"))
    config = live.config_for(tmp_path, case, engine)
    try:
        row = asyncio.run(live.run_trial(tmp_path, live.frozen_plan(), case, engine))
        assert row["status"] == "PASS", row.get("error")
        assert all(row["checks"].values())
        assert live.ledger(tmp_path).summary()["spent_cny"] == 0
        assert (config.db_path.parent / "requests.jsonl").is_file()
        if case == "restart":
            assert len(row["runtime_cycles"]) == 3
    finally:
        _cleanup(config.support_collection, config.knowledge_collection)
