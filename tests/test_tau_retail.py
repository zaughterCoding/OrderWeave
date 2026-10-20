"""T14 官方模拟用户桥与评分适配的离线回归。

官方子进程只在本地生成对话；Fake 处理器代替模型端点，不产生任何网络请求。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evaluation.retail_budget import BudgetExceeded
from evaluation.tau_bridge import (
    BridgeError,
    UserSimulatorBridge,
    to_anthropic_kwargs,
)

ROOT = Path(__file__).resolve().parents[1]
TAU_PY = ROOT / ".venv-tau" / "Scripts" / "python.exe"
TAU_DATA = ROOT / "evaluation" / "data" / "tau_retail" / "upstream_data"
TASKS = TAU_DATA / "tau2" / "domains" / "retail" / "tasks.json"
WORKER = ROOT / "evaluation" / "tau_user_worker.py"

pytestmark = pytest.mark.skipif(
    not TAU_PY.is_file(), reason="缺少独立 .venv-tau 评测环境"
)


def _bridge(tmp_path: Path, handler, task_id: str = "11") -> UserSimulatorBridge:
    return UserSimulatorBridge(
        python_exe=TAU_PY,
        worker_path=WORKER,
        data_dir=TAU_DATA,
        tasks_path=TASKS,
        task_id=task_id,
        llm="deepseek-flash",
        max_tokens=512,
        temperature=0.0,
        seed=300,
        stderr_path=tmp_path / "worker.log",
        timeout_s=60.0,
        llm_handler=handler,
    )


def test_official_user_simulator_role_flip_and_stop_markers(tmp_path):
    calls = []
    replies = [
        "I need to return an item. ###STOP###",
        "Please transfer me. ###TRANSFER###",
        "This is outside what you can do. ###OUT-OF-SCOPE###",
    ]

    async def handler(event):
        calls.append(event["request"])
        index = len(calls) - 1
        return {
            "content": replies[index],
            "usage": {"prompt_tokens": 100 + index, "completion_tokens": 8},
            "raw_data": {"fake": True},
        }

    async def run():
        bridge = _bridge(tmp_path, handler)
        ready = await bridge.start()
        assert ready["task_id"] == "11"
        assert ready["stop_markers"] == [
            "###STOP###",
            "###TRANSFER###",
            "###OUT-OF-SCOPE###",
        ]
        first = await bridge.send_agent_message("Hi! How can I help you today?")
        second = await bridge.send_agent_message("I can check that order.")
        third = await bridge.send_agent_message("I cannot do that here.")
        await bridge.close()
        return bridge, first, second, third

    bridge, first, second, third = asyncio.run(run())

    assert first["is_stop"] and first["stop_kind"] == "STOP"
    assert second["is_stop"] and second["stop_kind"] == "TRANSFER"
    assert third["is_stop"] and third["stop_kind"] == "OUT_OF_SCOPE"

    first_messages = calls[0]["messages"]
    assert first_messages[0]["role"] == "system"
    assert "User Simulation Guidelines" in first_messages[0]["content"]
    assert "Instructions:" in first_messages[0]["content"]
    # 首轮问候以 user 角色进入模拟用户，符合官方角色翻转。
    assert first_messages[1] == {
        "role": "user",
        "content": "Hi! How can I help you today?",
    }
    # 第二轮：模拟用户上一句翻成 assistant，Agent 回复为 user。
    second_messages = calls[1]["messages"]
    assert [item["role"] for item in second_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_messages[2]["content"] == replies[0]
    assert second_messages[3]["content"] == "I can check that order."

    assert calls[0]["seed"] == 300
    assert calls[0]["num_retries"] == 0
    assert all(event["type"] in {"ready", "agent_text", "llm_exchange", "user"}
               for event in bridge.protocol_log)


def test_bridge_does_not_leak_gold_into_user_prompt(tmp_path):
    captured = {}

    async def handler(event):
        captured.update(event["request"])
        return {
            "content": "Hello. ###STOP###",
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }

    async def run():
        bridge = _bridge(tmp_path, handler)
        await bridge.start()
        await bridge.send_agent_message("Hi! How can I help you today?")
        await bridge.close()

    asyncio.run(run())
    prompt = "\n".join(item["content"] for item in captured["messages"])
    # 任务 11 的 gold 订单、商品与支付方式不得进入模拟用户提示。
    assert "#W5490111" not in prompt
    assert "#W7387996" not in prompt
    assert "credit_card_3124723" not in prompt
    assert "paypal_9497703" not in prompt


def test_bridge_model_failure_is_reported(tmp_path):
    async def handler(event):
        raise RuntimeError("fake transport down")

    async def run():
        bridge = _bridge(tmp_path, handler)
        await bridge.start()
        with pytest.raises(BridgeError, match="fake transport down"):
            await bridge.send_agent_message("Hi! How can I help you today?")
        await bridge.close()

    asyncio.run(run())


def test_bridge_budget_rejection_happens_before_send(tmp_path):
    sends = 0

    async def handler(event):
        nonlocal sends
        sends += 1
        raise BudgetExceeded("预算不足，已在请求前停止")

    async def run():
        bridge = _bridge(tmp_path, handler)
        await bridge.start()
        with pytest.raises(BudgetExceeded):
            await bridge.send_agent_message("Hi! How can I help you today?")
        await bridge.close()

    asyncio.run(run())
    assert sends == 1


def test_anthropic_protocol_conversion_rejects_invalid_order():
    request = {
        "model": "deepseek-flash",
        "max_tokens": 128,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "guidelines"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
    }
    with pytest.raises(BridgeError, match="交替"):
        to_anthropic_kwargs(request)

    converted = to_anthropic_kwargs(
        {
            "model": "deepseek-flash",
            "max_tokens": 128,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "system", "content": "scenario"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "thanks"},
            ],
        }
    )
    assert converted["system"] == "rules\n\nscenario"
    assert converted["messages"][0] == {"role": "user", "content": "hello"}
    assert converted["max_tokens"] == 128
    assert "seed" not in converted
    assert "num_retries" not in converted
