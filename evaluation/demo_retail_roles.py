"""Fake 角色图演示：每轮新进程，保留业务库与独立 checkpoint。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from agents.agent_orchestrator import AgentOrchestrator
from agents.retail_agent import RetailService
from agents.retail_multi_agent import (
    RetailBillingAgent, RetailEscalationAgent, RetailGeneralAgent,
    RetailMultiAgent, RetailTechnicalAgent,
)
from agents.retail_tools import retail_tools
from evaluation.demo_retail_workflow import FakeMessages, ROOT, SEED
from mcp.retail_store import RetailStore

MESSAGES = {
    "technical": "technical：请说明商品资料的查询边界",
    "prepare": "我的邮箱 chen.silva2698@example.com，请为订单 #W3069600 准备退货",
    "resume": "确认，就按这些申请退货",
    "human": "转人工 human",
}


class RoleFakeMessages:
    """固定角色与工具脚本，不验证真实语言理解。"""

    def __init__(self, step):
        self.step = step
        self.script = FakeMessages("prepare" if step == "prepare" else "resume")
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs.get("system") or ""
        prompt = system + "\n" + "\n".join(m["content"] for m in kwargs.get("messages", []) if isinstance(m.get("content"), str))
        if "意图分析专家" in prompt or "intent analyst" in prompt:
            intent = {"technical": "technical", "human": "human_handoff"}.get(self.step, "refund")
            text = json.dumps({"intent": intent, "confidence": 0.95, "reasoning": "固定工程脚本"})
        elif "零售商品资料 Agent" in system:
            text = "商品资料角色只提供公开资料；账户及退货需账单角色核验。"
        elif "Response Composer" in system:
            text = "固定工程合成回复。"
        else:
            return await self.script.create(**kwargs)
        return SimpleNamespace(content=[{"type": "text", "text": text}])


def build(directory, step, mode="langgraph"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    store = RetailStore(directory / "business.sqlite", SEED)
    messages = RoleFakeMessages(step)
    orchestrator = AgentOrchestrator(
        client=SimpleNamespace(messages=messages), model="fake", retail_mode=True,
        policy_text=(ROOT / "evaluation/data/retail_sandbox/policy.md").read_text(encoding="utf-8"),
        retail_role_classes={"general": RetailGeneralAgent, "technical": RetailTechnicalAgent,
                             "billing": RetailBillingAgent, "escalation": RetailEscalationAgent},
    )
    orchestrator.set_shared_tools(retail_tools(store))
    service = RetailService(store, RetailMultiAgent(orchestrator), workflow_mode=mode,
                            event_log_path=str(directory / "events.jsonl"),
                            ledger_summary=lambda: {"requests": len(messages.calls)})
    return store, messages, service


async def worker(directory, step, mode):
    store, messages, service = build(directory, step, mode)
    try:
        result = await service.handle_turn(MESSAGES[step], "demo-client", "demo-conversation")
        checkpoint = None
        if service.workflow:
            state = await service.workflow._graph.aget_state(service.workflow.config("demo-client", "demo-conversation"))
            checkpoint = {"state": state.values, "next": list(state.next),
                          "interrupts": [i.value for task in state.tasks for i in task.interrupts]}
        row = {"step": step, "mode": mode, "pid": os.getpid(), "response": result.response,
               "workflow": result.workflow, "checkpoint": checkpoint,
               "model_calls": len(messages.calls), "tool_calls": len(result.tool_traces),
               "tools": result.tools_used, "operations": store.stats()["counts"]["return_operations"],
               "cost_cny": 0}
        print(json.dumps(row, ensure_ascii=False))
    finally:
        if service.workflow:
            await service.workflow.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--step", choices=MESSAGES)
    parser.add_argument("--mode", choices=["langgraph", "legacy"], default="langgraph")
    args = parser.parse_args()
    directory = (args.directory or ROOT / "data/demo" / ("u02-roles-" + uuid.uuid4().hex[:10])).resolve()
    if args.step:
        asyncio.run(worker(directory, args.step, args.mode))
        return
    if directory.exists():
        parser.error("完整演示必须使用新目录；已有会话续接使用 --step")
    rows = []
    for mode in ("legacy", "langgraph"):
        for step in MESSAGES:
            run = subprocess.run([sys.executable, "-m", "evaluation.demo_retail_roles", "--directory", str(directory / mode),
                                  "--step", step, "--mode", mode], check=True, capture_output=True, text=True, encoding="utf-8")
            row = json.loads(run.stdout)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))
    for old, graph in zip(rows[:4], rows[4:]):
        assert (old["model_calls"], old["tool_calls"], old["operations"]) == (graph["model_calls"], graph["tool_calls"], graph["operations"])
    assert rows[5]["checkpoint"]["next"] == ["wait_user"]
    assert rows[6]["workflow"]["resumed"]
    assert all(role in rows[index]["workflow"]["node_path"] for index, role in [(4, "technical"), (5, "billing"), (7, "escalation")])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
