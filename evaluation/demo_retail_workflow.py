"""用 Fake 模型和独立沙箱库展示工作图；两轮分别在新进程执行。"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from agents.retail_agent import RetailAgent, RetailService
from mcp.retail_store import RetailStore

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evaluation/data/retail_sandbox/db.json"
ORDER, ITEM, PAYMENT = "#W3069600", "5012998807", "credit_card_1565124"


class FakeMessages:
    """固定工具脚本仅验证工程机制，不执行任何网络模型请求。"""

    def __init__(self, step):
        args = {"order_id": ORDER, "item_ids": [ITEM], "payment_method_id": PAYMENT}
        self.script = ([
            ("find_user_id_by_email", {"email": "chen.silva2698@example.com"}),
            ("get_order_details", {"order_id": ORDER}),
            ("prepare_return", args),
        ] if step == "prepare" else [("return_delivered_order_items", args)])
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.script:
            name, args = self.script.pop(0)
            block = {"type": "tool_use", "id": f"fake-{self.calls}", "name": name, "input": args}
        else:
            block = {"type": "text", "text": "请查看本轮业务处理结果。"}
        return SimpleNamespace(content=[block])


async def worker(directory, step, mode, message):
    directory.mkdir(parents=True, exist_ok=True)
    store = RetailStore(directory / "business.sqlite", SEED)
    messages = FakeMessages(step)
    agent = RetailAgent(client=SimpleNamespace(messages=messages), model="fake", store=store,
                        policy_text=(ROOT / "evaluation/data/retail_sandbox/policy.md").read_text(encoding="utf-8"))
    service = RetailService(store, agent, workflow_mode=mode, event_log_path=str(directory / "events.jsonl"))
    try:
        result = await service.handle_turn(message, "demo-client", "demo-conversation")
        checkpoint = None
        if service.workflow:
            snapshot = await service.workflow._graph.aget_state(service.workflow.config("demo-client", "demo-conversation"))
            checkpoint = {"state": snapshot.values, "next": list(snapshot.next),
                          "interrupts": [i.value for task in snapshot.tasks for i in task.interrupts]}
        print(json.dumps({"step": step, "mode": mode, "pid": __import__("os").getpid(),
                          "response": result.response, "workflow": result.workflow, "checkpoint": checkpoint,
                          "model_calls": messages.calls, "tool_calls": len(result.tool_traces),
                          "tools": result.tools_used, "operations": store.stats()["counts"]["return_operations"],
                          "cost_cny": 0}, ensure_ascii=False))
    finally:
        if service.workflow:
            await service.workflow.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--step", choices=["prepare", "resume"])
    parser.add_argument("--mode", choices=["langgraph", "legacy"], default="langgraph")
    parser.add_argument("--message")
    args = parser.parse_args()
    directory = (args.directory or ROOT / "data/demo" / ("u02-" + uuid.uuid4().hex[:10])).resolve()
    if args.step:
        message = args.message or ("请准备退货" if args.step == "prepare" else "确认，就按这些申请退货")
        asyncio.run(worker(directory, args.step, args.mode, message))
        return
    if directory.exists():
        parser.error("完整演示需使用新目录，避免改写已有数据；续接请用 --step resume")
    rows = []
    for mode in ("legacy", "langgraph"):
        for step in ("prepare", "resume"):
            run = subprocess.run([sys.executable, "-m", "evaluation.demo_retail_workflow", "--directory", str(directory / mode),
                                  "--step", step, "--mode", mode], check=True, capture_output=True, text=True, encoding="utf-8")
            row = json.loads(run.stdout)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))
    assert rows[0]["operations"] == rows[2]["operations"] == 0
    assert rows[1]["operations"] == rows[3]["operations"] == 1
    assert rows[2]["checkpoint"]["next"] == ["wait_user"]
    assert rows[3]["workflow"]["resumed"]
    assert len({r["pid"] for r in rows}) == 4
    assert [r["model_calls"] for r in rows[:2]] == [r["model_calls"] for r in rows[2:]]
    assert [r["tool_calls"] for r in rows[:2]] == [r["tool_calls"] for r in rows[2:]]
    (directory / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {directory / 'summary.json'}")


if __name__ == "__main__":
    main()
