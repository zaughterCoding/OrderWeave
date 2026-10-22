"""U02 小量真实模型兼容验收；固定用例、独立五元账本，无 Judge。"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from agents.retail_runtime import build_retail_runtime
from evaluation.retail_ledger import RetailLedger
from evaluation.run_retail_support_live import (
    MODEL, ENDPOINT, _Recorder, _apply_runtime_adaptations, _trial_config,
)

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 5.0
INPUT_PRICE, OUTPUT_PRICE = 2.0, 8.0
ORDER = "#W3069600"
ITEM, TABLET = "5012998807", "8551474201"
CARD, GIFT = "credit_card_1565124", "gift_card_7250692"
USER, CONV = "u02-live-client", "u02-live-conversation"
PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
SOURCES = [
    "evaluation/run_retail_workflow_live.py", "evaluation/run_retail_support_live.py",
    "evaluation/retail_ledger.py", "evaluation/retail_budget.py",
    "agents/retail_workflow.py", "agents/retail_runtime.py", "agents/retail_agent.py",
    "agents/retail_multi_agent.py", "agents/agent_orchestrator.py", "mcp/retail_store.py",
    "agents/retail_tools.py", "agents/retail_response_guard.py", "core/intent_recognizer.py",
    "mcp/retail_client.py", "mcp/retail_protocol.py", "tests/test_retail_workflow_live.py",
    "evaluation/data/retail_live/cases.json", "evaluation/data/retail_sandbox/db.json",
    "evaluation/data/retail_sandbox/policy.md", "pyproject.toml", "uv.lock",
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def frozen_plan():
    old = json.loads((ROOT / "evaluation/data/retail_live/cases.json").read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in old["cases"]}
    return {
        "model": MODEL, "endpoint": ENDPOINT, "budget_cny": BUDGET,
        "input_cny_per_mtok": INPUT_PRICE, "output_cny_per_mtok": OUTPUT_PRICE,
        "pricing_url": PRICING_URL, "price_basis": "高峰缓存未命中价，保守结算全部输入",
        "initial": cases["V03"]["turns"][0],
        "confirm": cases["V03"]["turns"][2],
        "modify": cases["V03"]["turns"][1],
        "cancel": "取消这笔退货，我不退了。",
        "human": "这笔退货先不要提交，请帮我转人工继续处理。",
        "clarification": "我的邮箱 chen.silva2698@example.com，姓名 Chen Silva，邮编 46281；订单 #W3069600，只退 Makeup Kit（5012998807）一件，退款到原支付方式。请准备并展示详情，本轮不提交。",
        "modify_clarification": "订单 #W3069600，退 Makeup Kit（5012998807）和 Tablet（8551474201）各一件，退款到本人已有礼品卡 gift_card_7250692。请重新准备并展示详情，等我下一轮确认。",
        "order": [[case, mode] for case in ("restart_confirm", "modify", "cancel", "human") for mode in ("single", "multi")],
        "limits": {"max_user_turns": 5, "max_requests_per_trial": 40, "max_requests_per_turn": 32},
        "profile": {"language": "zh", "workflow": "langgraph", "memory": False, "knowledge": False, "support": False},
        "scope": "仅订单退货链；沿用既有允许关闭检索与记忆的验收配置，不改变生产默认，不验收RAG",
        "sources": hashes(),
    }


def ledger(directory):
    return RetailLedger(directory / "ledger.sqlite", BUDGET, INPUT_PRICE, OUTPUT_PRICE,
                        run_id=directory.name, model=MODEL)


def config_for(directory, case, mode):
    trial = directory / f"{case}-{mode}"
    config = _trial_config(run_id=directory.name, case_id=case,
                           arm="A" if mode == "single" else "B", trial_dir=trial,
                           collection="u02_unused", chroma_host="127.0.0.1", chroma_port=8002,
                           chroma_path=str(trial / "unused-chroma"))
    return replace(config, ledger_path=directory / "ledger.sqlite", enable_knowledge=False,
                   support_enabled=False, enable_memory=False, workflow_mode="langgraph")


def check_pending(rows, modified=False):
    wanted = sorted([ITEM, TABLET] if modified else [ITEM])
    payment = GIFT if modified else CARD
    amount = 119763 if modified else 25871
    return len(rows) == 1 and all((rows[0]["status"] == "displayed",
        rows[0]["order_id"] == ORDER, sorted(rows[0]["items"]["item_ids"]) == wanted,
        rows[0]["payment_method_id"] == payment, rows[0]["amount_cents"] == amount))


def check_final(case, snapshot, events, pending):
    order = snapshot["orders"][ORDER]
    submitted = [event for event in events if event["event_type"] == "return_requested"]
    handoffs = [event for event in events if event["event_type"] == "handoff_requested"]
    if case in {"restart_confirm", "modify"}:
        wanted = sorted([ITEM, TABLET] if case == "modify" else [ITEM])
        return (len(submitted) == 1 and not handoffs and not pending and
                order["status"] == "return requested" and sorted(order["return_items"]) == wanted and
                order["return_payment_method_id"] == (GIFT if case == "modify" else CARD))
    return (not submitted and not pending and order["status"] == "delivered" and
            len(handoffs) == (1 if case == "human" else 0))


async def run_trial(directory, plan, case, mode):
    config = config_for(directory, case, mode)
    trial = config.db_path.parent
    trial.mkdir(parents=True, exist_ok=True)
    turns, cycles = [], []
    runtime = None
    first_requests = ledger(directory).summary()["requests"]

    async def start():
        runtime = build_retail_runtime(config)
        recorder = _Recorder(trial, directory.name, case, mode)
        _apply_runtime_adaptations(runtime, runtime.ledger, recorder)
        original = runtime.client._inner.messages.create

        async def checked(**kwargs):
            if kwargs.get("model") != MODEL:
                raise RuntimeError("验收仅允许冻结模型，不允许角色覆盖为其他模型")
            if runtime.ledger.summary()["requests"] - first_requests > plan["limits"]["max_requests_per_trial"]:
                raise RuntimeError("本条验收已达到模型请求上限")
            return await original(**kwargs)
        runtime.client._inner.messages.create = checked
        try:
            await runtime.start()
        except BaseException:
            await runtime.close()
            raise
        cycles.append({"cycle": len(cycles) + 1, "checkpoint": str(runtime.service.workflow.path)})
        return runtime, recorder

    async def turn(message):
        current = runtime.ledger.summary()["requests"] - first_requests
        if current >= plan["limits"]["max_requests_per_trial"] or len(turns) >= plan["limits"]["max_user_turns"]:
            raise RuntimeError("验收达到冻结轮次或请求上限")
        runtime.ledger.begin_turn(f"{case}-{mode}-{len(turns)+1}", max_requests=32)
        recorder.set_turn(len(turns) + 1)
        result = await runtime.service.handle_turn(message, USER, CONV)
        pending = runtime.store.active_pending(USER, CONV)
        payload = asdict(result)
        payload["agent_type"] = result.agent_type.value
        record = {"message": message, "result": payload, "pending": pending}
        turns.append(record)
        write(trial / "turns.json", turns)
        return result, pending

    record = {"case": case, "mode": mode, "status": "RUNNING"}
    try:
        runtime, recorder = await start()
        before = runtime.store.business_snapshot()
        result, pending = await turn(plan["initial"])
        if not pending:
            result, pending = await turn(plan["clarification"])
        if not check_pending(pending) or result.workflow["next"] != ["wait_user"]:
            raise RuntimeError("未准备并展示正确详情或未进入真实 Interrupt")
        if any(event["event_type"] == "return_requested" for event in runtime.store.list_events(USER, CONV)):
            raise RuntimeError("用户尚未确认却出现退货写入")
        old_op = pending[0]["op_id"]
        if case == "restart_confirm":
            await runtime.close()
            runtime, recorder = await start()
            snapshot = await runtime.service.workflow._graph.aget_state(runtime.service.workflow.config(USER, CONV))
            record["restart_checkpoint_found"] = snapshot.next == ("wait_user",) and bool(snapshot.tasks[0].interrupts)
            if not record["restart_checkpoint_found"]:
                raise RuntimeError("重启后未找到等待点")
            result, pending = await turn(plan["confirm"])
            if not result.workflow["resumed"]:
                raise RuntimeError("确认没有经过 Resume")
        elif case == "modify":
            result, pending = await turn(plan["modify"])
            if not check_pending(pending, modified=True):
                result, pending = await turn(plan["modify_clarification"])
            if not check_pending(pending, modified=True) or pending[0]["op_id"] == old_op:
                raise RuntimeError("修改未生成新的正确详情")
            if any(event["event_type"] == "return_requested" for event in runtime.store.list_events(USER, CONV)):
                raise RuntimeError("修改信息被当作提交授权")
            result, pending = await turn(plan["confirm"])
        else:
            result, pending = await turn(plan[case])
            if result.workflow["next"]:
                raise RuntimeError("取消或人工未离开等待点")
        after = runtime.store.business_snapshot()
        events = runtime.store.list_events(USER, CONV)
        changed_orders = [key for key in before["orders"] if before["orders"][key] != after["orders"][key]]
        checks = {"target_state": check_final(case, after, events, pending),
                  "only_target_order_changed": changed_orders == ([ORDER] if case in {"restart_confirm", "modify"} else []),
                  "users_unchanged": before["users"] == after["users"],
                  "products_unchanged": before["products"] == after["products"]}
        record.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                      events=events, target_order=after["orders"][ORDER])
    except Exception as ex:
        record.update(status="FAIL", error=f"{type(ex).__name__}: {ex}")
    finally:
        if runtime is not None:
            await runtime.close()
        record.update(turns=turns, runtime_cycles=cycles,
                      request_count=ledger(directory).summary()["requests"] - first_requests,
                      tool_count=sum(len(row["result"]["tool_traces"]) for row in turns))
        write(trial / "result.json", record)
    return record


def preflight(directory):
    if directory.exists():
        raise ValueError("预检须使用新目录，不能覆盖已有证据或重置账本")
    directory.mkdir(parents=True)
    plan = frozen_plan()
    write(directory / "plan.json", plan)
    state = ledger(directory).summary()
    write(directory / "preflight.json", {"status": "READY", "ledger": state,
          "request_guard": "完整请求字节上界预留、usage结算、失败按上界、SDK重试0",
          "planned_requests": 100, "planned_input_tokens_each": 12000,
          "planned_output_tokens_each": 1024, "planned_cny": 3.2192,
          "note": "计划估算不替代逐请求硬上限；本轮8条，每条仅运行一次，不按失败补跑"})
    print(json.dumps({"status": "READY", "directory": str(directory), "budget_cny": BUDGET}, ensure_ascii=False))


async def live(directory):
    from dotenv import load_dotenv
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if plan != frozen_plan():
        raise RuntimeError("代码、用例或配置已改变，拒绝按旧冻结清单发送请求")
    if ledger(directory).summary()["requests"]:
        raise RuntimeError("本轮账本已有请求，不自动补跑")
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("未配置 API 密钥")
    with (directory / "started.json").open("x", encoding="utf-8") as file:
        json.dump({"budget_cny": BUDGET, "model": MODEL}, file)
    rows = []
    try:
        for case, mode in plan["order"]:
            state = ledger(directory).summary()
            if state["remaining_cny"] < 0.2 or state["cost_stop_reason"] or state["unfinished_reservations"]:
                break
            row = await run_trial(directory, plan, case, mode)
            rows.append(row)
            print(json.dumps({"case": case, "mode": mode, "status": row["status"],
                              "requests": row["request_count"], "spent_cny": ledger(directory).summary()["spent_cny"]}, ensure_ascii=False), flush=True)
            write(directory / "results.json", rows)
    finally:
        state = ledger(directory).summary()
        write(directory / "summary.json", {"status": "COMPLETE" if len(rows) == len(plan["order"]) else "PARTIAL",
              "passed": sum(row["status"] == "PASS" for row in rows), "completed": len(rows),
              "planned": len(plan["order"]), "ledger": state, "judge_calls": 0,
              "scope": plan["scope"], "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "live"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.expanduser().resolve()
    if args.command == "preflight":
        preflight(directory)
    else:
        asyncio.run(live(directory))


if __name__ == "__main__":
    main()
