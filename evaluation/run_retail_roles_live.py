"""角色图小量真实兼容验收；六条固定对话共用五元硬上限。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from agents.retail_runtime import build_retail_runtime
from evaluation import run_retail_workflow_live as previous
from evaluation.run_retail_support_live import _Recorder, _apply_runtime_adaptations

ROOT, MODEL, ENDPOINT = previous.ROOT, previous.MODEL, previous.ENDPOINT
USER, CONV = "u02-roles-live-client", "u02-roles-live-conversation"
ORDER, ITEM = "#W1067251", "6452271382"
CARD, GIFT = "credit_card_3362387", "gift_card_2259499"
SOURCES = list(dict.fromkeys(previous.SOURCES + [
    "agents/retail_handoff.py", "evaluation/run_retail_roles_live.py",
    "tests/test_retail_roles_live.py", "evaluation/RETAIL_ROLES_LIVE.md",
    "evaluation/data/retail_support/knowledge.jsonl", "evaluation/data/retail_support/manifest.json",
    "agents/retail_support.py", "mcp/qwen_embedding.py", "mcp/qwen_reranker.py",
    "mcp/hybrid_retrieval.py", "mcp/tool_manager.py",
]))
write, ledger = previous.write, previous.ledger


def frozen_plan():
    import hashlib
    return {
        "version": "u02-roles-live.1", "model": MODEL, "endpoint": ENDPOINT,
        "budget_cny": 5, "input_cny_per_mtok": 2, "output_cny_per_mtok": 8,
        "pricing_url": previous.PRICING_URL, "price_basis": "全部输入按高峰缓存未命中价保守结算",
        "order": [["restart", "langgraph"], ["restart", "legacy"], ["modify", "langgraph"],
                  ["human", "legacy"], ["human", "langgraph"], ["cancel", "langgraph"]],
        "technical": "我的无线耳机蓝牙搜不到。我已经重新配对过了（PAIR-03），还是连不上，请提供有限资料内的下一步建议。",
        "prepare": "还是不行，我要退货。我的邮箱是 raj.sanchez2046@example.com，订单 #W1067251，只退蓝色无线耳机这一件，退款到原支付方式。",
        "clarify": "本人姓名 Raj Sanchez，邮编 92147，邮箱 raj.sanchez2046@example.com；订单 #W1067251，退蓝色无线耳机一件，退款到原支付方式。请准备完整待确认详情，等我下一轮确认。",
        "modify": "先不要提交，修改退款去向：订单 #W1067251 仍只退蓝色无线耳机一件，改退到本人已有礼品卡 gift_card_2259499。请重新准备并展示详情，等我后续确认。",
        "negative": "我还没有确认，这些详情先留着，不要提交。",
        "question": "这笔退货的预计金额是多少？我只是在问问题，先不要提交。",
        "confirm": "确认，就按这些最新展示的详情申请退货。",
        "cancel": "取消这笔退货，我不退了。",
        "human": "这笔退货先不要提交，请帮我转人工继续处理。",
        "limits": {"max_user_turns": 6, "max_requests_per_trial": 40, "max_requests_per_turn": 32},
        "profile": {"agent_mode": "multi", "embedding": "qwen", "retrieval": "hybrid",
                    "reranker": "qwen", "strategy": "relevance", "support": True, "memory": False},
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
    }


def config_for(directory, case, engine):
    import hashlib
    config = previous.config_for(directory, case, "multi")
    trial = directory / f"{case}-{engine}"
    collection = "u02rl_" + hashlib.sha256(str(trial).encode()).hexdigest()[:20]
    return replace(config, db_path=trial / "retail.sqlite", event_log_path=trial / "events.jsonl",
                   mcp_stderr_log=trial / "mcp_stderr.log", chroma_path=trial / "unused-chroma",
                   enable_knowledge=True, support_enabled=True, support_collection=collection,
                   embedding_model="qwen", retrieval_mode="hybrid", reranker_model="qwen",
                   workflow_mode=engine)


def pending_matches(rows, modified=False):
    return len(rows) == 1 and all((rows[0]["status"] == "displayed", rows[0]["order_id"] == ORDER,
        rows[0]["items"]["item_ids"] == [ITEM], rows[0]["amount_cents"] == 25884,
        rows[0]["payment_method_id"] == (GIFT if modified else CARD)))


def role_packets(directory):
    packets = []
    path = directory / "requests.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        row = json.loads(line)
        for message in row["request"].get("messages", []):
            content = message.get("content")
            if isinstance(content, str) and content.startswith("[角色输入契约]\n"):
                packet = json.loads(content.split("\n", 1)[1])
                packets.append({"turn": row["turn"], "packet": packet})
    return packets


async def run_trial(directory, plan, case, engine):
    config = config_for(directory, case, engine)
    trial = config.db_path.parent
    trial.mkdir(parents=True, exist_ok=False)
    first_requests = ledger(directory).summary()["requests"]
    runtime = None
    turns, cycles = [], []
    record = {"case": case, "engine": engine, "status": "RUNNING", "checks": {}}

    def require(name, condition):
        record["checks"][name] = bool(condition)
        if not condition:
            raise RuntimeError(name)

    def no_submit():
        return not any(e["event_type"] == "return_requested" for e in runtime.store.list_events(USER, CONV))

    async def start():
        current = build_retail_runtime(config)
        recorder = _Recorder(trial, directory.name, case, engine)
        _apply_runtime_adaptations(current, current.ledger, recorder)
        original = current.client._inner.messages.create
        async def checked(**kwargs):
            if kwargs.get("model") != MODEL or current.client._inner.max_retries != 0:
                raise RuntimeError("冻结模型或 SDK 零重试约束失效")
            # 到达底层前已预留，因此等于上限的最后一次请求仍允许发送。
            if current.ledger.summary()["requests"] - first_requests > plan["limits"]["max_requests_per_trial"]:
                raise RuntimeError("达到单条验收模型请求上限")
            return await original(**kwargs)
        current.client._inner.messages.create = checked
        try:
            await current.start()
        except BaseException:
            await current.close()
            raise
        cycles.append({"cycle": len(cycles) + 1, "checkpoint": str(current.service.workflow.path) if current.service.workflow else None})
        return current, recorder

    async def turn(text):
        if len(turns) >= plan["limits"]["max_user_turns"] or ledger(directory).summary()["requests"] - first_requests >= plan["limits"]["max_requests_per_trial"]:
            raise RuntimeError("达到冻结轮次或请求上限")
        current_turn = len(turns) + 1
        runtime.ledger.begin_turn(f"{case}-{engine}-{current_turn}", max_requests=32)
        recorder.set_turn(current_turn)
        before = runtime.ledger.summary()["requests"]
        result = await runtime.service.handle_turn(text, USER, CONV)
        payload = asdict(result)
        payload["agent_type"] = result.agent_type.value
        pending = runtime.store.active_pending(USER, CONV)
        turns.append({"message": text, "result": payload, "pending": pending,
                      "model_calls": runtime.ledger.summary()["requests"] - before})
        write(trial / "turns.json", turns)
        print(json.dumps({"case": case, "engine": engine, "turn": current_turn,
                          "role": result.primary_agent, "requests": turns[-1]["model_calls"],
                          "spent_cny": runtime.ledger.summary()["spent_cny"]}, ensure_ascii=False), flush=True)
        return result, pending

    try:
        runtime, recorder = await start()
        before = runtime.store.business_snapshot()
        if case in {"restart", "human"}:
            result, _ = await turn(plan["technical"])
            require("technical_executed", result.primary_agent == "technical")
            require("technical_retrieved", any(t["tool_name"] == "search_knowledge_base" and t.get("result_success") for t in result.tool_traces))
            require("technical_no_return_write", no_submit())
            if case == "restart":
                await runtime.close()
                runtime, recorder = await start()
                if runtime.service.workflow:
                    snapshot = await runtime.service.workflow._graph.aget_state(runtime.service.workflow.config(USER, CONV))
                    require("restart_kept_technical_role", snapshot.values.get("role_result", {}).get("role") == "technical")
        result, pending = await turn(plan["prepare"])
        if not pending:
            result, pending = await turn(plan["clarify"])
        require("correct_displayed_details", pending_matches(pending))
        require("not_submitted_before_confirmation", no_submit())
        if engine == "langgraph":
            require("interrupt_waiting", result.workflow["next"] == ["wait_user"])
        old_op = pending[0]["op_id"]
        if case == "restart":
            await runtime.close()
            runtime, recorder = await start()
            if runtime.service.workflow:
                snapshot = await runtime.service.workflow._graph.aget_state(runtime.service.workflow.config(USER, CONV))
                require("restart_found_real_interrupt", snapshot.next == ("wait_user",) and any(t.interrupts for t in snapshot.tasks))
            result, pending = await turn(plan["confirm"])
            if engine == "langgraph":
                require("confirmation_resumed", result.workflow["resumed"])
        elif case == "modify":
            _, pending = await turn(plan["modify"])
            require("new_details_old_confirmation_invalid", pending_matches(pending, True) and pending[0]["op_id"] != old_op)
            require("modify_not_authorization", no_submit())
            _, pending = await turn(plan["negative"])
            require("negative_not_authorization", no_submit() and pending_matches(pending, True))
            result, pending = await turn(plan["confirm"])
        elif case == "human":
            result, pending = await turn(plan["human"])
            require("human_is_escalation", result.primary_agent == "escalation")
            require("human_exits_wait", not pending and not result.workflow.get("next"))
            if engine == "langgraph":
                require("human_resume_branch", result.workflow["resumed"] and "leave_return" in result.workflow["node_path"])
                require("handoff_used_by_real_tool", any("billing→escalation" in t.get("input", {}).get("summary", "") for t in result.tool_traces))
        else:
            _, pending = await turn(plan["question"])
            require("question_not_authorization", no_submit() and pending_matches(pending))
            result, pending = await turn(plan["cancel"])
            require("cancel_exits_wait", not pending and "cancel_return" in result.workflow["node_path"])
            require("cancel_no_model", turns[-1]["model_calls"] == 0)
        after = runtime.store.business_snapshot()
        events = runtime.store.list_events(USER, CONV)
        submitted = [e for e in events if e["event_type"] == "return_requested"]
        handoffs = [e for e in events if e["event_type"] == "handoff_requested"]
        require("exact_write_counts", len(submitted) == (1 if case in {"restart", "modify"} else 0) and len(handoffs) == (1 if case == "human" else 0))
        require("target_order_state", after["orders"][ORDER]["status"] == ("return requested" if submitted else "delivered"))
        if submitted:
            require("submitted_latest_payment", after["orders"][ORDER]["return_payment_method_id"] == (GIFT if case == "modify" else CARD))
        changed = [key for key in before["orders"] if before["orders"][key] != after["orders"][key]]
        require("unrelated_business_unchanged", changed == ([ORDER] if submitted else []) and before["users"] == after["users"] and before["products"] == after["products"])
        packets = role_packets(trial)
        if engine == "langgraph":
            require("role_packets_have_handoff", bool(packets) and all(p["packet"].get("role_handoff", {}).get("version") == 1 for p in packets))
            if case in {"restart", "human"}:
                target = next((p["packet"]["role_handoff"] for p in packets if p["packet"]["agent_type"] == "billing"), {})
                require("technical_to_billing_real_input", target.get("from_role") == "technical" and target.get("source_turn") == 1)
                require("user_report_preserved", "PAIR-03" in target.get("support_refs", {}).get("user_reported_steps", []))
                require("evidence_refs_preserved", bool(target.get("previous_role", {}).get("evidence_ids")))
            require("no_whole_multi_agent_node", all("run_agent" not in t["result"]["workflow"]["node_path"] for t in turns))
        record.update(status="PASS", events=events, target_order=after["orders"][ORDER], observed_role_packets=packets)
    except Exception as ex:
        record.update(status="FAIL", error=f"{type(ex).__name__}: {ex}")
    finally:
        if runtime is not None:
            await runtime.close()
        record.update(turns=turns, runtime_cycles=cycles,
                      request_count=ledger(directory).summary()["requests"] - first_requests,
                      tool_trace_count=sum(len(t["result"]["tool_traces"]) for t in turns))
        write(trial / "result.json", record)
    return record


def preflight(directory):
    if directory.exists():
        raise ValueError("预检必须使用新目录，不覆盖历史证据或账本")
    directory.mkdir(parents=True)
    write(directory / "plan.json", frozen_plan())
    write(directory / "preflight.json", {"status": "READY", "ledger": ledger(directory).summary(),
        "planned_requests": 100, "planned_cny": 4.0384,
        "estimate": "100次×(16000输入×2+1048输出×8)/百万；估算不替代请求前硬上限",
        "request_guard": "完整请求字节上界预留、usage结算、失败按上界、SDK重试0"})
    print(json.dumps({"status": "READY", "directory": str(directory), "budget_cny": 5}, ensure_ascii=False))


async def live(directory):
    from dotenv import load_dotenv
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if plan != frozen_plan():
        raise RuntimeError("冻结源码、配置或场景改变，拒绝发送付费请求")
    if ledger(directory).summary()["requests"]:
        raise RuntimeError("账本已有请求，禁止自动补跑")
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("未配置 API 密钥")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with (directory / "started.json").open("x", encoding="utf-8") as handle:
        json.dump({"budget_cny": 5, "model": MODEL}, handle)
    rows = []
    try:
        for case, engine in plan["order"]:
            state = ledger(directory).summary()
            if state["remaining_cny"] < 0.2 or state["cost_stop_reason"] or state["unfinished_reservations"]:
                break
            row = await run_trial(directory, plan, case, engine)
            rows.append(row)
            write(directory / "results.json", rows)
            print(json.dumps({"case": case, "engine": engine, "status": row["status"], "error": row.get("error"), "spent_cny": ledger(directory).summary()["spent_cny"]}, ensure_ascii=False), flush=True)
    finally:
        write(directory / "summary.json", {"status": "COMPLETE" if len(rows) == len(plan["order"]) else "PARTIAL",
              "passed": sum(r["status"] == "PASS" for r in rows), "completed": len(rows), "planned": len(plan["order"]),
              "ledger": ledger(directory).summary(), "judge_calls": 0, "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "live"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    preflight(directory) if args.command == "preflight" else asyncio.run(live(directory))


if __name__ == "__main__":
    main()
