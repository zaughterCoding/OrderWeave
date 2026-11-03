"""单个评测会话：独立业务库、checkpoint 与账本，结束后写 result.json。

由调度器以子进程启动，避免同进程并发共享单轮请求计数等进程内状态。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.harness import config as cfg  # noqa: E402



def is_write_event(event_type: str) -> bool:
    """退货沿用原事件名，其余写操作统一以 _executed 结尾。"""
    return event_type == "return_requested" or event_type.endswith("_executed")
# 评测设施原因的终止不计入有效分母；Agent 自身失败（报错、步数、工具错误）计为有效失败。
INFRA_TERMINATIONS = {"USER_ERROR", "BRIDGE_ERROR", "RUN_ERROR", "BUDGET_STOP", "INPUT_LIMIT"}


def classify(termination: str, reward, eval_error: str) -> tuple:
    """按官方口径：只有用户正常结束且奖励为 1 才算成功。"""
    valid = termination not in INFRA_TERMINATIONS and reward is not None and not eval_error
    return valid, bool(valid and termination == "USER_STOP" and reward == 1.0)


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def trajectory_messages(turns: List[Dict[str, Any]], final_user_stop: str) -> List[Dict[str, Any]]:
    """把项目回合映射为官方消息序列；工具调用与结果按原序一一对应。"""
    messages: List[Dict[str, Any]] = [{"role": "assistant", "content": cfg.GREETING}]
    for turn_index, turn in enumerate(turns, start=1):
        messages.append({"role": "user", "content": str(turn.get("user") or "")})
        for trace_index, trace in enumerate(turn.get("tool_traces") or []):
            tool_id = str(trace.get("tool_use_id") or f"tool-{turn_index}-{trace_index}")
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": tool_id, "name": str(trace.get("tool_name") or ""),
                "arguments": dict(trace.get("input") or {}), "requestor": "assistant"}]})
            messages.append({"role": "tool", "id": tool_id, "requestor": "assistant",
                             "content": json.dumps(trace.get("result"), ensure_ascii=False, default=str),
                             "error": bool(trace.get("success") is False or trace.get("result_success") is False)})
        messages.append({"role": "assistant", "content": str(turn.get("assistant") or "")})
    if final_user_stop:
        messages.append({"role": "user", "content": final_user_stop})
    return messages


def changed_keys(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, List[str]]:
    return {table: sorted(k for k in set(before.get(table, {})) | set(after.get(table, {}))
                          if before.get(table, {}).get(k) != after.get(table, {}).get(k))
            for table in ("users", "orders", "products")}


def safety_checks(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """写操作必须先有同一操作的展示事件；身份未绑定时不能有写入。"""
    displayed = {str(e.get("op_id")) for e in events if str(e["event_type"]).endswith("_displayed")}
    writes = [e for e in events if is_write_event(str(e["event_type"]))]
    unconfirmed = [str(e.get("op_id")) for e in writes if str(e.get("op_id")) not in displayed]
    ops = [str(e.get("op_id")) for e in writes]
    duplicates = sorted({op for op in ops if ops.count(op) > 1})
    return {"writes": len(writes), "unconfirmed_writes": unconfirmed, "duplicate_writes": duplicates,
            "identity_bound": any(e["event_type"] == "identity_bound" for e in events)}


def _usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    return {} if usage is None else {"prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                                     "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0)}


def _raw(response: Any) -> Dict[str, Any]:
    return {"id": getattr(response, "id", ""), "model": getattr(response, "model", ""),
            "stop_reason": getattr(response, "stop_reason", "")}


def ledger_cost(path: Path) -> Dict[str, Any]:
    import sqlite3
    if not path.exists():
        return {}
    with sqlite3.connect(str(path)) as connection:
        rows = connection.execute(
            "SELECT stage, COUNT(*), COALESCE(SUM(actual_cny),0), COALESCE(SUM(input_tokens),0),"
            " COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(output_tokens),0),"
            " SUM(CASE WHEN status!='success' THEN 1 ELSE 0 END) FROM entries GROUP BY stage").fetchall()
    stages = {r[0]: {"requests": r[1], "cny": round(r[2], 8), "input_tokens": r[3],
                     "cache_read_tokens": r[4], "output_tokens": r[5], "non_success": r[6]} for r in rows}

    def total(keys):
        return round(sum(v["cny"] for k, v in stages.items() if keys(k)), 8)
    return {"stages": stages, "system_cny": total(lambda k: k not in {"user", "judge"}),
            "user_cny": total(lambda k: k == "user"), "judge_cny": total(lambda k: k == "judge"),
            "total_cny": total(lambda k: True)}


async def run_session(args: argparse.Namespace) -> Dict[str, Any]:
    cfg.load_env()
    from agents.retail_runtime import RetailRuntimeConfig, build_retail_runtime
    from core.llm_utils import extract_text_content
    from evaluation.tau_bridge import BridgeError, UserSimulatorBridge, to_anthropic_kwargs
    from evaluation.tau_eval_bridge import EvalBridge

    case_dir = Path(args.case_dir)
    if (case_dir / "result.json").exists():
        return json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    if case_dir.exists():
        # 中断的半成品保留为证据，从干净状态重跑，不复用旧库或旧对话。
        shutil.move(str(case_dir), str(case_dir.with_name(case_dir.name + ".aborted-" + str(int(time.time())))))
    case_dir.mkdir(parents=True)
    profile = cfg.PROFILES[args.profile]
    model = cfg.model_settings()
    ledger_path = case_dir / "ledger.sqlite"
    runtime_config = RetailRuntimeConfig(
        db_path=case_dir / "retail.sqlite", seed_path=cfg.SEED_PATH, policy_path=cfg.POLICY_PATH,
        knowledge_path=case_dir / "unused-knowledge.jsonl", knowledge_collection="harness_unused",
        ledger_path=ledger_path, event_log_path=case_dir / "events.jsonl",
        budget_cny=cfg.SESSION_BUDGET_CNY, input_price_cny_per_mtok=cfg.INPUT_CNY_PER_MTOK,
        output_price_cny_per_mtok=cfg.OUTPUT_CNY_PER_MTOK, model=model["model"], base_url=model["base_url"],
        redis_url="", chroma_host="127.0.0.1", chroma_port=0, chroma_path=case_dir / "chroma",
        case_id=f"{args.task_id}-t{args.trial}", run_id=args.run_id, profile_enabled=False,
        memory_key_prefix=f"harness:{args.run_id}:", memory_collection_suffix="_harness",
        max_tokens=profile.max_tokens, max_message_chars=cfg.MAX_MESSAGE_CHARS, max_context_chars=12000,
        mcp_timeout_seconds=60.0, mcp_stderr_log=case_dir / "mcp.log", agent_mode=profile.agent_mode,
        enable_memory=profile.enable_memory, enable_knowledge=False, language="en",
        workflow_mode=profile.workflow_mode, checkpoint_path=case_dir / "checkpoints.sqlite",
        semantic_routing=False, ledger_settlement_policy=model["settlement_policy"],
        model_thinking=profile.thinking, context_mode=profile.context_mode)
    runtime = build_retail_runtime(runtime_config)
    await runtime.start()
    user_client = runtime.client.for_stage("user")
    judge_client = runtime.client.for_stage("judge")

    async def call(client, event):
        response = await client.messages.create(**to_anthropic_kwargs(event["request"]))
        return {"content": extract_text_content(response.content), "usage": _usage(response), "raw_data": _raw(response)}

    user_bridge = UserSimulatorBridge(
        python_exe=cfg.TAU_PYTHON, worker_path=cfg.USER_WORKER, data_dir=cfg.TAU_DATA,
        tasks_path=args.tasks, task_id=args.task_id, llm=model["user_model"],
        max_tokens=cfg.USER_MAX_TOKENS, temperature=0.0, seed=cfg.SEED + int(args.trial),
        stderr_path=case_dir / "user-worker.log", timeout_s=cfg.USER_TIMEOUT_S,
        llm_handler=lambda event: call(user_client, event))
    before = runtime.store.business_snapshot()
    turns: List[Dict[str, Any]] = []
    termination, detail, final_stop = "AGENT_ERROR", "", ""
    steps, errors = 1, 0
    latencies: List[float] = []
    started = time.monotonic()
    client_id, conv_id = f"client-{args.task_id}", f"conv-{args.task_id}-t{args.trial}"
    try:
        await user_bridge.start()
        event = await user_bridge.send_agent_message(cfg.GREETING)
        while True:
            text = str(event.get("content") or "")
            if event.get("is_stop"):
                termination, detail, final_stop = "USER_STOP", str(event.get("stop_kind") or ""), text
                break
            if not text.strip():
                termination, detail = "USER_ERROR", "模拟用户返回空内容"
                break
            if len(text) > cfg.MAX_MESSAGE_CHARS or sum(
                    len(t["user"]) + len(t["assistant"]) for t in turns) > cfg.MAX_CONVERSATION_CHARS:
                termination, detail = "INPUT_LIMIT", "消息或累计对话超过上限"
                break
            turn = {"user": text, "assistant": "", "tool_traces": []}
            turns.append(turn)
            turn_started = time.monotonic()
            try:
                result = await runtime.service.handle_turn(text, client_id, conv_id)
            except Exception as exc:
                termination, detail = "AGENT_ERROR", f"{type(exc).__name__}: {exc}"
                break
            latencies.append(round((time.monotonic() - turn_started) * 1000, 1))
            turn.update(assistant=result.response, tool_traces=list(result.tool_traces),
                        tools_used=list(result.tools_used), turn_success=bool(result.success))
            if not result.success:
                stop = runtime.ledger.summary().get("cost_stop_reason") or runtime.ledger.last_rejection_reason
                termination, detail = ("BUDGET_STOP", stop) if stop else ("AGENT_ERROR", "Agent 返回失败")
                break
            errors += sum(1 for t in turn["tool_traces"] if t.get("success") is False or t.get("result_success") is False)
            steps += 2 + 2 * len(turn["tool_traces"])
            if errors >= cfg.MAX_ERRORS or steps >= cfg.MAX_STEPS:
                termination, detail = ("TOO_MANY_ERRORS", str(errors)) if errors >= cfg.MAX_ERRORS else ("MAX_STEPS", str(steps))
                break
            if not str(result.response).strip():
                termination, detail = "AGENT_ERROR", "Agent 回复为空"
                break
            try:
                event = await user_bridge.send_agent_message(str(result.response))
            except BridgeError as exc:
                termination, detail = "BRIDGE_ERROR", str(exc)
                break
    except Exception as exc:
        termination, detail = "RUN_ERROR", f"{type(exc).__name__}: {exc}"
    finally:
        await user_bridge.close()

    after = runtime.store.business_snapshot()
    events = runtime.store.list_events(client_id, conv_id)
    trajectory = trajectory_messages(turns, final_stop)
    write_json(case_dir / "snapshot-after.json", after)
    write_json(case_dir / "trajectory.json", trajectory)
    write_json(case_dir / "dialogue.json", {"turns": turns, "events": events})

    eval_result: Dict[str, Any] = {}
    eval_error = ""
    bridge = EvalBridge(
        python_exe=cfg.TAU_PYTHON, worker_path=cfg.EVAL_WORKER, data_dir=cfg.TAU_DATA,
        tasks_path=args.tasks, task_id=args.task_id, snapshot_path=case_dir / "snapshot-after.json",
        trajectory_path=case_dir / "trajectory.json", judge_model=model["judge_model"],
        judge_max_tokens=cfg.JUDGE_MAX_TOKENS, judge_seed=cfg.SEED, stderr_path=case_dir / "eval-worker.log",
        timeout_s=cfg.EVAL_TIMEOUT_S, judge_handler=lambda event: call(judge_client, event),
        skip_nl=termination == "BUDGET_STOP")
    try:
        eval_result = await bridge.run()
    except Exception as exc:
        eval_error = f"{type(exc).__name__}: {exc}"
        eval_result = {"db": bridge.db_result or {}, "nl": bridge.nl_result or {}}
    await runtime.close()

    db, nl = eval_result.get("db") or {}, eval_result.get("nl") or {}
    # 官方回放把参考动作报错转成错误消息并照常计分；参考问题单列，不作废该题。
    db_reward = (db.get("official_reward") or {}).get("reward") if db.get("status") in {"OK", "INVALID_REFERENCE"} else None
    nl_reward = 1.0 if nl.get("status") == "NOT_APPLICABLE" else nl.get("reward") if nl.get("status") == "OK" else None
    reward = None if db_reward is None or nl_reward is None else float(db_reward) * float(nl_reward)
    valid, success = classify(termination, reward, eval_error)
    result = {
        "suite": args.suite, "task_id": args.task_id, "trial": int(args.trial), "profile": args.profile,
        "valid": valid, "success": success, "reward": reward,
        "db_reward": db_reward, "nl_reward": nl_reward, "db_status": db.get("status"), "nl_status": nl.get("status"),
        "reference_errors": db.get("gold_errors") or [], "eval_error": eval_error,
        "nl_checks": nl.get("checks") or [],
        "termination": termination, "termination_detail": detail, "turns": len(turns), "steps": steps,
        "tool_errors": errors, "turn_latency_ms": latencies, "duration_ms": round((time.monotonic() - started) * 1000, 1),
        "safety": safety_checks(events), "changed": changed_keys(before, after),
        "cost": ledger_cost(ledger_path), "model": model["model"],
        "finished_at": datetime.now(timezone.utc).isoformat()}
    write_json(case_dir / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="运行单个评测会话")
    for name in ("suite", "tasks", "task-id", "trial", "profile", "run-id", "case-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    asyncio.run(run_session(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
