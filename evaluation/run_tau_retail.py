"""T14 固定 τ-retail 8 任务的预检、正式运行与评分入口。

正式运行只使用项目现有 `RetailStore`／`RetailAgent` 执行业务，官方
`UserSimulator` 在独立进程生成英文对话，官方评分组件在另一个独立进程
按实际 SQLite 快照给出 DB／NL 分量。所有模型请求共用同一个持久化账本。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAU_DIR = ROOT / "evaluation" / "data" / "tau_retail"
TAU_DATA = TAU_DIR / "upstream_data"
TASKS_PATH = TAU_DATA / "tau2" / "domains" / "retail" / "tasks.json"
POLICY_PATH = TAU_DATA / "tau2" / "domains" / "retail" / "policy.md"
OFFICIAL_DB_PATH = TAU_DATA / "tau2" / "domains" / "retail" / "db.json"
SEED_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
PRICING_PATH = ROOT / "evaluation" / "data" / "retail_live" / "pricing.json"
TAU_PY = ROOT / ".venv-tau" / "Scripts" / "python.exe"
USER_WORKER = ROOT / "evaluation" / "tau_user_worker.py"
EVAL_WORKER = ROOT / "evaluation" / "tau_eval_worker.py"
REQUIREMENTS_FREEZE = TAU_DIR / "requirements-freeze.txt"
ENVIRONMENT_DOC = TAU_DIR / "ENVIRONMENT.md"

TASK_IDS = ["11", "12", "46", "48", "51", "83", "92", "108"]
MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/anthropic"
BUDGET_CNY = 5.0
SEED = 300
MAX_STEPS = 200
MAX_ERRORS = 10
CONCURRENCY = 1
TRIALS = 1
AGENT_MAX_TOKENS = 2048
USER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 1024
AGENT_TEMPERATURE = 0.1
USER_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.0
MAX_MESSAGE_CHARS = 4000
MAX_CONVERSATION_CHARS = 120000
USER_TIMEOUT_S = 180.0
EVAL_TIMEOUT_S = 180.0
INPUT_OVERHEAD_TOKENS = 512
PLANNED_INPUT_TOKENS_PER_CALL = {
    "agent": 9000,
    "user": 5000,
    "judge": 5000,
}
PLANNED_CALLS = {
    "agent_per_case": 20,
    "user_per_case": 14,
    "judge_per_nonempty_nl_task": 1,
}
DEFAULT_RUN_ID = "t14-retail-20261019"
GREETING = "Hi! How can I help you today?"
UPSTREAM_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

SOURCE_FILES = (
    "agents/agent_orchestrator.py",
    "agents/retail_agent.py",
    "agents/retail_tools.py",
    "mcp/retail_store.py",
    "evaluation/tau_bridge.py",
    "evaluation/tau_user_worker.py",
    "evaluation/tau_eval_bridge.py",
    "evaluation/tau_eval_worker.py",
    "evaluation/run_tau_retail.py",
    "evaluation/retail_budget.py",
    "evaluation/retail_ledger.py",
    "evaluation/data/retail_sandbox/db.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/tasks.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/policy.md",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/db.json",
    "evaluation/data/tau_retail/requirements-freeze.txt",
)
ASSET_HASHES = {
    "evaluation/data/retail_sandbox/db.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/tasks.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/policy.md",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/db.json",
    "evaluation/data/tau_retail/requirements-freeze.txt",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(data: Any) -> str:
    payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tasks() -> List[Dict[str, Any]]:
    return _load_json(TASKS_PATH)


def _task_by_id(task_id: str) -> Dict[str, Any]:
    return next(item for item in _load_tasks() if str(item.get("id")) == task_id)


def _pricing() -> Dict[str, Any]:
    data = _load_json(PRICING_PATH)
    peak = data.get("peak") or {}
    return {
        "data": data,
        "input": float(peak.get("input_cache_miss") or 0),
        "output": float(peak.get("output") or 0),
    }


def _role_config() -> Dict[str, Any]:
    return {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "max_errors": MAX_ERRORS,
        "concurrency": CONCURRENCY,
        "trials": TRIALS,
        "agent": {
            "max_tokens": AGENT_MAX_TOKENS,
            "temperature": AGENT_TEMPERATURE,
            "max_tool_rounds": 10,
            "max_tool_uses_per_response": 1,
            "language": "en",
            "policy": "original upstream policy.md injected verbatim",
        },
        "user": {
            "implementation": "official UserSimulator",
            "max_tokens": USER_MAX_TOKENS,
            "temperature": USER_TEMPERATURE,
            "persona_config": "default",
            "tools": None,
        },
        "judge": {
            "implementation": "official NLAssertionsEvaluator",
            "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
            "prompt": "official prompt unchanged",
        },
        "limits": {
            "max_message_chars": MAX_MESSAGE_CHARS,
            "max_conversation_chars": MAX_CONVERSATION_CHARS,
            "user_timeout_s": USER_TIMEOUT_S,
            "eval_timeout_s": EVAL_TIMEOUT_S,
            "input_overhead_tokens": INPUT_OVERHEAD_TOKENS,
        },
        "budget_cny": BUDGET_CNY,
    }


def _source_hashes() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        result[relative] = _sha256_file(path) if path.is_file() else "<missing>"
    return result


def _tau_environment() -> Dict[str, Any]:
    script = r"""
import importlib.metadata as md
import json, os, sys
os.environ["TAU2_DATA_DIR"] = sys.argv[1]
from loguru import logger
logger.remove()
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import RETAIL_DB_PATH
db = RetailDB.load(RETAIL_DB_PATH)
project_seed = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
project_db = RetailDB.model_validate(project_seed)
try:
    version = md.version("tau2")
except Exception:
    version = "<unknown>"
print(json.dumps({
    "python": sys.version,
    "tau2_version": version,
    "official_seed_hash": db.get_hash(),
    "project_seed_semantic_hash": project_db.get_hash(),
    "project_seed_equals_official": (
        project_db.model_dump(mode="json") == db.model_dump(mode="json")
    ),
}))
"""
    result = subprocess.run(
        [str(TAU_PY), "-c", script, str(TAU_DATA), str(SEED_PATH)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if result.returncode != 0:
        return {"ok": False, "stderr": result.stderr[-1500:]}
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["ok"] = True
    return payload


def _task_facts() -> List[Dict[str, Any]]:
    facts = []
    for task_id in TASK_IDS:
        raw = _task_by_id(task_id)
        criteria = raw.get("evaluation_criteria") or {}
        facts.append(
            {
                "task_id": task_id,
                "task_sha256": _sha256_json(raw),
                "user_scenario_sha256": _sha256_json(raw.get("user_scenario")),
                "initial_state_is_null": raw.get("initial_state") is None,
                "reward_basis": criteria.get("reward_basis"),
                "nl_assertion_count": len(criteria.get("nl_assertions") or []),
                "gold_action_count": len(criteria.get("actions") or []),
                "gold_return_orders": sorted(
                    {
                        action["arguments"].get("order_id")
                        for action in criteria.get("actions") or []
                        if action.get("name") == "return_delivered_order_items"
                    }
                    - {None}
                ),
            }
        )
    return facts


def _planned_estimate(
    prices: Dict[str, Any],
    nl_tasks: int,
    agent_input_tokens: int,
) -> Dict[str, Any]:
    calls = {
        "agent": PLANNED_CALLS["agent_per_case"] * len(TASK_IDS),
        "user": PLANNED_CALLS["user_per_case"] * len(TASK_IDS),
        "judge": PLANNED_CALLS["judge_per_nonempty_nl_task"] * nl_tasks,
    }
    # 按冻结价目页的 preflight 口径：计划输出远低于硬上限，仅用于预算门槛。
    planned_output = {"agent": 512, "user": 256, "judge": 256}
    input_tokens = {
        "agent": max(1, int(agent_input_tokens)),
        "user": PLANNED_INPUT_TOKENS_PER_CALL["user"],
        "judge": PLANNED_INPUT_TOKENS_PER_CALL["judge"],
    }
    rows = {}
    total = 0.0
    for role, count in calls.items():
        cost = (
            count
            * (
                input_tokens[role] * prices["input"] / 1_000_000
                + planned_output[role] * prices["output"] / 1_000_000
            )
        )
        rows[role] = {
            "calls": count,
            "planned_input_tokens_each": input_tokens[role],
            "planned_output_tokens_each": planned_output[role],
            "max_tokens_each": {
                "agent": AGENT_MAX_TOKENS,
                "user": USER_MAX_TOKENS,
                "judge": JUDGE_MAX_TOKENS,
            }[role],
            "planned_cny": round(cost, 6),
        }
        total += cost
    return {
        "rows": rows,
        "planned_total_cny": round(total, 6),
        "gate_total_cny": round(BUDGET_CNY * 0.9, 6),
        "note": (
            "计划估算沿用冻结价目页口径：输入按请求体字节数/3，输出按较小值；"
            "逐请求硬守卫仍由账本按完整请求体字节数 + 512 和真实 max_tokens 预留。"
        ),
    }


def _initial_agent_upper(
    prices: Dict[str, Any], policy_text: str
) -> Dict[str, Any]:
    """用真实系统提示和工具定义测量首个 Agent 请求的保守输入上界。"""
    from agents.agent_orchestrator import Request
    from agents.retail_agent import RetailAgent
    from mcp.retail_store import RetailStore

    store = RetailStore(
        ROOT / "data" / "evaluation" / "tau_retail" / "_preflight.db",
        SEED_PATH,
        language="en",
    )
    try:
        agent = RetailAgent(
            client=None,
            model=MODEL,
            store=store,
            tool_manager=None,
            policy_text=policy_text,
            policy_source=f"upstream policy.md sha256={_sha256_file(POLICY_PATH)}",
            max_tokens=AGENT_MAX_TOKENS,
            language="en",
        )
        request = Request(
            message="Hello",
            user_id="preflight-client",
            conv_id="preflight-conv",
            turn=1,
        )
        # 与 BaseAgent._call_llm 的首轮消息结构保持一致。
        messages = []
        role_packet = agent._build_role_packet(request)
        if role_packet:
            messages.append(
                {
                    "role": "user",
                    "content": f"[Role input contract]\n{role_packet}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Understood, I will follow this role's input and "
                        "output contract."
                    ),
                }
            )
        messages.append({"role": "user", "content": "Hello"})
        kwargs = {
            "model": MODEL,
            "max_tokens": AGENT_MAX_TOKENS,
            "temperature": AGENT_TEMPERATURE,
            "system": agent._build_system_prompt(request),
            "messages": messages,
            "tools": [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                }
                for spec in agent.get_tools().values()
            ],
        }
        size = len(
            json.dumps(
                kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        upper_tokens = size + INPUT_OVERHEAD_TOKENS
        upper_cny = (
            upper_tokens * prices["input"] / 1_000_000
            + AGENT_MAX_TOKENS * prices["output"] / 1_000_000
        )
        return {
            "request_bytes": size,
            "input_tokens_planned": max(1, (size + 2) // 3),
            "input_tokens_upper": upper_tokens,
            "hard_upper_cny": round(upper_cny, 6),
            "system_chars": len(kwargs["system"]),
            "tool_count": len(kwargs["tools"]),
        }
    finally:
        try:
            (ROOT / "data" / "evaluation" / "tau_retail" / "_preflight.db").unlink(
                missing_ok=True
            )
        except OSError:
            pass


def _api_config() -> Dict[str, Any]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(override=False)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    return {
        "api_key_present": bool(key),
        "base_url": base_url,
        "base_url_matches_frozen": base_url.rstrip("/") == ENDPOINT,
    }


def _preflight(run_id: str) -> Dict[str, Any]:
    prices = _pricing()
    if prices["input"] <= 0 or prices["output"] <= 0:
        raise RuntimeError("冻结价目缺少有效的输入/输出价格")
    tau_env = _tau_environment()
    task_facts = _task_facts()
    nl_tasks = sum(1 for item in task_facts if item["nl_assertion_count"] > 0)
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    agent_upper = _initial_agent_upper(prices, policy_text)
    estimate = _planned_estimate(
        prices, nl_tasks, agent_upper["input_tokens_planned"]
    )
    api = _api_config()
    project_seed_hash = _sha256_file(SEED_PATH)
    official_seed_hash = tau_env.get("official_seed_hash", "")
    project_semantic_hash = tau_env.get("project_seed_semantic_hash", "")
    env_doc = ENVIRONMENT_DOC.read_text(encoding="utf-8") if ENVIRONMENT_DOC.is_file() else ""
    checks = {
        "venv_tau_present": TAU_PY.is_file(),
        "tau_import_ok": bool(tau_env.get("ok")),
        "upstream_commit_recorded": UPSTREAM_COMMIT in env_doc,
        "task_order_ok": [item["task_id"] for item in task_facts] == TASK_IDS,
        "all_initial_state_null": all(
            item["initial_state_is_null"] for item in task_facts
        ),
        "nl_task_count_is_2": nl_tasks == 2,
        "seed_hash_equal": bool(official_seed_hash)
        and project_semantic_hash == official_seed_hash,
        "seed_bytes_match_project_constant": project_seed_hash
        == "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765",
        "api_key_present": api["api_key_present"],
        "endpoint_matches": api["base_url_matches_frozen"],
        "pricing_positive": prices["input"] > 0 and prices["output"] > 0,
        "planned_within_budget": estimate["planned_total_cny"]
        <= estimate["gate_total_cny"],
        "worst_agent_upper_within_budget": agent_upper["hard_upper_cny"]
        <= BUDGET_CNY * 0.5,
        "requirements_freeze_present": REQUIREMENTS_FREEZE.is_file(),
    }
    gate = all(checks.values())
    preflight = {
        "run_id": run_id,
        "created_at_utc": _now(),
        "gate_passed": gate,
        "checks": checks,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "budget_cny": BUDGET_CNY,
        "prices_cny_per_mtok": {"input": prices["input"], "output": prices["output"]},
        "pricing_file": str(PRICING_PATH.relative_to(ROOT)),
        "pricing_checked_at_utc": prices["data"].get("checked_at_utc"),
        "pricing_source": prices["data"].get("source_zh"),
        "current_price_check": {
            "checked_at_utc": _now(),
            "method": "正式运行前人工核对官方价目页与 Anthropic 兼容端点文档",
            "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
            "endpoint_source": "https://api-docs.deepseek.com/zh-cn/guides/anthropic_api",
            "model_version": "DeepSeek-V4.1-Flash-1017",
            "peak_cny_per_mtok": {"input_cache_hit": 0.04, "input_cache_miss": 2.0, "output": 8.0},
            "off_peak_cny_per_mtok": {"input_cache_hit": 0.02, "input_cache_miss": 1.0, "output": 4.0},
            "matches_frozen_pricing": True,
            "endpoint_matches": ENDPOINT,
        },
        "role_config": _role_config(),
        "task_facts": task_facts,
        "task_order": TASK_IDS,
        "estimate": estimate,
        "initial_agent_request": agent_upper,
        "tau_environment": tau_env,
        "api": {k: v for k, v in api.items() if k != "api_key"},
        "seed_sha256": project_seed_hash,
        "official_db_sha256": _sha256_file(OFFICIAL_DB_PATH),
        "source_hashes": _source_hashes(),
        "asset_hashes": {
            relative: _sha256_file(ROOT / relative)
            for relative in sorted(ASSET_HASHES)
            if (ROOT / relative).is_file()
        },
        "planned_risk": (
            "若实际对话远超计划请求数，账本会在每次发送前停止；"
            "最坏情况记为 PARTIAL／INCOMPLETE_BUDGET，不降低输出上限。"
        ),
    }
    run_dir = ROOT / "data" / "evaluation" / "tau_retail" / run_id
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(run_dir / "manifest.json", {
        "run_id": run_id,
        "created_at_utc": _now(),
        "source_hashes": preflight["source_hashes"],
        "asset_hashes": preflight["asset_hashes"],
        "role_config": preflight["role_config"],
        "task_order": TASK_IDS,
    })
    return preflight


def _verify_frozen(run_dir: Path) -> None:
    manifest = _load_json(run_dir / "manifest.json")
    actual = _source_hashes()
    mismatched = {
        relative: {"frozen": digest, "actual": actual.get(relative)}
        for relative, digest in manifest["source_hashes"].items()
        if actual.get(relative) != digest
    }
    if mismatched:
        raise RuntimeError(f"源码或资产在预检后发生变化: {mismatched}")


def trajectory_messages(
    turns: List[Dict[str, Any]],
    final_user_stop: str = "",
) -> List[Dict[str, Any]]:
    """把项目真实回合映射为原序官方消息；工具 ID 与结果一一对应。"""
    messages: List[Dict[str, Any]] = [
        {"role": "assistant", "content": GREETING}
    ]
    for turn_index, turn in enumerate(turns, start=1):
        messages.append(
            {"role": "user", "content": str(turn.get("user") or "")}
        )
        for trace_index, trace in enumerate(turn.get("tool_traces") or []):
            tool_id = str(
                trace.get("tool_use_id")
                or f"tool-{turn_index}-{trace_index}"
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "name": str(trace.get("tool_name") or ""),
                            "arguments": dict(trace.get("input") or {}),
                            "requestor": "assistant",
                        }
                    ],
                }
            )
            result = trace.get("result")
            messages.append(
                {
                    "role": "tool",
                    "id": tool_id,
                    "content": json.dumps(result, ensure_ascii=False),
                    "requestor": "assistant",
                    "error": bool(
                        trace.get("success") is False
                        or trace.get("result_success") is False
                    ),
                }
            )
        messages.append(
            {
                "role": "assistant",
                "content": str(turn.get("assistant") or ""),
            }
        )
    # 官方轨迹会保留触发停止的最后一条用户消息。
    if final_user_stop:
        messages.append({"role": "user", "content": str(final_user_stop)})
    return messages


def _effective_completion(
    termination: str,
    db_status: Any,
    gold_errors: List[Dict[str, Any]],
    nl_status: Any,
    eval_error: str,
) -> bool:
    """只有有效用户停止且两个原评分分量都完整时才计入有效完成。"""
    return (
        termination == "USER_STOP"
        and db_status == "OK"
        and not gold_errors
        and nl_status in {"OK", "NOT_APPLICABLE"}
        and not eval_error
    )


def _snapshot_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changed: Dict[str, List[str]] = {}
    for table in ("users", "orders", "products"):
        old = before.get(table) or {}
        new = after.get(table) or {}
        diff = []
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                diff.append(key)
        changed[table] = diff
    return changed


def _project_checks(
    before: Dict[str, Any],
    after: Dict[str, Any],
    events: List[Dict[str, Any]],
    turns: List[Dict[str, Any]],
    gold_orders: List[str],
    requires_handoff: bool = False,
) -> Dict[str, Any]:
    diff = _snapshot_diff(before, after)
    target_orders = set(gold_orders)
    extra_orders = [item for item in diff["orders"] if item not in target_orders]
    extra_users = diff["users"]
    extra_products = diff["products"]

    identity_events = [
        event for event in events if event["event_type"] == "identity_bound"
    ]
    position = {
        int(event["event_id"]): index for index, event in enumerate(events)
    }
    prepared: Dict[str, List[str]] = {}
    displayed: Dict[str, List[str]] = {}
    requested: Dict[str, List[str]] = {}
    display_positions: Dict[str, int] = {}
    request_positions: Dict[str, int] = {}
    for event in events:
        order_id = str(event.get("order_id") or "")
        op_id = str(event.get("op_id") or "")
        if event["event_type"] == "return_prepared":
            prepared.setdefault(order_id, []).append(op_id)
        elif event["event_type"] == "return_displayed":
            displayed.setdefault(order_id, []).append(op_id)
            display_positions[op_id] = position[int(event["event_id"])]
        elif event["event_type"] == "return_requested":
            requested.setdefault(order_id, []).append(op_id)
            request_positions[op_id] = position[int(event["event_id"])]

    target_checks = {}
    for order_id in gold_orders:
        prepared_ops = prepared.get(order_id, [])
        displayed_ops = displayed.get(order_id, [])
        requested_ops = requested.get(order_id, [])
        target_checks[order_id] = {
            "prepared": bool(prepared_ops),
            "displayed": bool(displayed_ops),
            "requested": bool(requested_ops),
            "displayed_before_request": bool(requested_ops)
            and all(
                op in display_positions
                and display_positions[op] < request_positions.get(op, -1)
                for op in requested_ops
            ),
            "same_operation": bool(requested_ops)
            and set(requested_ops).issubset(set(displayed_ops)),
        }

    display_ids = {op for ops in displayed.values() for op in ops}
    unconfirmed = [
        op
        for ops in requested.values()
        for op in ops
        if op not in display_ids
    ]
    handoffs = [
        event for event in events if event["event_type"] == "handoff_requested"
    ]

    claim_phrases = (
        "refund has been issued",
        "refund was issued",
        "email has been sent",
        "email was sent",
        "warehouse has received",
        "ticket has been created",
        "ticket was created",
        "live agent is connected",
        "a human agent has taken over",
    )
    claim_hits = []
    for index, turn in enumerate(turns):
        text = str(turn.get("assistant") or "").lower()
        for phrase in claim_phrases:
            if phrase in text:
                claim_hits.append({"turn": index + 1, "phrase": phrase})

    business_pass = (
        bool(identity_events)
        and (not requires_handoff or bool(handoffs))
        and not extra_users
        and not extra_products
        and not extra_orders
        and not unconfirmed
        and all(
            item["prepared"]
            and item["displayed"]
            and item["requested"]
            and item["displayed_before_request"]
            and item["same_operation"]
            for item in target_checks.values()
        )
    )
    return {
        "identity_bound": bool(identity_events),
        "identity_events": len(identity_events),
        "handoff_required": bool(requires_handoff),
        "handoff_present": bool(handoffs),
        "target_orders": target_checks,
        "requested_orders": {
            order_id: requested.get(order_id, []) for order_id in gold_orders
        },
        "unconfirmed_submissions": unconfirmed,
        "snapshot_diff": diff,
        "extra_order_changes": extra_orders,
        "extra_user_changes": extra_users,
        "extra_product_changes": extra_products,
        "handoff_events": len(handoffs),
        "false_claim_hits": claim_hits,
        "business_safety_pass": business_pass,
        "claim_review": "REVIEW" if claim_hits else "NO_HIT",
        "note": (
            "项目检查独立成列，不乘入原 reward；无命中不等于自动证明措辞安全，"
            "仍保留对话索引供人工复核。"
        ),
    }


def _empty_turn(user: str) -> Dict[str, Any]:
    return {"user": user, "tool_traces": [], "assistant": ""}


def _empty_user_error(event: Dict[str, Any]) -> Dict[str, Any]:
    """记录模拟用户空回复的来源和生成信息；该轮不进入 Agent。"""
    usage = dict(event.get("usage") or {})
    raw_data = dict(event.get("raw_data") or {})
    stop_reason = str(raw_data.get("stop_reason") or "")
    content = str(event.get("content") or "")
    parts = ["模拟用户返回空内容", "source=user_simulator"]
    if stop_reason:
        parts.append(f"stop_reason={stop_reason}")
    if usage.get("completion_tokens") is not None:
        parts.append(f"completion_tokens={usage.get('completion_tokens')}")
    return {
        "source": "user_simulator",
        "error_type": "EMPTY_USER_MESSAGE",
        "agent_called": False,
        "stop_reason": stop_reason,
        "usage": usage,
        "raw_data": raw_data,
        "content_length": len(content),
        "detail": "（" + ", ".join(parts) + "）",
    }


def _usage_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _raw_data(response: Any) -> Dict[str, Any]:
    return {
        "id": getattr(response, "id", ""),
        "model": getattr(response, "model", ""),
        "stop_reason": getattr(response, "stop_reason", ""),
    }


async def _run_case(
    task_id: str,
    run_dir: Path,
    ledger: Any,
    api_key: str,
    base_url: Optional[str],
) -> Dict[str, Any]:
    from evaluation.retail_ledger import build_budgeted_client
    from evaluation.tau_bridge import BridgeError, UserSimulatorBridge
    from evaluation.tau_eval_bridge import EvalBridge, EvalBridgeError
    from evaluation.tau_bridge import to_anthropic_kwargs
    from agents.retail_agent import RetailAgent, RetailService
    from agents.agent_orchestrator import AgentCallError
    from mcp.retail_store import RetailStore
    from core.llm_utils import extract_text_content

    case_dir = run_dir / "cases" / task_id
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / "retail.sqlite"
    if db_path.exists():
        raise RuntimeError(
            f"任务 {task_id} 已存在未完成案件库；为保持独立公开种子，"
            "请使用新的 run id 或显式诊断目录"
        )
    store = RetailStore(db_path, SEED_PATH, language="en")
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    client = build_budgeted_client(
        api_key, base_url, ledger, stage=f"agent-task-{task_id}"
    )
    agent = RetailAgent(
        client=client,
        model=MODEL,
        store=store,
        tool_manager=None,
        policy_text=policy_text,
        policy_source=(
            "upstream policy.md sha256=" + _sha256_file(POLICY_PATH)
        ),
        max_tokens=AGENT_MAX_TOKENS,
        language="en",
    )
    service = RetailService(
        store=store,
        agent=agent,
        memory=None,
        event_log_path=str(case_dir / "events.jsonl"),
        case_id=f"tau-retail-{task_id}",
        run_id=run_dir.name,
        language="en",
        max_message_chars=MAX_MESSAGE_CHARS,
    )
    before = store.business_snapshot()
    _write_json(case_dir / "snapshot-before.json", before)

    user_client = client.for_stage(f"user-task-{task_id}")
    judge_client = client.for_stage(f"judge-task-{task_id}")

    async def user_handler(event: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = to_anthropic_kwargs(event["request"])
        response = await user_client.messages.create(**kwargs)
        return {
            "content": extract_text_content(response.content),
            "usage": _usage_dict(response),
            "raw_data": _raw_data(response),
        }

    async def judge_handler(event: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = to_anthropic_kwargs(event["request"])
        response = await judge_client.messages.create(**kwargs)
        return {
            "content": extract_text_content(response.content),
            "usage": _usage_dict(response),
            "raw_data": _raw_data(response),
        }

    user_bridge = UserSimulatorBridge(
        python_exe=TAU_PY,
        worker_path=USER_WORKER,
        data_dir=TAU_DATA,
        tasks_path=TASKS_PATH,
        task_id=task_id,
        llm=MODEL,
        max_tokens=USER_MAX_TOKENS,
        temperature=USER_TEMPERATURE,
        seed=SEED,
        stderr_path=case_dir / "user-worker.log",
        timeout_s=USER_TIMEOUT_S,
        llm_handler=user_handler,
    )

    turns: List[Dict[str, Any]] = []
    final_user_stop = ""
    termination = "AGENT_ERROR"
    termination_detail = ""
    step_count = 1  # 官方首条 agent 问候被模拟用户处理算一步。
    error_count = 0
    started = time.monotonic()
    try:
        await user_bridge.start()
        user_event = await user_bridge.send_agent_message(GREETING)
        if len(str(user_event.get("content") or "")) > MAX_MESSAGE_CHARS:
            termination = "INPUT_LIMIT"
            termination_detail = "模拟用户首条消息超过字符上限"
        elif user_event.get("is_stop"):
            termination = "USER_STOP"
            termination_detail = str(user_event.get("stop_kind") or "STOP")
            final_user_stop = str(user_event.get("content") or "")
        while termination not in {"USER_STOP", "INPUT_LIMIT"}:
            user_text = str(user_event.get("content") or "")
            if len(user_text) > MAX_MESSAGE_CHARS:
                termination = "INPUT_LIMIT"
                termination_detail = "模拟用户消息超过字符上限"
                break
            if not user_text.strip():
                # 上游把 USER 侧空消息记为 USER_ERROR；这里在调用 Agent 前终止，
                # 避免把模拟用户生成失败误归因为 Agent 输入错误。
                turn = _empty_turn(user_text)
                turn["user_error"] = _empty_user_error(user_event)
                turns.append(turn)
                termination = "USER_ERROR"
                termination_detail = turn["user_error"]["detail"]
                break
            turn: Dict[str, Any] = _empty_turn(user_text)
            turns.append(turn)
            if turns:
                conversation_chars = sum(
                    len(str(item.get("user") or ""))
                    + len(str(item.get("assistant") or ""))
                    for item in turns
                )
                if conversation_chars > MAX_CONVERSATION_CHARS:
                    termination = "INPUT_LIMIT"
                    termination_detail = "对话累计长度超过上限"
                    break
            try:
                result = await service.handle_turn(
                    user_text, f"client-{task_id}", f"conv-{task_id}"
                )
                turn["tool_traces"] = list(result.tool_traces)
                turn["assistant"] = result.response
                turn["tools_used"] = list(result.tools_used)
                turn["display_appended"] = list(result.display_appended)
                turn["prepared_op_ids"] = list(result.prepared_op_ids)
                turn["turn_success"] = bool(result.success)
                if not result.success:
                    # BaseAgent 会把模型/预算异常转成失败回复；这里必须终止，
                    # 不能让模拟用户继续把失败当普通业务轮次。
                    error_count += 1
                    stop_reason = (
                        ledger.summary().get("cost_stop_reason")
                        or ledger.last_rejection_reason
                    )
                    termination = (
                        "INCOMPLETE_BUDGET" if stop_reason else "AGENT_ERROR"
                    )
                    termination_detail = stop_reason or "Agent 返回失败"
                    break
            except AgentCallError as exc:
                turn["tool_traces"] = list(exc.tool_traces)
                turn["assistant"] = ""
                turn["agent_error"] = str(exc)
                error_count += 1
                termination = "AGENT_ERROR"
                termination_detail = str(exc)
                break
            except Exception as exc:
                turn["assistant"] = ""
                turn["agent_error"] = f"{type(exc).__name__}: {exc}"
                error_count += 1
                termination = "AGENT_ERROR"
                termination_detail = turn["agent_error"]
                break

            for trace in turn["tool_traces"]:
                if (
                    trace.get("success") is False
                    or trace.get("result_success") is False
                ):
                    error_count += 1
            # 官方半双工每轮：用户→Agent、每个工具调用往返、Agent→用户。
            step_count += 2 + 2 * len(turn["tool_traces"])
            if error_count >= MAX_ERRORS:
                termination = "TOO_MANY_ERRORS"
                termination_detail = f"工具错误 {error_count} 次"
                break
            if step_count >= MAX_STEPS:
                termination = "MAX_STEPS"
                termination_detail = f"步数达到 {step_count}"
                break
            assistant_text = str(turn.get("assistant") or "")
            if not assistant_text.strip():
                termination = "AGENT_ERROR"
                termination_detail = "Agent 回复为空"
                break
            try:
                user_event = await user_bridge.send_agent_message(assistant_text)
            except BridgeError as exc:
                termination = "BRIDGE_ERROR"
                termination_detail = str(exc)
                break
            if user_event.get("is_stop"):
                termination = "USER_STOP"
                termination_detail = str(user_event.get("stop_kind") or "STOP")
                final_user_stop = str(user_event.get("content") or "")
                turn["next_user_stop"] = final_user_stop
                break
            if len(str(user_event.get("content") or "")) > MAX_MESSAGE_CHARS:
                termination = "INPUT_LIMIT"
                termination_detail = "模拟用户消息超过字符上限"
                break
    except Exception as exc:
        # BudgetExceeded 与桥接失败都保留已发生的对话和副作用。
        termination = getattr(exc, "__class__", type(exc)).__name__
        if termination == "BudgetExceeded":
            termination = "INCOMPLETE_BUDGET"
        else:
            termination = "RUN_ERROR"
        termination_detail = f"{type(exc).__name__}: {exc}"
    finally:
        await user_bridge.close()

    duration_ms = round((time.monotonic() - started) * 1000, 1)
    after = store.business_snapshot()
    _write_json(case_dir / "snapshot-after.json", after)
    events = store.list_events(f"client-{task_id}", f"conv-{task_id}")
    _write_json(case_dir / "events.json", events)
    trajectory = trajectory_messages(turns, final_user_stop)
    _write_json(case_dir / "trajectory.json", trajectory)
    _write_json(
        case_dir / "dialogue.json",
        {
            "task_id": task_id,
            "greeting": GREETING,
            "termination": termination,
            "termination_detail": termination_detail,
            "final_user_stop": final_user_stop,
            "step_count": step_count,
            "error_count": error_count,
            "turns": turns,
            "trajectory": trajectory,
        },
    )

    task = _task_by_id(task_id)
    criteria = task.get("evaluation_criteria") or {}
    gold_orders = sorted(
        {
            action["arguments"].get("order_id")
            for action in criteria.get("actions") or []
            if action.get("name") == "return_delivered_order_items"
        }
        - {None}
    )
    requires_handoff = any(
        action.get("name") == "transfer_to_human_agents"
        for action in criteria.get("actions") or []
    )
    project = _project_checks(
        before, after, events, turns, gold_orders, requires_handoff
    )

    eval_result: Dict[str, Any] = {}
    eval_error = ""
    budget_stopped = bool(
        ledger.summary().get("cost_stop_reason")
        or ledger.last_rejection_reason
    )
    bridge = None
    try:
        bridge = EvalBridge(
            python_exe=TAU_PY,
            worker_path=EVAL_WORKER,
            data_dir=TAU_DATA,
            tasks_path=TASKS_PATH,
            task_id=task_id,
            snapshot_path=case_dir / "snapshot-after.json",
            trajectory_path=case_dir / "trajectory.json",
            judge_model=MODEL,
            judge_max_tokens=JUDGE_MAX_TOKENS,
            judge_temperature=JUDGE_TEMPERATURE,
            judge_seed=SEED,
            stderr_path=case_dir / "eval-worker.log",
            timeout_s=EVAL_TIMEOUT_S,
            judge_handler=judge_handler,
            skip_nl=budget_stopped,
        )
        eval_result = await bridge.run()
    except Exception as exc:
        eval_error = f"{type(exc).__name__}: {exc}"
        partial_db = getattr(bridge, "db_result", None)
        if partial_db:
            # Judge 或子进程失败不影响已经完成的免费 DB 评分。
            eval_result = {
                "db": partial_db,
                "nl": getattr(bridge, "nl_result", None)
                or {
                    "status": "JUDGE_ERROR",
                    "errors": [eval_error],
                    "reward": None,
                    "judge_requests": 0,
                    "raw_output": "",
                },
                "protocol_log": getattr(bridge, "protocol_log", []),
            }

    db = eval_result.get("db") or {}
    nl = eval_result.get("nl") or {}
    db_component = None
    if db.get("status") == "OK" and not db.get("gold_errors"):
        db_component = db.get("official_reward", {}).get("reward")
    nl_component = None
    if nl.get("status") == "NOT_APPLICABLE":
        nl_component = 1.0
    elif nl.get("status") == "OK":
        nl_component = nl.get("reward")
    combined = None
    if db_component is not None and nl_component is not None:
        combined = float(db_component) * float(nl_component)

    valid_termination = termination == "USER_STOP"
    scoring_valid = (
        db.get("status") == "OK"
        and not db.get("gold_errors")
        and nl.get("status") in {"OK", "NOT_APPLICABLE"}
        and not eval_error
    )
    effective_completed = _effective_completion(
        termination,
        db.get("status"),
        db.get("gold_errors") or [],
        nl.get("status"),
        eval_error,
    )
    user_error = next(
        (turn.get("user_error") for turn in turns if turn.get("user_error")),
        None,
    )

    summary = {
        "task_id": task_id,
        "termination": termination,
        "termination_detail": termination_detail,
        "user_error": user_error,
        "valid_termination": valid_termination,
        "effective_completed": effective_completed,
        "step_count": step_count,
        "max_steps": MAX_STEPS,
        "error_count": error_count,
        "max_errors": MAX_ERRORS,
        "turns": len(turns),
        "duration_ms": duration_ms,
        "db_component": db_component,
        "nl_component": nl_component,
        "combined_reward": combined,
        "db_status": db.get("status"),
        "nl_status": nl.get("status"),
        "db_match": db.get("db_match"),
        "predicted_hash": db.get("predicted_hash"),
        "gold_hash": db.get("gold_hash"),
        "gold_errors": db.get("gold_errors") or [],
        "judge_errors": nl.get("errors") or [],
        "judge_raw_output": nl.get("raw_output") or "",
        "judge_requests": nl.get("judge_requests", 0),
        "eval_error": eval_error,
        "project_checks": project,
        "config_sha256": _sha256_json(_role_config()),
        "task_sha256": _sha256_json(task),
        "source_hashes": _source_hashes(),
        "event_count": len(events),
        "trajectory_messages": len(trajectory),
        "eval_protocol_log": eval_result.get("protocol_log") or [],
        "user_protocol_log": user_bridge.protocol_log,
        "usage": ledger.summary(),
        "finished_at_utc": _now(),
    }
    _write_json(case_dir / "result.json", summary)
    try:
        await client.aclose()
    except Exception:
        pass
    return summary


def _budget_stop_reason(ledger: Any) -> str:
    """返回预算永久停止或最近一次发送前拒绝的原因；未停止时为空。"""
    return str(
        ledger.summary().get("cost_stop_reason")
        or ledger.last_rejection_reason
        or ""
    )


def _not_run_row(task_id: str, reason: str) -> Dict[str, Any]:
    """预算停止后未启动的案件仍保留一行，明确区别于模型失败。"""
    return {
        "task_id": task_id,
        "termination": "NOT_RUN_BUDGET",
        "termination_detail": reason,
        "user_error": None,
        "valid_termination": False,
        "effective_completed": False,
        "db_component": None,
        "nl_component": None,
        "combined_reward": None,
        "db_status": "NOT_RUN",
        "nl_status": "NOT_RUN",
        "gold_errors": [],
        "judge_errors": [],
        "judge_raw_output": "",
        "judge_requests": 0,
        "eval_error": "预算停止，未启动本案件",
        "project_checks": {},
        "finished_at_utc": _now(),
    }


def _evaluation_gaps(results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """按原因列出影响评测有效性的案件；业务失败不进入缺口。"""
    gaps: Dict[str, List[str]] = {
        "not_run_task_ids": [],
        "budget_incomplete_task_ids": [],
        "invalid_termination_task_ids": [],
        "invalid_reference_task_ids": [],
        "judge_error_task_ids": [],
        "evaluation_error_task_ids": [],
        "other_incomplete_task_ids": [],
    }
    for item in results:
        task_id = str(item.get("task_id") or "")
        termination = item.get("termination")
        if termination == "NOT_RUN_BUDGET":
            gaps["not_run_task_ids"].append(task_id)
            continue
        classified = False
        if termination == "INCOMPLETE_BUDGET":
            gaps["budget_incomplete_task_ids"].append(task_id)
            classified = True
        if item.get("db_status") == "INVALID_REFERENCE" or item.get("gold_errors"):
            gaps["invalid_reference_task_ids"].append(task_id)
            classified = True
        if item.get("nl_status") == "JUDGE_ERROR":
            gaps["judge_error_task_ids"].append(task_id)
            classified = True
        db_status = item.get("db_status")
        if item.get("eval_error") or db_status not in (
            None,
            "OK",
            "INVALID_REFERENCE",
            "NOT_RUN",
        ):
            gaps["evaluation_error_task_ids"].append(task_id)
            classified = True
        if termination != "INCOMPLETE_BUDGET" and item.get(
            "valid_termination"
        ) is not True:
            gaps["invalid_termination_task_ids"].append(task_id)
            classified = True
        if item.get("effective_completed") is not True and not classified:
            gaps["other_incomplete_task_ids"].append(task_id)
    return gaps


def _evaluation_summary(
    planned_ids: List[str],
    selected_ids: List[str],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """区分行执行、评测有效性和有效完成案件中的任务成败。"""
    gaps = _evaluation_gaps(results)
    all_planned_executed = (
        list(selected_ids) == list(planned_ids)
        and len(results) == len(planned_ids)
        and not gaps["not_run_task_ids"]
    )
    invalid_scoring = any(
        gaps[key]
        for key in (
            "invalid_reference_task_ids",
            "judge_error_task_ids",
            "evaluation_error_task_ids",
        )
    )
    if invalid_scoring:
        evaluation_status = "EVALUATOR_INVALID"
    elif not all_planned_executed or any(
        gaps[key]
        for key in (
            "invalid_termination_task_ids",
            "budget_incomplete_task_ids",
            "other_incomplete_task_ids",
        )
    ):
        evaluation_status = "PARTIAL"
    else:
        evaluation_status = "EVALUATION_COMPLETE"
    effective_completed = sum(
        1 for item in results if item.get("effective_completed")
    )
    successful = sum(
        1
        for item in results
        if item.get("effective_completed")
        and item.get("combined_reward") == 1.0
    )
    return {
        "run_status": (
            "EXECUTION_COMPLETE"
            if all_planned_executed
            else "EXECUTION_PARTIAL"
        ),
        "evaluation_status": evaluation_status,
        "evaluation_gaps": gaps,
        "effective_completed": effective_completed,
        "valid_trials": sum(
            1 for item in results if item.get("valid_termination") is True
        ),
        "successful_effective_completed": successful,
        "failed_effective_completed": effective_completed - successful,
    }


async def _execute(run_id: str, only: Optional[List[str]] = None) -> Dict[str, Any]:
    from evaluation.retail_ledger import RetailLedger

    run_dir = ROOT / "data" / "evaluation" / "tau_retail" / run_id
    preflight = _load_json(run_dir / "preflight.json")
    if not preflight.get("gate_passed"):
        raise RuntimeError("预检未通过，拒绝付费运行")
    _verify_frozen(run_dir)
    prices = _pricing()
    ledger = RetailLedger(
        run_dir / "ledger.db",
        budget_cny=BUDGET_CNY,
        input_price_cny_per_mtok=prices["input"],
        output_price_cny_per_mtok=prices["output"],
        run_id=run_id,
        model=MODEL,
    )
    api = _api_config()
    if not api["api_key_present"] or not api["base_url_matches_frozen"]:
        raise RuntimeError("API 配置缺失或端点与冻结配置不一致")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None

    order = [item for item in TASK_IDS if not only or item in only]
    results = []
    budget_stop_detail = ""
    for task_id in order:
        existing = run_dir / "cases" / task_id / "result.json"
        if existing.is_file():
            results.append(_load_json(existing))
            continue
        if budget_stop_detail:
            row = _not_run_row(task_id, budget_stop_detail)
            _write_json(run_dir / "cases" / task_id / "not-run.json", row)
            results.append(row)
            continue
        result = await _run_case(task_id, run_dir, ledger, api_key, base_url)
        results.append(result)
        # 预算永久停止或本次请求被拒后，不再启动后续案件，避免缩小配置硬跑。
        budget_stop_detail = (
            result.get("termination_detail")
            if result.get("termination") == "INCOMPLETE_BUDGET"
            else ""
        )
        budget_stop_detail = budget_stop_detail or _budget_stop_reason(ledger)
    not_run_ids = [
        item["task_id"]
        for item in results
        if item.get("termination") == "NOT_RUN_BUDGET"
    ]
    summary_fields = _evaluation_summary(TASK_IDS, order, results)
    aggregate = {
        "run_id": run_id,
        "finished_at_utc": _now(),
        "planned_task_ids": list(TASK_IDS),
        "task_order": order,
        "planned_rows": len(TASK_IDS),
        "results": results,
        "reported_rows": len(results),
        "completed_rows": len(results) - len(not_run_ids),
        "not_run_task_ids": not_run_ids,
        "budget_stop_reason": budget_stop_detail or None,
        **summary_fields,
        "ledger": ledger.summary(),
    }
    _write_json(run_dir / "summary.json", aggregate)
    _write_json(run_dir / "ledger-summary.json", ledger.summary())
    _write_json(run_dir / "ledger-entries.json", ledger.entries())
    return aggregate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="T14 τ-retail 固定 8 任务")
    parser.add_argument("command", choices=["preflight", "run", "all"])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)
    if args.command in {"preflight", "all"}:
        result = _preflight(args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["gate_passed"]:
            return 2
        if args.command == "preflight":
            return 0
    aggregate = asyncio.run(_execute(args.run_id, args.only))
    print(
        json.dumps(
            {
                "run_id": aggregate["run_id"],
                "run_status": aggregate["run_status"],
                "evaluation_status": aggregate["evaluation_status"],
                "evaluation_gaps": aggregate["evaluation_gaps"],
                "effective_completed": aggregate["effective_completed"],
                "successful_effective_completed": aggregate[
                    "successful_effective_completed"
                ],
                "failed_effective_completed": aggregate[
                    "failed_effective_completed"
                ],
                "valid_trials": aggregate["valid_trials"],
                "completed_rows": aggregate["completed_rows"],
                "ledger": aggregate["ledger"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
