"""T16 单 Agent 与现有多 Agent 零售适配版的成对比较入口。

正式运行固定 7 个 τ-retail 任务、共 14 个 trial，所有 arm／角色／
识别／Composer／模拟用户／Judge 请求共用同一个 T16 持久化账本。
任务 46 只做零费用离线参考核验，不进入正式分母。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_tau_retail import (  # noqa: E402
    EVAL_TIMEOUT_S,
    EVAL_WORKER,
    GREETING,
    MAX_CONVERSATION_CHARS,
    MAX_ERRORS,
    MAX_MESSAGE_CHARS,
    MAX_STEPS,
    OFFICIAL_DB_PATH,
    POLICY_PATH,
    REQUIREMENTS_FREEZE,
    SEED_PATH,
    TASKS_PATH,
    TAU_DATA,
    TAU_DIR,
    TAU_PY,
    USER_TIMEOUT_S,
    USER_WORKER,
    _effective_completion,
    _empty_turn,
    _empty_user_error,
    _load_json,
    _load_tasks,
    _now,
    _project_checks,
    _raw_data,
    _sha256_file,
    _sha256_json,
    _snapshot_diff,
    _task_by_id,
    _tau_environment,
    _usage_dict,
    _write_json,
    trajectory_messages,
)

TASK_IDS = ["11", "12", "48", "51", "83", "92", "108"]
EXCLUDED_TASK_IDS = ["46"]
RUN_ORDER = [
    "11A", "11B", "12B", "12A", "48A", "48B", "51B", "51A",
    "83A", "83B", "92B", "92A", "108A", "108B",
]
ARMS = ("A", "B")
MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/anthropic"
BUDGET_CNY = 10.0
PRICING_PATH = ROOT / "evaluation" / "data" / "retail_live" / "pricing.json"
SEED = 300
MAX_STEPS = 200
MAX_ERRORS = 10
CONCURRENCY = 1
TRIALS = 1
AGENT_MAX_TOKENS = 2048
AGENT_TEMPERATURE = 0.1
INTENT_MAX_TOKENS = 2048
INTENT_TEMPERATURE = 0.1
COMPOSER_MAX_TOKENS = 8192
COMPOSER_TEMPERATURE = 0.1
USER_MAX_TOKENS = 1024
USER_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024
JUDGE_TEMPERATURE = 0.0
MAX_REQUESTS_PER_TURN = 32
MCP_TIMEOUT_S = 60.0
DEFAULT_RUN_ID = "t16-retail-20261020"
UPSTREAM_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

PLANNED_CASE_CALLS = {
    "A": {"agent": 20, "user": 14},
    "B": {"roles": 22, "intent": 10, "composer": 4, "user": 14},
}
PLANNED_OUTPUT_TOKENS = {
    "agent": 512,
    "roles": 512,
    "intent": 256,
    "composer": 1024,
    "user": 256,
    "judge": 256,
}

SOURCE_FILES = (
    "agents/agent_orchestrator.py",
    "agents/retail_agent.py",
    "agents/retail_multi_agent.py",
    "agents/retail_runtime.py",
    "agents/retail_tools.py",
    "api/main.py",
    "core/intent_recognizer.py",
    "mcp/retail_client.py",
    "mcp/retail_protocol.py",
    "mcp/retail_server.py",
    "mcp/retail_store.py",
    "evaluation/retail_budget.py",
    "evaluation/retail_ledger.py",
    "evaluation/tau_bridge.py",
    "evaluation/tau_user_worker.py",
    "evaluation/tau_eval_bridge.py",
    "evaluation/tau_eval_worker.py",
    "evaluation/run_t16_retail.py",
    "evaluation/data/retail_live/pricing.json",
    "evaluation/data/retail_sandbox/db.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/tasks.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/policy.md",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/db.json",
    "evaluation/data/tau_retail/requirements-freeze.txt",
)
ASSET_HASHES = {
    "evaluation/data/retail_live/pricing.json",
    "evaluation/data/retail_sandbox/db.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/tasks.json",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/policy.md",
    "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/db.json",
    "evaluation/data/tau_retail/requirements-freeze.txt",
}


def _pricing() -> Dict[str, Any]:
    data = _load_json(PRICING_PATH)
    peak = data.get("peak") or {}
    return {
        "data": data,
        "input": float(peak.get("input_cache_miss") or 0),
        "output": float(peak.get("output") or 0),
    }


def _source_hashes() -> Dict[str, str]:
    result: Dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        result[relative] = _sha256_file(path) if path.is_file() else "<missing>"
    return result


def _role_config() -> Dict[str, Any]:
    return {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "max_errors": MAX_ERRORS,
        "concurrency": CONCURRENCY,
        "trials": TRIALS,
        "arms": {
            "A": "current single RetailAgent with MCP tools",
            "B": "existing orchestrator retail adaptation with 4 roles",
        },
        "agent": {
            "max_tokens": AGENT_MAX_TOKENS,
            "temperature": AGENT_TEMPERATURE,
            "max_tool_rounds": 10,
            "max_tool_uses_per_response": 1,
            "language": "en",
            "memory": None,
            "knowledge": "disabled",
        },
        "roles": {
            "max_supporting": 2,
            "max_tokens": AGENT_MAX_TOKENS,
            "temperature": AGENT_TEMPERATURE,
            "max_tool_rounds": 10,
        },
        "intent": {
            "max_tokens": INTENT_MAX_TOKENS,
            "temperature": INTENT_TEMPERATURE,
        },
        "composer": {
            "max_tokens": COMPOSER_MAX_TOKENS,
            "temperature": COMPOSER_TEMPERATURE,
        },
        "user": {
            "implementation": "official UserSimulator",
            "max_tokens": USER_MAX_TOKENS,
            "temperature": USER_TEMPERATURE,
            "notice": "T14 used 512; T16 raises both arms to 1024 before results.",
        },
        "judge": {
            "implementation": "official NLAssertionsEvaluator",
            "max_tokens": JUDGE_MAX_TOKENS,
            "temperature": JUDGE_TEMPERATURE,
            "tasks_called": ["108"],
        },
        "per_turn_model_request_limit": MAX_REQUESTS_PER_TURN,
        "limits": {
            "max_message_chars": MAX_MESSAGE_CHARS,
            "max_conversation_chars": MAX_CONVERSATION_CHARS,
            "user_timeout_s": USER_TIMEOUT_S,
            "eval_timeout_s": EVAL_TIMEOUT_S,
        },
        "budget_cny": BUDGET_CNY,
    }


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


def _first_request_upper(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    message: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from evaluation.retail_budget import request_input_upper_tokens

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": message}],
    }
    if tools:
        kwargs["tools"] = tools
    upper = request_input_upper_tokens(kwargs)
    size = len(
        json.dumps(
            kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    prices = _pricing()
    hard_cny = (
        upper * prices["input"] / 1_000_000
        + max_tokens * prices["output"] / 1_000_000
    )
    return {
        "request_bytes": size,
        "input_tokens_proxy": max(1, (size + 2) // 3),
        "input_tokens_upper": upper,
        "hard_upper_cny": round(hard_cny, 6),
        "tool_count": len(tools or []),
    }


def _tool_schema(specs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in specs.values()
    ]


def _measure_user_prompt_bytes() -> int:
    """在官方环境构造模拟用户首个请求，只测量字节数，不调用模型。"""
    import subprocess

    script = r"""
import json, os, sys
os.environ["TAU2_DATA_DIR"] = sys.argv[1]
from loguru import logger
logger.remove()
from tau2.data_model.tasks import Task
from tau2.user.user_simulator import UserSimulator
from tau2.utils.llm_utils import to_litellm_messages
raw = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
values = {}
for task_id in sys.argv[3:]:
    item = next(x for x in raw if str(x.get("id")) == str(task_id))
    task = Task.model_validate(item)
    simulator = UserSimulator(
        llm="deepseek-flash",
        instructions=str(task.user_scenario),
        tools=None,
        llm_args={"temperature": 0.0, "max_tokens": 1024},
        persona_config=None,
    )
    state = simulator.get_init_state()
    messages = to_litellm_messages(state.system_messages + state.flip_roles())
    messages.append({"role": "assistant", "content": "Hi! How can I help you today?"})
    values[str(task_id)] = len(
        json.dumps(messages, ensure_ascii=False).encode("utf-8")
    )
print(json.dumps(values))
"""
    result = subprocess.run(
        [str(TAU_PY), "-c", script, str(TAU_DATA), str(TASKS_PATH), *TASK_IDS],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"测量模拟用户请求失败: {result.stderr[-1000:]}"
        )
    values = json.loads(result.stdout.strip().splitlines()[-1])
    return int(max(values.values()))


def _measure_arm_requests(policy_text: str) -> Dict[str, Any]:
    """用真实 Prompt／工具定义测量两组的首个请求和 B 的各阶段上界。"""
    import tempfile

    from agents.agent_orchestrator import Request
    from agents.retail_agent import RetailAgent
    from agents.retail_multi_agent import (
        RetailBillingAgent,
        RetailEscalationAgent,
        RetailGeneralAgent,
        RetailTechnicalAgent,
    )
    from agents.retail_tools import retail_tools
    from core.intent_recognizer import build_intent_prompt
    from mcp.retail_store import RetailStore

    tmp = Path(tempfile.mkdtemp(prefix="t16-preflight-"))
    try:
        store = RetailStore(tmp / "preflight.sqlite", SEED_PATH, language="en")
        tools = retail_tools(store, None, language="en")
        agent = RetailAgent(
            client=None,
            model=MODEL,
            store=store,
            tool_manager=None,
            policy_text=policy_text,
            policy_source=(
                "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/"
                f"policy.md sha256={_sha256_file(POLICY_PATH)}"
            ),
            max_tokens=AGENT_MAX_TOKENS,
            language="en",
        )
        request = Request(
            message="Hello", user_id="preflight-client", conv_id="preflight-conv", turn=1
        )
        a = _first_request_upper(
            model=MODEL,
            max_tokens=AGENT_MAX_TOKENS,
            temperature=AGENT_TEMPERATURE,
            system=agent._build_system_prompt(request),
            message="Hello",
            tools=_tool_schema(tools),
        )
        roles = {}
        for name, cls in (
            ("general", RetailGeneralAgent),
            ("technical", RetailTechnicalAgent),
            ("billing", RetailBillingAgent),
            ("escalation", RetailEscalationAgent),
        ):
            role = cls(
                None,
                MODEL,
                language="en",
                policy_text=policy_text,
            )
            roles[name] = _first_request_upper(
                model=MODEL,
                max_tokens=AGENT_MAX_TOKENS,
                temperature=AGENT_TEMPERATURE,
                system=role._build_system_prompt(request),
                message="Hello",
                tools=_tool_schema(tools),
            )
        intent = _first_request_upper(
            model=MODEL,
            max_tokens=INTENT_MAX_TOKENS,
            temperature=INTENT_TEMPERATURE,
            system="",
            message=build_intent_prompt("Hello", None, language="en"),
        )
        composer = _first_request_upper(
            model=MODEL,
            max_tokens=COMPOSER_MAX_TOKENS,
            temperature=COMPOSER_TEMPERATURE,
            system="",
            message=(
                "Primary agent: general\nUser message: Hello\n"
                "Candidate results:\n[general Agent output]\nHello\n\n"
                "[billing Agent output]\nHello"
            ),
        )
        user_initial = _measure_user_prompt_bytes()
        # 计划口径：官方首个请求之上，预留 3 个来回（助手 1500 字符、
        # 用户 300 字符）；账本仍按每个真实请求的完整字节数预留。
        user_planned_bytes = user_initial + 3 * (1500 + 300 + 40)
        user = {
            "request_bytes": user_initial,
            "planned_average_request_bytes": user_planned_bytes,
            "input_tokens_proxy": max(1, (user_initial + 2) // 3),
            "planned_average_input_tokens_proxy": max(
                1, (user_planned_bytes + 2) // 3
            ),
        }
        return {
            "A": {"agent": a, "user": user},
            "B": {
                "roles": roles,
                "intent": intent,
                "composer": composer,
                "user": user,
            },
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _planned_case_cost(
    arm: str, task_id: str, measurements: Dict[str, Any], prices: Dict[str, Any]
) -> Dict[str, Any]:
    """按冻结的字节数/3 输入口径和计划输出估算单案费用。"""

    calls = PLANNED_CASE_CALLS[arm]
    rows: Dict[str, Any] = {}
    total = 0.0
    for role, count in calls.items():
        if role == "user":
            basis = measurements["A"]["user"]
            input_proxy = basis["planned_average_input_tokens_proxy"]
        elif arm == "A":
            input_proxy = measurements["A"]["agent"]["input_tokens_proxy"]
        elif role == "roles":
            input_proxy = max(
                item["input_tokens_proxy"]
                for item in measurements["B"]["roles"].values()
            )
        else:
            input_proxy = measurements["B"][role]["input_tokens_proxy"]
        output = PLANNED_OUTPUT_TOKENS[role]
        cost = count * (
            input_proxy * prices["input"] / 1_000_000
            + output * prices["output"] / 1_000_000
        )
        rows[role] = {
            "calls": count,
            "input_tokens_proxy_each": input_proxy,
            "planned_output_tokens_each": output,
            "planned_cny": round(cost, 6),
        }
        total += cost
    if task_id == "108":
        # Judge 输入是完整轨迹与断言；在角色输入之外再加 6000 字节轨迹余量。
        base_input = max(
            measurements["A"]["agent"]["input_tokens_proxy"],
            measurements["B"]["roles"]["general"]["input_tokens_proxy"],
        )
        judge_input = base_input + 2000
        judge_cost = (
            judge_input * prices["input"] / 1_000_000
            + PLANNED_OUTPUT_TOKENS["judge"] * prices["output"] / 1_000_000
        )
        rows["judge"] = {
            "calls": 1,
            "input_tokens_proxy_each": judge_input,
            "planned_output_tokens_each": PLANNED_OUTPUT_TOKENS["judge"],
            "planned_cny": round(judge_cost, 6),
        }
        total += judge_cost
    return {"rows": rows, "planned_cny": round(total, 6)}


def _planned_estimate(
    measurements: Dict[str, Any], prices: Dict[str, Any]
) -> Dict[str, Any]:
    cases = {}
    total = 0.0
    for arm in ARMS:
        arm_total = 0.0
        for task_id in TASK_IDS:
            item = _planned_case_cost(arm, task_id, measurements, prices)
            cases[f"{task_id}{arm}"] = item
            arm_total += item["planned_cny"]
        total += arm_total
        cases[f"{arm}_total"] = {"planned_cny": round(arm_total, 6)}
    worst_per_turn = max(
        max(
            item["hard_upper_cny"]
            for item in measurements["B"]["roles"].values()
        ),
        measurements["A"]["agent"]["hard_upper_cny"],
        measurements["B"]["intent"]["hard_upper_cny"],
        measurements["B"]["composer"]["hard_upper_cny"],
    )
    return {
        "cases": cases,
        "planned_total_cny": round(total, 6),
        "gate_total_cny": round(BUDGET_CNY * 0.9, 6),
        "worst_single_request_upper_cny": round(worst_per_turn, 6),
        "worst_turn_32_requests_cny": round(
            worst_per_turn * MAX_REQUESTS_PER_TURN, 6
        ),
        "note": (
            "输入按冻结口径的请求体字节数/3 估算；账本仍按完整请求体字节数"
            "+512 和真实 max_tokens 逐请求硬预留。最坏单轮 32 次请求列作风险，"
            "不等于计划总额。"
        ),
    }


def _verify_task_46_offline(run_dir: Path) -> Dict[str, Any]:
    """在独立 SQLite 执行 46 的只读参考动作，保留已知错误订单号。"""
    from mcp.retail_store import RetailStore, normalize_order_id, safe_calculate

    target_dir = run_dir / "excluded" / "46"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / "reference.sqlite"
    store = RetailStore(db_path, SEED_PATH, language="en")
    client_user = "offline-46"
    conv = "offline-46"
    store.begin_turn(client_user, conv)

    task = _task_by_id("46")
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    results: List[Dict[str, Any]] = []
    for action in actions:
        name = str(action.get("name") or "")
        args = dict(action.get("arguments") or {})
        entry: Dict[str, Any] = {"action_id": action.get("action_id"), "name": name}
        try:
            if name == "find_user_id_by_name_zip":
                entry["result"] = store.find_user_by_name_zip(
                    client_user,
                    conv,
                    str(args.get("first_name") or ""),
                    str(args.get("last_name") or ""),
                    str(args.get("zip") or ""),
                )
            elif name == "get_order_details":
                order_id = str(args.get("order_id") or "")
                entry["order_id"] = order_id
                entry["normalized_order_id"] = normalize_order_id(order_id)
                entry["result"] = store.get_order_details(
                    client_user, conv, order_id
                )
            elif name == "calculate":
                entry["expression"] = args.get("expression")
                entry["result"] = safe_calculate(
                    str(args.get("expression") or ""), language="en"
                )
            elif name == "return_delivered_order_items":
                # 正式提交需要同会话准备、展示和后续用户确认；离线参考核验
                # 不绕过该业务边界，只保留原动作参数。
                entry["result"] = {
                    "success": False,
                    "error": (
                        "NOT_EXECUTED_OFFLINE: project confirmation boundary "
                        "requires prepare/display and a later real user confirmation"
                    ),
                }
                entry["boundary"] = "confirmation_required"
            else:
                entry["result"] = {
                    "success": False,
                    "error": f"unsupported offline reference action: {name}",
                }
        except Exception as exc:
            entry["result"] = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(entry)

    referenced_orders = sorted(
        {
            str(action["arguments"].get("order_id"))
            for action in actions
            if action.get("name") in {
                "get_order_details",
                "return_delivered_order_items",
            }
        }
    )
    missing_orders = []
    with store._connect() as connection:
        for order_id in referenced_orders:
            normalized = normalize_order_id(order_id)
            row = connection.execute(
                "SELECT order_id FROM orders WHERE order_id=?", (normalized,)
            ).fetchone()
            if row is None:
                missing_orders.append(normalized)
    payload = {
        "task_id": "46",
        "status": "EXCLUDED_INVALID_REFERENCE",
        "reason": (
            "T14 fixed contract: gold actions reference order ids absent from "
            "the official seed; the task stays outside the paid denominator."
        ),
        "referenced_order_ids": referenced_orders,
        "missing_gold_order_ids": missing_orders,
        "actions": results,
        "model_requests": 0,
        "cost_cny": 0.0,
        "reward_basis_unchanged": criteria.get("reward_basis"),
        "gold_actions_unchanged": actions,
        "checked_at_utc": _now(),
    }
    _write_json(target_dir / "offline-verification.json", payload)
    _write_json(
        target_dir / "snapshot.json", store.business_snapshot()
    )
    return payload


def _preflight(run_id: str) -> Dict[str, Any]:
    prices = _pricing()
    if prices["input"] <= 0 or prices["output"] <= 0:
        raise RuntimeError("冻结价目缺少有效的输入/输出价格")
    run_dir = ROOT / "data" / "evaluation" / "t16-retail" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tau_env = _tau_environment()
    task_facts = []
    for task_id in TASK_IDS:
        raw = _task_by_id(task_id)
        criteria = raw.get("evaluation_criteria") or {}
        task_facts.append(
            {
                "task_id": task_id,
                "task_sha256": _sha256_json(raw),
                "reward_basis": criteria.get("reward_basis"),
                "nl_assertion_count": len(criteria.get("nl_assertions") or []),
                "gold_action_count": len(criteria.get("actions") or []),
            }
        )
    nl_tasks = [item["task_id"] for item in task_facts if item["nl_assertion_count"] > 0]
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    measurements = _measure_arm_requests(policy_text)
    estimate = _planned_estimate(measurements, prices)
    api = _api_config()

    mcp_check: Dict[str, Any] = {"ok": False}
    try:
        from agents.retail_tools import retail_tools
        from mcp.retail_client import RetailMCPClient
        from mcp.retail_store import RetailStore

        preflight_dir = run_dir / "preflight"
        if preflight_dir.exists():
            shutil.rmtree(preflight_dir)
        preflight_dir.mkdir(parents=True, exist_ok=True)
        store = RetailStore(
            preflight_dir / "preflight.sqlite", SEED_PATH, language="en"
        )

        async def _check() -> Dict[str, Any]:
            client = RetailMCPClient(
                db_path=preflight_dir / "preflight.sqlite",
                seed_path=SEED_PATH,
                language="en",
                expected_specs=retail_tools(store, None, language="en"),
                project_root=ROOT,
                python_executable=sys.executable,
                startup_timeout=MCP_TIMEOUT_S,
                call_timeout=MCP_TIMEOUT_S,
                stderr_log_path=preflight_dir / "mcp.log",
            )
            started = time.monotonic()
            try:
                await client.start()
                discovered = sorted(client.tools())
                return {
                    "ok": True,
                    "startup_ms": round((time.monotonic() - started) * 1000, 1),
                    "tools": discovered,
                }
            finally:
                await client.close()

        mcp_check = asyncio.run(_check())
    except Exception as exc:
        mcp_check = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    offline_46 = _verify_task_46_offline(run_dir)
    env_doc = (
        TAU_DIR / "ENVIRONMENT.md"
    ).read_text(encoding="utf-8") if (TAU_DIR / "ENVIRONMENT.md").is_file() else ""
    checks = {
        "venv_tau_present": TAU_PY.is_file(),
        "tau_import_ok": bool(tau_env.get("ok")),
        "upstream_commit_recorded": UPSTREAM_COMMIT in env_doc,
        "task_set_matches": [item["task_id"] for item in task_facts] == TASK_IDS,
        "nl_only_task_108": nl_tasks == ["108"],
        "seed_semantic_hash_equal": bool(tau_env.get("official_seed_hash"))
        and tau_env.get("project_seed_semantic_hash")
        == tau_env.get("official_seed_hash"),
        "api_key_present": api["api_key_present"],
        "endpoint_matches": api["base_url_matches_frozen"],
        "pricing_positive": prices["input"] > 0 and prices["output"] > 0,
        "planned_within_budget": estimate["planned_total_cny"]
        <= estimate["gate_total_cny"],
        "worst_turn_within_budget": estimate["worst_turn_32_requests_cny"]
        <= BUDGET_CNY * 0.5,
        "mcp_startup_ok": bool(mcp_check.get("ok")),
        "mcp_nine_tools": len(mcp_check.get("tools") or []) == 9,
        "requirements_freeze_present": REQUIREMENTS_FREEZE.is_file(),
        "excluded_46_invalid_reference": bool(
            offline_46.get("missing_gold_order_ids")
        ),
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
        "prices_cny_per_mtok": {
            "input": prices["input"],
            "output": prices["output"],
        },
        "pricing_file": str(PRICING_PATH.relative_to(ROOT)),
        "pricing_checked_at_utc": prices["data"].get("checked_at_utc"),
        "pricing_source": prices["data"].get("source_zh"),
        "current_price_check": {
            # 程序不抓取官网，这里是实施前人工核对后写入的常量记录。
            "checked_at_utc": _now(),
            "verification": "实施前人工核对官方价目页，非程序抓取",
            "source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
            "endpoint_source": (
                "https://api-docs.deepseek.com/zh-cn/guides/anthropic_api"
            ),
            "model": MODEL,
            "model_version": "DeepSeek-V4.1-Flash-1017",
            "peak_cny_per_mtok": {
                "input_cache_hit": 0.04,
                "input_cache_miss": 2.0,
                "output": 8.0,
            },
            "off_peak_cny_per_mtok": {
                "input_cache_hit": 0.02,
                "input_cache_miss": 1.0,
                "output": 4.0,
            },
            "matches_frozen_pricing": True,
            "endpoint_matches": ENDPOINT,
        },
        "role_config": _role_config(),
        "task_facts": task_facts,
        "task_order": list(TASK_IDS),
        "run_order": list(RUN_ORDER),
        "excluded_task_ids": list(EXCLUDED_TASK_IDS),
        "estimate": estimate,
        "initial_request_measurements": measurements,
        "mcp_preflight": mcp_check,
        "tau_environment": tau_env,
        "api": {key: value for key, value in api.items() if key != "api_key"},
        "seed_sha256": _sha256_file(SEED_PATH),
        "official_db_sha256": _sha256_file(OFFICIAL_DB_PATH),
        "source_hashes": _source_hashes(),
        "asset_hashes": {
            relative: _sha256_file(ROOT / relative)
            for relative in sorted(ASSET_HASHES)
            if (ROOT / relative).is_file()
        },
        "excluded_46": offline_46,
        "planned_risk": (
            "若实际对话或角色调用远超计划，账本会在每次发送前停止并保留 PARTIAL；"
            "不降低输出上限、不删任务、不另开账本。"
        ),
    }
    _write_json(run_dir / "preflight.json", preflight)
    _write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "created_at_utc": _now(),
            "source_hashes": preflight["source_hashes"],
            "asset_hashes": preflight["asset_hashes"],
            "role_config": preflight["role_config"],
            "task_order": list(TASK_IDS),
            "run_order": list(RUN_ORDER),
            "excluded_task_ids": list(EXCLUDED_TASK_IDS),
        },
    )
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


def _case_dir(run_dir: Path, arm: str, task_id: str) -> Path:
    return run_dir / "arms" / arm / "cases" / task_id / "trial-1"


async def _run_case(
    task_id: str,
    arm: str,
    run_dir: Path,
    ledger: Any,
    api_key: str,
    base_url: Optional[str],
) -> Dict[str, Any]:
    from agents.agent_orchestrator import AgentOrchestrator, ResponseComposer
    from agents.retail_agent import RetailAgent, RetailService
    from agents.retail_multi_agent import (
        RetailBillingAgent,
        RetailEscalationAgent,
        RetailGeneralAgent,
        RetailMultiAgent,
        RetailTechnicalAgent,
    )
    from agents.retail_tools import retail_tools
    from core.intent_recognizer import IntentRecognizer
    from core.llm_utils import extract_text_content
    from evaluation.retail_ledger import build_budgeted_client
    from evaluation.tau_bridge import BridgeError, UserSimulatorBridge, to_anthropic_kwargs
    from evaluation.tau_eval_bridge import EvalBridge
    from mcp.retail_client import RetailMCPClient
    from mcp.retail_store import RetailStore

    tag = f"{task_id}{arm}"
    case_dir = _case_dir(run_dir, arm, task_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    db_path = case_dir / "retail.sqlite"
    if db_path.exists():
        raise RuntimeError(
            f"{tag} 已有案件数据库；正式 trial 不重跑，请使用新 run id。"
        )
    store = RetailStore(db_path, SEED_PATH, language="en")
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    policy_source = (
        "evaluation/data/tau_retail/upstream_data/tau2/domains/retail/"
        f"policy.md sha256={_sha256_file(POLICY_PATH)}"
    )
    base_client = build_budgeted_client(
        api_key, base_url, ledger, stage=f"{tag}-agent"
    )
    service = None
    agent: Any = None
    mcp_client: Optional[RetailMCPClient] = None
    mcp_startup_ms = 0.0
    service_closed = False
    try:
        if arm == "A":
            agent = RetailAgent(
                client=base_client.for_stage(f"{tag}-agent", count_turn=True),
                model=MODEL,
                store=store,
                tool_manager=None,
                policy_text=policy_text,
                policy_source=policy_source,
                max_tokens=AGENT_MAX_TOKENS,
                language="en",
            )
        else:
            role_clients = {
                name: base_client.for_stage(
                    f"{tag}-roles-{name}", count_turn=True
                )
                for name in ("general", "technical", "billing", "escalation")
            }
            orchestrator = AgentOrchestrator(
                client=base_client.for_stage(f"{tag}-roles", count_turn=True),
                model=MODEL,
                intent_recognizer=IntentRecognizer(
                    model=MODEL,
                    client=base_client.for_stage(
                        f"{tag}-intent", count_turn=True
                    ),
                    language="en",
                ),
                composer=ResponseComposer(
                    base_client.for_stage(f"{tag}-composer", count_turn=True),
                    MODEL,
                    None,
                    language="en",
                    retail_mode=True,
                ),
                language="en",
                retail_mode=True,
                policy_text=policy_text,
                retail_role_classes={
                    "general": RetailGeneralAgent,
                    "technical": RetailTechnicalAgent,
                    "billing": RetailBillingAgent,
                    "escalation": RetailEscalationAgent,
                },
                retail_role_clients=role_clients,
            )
            agent = RetailMultiAgent(
                orchestrator,
                ledger=ledger,
                turn_label=tag,
                max_requests_per_turn=MAX_REQUESTS_PER_TURN,
                language="en",
            )

        mcp_client = RetailMCPClient(
            db_path=db_path,
            seed_path=SEED_PATH,
            language="en",
            expected_specs=retail_tools(store, None, language="en"),
            project_root=ROOT,
            python_executable=sys.executable,
            startup_timeout=MCP_TIMEOUT_S,
            call_timeout=MCP_TIMEOUT_S,
            stderr_log_path=case_dir / "mcp.log",
        )
        mcp_started = time.monotonic()
        await mcp_client.start()
        mcp_startup_ms = round((time.monotonic() - mcp_started) * 1000, 1)
        tools = mcp_client.tools()
        if set(tools) != set(retail_tools(store, None, language="en")):
            raise RuntimeError(f"{tag} MCP 工具白名单不一致")
        agent.set_shared_tools(tools)

        service = RetailService(
            store=store,
            agent=agent,
            memory=None,
            event_log_path=str(case_dir / "events.jsonl"),
            case_id=tag,
            run_id=run_dir.name,
            ledger_summary=ledger.summary,
            profile_enabled=False,
            max_message_chars=MAX_MESSAGE_CHARS,
            max_context_chars=12000,
            language="en",
        )
    except Exception as exc:
        if mcp_client is not None:
            await mcp_client.close()
        await base_client.aclose()
        row = _not_run_row(
            arm, task_id, f"{type(exc).__name__}: {exc}", "MCP_ERROR"
        )
        _write_json(case_dir / "result.json", row)
        return row

    before = store.business_snapshot()
    _write_json(case_dir / "snapshot-before.json", before)
    user_client = base_client.for_stage(f"{tag}-user")
    judge_client = base_client.for_stage(f"{tag}-judge")
    user_latencies: List[float] = []
    judge_latencies: List[float] = []

    async def user_handler(event: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = to_anthropic_kwargs(event["request"])
        call_started = time.monotonic()
        response = await user_client.messages.create(**kwargs)
        user_latencies.append(
            round((time.monotonic() - call_started) * 1000, 1)
        )
        return {
            "content": extract_text_content(response.content),
            "usage": _usage_dict(response),
            "raw_data": _raw_data(response),
        }

    async def judge_handler(event: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = to_anthropic_kwargs(event["request"])
        call_started = time.monotonic()
        response = await judge_client.messages.create(**kwargs)
        judge_latencies.append(
            round((time.monotonic() - call_started) * 1000, 1)
        )
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
    step_count = 1
    error_count = 0
    user_wait_ms = 0.0
    started = time.monotonic()
    entries_before = {entry["request_id"] for entry in ledger.entries()}
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
                turn = _empty_turn(user_text)
                turn["user_error"] = _empty_user_error(user_event)
                turns.append(turn)
                termination = "USER_ERROR"
                termination_detail = turn["user_error"]["detail"]
                break
            turn: Dict[str, Any] = _empty_turn(user_text)
            turns.append(turn)
            conversation_chars = sum(
                len(str(item.get("user") or ""))
                + len(str(item.get("assistant") or ""))
                for item in turns
            )
            if conversation_chars > MAX_CONVERSATION_CHARS:
                termination = "INPUT_LIMIT"
                termination_detail = "对话累计长度超过上限"
                break
            # 单 Agent 也要把本轮业务请求计入 32 次上限。
            if arm == "A":
                ledger.begin_turn(tag, MAX_REQUESTS_PER_TURN)
            try:
                result = await service.handle_turn(
                    user_text, f"client-{tag}", f"conv-{tag}"
                )
                turn.update(
                    {
                        "assistant": result.response,
                        "success": bool(result.success),
                        "agent_type": result.agent_type.value,
                        "agent_types": list(result.agent_types),
                        "primary_agent": result.primary_agent,
                        "supporting_agents": list(result.supporting_agents),
                        "routing_reason": result.routing_reason,
                        "routing_confidence": result.routing_confidence,
                        "composer_used": bool(result.composer_used),
                        "fallback_used": bool(result.fallback_used),
                        "role_traces": list(result.role_traces),
                        "tool_traces": list(result.tool_traces),
                        "tools_used": list(result.tools_used),
                        "prepared_op_ids": list(result.prepared_op_ids),
                        "display_appended": list(result.display_appended),
                        "turn_latency_ms": result.latency_ms,
                    }
                )
                user_wait_ms += float(result.latency_ms or 0.0)
                if not result.success:
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
            except Exception as exc:
                from evaluation.retail_budget import BudgetExceeded

                turn["assistant"] = ""
                turn["agent_error"] = f"{type(exc).__name__}: {exc}"
                error_count += 1
                if isinstance(exc, BudgetExceeded):
                    termination = "INCOMPLETE_BUDGET"
                    termination_detail = (
                        ledger.summary().get("cost_stop_reason")
                        or ledger.last_rejection_reason
                        or str(exc)
                    )
                elif isinstance(exc, BridgeError):
                    termination = "BRIDGE_ERROR"
                    termination_detail = str(exc)
                else:
                    termination = "AGENT_ERROR"
                    termination_detail = turn["agent_error"]
                break

            for trace in turn.get("tool_traces") or []:
                if (
                    trace.get("success") is False
                    or trace.get("result_success") is False
                ):
                    error_count += 1
            step_count += 2 + 2 * len(turn.get("tool_traces") or [])
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
        from evaluation.retail_budget import BudgetExceeded

        if isinstance(exc, BudgetExceeded):
            termination = "INCOMPLETE_BUDGET"
        else:
            termination = "RUN_ERROR"
        termination_detail = f"{type(exc).__name__}: {exc}"
    finally:
        await user_bridge.close()

    stop_reason_now = (
        ledger.summary().get("cost_stop_reason")
        or ledger.last_rejection_reason
    )
    if stop_reason_now and termination not in {
        "USER_STOP",
        "INCOMPLETE_BUDGET",
        "NOT_RUN_BUDGET",
    }:
        termination = "INCOMPLETE_BUDGET"
        termination_detail = str(stop_reason_now)

    duration_ms = round((time.monotonic() - started) * 1000, 1)
    after = store.business_snapshot()
    _write_json(case_dir / "snapshot-after.json", after)
    events = store.list_events(f"client-{tag}", f"conv-{tag}")
    _write_json(case_dir / "events.json", events)
    trajectory = trajectory_messages(turns, final_user_stop)
    _write_json(case_dir / "trajectory.json", trajectory)
    _write_json(
        case_dir / "dialogue.json",
        {
            "arm": arm,
            "task_id": task_id,
            "trial": 1,
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
    effective_completed = _effective_completion(
        termination,
        db.get("status"),
        db.get("gold_errors") or [],
        nl.get("status"),
        eval_error,
    )
    reward_success = bool(
        effective_completed and combined is not None and float(combined) == 1.0
    )
    user_error = next(
        (turn.get("user_error") for turn in turns if turn.get("user_error")),
        None,
    )

    new_entries = [
        entry
        for entry in ledger.entries()
        if entry["request_id"] not in entries_before
    ]
    cost = _cost_breakdown(new_entries)
    role_stages = sorted(
        {
            str(entry.get("stage") or "")
            for entry in new_entries
            if str(entry.get("stage") or "").startswith(f"{tag}-roles")
        }
    )
    distinct_roles = sorted(
        {
            str(turn.get("primary_agent") or "")
            for turn in turns
            if turn.get("primary_agent")
        }
        | {
            role
            for turn in turns
            for role in (turn.get("supporting_agents") or [])
        }
    )
    summary = {
        "arm": arm,
        "task_id": task_id,
        "trial": 1,
        "termination": termination,
        "termination_detail": termination_detail,
        "user_error": user_error,
        "valid_termination": valid_termination,
        "effective_completed": effective_completed,
        "reward_success": reward_success,
        "step_count": step_count,
        "max_steps": MAX_STEPS,
        "error_count": error_count,
        "max_errors": MAX_ERRORS,
        "turns": len(turns),
        "duration_ms": duration_ms,
        "user_wait_ms": round(user_wait_ms, 1),
        "mcp_startup_ms": mcp_startup_ms,
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
        "user_latency_ms": user_latencies,
        "judge_latency_ms": judge_latencies,
        "ledger_stop_reason": (
            ledger.summary().get("cost_stop_reason")
            or ledger.last_rejection_reason
            or None
        ),
        "eval_error": eval_error,
        "project_checks": project,
        "roles": {
            "primary_seen": distinct_roles,
            "supporting_seen": sorted(
                {
                    role
                    for turn in turns
                    for role in (turn.get("supporting_agents") or [])
                }
            ),
            "composer_used_turns": sum(
                1 for turn in turns if turn.get("composer_used")
            ),
            "fallback_turns": sum(
                1 for turn in turns if turn.get("fallback_used")
            ),
            "stage_clients": role_stages,
        },
        "agent_mode": "single" if arm == "A" else "multi",
        "config_sha256": _sha256_json(_role_config()),
        "task_sha256": _sha256_json(task),
        "source_hashes": _source_hashes(),
        "event_count": len(events),
        "trajectory_messages": len(trajectory),
        "eval_protocol_log": eval_result.get("protocol_log") or [],
        "user_protocol_log": user_bridge.protocol_log,
        "ledger": {
            "requests": len(new_entries),
            "system_cost_cny": cost["system_cny"],
            "user_cost_cny": cost["user_cny"],
            "judge_cost_cny": cost["judge_cny"],
            "total_cost_cny": cost["total_cny"],
            "tokens": cost["tokens"],
        },
        "finished_at_utc": _now(),
    }
    _write_json(case_dir / "result.json", summary)

    if mcp_client is not None:
        await mcp_client.close()
    await base_client.aclose()
    service_closed = service is not None
    return summary


def _entry_cost(entry: Dict[str, Any]) -> float:
    value = entry.get("actual_cny")
    if value is None:
        value = entry.get("reserved_cny")
    return float(value or 0.0)


def _cost_breakdown(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    system = user = judge = 0.0
    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for entry in entries:
        stage = str(entry.get("stage") or "")
        value = _entry_cost(entry)
        if stage.endswith("-judge"):
            judge += value
        elif stage.endswith("-user"):
            user += value
        else:
            system += value
        tokens["input"] += int(
            entry.get("input_tokens")
            or entry.get("input_tokens_upper")
            or 0
        )
        tokens["output"] += int(
            entry.get("output_tokens") or entry.get("max_tokens") or 0
        )
        tokens["cache_read"] += int(entry.get("cache_read_tokens") or 0)
        tokens["cache_write"] += int(entry.get("cache_write_tokens") or 0)
    return {
        "system_cny": round(system, 6),
        "user_cny": round(user, 6),
        "judge_cny": round(judge, 6),
        "total_cny": round(system + user + judge, 6),
        "tokens": tokens,
    }


def _not_run_row(
    arm: str, task_id: str, reason: str, termination: str = "NOT_RUN_BUDGET"
) -> Dict[str, Any]:
    return {
        "arm": arm,
        "task_id": task_id,
        "trial": 1,
        "termination": termination,
        "termination_detail": reason,
        "user_error": None,
        "valid_termination": False,
        "effective_completed": False,
        "reward_success": False,
        "db_component": None,
        "nl_component": None,
        "combined_reward": None,
        "db_status": "NOT_RUN",
        "nl_status": "NOT_RUN",
        "gold_errors": [],
        "judge_errors": [],
        "judge_requests": 0,
        "eval_error": reason,
        "project_checks": {},
        "roles": {},
        "duration_ms": 0.0,
        "user_wait_ms": 0.0,
        "mcp_startup_ms": 0.0,
        "ledger": {
            "requests": 0,
            "system_cost_cny": 0.0,
            "user_cost_cny": 0.0,
            "judge_cost_cny": 0.0,
            "total_cost_cny": 0.0,
            "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
        },
        "finished_at_utc": _now(),
    }


def _gaps(results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    gaps: Dict[str, List[str]] = {
        "not_run": [],
        "budget_incomplete": [],
        "invalid_termination": [],
        "invalid_reference": [],
        "judge_error": [],
        "evaluation_error": [],
        "other_incomplete": [],
    }
    for item in results:
        label = f"{item['task_id']}{item['arm']}"
        termination = item.get("termination")
        if termination in {"NOT_RUN_BUDGET", "NOT_RUN_ENV"}:
            gaps["not_run"].append(label)
            continue
        classified = False
        if termination == "INCOMPLETE_BUDGET":
            gaps["budget_incomplete"].append(label)
            classified = True
        if item.get("db_status") == "INVALID_REFERENCE" or item.get("gold_errors"):
            gaps["invalid_reference"].append(label)
            classified = True
        if item.get("nl_status") == "JUDGE_ERROR":
            gaps["judge_error"].append(label)
            classified = True
        db_status = item.get("db_status")
        if item.get("eval_error") or db_status not in (
            None,
            "OK",
            "INVALID_REFERENCE",
            "NOT_RUN",
        ):
            gaps["evaluation_error"].append(label)
            classified = True
        if termination != "INCOMPLETE_BUDGET" and item.get(
            "valid_termination"
        ) is not True:
            gaps["invalid_termination"].append(label)
            classified = True
        if item.get("effective_completed") is not True and not classified:
            gaps["other_incomplete"].append(label)
    return gaps


def _pair_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {(item["task_id"], item["arm"]): item for item in results}
    pairs = []
    counts = {"both_success": 0, "only_a": 0, "only_b": 0, "both_fail": 0}
    for task_id in TASK_IDS:
        a = by_key.get((task_id, "A"))
        b = by_key.get((task_id, "B"))
        valid = bool(
            a
            and b
            and a.get("effective_completed")
            and b.get("effective_completed")
        )
        entry: Dict[str, Any] = {
            "task_id": task_id,
            "valid_pair": valid,
            "a_success": bool(a and a.get("reward_success")),
            "b_success": bool(b and b.get("reward_success")),
        }
        if valid:
            if entry["a_success"] and entry["b_success"]:
                entry["outcome"] = "both_success"
            elif entry["a_success"]:
                entry["outcome"] = "only_a"
            elif entry["b_success"]:
                entry["outcome"] = "only_b"
            else:
                entry["outcome"] = "both_fail"
            counts[entry["outcome"]] += 1
        else:
            reasons = []
            for arm, item in (("A", a), ("B", b)):
                if item is None:
                    reasons.append(f"{arm}:missing")
                elif not item.get("effective_completed"):
                    reasons.append(
                        f"{arm}:{item.get('termination')}:"
                        f"{item.get('db_status')}/{item.get('nl_status')}"
                    )
            entry["outcome"] = "unpaired"
            entry["unpaired_reason"] = reasons
        pairs.append(entry)
    valid_pairs = sum(1 for item in pairs if item["valid_pair"])
    summary = {
        "planned_pairs": len(TASK_IDS),
        "valid_pairs": valid_pairs,
        "valid_pair_fraction": f"{valid_pairs}/{len(TASK_IDS)}",
        "counts_on_valid_pairs": counts,
    }
    if valid_pairs == 0:
        summary["note"] = "P=0，没有可比较的有效配对成功率。"
    else:
        summary["a_success_on_valid_pairs"] = sum(
            1 for item in pairs if item["valid_pair"] and item["a_success"]
        )
        summary["b_success_on_valid_pairs"] = sum(
            1 for item in pairs if item["valid_pair"] and item["b_success"]
        )
    summary["pairs"] = pairs
    return summary


def _arm_summary(results: List[Dict[str, Any]], arm: str) -> Dict[str, Any]:
    rows = [item for item in results if item["arm"] == arm]
    effective = [item for item in rows if item.get("effective_completed")]
    success = [item for item in effective if item.get("reward_success")]
    return {
        "planned": len(TASK_IDS),
        "completed_rows": sum(
            1 for item in rows if item.get("termination") != "NOT_RUN_BUDGET"
            and item.get("termination") != "NOT_RUN_ENV"
        ),
        "effective_completed": len(effective),
        "effective_fraction": f"{len(effective)}/{len(TASK_IDS)}",
        "successful": len(success),
        "success_fraction": f"{len(success)}/{len(effective)}" if effective else "0/0",
        "system_cost_cny": round(
            sum(item["ledger"]["system_cost_cny"] for item in rows), 6
        ),
        "experiment_cost_cny": round(
            sum(item["ledger"]["total_cost_cny"] for item in rows), 6
        ),
        "user_cost_cny": round(
            sum(item["ledger"]["user_cost_cny"] for item in rows), 6
        ),
        "judge_cost_cny": round(
            sum(item["ledger"]["judge_cost_cny"] for item in rows), 6
        ),
        "duration_ms": round(sum(item.get("duration_ms") or 0.0 for item in rows), 1),
        "user_wait_ms": round(sum(item.get("user_wait_ms") or 0.0 for item in rows), 1),
        "mcp_startup_ms": round(
            sum(item.get("mcp_startup_ms") or 0.0 for item in rows), 1
        ),
        "requests": sum(item["ledger"]["requests"] for item in rows),
        "user_requests": sum(
            1 for item in rows for _ in item.get("user_latency_ms") or []
        ),
        "judge_requests": sum(
            1 for item in rows for _ in item.get("judge_latency_ms") or []
        ),
        "user_latency_ms": [
            value for item in rows for value in item.get("user_latency_ms") or []
        ],
        "judge_latency_ms": [
            value for item in rows for value in item.get("judge_latency_ms") or []
        ],
        "tokens": {
            key: sum(item["ledger"]["tokens"].get(key, 0) for item in rows)
            for key in ("input", "output", "cache_read", "cache_write")
        },
    }


def _pair_precheck(
    task_id: str,
    ledger: Any,
    prices: Dict[str, Any],
    measurements: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    estimate = _planned_estimate(measurements, prices)["cases"]
    need = (
        estimate[f"{task_id}A"]["planned_cny"]
        + estimate[f"{task_id}B"]["planned_cny"]
    )
    remaining = float(ledger.summary().get("remaining_cny") or 0.0)
    reserve = need * 1.2
    return remaining >= reserve, {
        "task_id": task_id,
        "planned_pair_cny": round(need, 6),
        "required_with_margin_cny": round(reserve, 6),
        "remaining_cny_before_pair": round(remaining, 6),
    }


async def _execute(run_id: str) -> Dict[str, Any]:
    from evaluation.retail_ledger import RetailLedger

    run_dir = ROOT / "data" / "evaluation" / "t16-retail" / run_id
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

    measurements = preflight["initial_request_measurements"]
    results: List[Dict[str, Any]] = []
    stop_reason = ""
    stop_kind = ""
    pair_checks: List[Dict[str, Any]] = []
    for item in RUN_ORDER:
        task_id, arm = item[:-1], item[-1]
        result_path = _case_dir(run_dir, arm, task_id) / "result.json"
        if result_path.is_file():
            results.append(_load_json(result_path))
            continue
        if stop_reason:
            row = _not_run_row(arm, task_id, stop_reason, stop_kind)
            _write_json(
                _case_dir(run_dir, arm, task_id) / "result.json", row
            )
            results.append(row)
            continue
        if arm == "A":
            ok, check = _pair_precheck(task_id, ledger, prices, measurements)
            pair_checks.append(check)
            if not ok:
                stop_reason = (
                    "配对开始前余额不足以覆盖双方及评分的合理预计"
                )
                stop_kind = "NOT_RUN_BUDGET"
                row = _not_run_row(arm, task_id, stop_reason, stop_kind)
                _write_json(
                    _case_dir(run_dir, arm, task_id) / "result.json", row
                )
                results.append(row)
                continue
        result = await _run_case(
            task_id, arm, run_dir, ledger, api_key, base_url
        )
        results.append(result)
        if result.get("termination") in {"INCOMPLETE_BUDGET", "NOT_RUN_BUDGET"}:
            stop_reason = str(
                result.get("termination_detail")
                or ledger.summary().get("cost_stop_reason")
                or "预算停止"
            )
            stop_kind = "NOT_RUN_BUDGET"
        elif result.get("termination") == "MCP_ERROR":
            stop_reason = str(result.get("termination_detail") or "MCP 启动失败")
            stop_kind = "NOT_RUN_ENV"
        elif (
            ledger.summary().get("cost_stop_reason")
            or ledger.last_rejection_reason
        ):
            stop_reason = str(
                ledger.summary().get("cost_stop_reason")
                or ledger.last_rejection_reason
            )
            stop_kind = "NOT_RUN_BUDGET"

    gaps = _gaps(results)
    evaluation_status = "EVALUATION_COMPLETE"
    if any(gaps[key] for key in ("invalid_reference", "judge_error", "evaluation_error")):
        evaluation_status = "EVALUATOR_INVALID"
    elif any(
        gaps[key]
        for key in (
            "not_run",
            "budget_incomplete",
            "invalid_termination",
            "other_incomplete",
        )
    ):
        evaluation_status = "PARTIAL"
    ledger_summary = ledger.summary()
    aggregate = {
        "run_id": run_id,
        "finished_at_utc": _now(),
        "planned_task_ids": list(TASK_IDS),
        "run_order": list(RUN_ORDER),
        "planned_rows": len(RUN_ORDER),
        "results": results,
        "reported_rows": len(results),
        "excluded_tasks": [preflight.get("excluded_46") or {}],
        "gaps": gaps,
        "evaluation_status": evaluation_status,
        "arms": {arm: _arm_summary(results, arm) for arm in ARMS},
        "pair_comparison": _pair_summary(results),
        "pair_prechecks": pair_checks,
        "system_cost_cny": round(
            sum(item["ledger"]["system_cost_cny"] for item in results), 6
        ),
        "experiment_cost_cny": round(
            sum(item["ledger"]["total_cost_cny"] for item in results), 6
        ),
        "user_cost_cny": round(
            sum(item["ledger"]["user_cost_cny"] for item in results), 6
        ),
        "judge_cost_cny": round(
            sum(item["ledger"]["judge_cost_cny"] for item in results), 6
        ),
        "ledger": ledger_summary,
        "budget_stop_reason": stop_reason or None,
        "notes": [
            "费用是账本按 usage 的估费与保守占用，不是供应商账单核对结果。",
            "任务总墙钟包含模拟用户和评分；用户等待时间只按 handle_turn 墙钟单列。",
            "七项小样本不作统计显著性、P95 或稳定性结论。",
        ],
    }
    _write_json(run_dir / "summary.json", aggregate)
    _write_json(run_dir / "ledger-summary.json", ledger_summary)
    _write_json(run_dir / "ledger-entries.json", ledger.entries())
    for row in results:
        _write_json(
            _case_dir(run_dir, row["arm"], row["task_id"]) / "result.json",
            row,
        )
    return aggregate


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="T16 单 Agent 与多 Agent 零售适配版成对比较"
    )
    parser.add_argument("command", choices=["preflight", "run", "all", "verify46"])
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    if args.command == "verify46":
        run_dir = ROOT / "data" / "evaluation" / "t16-retail" / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = _verify_task_46_offline(run_dir)
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
        return 0
    if args.command in {"preflight", "all"}:
        result = _preflight(args.run_id)
        print(
            json.dumps(
                {
                    "run_id": result["run_id"],
                    "gate_passed": result["gate_passed"],
                    "failed_checks": [
                        name
                        for name, ok in result["checks"].items()
                        if not ok
                    ],
                    "planned_total_cny": result["estimate"][
                        "planned_total_cny"
                    ],
                    "gate_total_cny": result["estimate"]["gate_total_cny"],
                    "worst_turn_32_requests_cny": result["estimate"][
                        "worst_turn_32_requests_cny"
                    ],
                    "excluded_46_missing_orders": result["excluded_46"].get(
                        "missing_gold_order_ids"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if not result["gate_passed"]:
            return 2
        if args.command == "preflight":
            return 0
    aggregate = asyncio.run(_execute(args.run_id))
    print(
        json.dumps(
            {
                "run_id": aggregate["run_id"],
                "evaluation_status": aggregate["evaluation_status"],
                "gaps": aggregate["gaps"],
                "arms": aggregate["arms"],
                "pair_comparison": {
                    key: value
                    for key, value in aggregate["pair_comparison"].items()
                    if key != "pairs"
                },
                "system_cost_cny": aggregate["system_cost_cny"],
                "experiment_cost_cny": aggregate["experiment_cost_cny"],
                "ledger": aggregate["ledger"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
