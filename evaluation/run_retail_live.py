"""T13 零售沙箱 V01-V08 的预检、真实多轮运行与检查。

本模块只读取冻结的公开种子、政策和知识，不修改源项目。预检只构造真实
Agent 的请求体做费用估算，不创建可发送客户端；真实运行才启动 API/CLI
并写入固定预算账本。所有收费请求共用同一个账本，失败或超界都会永久停止。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASES_PATH = ROOT / "evaluation" / "data" / "retail_live" / "cases.json"
PRICING_PATH = ROOT / "evaluation" / "data" / "retail_live" / "pricing.json"
SEED_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
POLICY_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "policy.md"
KNOWLEDGE_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "knowledge.jsonl"
LICENSE_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "LICENSE"
SOURCES_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "sources.json"
SKILLS_DIR = ROOT / "skills"
LEDGER_PATH = ROOT / "data" / "retail" / "t13_live" / "ledger.db"
RUNS_DIR = ROOT / "data" / "retail" / "t13_live" / "runs"

RUN_ID = "t13-live"
EXPECTED_SEED_SHA256 = (
    "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"
)
EXPECTED_POLICY_SHA256 = (
    "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467"
)
EXPECTED_KNOWLEDGE_SHA256 = (
    "e7360f9d22531ed836da12fe34d90c971a9300a3ed943031d75f8f6dc3b734c0"
)
EXPECTED_LICENSE_SHA256 = (
    "e67c5aa0074dfcaefd3c3a1aedb94cb539234aecd15d5a972574e3200e6252fe"
)
EXPECTED_KNOWLEDGE_COUNT = 29
EXPECTED_SEED_COUNTS = {"users": 500, "orders": 1000, "products": 50}
AGENT_MAX_TOKENS = 2048
REWRITE_MAX_TOKENS = 1024
RERANK_MAX_TOKENS = 8192
PLAN_AGENT_OUTPUT_TOKENS = 512
PLAN_REWRITE_OUTPUT_TOKENS = 128
PLAN_RERANK_OUTPUT_TOKENS = 64
MAX_CONTEXT_CHARS = 12000

_CODE_FILES = (
    "api/main.py",
    "agents/agent_orchestrator.py",
    "agents/retail_agent.py",
    "agents/retail_runtime.py",
    "agents/retail_tools.py",
    "agents/tools.py",
    "core/skill_loader.py",
    "evaluation/retail_budget.py",
    "evaluation/retail_ledger.py",
    "evaluation/run_retail_live.py",
    "mcp/knowledge_base.py",
    "mcp/retail_store.py",
    "mcp/tool_manager.py",
    "memory/conversation_memory.py",
    "skills/retail_support/SKILL.md",
    "evaluation/data/retail_live/cases.json",
    "evaluation/data/retail_live/pricing.json",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_knowledge_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in KNOWLEDGE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_state() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as ex:
            return f"<error: {ex}>"
        return (result.stdout or result.stderr or "").strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def _code_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in _CODE_FILES:
        path = ROOT / relative
        hashes[relative] = _sha256_file(path) if path.is_file() else "<missing>"
    return hashes


def _asset_record() -> Dict[str, Any]:
    assets = {
        "seed": (SEED_PATH, EXPECTED_SEED_SHA256),
        "policy": (POLICY_PATH, EXPECTED_POLICY_SHA256),
        "knowledge": (KNOWLEDGE_PATH, EXPECTED_KNOWLEDGE_SHA256),
        "license": (LICENSE_PATH, EXPECTED_LICENSE_SHA256),
    }
    record: Dict[str, Any] = {}
    for name, (path, expected) in assets.items():
        actual = _sha256_file(path) if path.is_file() else "<missing>"
        record[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": actual,
            "expected": expected,
            "ok": actual == expected,
        }
    knowledge_rows = _load_knowledge_rows()
    record["knowledge_count"] = {
        "actual": len(knowledge_rows),
        "expected": EXPECTED_KNOWLEDGE_COUNT,
        "ok": len(knowledge_rows) == EXPECTED_KNOWLEDGE_COUNT,
    }
    seed = _load_json(SEED_PATH)
    record["seed_counts"] = {
        key: {
            "actual": len(seed.get(key) or {}) if isinstance(seed, dict) else None,
            "expected": count,
        }
        for key, count in EXPECTED_SEED_COUNTS.items()
    }
    record["seed_counts_ok"] = all(
        item["actual"] == item["expected"]
        for item in record["seed_counts"].values()
    )
    if SOURCES_PATH.is_file():
        record["sources"] = _load_json(SOURCES_PATH)
        record["sources_sha256"] = _sha256_file(SOURCES_PATH)
    return record


def _pricing() -> Dict[str, Any]:
    data = _load_json(PRICING_PATH)
    peak = data.get("peak") or {}
    input_price = float(peak.get("input_cache_miss") or 0)
    output_price = float(peak.get("output") or 0)
    if input_price <= 0 or output_price <= 0:
        raise RuntimeError("pricing.json 缺少有效的高峰输入/输出价格")
    return {
        "data": data,
        "input_price_cny_per_mtok": input_price,
        "output_price_cny_per_mtok": output_price,
    }


def _ledger_state(budget_cny: float) -> Dict[str, Any]:
    """直接读账本，不在预检阶段创建或写入账本。"""
    if not LEDGER_PATH.is_file():
        return {
            "exists": False,
            "path": str(LEDGER_PATH),
            "budget_cny": budget_cny,
            "spent_cny": 0.0,
            "reserved_cny": 0.0,
            "remaining_cny": budget_cny,
            "requests": 0,
            "run_id": "",
            "model": "",
            "cost_stop_reason": None,
        }
    with sqlite3.connect(str(LEDGER_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        meta = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM meta")
        }
        totals = connection.execute(
            "SELECT"
            " COALESCE(SUM(CASE WHEN status IN ('success','failed','crashed')"
            " THEN actual_cny ELSE 0 END), 0) AS spent,"
            " COALESCE(SUM(CASE WHEN status='reserved'"
            " THEN reserved_cny ELSE 0 END), 0) AS reserved,"
            " COUNT(*) AS total"
            " FROM entries"
        ).fetchone()
    spent = float(totals["spent"] or 0.0)
    reserved = float(totals["reserved"] or 0.0)
    return {
        "exists": True,
        "path": str(LEDGER_PATH),
        "budget_cny": budget_cny,
        "spent_cny": spent,
        "reserved_cny": reserved,
        "remaining_cny": budget_cny - spent - reserved,
        "requests": int(totals["total"] or 0),
        "run_id": meta.get("run_id", ""),
        "model": meta.get("model", ""),
        "cost_stop_reason": meta.get("overrun_reason") or None,
    }


def _ledger_rows() -> List[Dict[str, Any]]:
    if not LEDGER_PATH.is_file():
        return []
    with sqlite3.connect(str(LEDGER_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM entries ORDER BY created_at, request_id"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def _service_status() -> Dict[str, Any]:
    """只做连接与心跳检查，不写入、不创建集合。"""
    status: Dict[str, Any] = {
        "redis": {"ok": False, "url": "", "error": ""},
        "chroma": {"ok": False, "host": "", "port": 0, "error": ""},
        "embedding_cache": {"ok": False, "path": "", "error": ""},
    }
    try:
        import redis

        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip()
        status["redis"]["url"] = url
        client = redis.Redis.from_url(
            url,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        status["redis"]["ok"] = bool(client.ping())
    except Exception as ex:  # pragma: no cover - 环境不可用时记录原因
        status["redis"]["error"] = f"{type(ex).__name__}: {ex}"

    try:
        import chromadb
        from chromadb.config import Settings

        host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
        port = int(os.getenv("CHROMA_PORT", "8000"))
        status["chroma"]["host"] = host
        status["chroma"]["port"] = port
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(anonymized_telemetry=False),
        )
        status["chroma"]["ok"] = client.heartbeat() is not None
    except Exception as ex:  # pragma: no cover - 环境不可用时记录原因
        status["chroma"]["error"] = f"{type(ex).__name__}: {ex}"

    try:
        from mcp.knowledge_base import _embedding_cache_ready

        status["embedding_cache"]["ok"] = bool(_embedding_cache_ready())
        status["embedding_cache"]["path"] = os.getenv(
            "CHROMA_EMBEDDING_CACHE",
            str(ROOT / "data" / "chroma_cache"),
        )
    except Exception as ex:
        status["embedding_cache"]["error"] = f"{type(ex).__name__}: {ex}"
    return status


def _seed_snapshot() -> Dict[str, Dict[str, Any]]:
    seed = _load_json(SEED_PATH)
    return {
        table: dict(seed.get(table) or {})
        for table in ("users", "orders", "products")
    }


def _json_bytes(payload: Any) -> int:
    return len(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _plan_tokens(payload: Any) -> int:
    """计划估算：请求体 UTF-8 字节数除以 3，向上取整。"""
    return max(1, math.ceil(_json_bytes(payload) / 3))


def _hard_tokens(payload: Any) -> int:
    from evaluation.retail_budget import request_input_upper_tokens

    return request_input_upper_tokens(payload)


def _request_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float,
    output_price: float,
) -> float:
    return (
        input_tokens * input_price / 1_000_000
        + output_tokens * output_price / 1_000_000
    )


def _case_assets(case: Dict[str, Any], run_short: str) -> Dict[str, Any]:
    cid = str(case["id"]).lower()
    suffix = f"_t13{cid}_{run_short}"
    return {
        "collection": f"t13{cid}_{run_short}",
        "memory_suffix": suffix,
        "memory_collections": [f"episodic{suffix}", f"user_profile{suffix}"],
        "redis_prefix": f"t13live:{run_short}:{case['id']}:",
        "client_user": f"t13-client-{case['id']}-{run_short}",
        "conv_id": f"t13-conv-{case['id']}-{run_short}",
    }


def _base_env(
    case: Dict[str, Any],
    case_dir: Path,
    assets: Dict[str, str],
    prices: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "RETAIL_MODE": "true",
            "RETAIL_DB_PATH": str((case_dir / "retail.db").resolve()),
            "RETAIL_SEED_PATH": str(SEED_PATH),
            "RETAIL_POLICY_PATH": str(POLICY_PATH),
            "RETAIL_KNOWLEDGE_PATH": str(KNOWLEDGE_PATH),
            "RETAIL_COLLECTION": assets["collection"],
            "RETAIL_MEMORY_COLLECTION_SUFFIX": assets["memory_suffix"],
            "RETAIL_MEMORY_KEY_PREFIX": assets["redis_prefix"],
            "RETAIL_EVENT_LOG": str((case_dir / "events.jsonl").resolve()),
            "RETAIL_CASE_ID": str(case["id"]),
            "RETAIL_RUN_ID": RUN_ID,
            "RETAIL_CLIENT_USER": assets["client_user"],
            "RETAIL_CONV_ID": assets["conv_id"],
            "RETAIL_BUDGET_CNY": str(config["budget_cny"]),
            "RETAIL_INPUT_CNY_PER_MTOK": str(
                prices["input_price_cny_per_mtok"]
            ),
            "RETAIL_OUTPUT_CNY_PER_MTOK": str(
                prices["output_price_cny_per_mtok"]
            ),
            "RETAIL_LEDGER_PATH": str(LEDGER_PATH),
            "RETAIL_BUDGET_LEDGER": str(LEDGER_PATH),
            "RETAIL_PROFILE_ENABLED": "false",
            "RETAIL_AGENT_MAX_TOKENS": str(AGENT_MAX_TOKENS),
            "RETAIL_MAX_MESSAGE_CHARS": "4000",
            "RETAIL_MAX_CONTEXT_CHARS": str(MAX_CONTEXT_CHARS),
            "ECHOMIND_SKILLS_DIR": str(SKILLS_DIR),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _apply_env(env: Dict[str, str]) -> Dict[str, Optional[str]]:
    keys = [
        key
        for key in env
        if key.startswith("RETAIL_")
        or key in {
            "ECHOMIND_SKILLS_DIR",
            "PYTHONIOENCODING",
            "PYTHONUNBUFFERED",
        }
    ]
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ[key] = env[key]
    return previous


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _read_event_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _turn_events(path: Path) -> List[Dict[str, Any]]:
    return [
        row
        for row in _read_event_rows(path)
        if str(row.get("type") or "") == "turn"
    ]


def _pending_rows(db_path: Path, client_user: str, conv_id: str) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT op_id, order_id, status, items_json, payment_method_id,"
                " amount_cents, display_text, prepared_turn, displayed_turn"
                " FROM pending_returns"
                " WHERE client_user_id=? AND conv_id=? ORDER BY created_at, op_id",
                (client_user, conv_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def _return_operations(db_path: Path, client_user: str, conv_id: str) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM return_operations"
                " WHERE client_user_id=? AND conv_id=? ORDER BY created_at, op_id",
                (client_user, conv_id),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(row) for row in rows]


def _order_diff(
    baseline: Dict[str, Dict[str, Any]],
    final: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"users": [], "products": [], "orders": []}
    for table in ("users", "products"):
        before = baseline.get(table) or {}
        after = final.get(table) or {}
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                result[table].append(key)
    before_orders = baseline.get("orders") or {}
    after_orders = final.get("orders") or {}
    for order_id in sorted(set(before_orders) | set(after_orders)):
        before = before_orders.get(order_id)
        after = after_orders.get(order_id)
        if before == after:
            continue
        fields = sorted(
            key
            for key in set(before or {}) | set(after or {})
            if (before or {}).get(key) != (after or {}).get(key)
        )
        result["orders"].append(
            {
                "order_id": order_id,
                "changed_fields": fields,
                "before": {key: (before or {}).get(key) for key in fields},
                "after": {key: (after or {}).get(key) for key in fields},
            }
        )
    return result


def _only_allowed_order_fields(
    diff: Dict[str, Any],
    expected_orders: Dict[str, Dict[str, Any]],
) -> bool:
    if diff.get("users") or diff.get("products"):
        return False
    allowed = {"status", "return_items", "return_payment_method_id"}
    for item in diff.get("orders") or []:
        if item["order_id"] not in expected_orders:
            return False
        if set(item["changed_fields"]) - allowed:
            return False
    return True


def _norm_items(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value)


_NEGATION_MARKERS = (
    "不代表",
    "不表示",
    "并非",
    "不等于",
    "不会",
    "没有",
    "尚未",
    "无法",
    "不能",
    "未",
)


def _forbidden_claims(text: str, patterns: List[str]) -> List[Dict[str, str]]:
    """只统计肯定的禁用说法；同一分句里带否定词的属于澄清，不算违规。"""
    hits: List[Dict[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start = match.start()
            boundary = max(text.rfind(char, 0, start) for char in "。！？；\n")
            clause = text[boundary + 1 : start]
            if any(marker in clause for marker in _NEGATION_MARKERS):
                continue
            hits.append({"pattern": pattern, "match": match.group(0)})
    return hits


def _check_case(
    case: Dict[str, Any],
    turns: List[Dict[str, Any]],
    meta: Dict[str, Any],
    db_path: Path,
    baseline: Dict[str, Dict[str, Any]],
    final: Optional[Dict[str, Dict[str, Any]]],
    assets: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    responses = [str(turn.get("response") or "") for turn in turns]
    text_all = "\n".join(responses)
    store_events = _store_events(db_path, assets)
    event_types = [str(event.get("event_type") or "") for event in store_events]

    add(
        "turn_count",
        len(turns) == int(case["max_turns"]),
        {"actual": len(turns), "expected": case["max_turns"]},
    )
    add(
        "all_turns_succeeded",
        bool(turns) and all(turn.get("agent_success") for turn in turns),
        [turn.get("agent_success") for turn in turns],
    )
    add(
        "request_limit",
        int(meta.get("request_count") or 0) <= int(case["max_requests"]),
        {
            "actual": int(meta.get("request_count") or 0),
            "max": int(case["max_requests"]),
        },
    )
    per_turn_caps = [int(value) for value in case.get("max_requests_per_turn") or []]
    for index, turn in enumerate(turns, start=1):
        if index <= len(per_turn_caps):
            actual = int(turn.get("requests_in_turn") or 0)
            add(
                f"turn_request_limit:{index}",
                actual <= per_turn_caps[index - 1],
                {"actual": actual, "max": per_turn_caps[index - 1]},
            )
    add(
        "tool_limit",
        int(meta.get("tool_count") or 0) <= int(case["max_tool_calls"]),
        {
            "actual": int(meta.get("tool_count") or 0),
            "max": int(case["max_tool_calls"]),
        },
    )
    add(
        "channel",
        str(case.get("channel")) == str(meta.get("channel")),
        {"expected": case.get("channel"), "actual": meta.get("channel")},
    )

    tools_by_turn = {
        int(turn.get("turn") or 0): list(turn.get("tools_used") or [])
        for turn in turns
    }
    tools_all = [name for names in tools_by_turn.values() for name in names]
    for name in case.get("required_tools") or []:
        add(f"required_tool:{name}", name in tools_all, tools_all)
    for group in case.get("required_any_tools") or []:
        add(
            "required_any:" + "|".join(group),
            any(name in tools_all for name in group),
            tools_all,
        )
    for turn_no in case.get("knowledge_turns") or []:
        turn = _turn_by_number(turns, int(turn_no))
        add(
            f"knowledge_turn:{turn_no}",
            bool(turn)
            and (
                int(turn.get("knowledge_hits") or 0) > 0
                or "search_knowledge_base" in (turn.get("tools_used") or [])
            ),
            {
                "knowledge_hits": (turn or {}).get("knowledge_hits"),
                "tools_used": (turn or {}).get("tools_used"),
            },
        )

    for check in case.get("turn_checks") or []:
        turn_no = int(check.get("turn") or 0)
        turn = _turn_by_number(turns, turn_no)
        text = str((turn or {}).get("response") or "")
        detail = {"turn": turn_no}
        ok = bool(turn)
        for needle in check.get("contains_all") or []:
            detail[f"contains_all:{needle}"] = needle in text
            ok = ok and needle in text
        any_list = check.get("contains_any") or []
        if any_list:
            hits = [needle for needle in any_list if needle in text]
            detail["contains_any_hits"] = hits
            ok = ok and bool(hits)
        add(f"turn_check:{turn_no}", ok, detail)

    for needle in case.get("contains_all") or []:
        add(f"contains_all:{needle}", needle in text_all)
    any_list = case.get("contains_any") or []
    if any_list:
        hits = [needle for needle in any_list if needle in text_all]
        add("contains_any", bool(hits), hits)
    last_any = case.get("last_contains_any") or []
    if last_any:
        last_text = responses[-1] if responses else ""
        hits = [needle for needle in last_any if needle in last_text]
        add("last_contains_any", bool(hits), hits)

    forbidden_hits = [
        needle for needle in (case.get("forbidden_any") or []) if needle in text_all
    ]
    add("forbidden_any", not forbidden_hits, forbidden_hits)
    regex_hits = _forbidden_claims(text_all, case.get("forbidden_regex") or [])
    add("forbidden_regex", not regex_hits, regex_hits)

    expect = case.get("expect") or {}
    expected_orders = expect.get("orders") or {}
    pending_rows = _pending_rows(db_path, assets["client_user"], assets["conv_id"])
    operations = _return_operations(
        db_path, assets["client_user"], assets["conv_id"]
    )
    bound_user = _bound_user(db_path, assets["client_user"], assets["conv_id"])
    final_snapshot = final or {}
    diff = _order_diff(baseline, final_snapshot)

    add("db_business_diff", _only_allowed_order_fields(diff, expected_orders), diff)
    for order_id, expected in expected_orders.items():
        order = (final_snapshot.get("orders") or {}).get(order_id) or {}
        mismatches: Dict[str, Any] = {}
        if "status" in expected and order.get("status") != expected["status"]:
            mismatches["status"] = {
                "actual": order.get("status"),
                "expected": expected["status"],
            }
        if "return_items" in expected and _norm_items(
            order.get("return_items")
        ) != _norm_items(expected["return_items"]):
            mismatches["return_items"] = {
                "actual": order.get("return_items"),
                "expected": expected["return_items"],
            }
        if "return_payment_method_id" in expected and order.get(
            "return_payment_method_id"
        ) != expected["return_payment_method_id"]:
            mismatches["return_payment_method_id"] = {
                "actual": order.get("return_payment_method_id"),
                "expected": expected["return_payment_method_id"],
            }
        add(f"order:{order_id}", not mismatches, mismatches)

    add(
        "write_expectation",
        (bool(expected_orders) and bool(operations))
        if expect.get("write")
        else (not operations and not diff.get("orders")),
        {
            "expected_write": bool(expect.get("write")),
            "operations": len(operations),
            "changed_orders": diff.get("orders"),
        },
    )
    add(
        "return_operations",
        len(operations) == int(expect.get("return_operations", 0)),
        {"actual": len(operations), "expected": expect.get("return_operations", 0)},
    )
    requested_count = sum(1 for name in event_types if name == "return_requested")
    add(
        "return_requested_events",
        requested_count == int(expect.get("return_requested_events", 0)),
        {
            "actual": requested_count,
            "expected": expect.get("return_requested_events", 0),
        },
    )
    add(
        "return_operations_amounts",
        _operations_amounts_ok(operations, expected_orders),
        [
            {
                "order_id": row.get("order_id"),
                "amount_cents": row.get("amount_cents"),
                "items_json": row.get("items_json"),
                "payment_method_id": row.get("payment_method_id"),
            }
            for row in operations
        ],
    )
    add(
        "pending_total",
        len(pending_rows) == int(expect.get("pending_total", 0)),
        {"actual": len(pending_rows), "expected": expect.get("pending_total", 0)},
    )
    displayed = [row for row in pending_rows if row.get("status") == "displayed"]
    ever_displayed = [
        row for row in pending_rows if row.get("displayed_turn") is not None
    ]
    add(
        "pending_displayed_min",
        len(ever_displayed) >= int(expect.get("pending_displayed_min", 0)),
        {
            "actual": len(ever_displayed),
            "expected_min": expect.get("pending_displayed_min", 0),
        },
    )
    add(
        "pending_displayed_at_end_min",
        len(displayed) >= int(expect.get("pending_displayed_at_end_min", 0)),
        {
            "actual": len(displayed),
            "expected_min": expect.get("pending_displayed_at_end_min", 0),
        },
    )
    consumed_all = bool(pending_rows) and all(
        row.get("status") == "consumed" for row in pending_rows
    )
    if expect.get("pending_consumed_all"):
        add("pending_consumed_all", consumed_all, [row.get("status") for row in pending_rows])
    superseded = [row for row in pending_rows if row.get("status") == "superseded"]
    add(
        "superseded_min",
        len(superseded) >= int(expect.get("superseded_min", 0)),
        {"actual": len(superseded), "expected_min": expect.get("superseded_min", 0)},
    )
    add(
        "bound_user_id",
        bound_user == expect.get("bound_user_id"),
        {"actual": bound_user, "expected": expect.get("bound_user_id")},
    )
    for name in case.get("expect_events") or []:
        add(f"event:{name}", name in event_types, event_types)
    for name in case.get("forbid_events") or []:
        add(f"forbid_event:{name}", name not in event_types, event_types)
    handoff = any(
        bool(turn.get("escalated")) for turn in turns
    ) or "handoff_requested" in event_types
    add(
        "handoff_expectation",
        handoff == bool(case.get("expect_handoff")),
        {"actual": handoff, "expected": bool(case.get("expect_handoff"))},
    )
    other_facts = [
        needle
        for needle in (case.get("forbid_other_user_facts") or [])
        if needle in text_all
    ]
    add("forbid_other_user_facts", not other_facts, other_facts)

    cost = float(meta.get("case_cost_cny") or 0.0)
    add("cost_recorded", cost >= 0.0, cost)
    add("budget_within_total", cost <= float(config["budget_cny"]), cost)
    return {
        "case_id": case["id"],
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
        "cost_cny": round(cost, 8),
        "requests": int(meta.get("request_count") or 0),
        "tools": int(meta.get("tool_count") or 0),
        "duration_ms": float(meta.get("duration_ms") or 0.0),
    }


def _turn_by_number(
    turns: List[Dict[str, Any]], number: int
) -> Optional[Dict[str, Any]]:
    for turn in turns:
        if int(turn.get("turn") or 0) == number:
            return turn
    return None


def _store_events(db_path: Path, assets: Dict[str, str]) -> List[Dict[str, Any]]:
    if not db_path.is_file():
        return []
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT event_type, order_id, op_id, payload_json, created_at"
                " FROM retail_events WHERE client_user_id=? AND conv_id=?"
                " ORDER BY event_id",
                (assets["client_user"], assets["conv_id"]),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    result = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except json.JSONDecodeError:
            item["payload"] = {}
        result.append(item)
    return result


def _bound_user(db_path: Path, client_user: str, conv_id: str) -> Optional[str]:
    if not db_path.is_file():
        return None
    with sqlite3.connect(str(db_path)) as connection:
        try:
            row = connection.execute(
                "SELECT retail_user_id FROM sessions WHERE client_user_id=? AND conv_id=?",
                (client_user, conv_id),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return str(row[0]) if row else None


def _operations_amounts_ok(
    operations: List[Dict[str, Any]],
    expected_orders: Dict[str, Dict[str, Any]],
) -> bool:
    """金额必须来自订单价；这里只做结构与非负检查，精确金额由订单检查覆盖。"""
    for row in operations:
        try:
            amount = int(row.get("amount_cents"))
        except (TypeError, ValueError):
            return False
        if amount <= 0:
            return False
    return True


def _sample_payloads(
    store: Any,
    email: str,
    order_id: str,
    item_ids: List[str],
    payment_method_id: str,
) -> Dict[str, Any]:
    client_user = "preflight-client"
    conv_id = "preflight-conv"
    user = store.find_user_by_email(client_user, conv_id, email)
    order = store.get_order_details(client_user, conv_id, order_id)
    prepared = store.prepare_return(
        client_user, conv_id, order_id, item_ids, payment_method_id
    )
    knowledge_rows = _load_knowledge_rows()
    fragments = []
    for segment in knowledge_rows[:5]:
        fragments.append(
            {
                "id": segment.get("id"),
                "evidence_id": segment.get("evidence_id"),
                "doc_id": segment.get("doc_id", ""),
                "title": segment.get("title", ""),
                "content": str(segment.get("content", ""))[:500],
                "score": 0.9,
                "source_kind": segment.get("source_kind", ""),
                "source_refs": segment.get("source_refs") or [],
            }
        )
    knowledge = {
        "success": True,
        "query": "退货期限、运费与准备建议",
        "top_k": 5,
        "results": fragments,
        "reranked": True,
        "degraded": False,
    }
    return {"user": user, "order": order, "prepare": prepared, "knowledge": knowledge}


def _build_preflight_agent(
    work_dir: Path,
    model: str,
    policy_text: str,
) -> Tuple[Any, Any]:
    from agents.retail_agent import RetailAgent
    from core.skill_loader import SkillManager
    from mcp.retail_store import RetailStore

    store = RetailStore(work_dir / "preflight.db", SEED_PATH)
    skill_manager = SkillManager(
        root_dir=SKILLS_DIR,
        max_prompt_chars=int(
            os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")
        ),
    )
    skill_manager.load()
    agent = RetailAgent(
        client=None,
        model=model,
        store=store,
        tool_manager=object(),
        skill_manager=skill_manager,
        policy_text=policy_text,
        policy_source=(
            "evaluation/data/retail_sandbox/policy.md "
            f"sha256={EXPECTED_POLICY_SHA256}"
        ),
        max_tokens=AGENT_MAX_TOKENS,
    )
    return agent, store


def _request_kwargs(agent: Any, system: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tools = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in agent.get_tools().values()
    ]
    return {
        "model": agent._model,
        "max_tokens": agent.profile.max_tokens,
        "temperature": agent.profile.temperature,
        "system": system,
        "messages": messages,
        "tools": tools,
    }


def _tool_use_block(name: str, tool_input: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "id": f"toolu_preflight_{index}",
        "name": name,
        "input": tool_input,
    }


def _tool_result_block(
    tool_use_id: str, result: Any
) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


def _simulate_case_plan(
    agent: Any,
    case: Dict[str, Any],
    sample: Dict[str, Any],
    customer_id: str,
    prices: Dict[str, Any],
) -> Dict[str, Any]:
    """按每轮允许的最大请求数构造最坏情况的请求体，只做估算。"""
    from agents.agent_orchestrator import Request

    per_turn = [int(value) for value in case.get("max_requests_per_turn") or []]
    turns = list(case.get("turns") or [])
    knowledge_turns = set(int(value) for value in case.get("knowledge_turns") or [])
    requests: List[Dict[str, Any]] = []
    pending = [
        {
            "op_id": "op-preflight",
            "order_id": "#W8171054",
            "item_ids": ["9747045638"],
            "payment_method_id": "credit_card_1565124",
            "amount": "94.01",
            "status": "displayed",
            "prepared_turn": 1,
            "displayed_turn": 1,
        }
    ]
    for index, message in enumerate(turns, start=1):
        turn_limit = per_turn[index - 1] if index - 1 < len(per_turn) else 2
        context = ""
        if index > 1:
            context = "[较早上下文已截断]\n" + ("上" * MAX_CONTEXT_CHARS)
        request = Request(
            message=message,
            user_id="preflight-client",
            conv_id="preflight-conv",
            context=context,
            turn=index,
            retail_user_id=customer_id,
            retail_pending=pending if index > 1 else [],
        )
        system = agent._build_system_prompt(request)
        messages: List[Dict[str, Any]] = []
        if request.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{request.context}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        role_packet = agent._build_role_packet(request)
        if role_packet:
            messages.append({"role": "user", "content": f"[角色输入契约]\n{role_packet}"})
            messages.append(
                {"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"}
            )
        messages.append({"role": "user", "content": message})

        result_order = [
            "knowledge" if index in knowledge_turns else "order",
            "prepare",
            "knowledge",
        ]
        for request_no in range(turn_limit):
            payload = _request_kwargs(agent, system, messages)
            hard = _hard_tokens(payload)
            plan = _plan_tokens(payload)
            requests.append(
                {
                    "turn": index,
                    "request": request_no + 1,
                    "stage": "agent",
                    "plan_input_tokens": plan,
                    "hard_input_tokens": hard,
                    "plan_cost_cny": _request_cost(
                        plan,
                        PLAN_AGENT_OUTPUT_TOKENS,
                        prices["input_price_cny_per_mtok"],
                        prices["output_price_cny_per_mtok"],
                    ),
                    "hard_cost_cny": _request_cost(
                        hard,
                        AGENT_MAX_TOKENS,
                        prices["input_price_cny_per_mtok"],
                        prices["output_price_cny_per_mtok"],
                    ),
                }
            )
            if request_no + 1 >= turn_limit:
                break
            kind = result_order[request_no % len(result_order)]
            name = {
                "knowledge": "search_knowledge_base",
                "prepare": "prepare_return",
                "order": "get_order_details",
            }[kind]
            tool_use = _tool_use_block(name, {"query": message}, request_no)
            messages.append({"role": "assistant", "content": [tool_use]})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        _tool_result_block(tool_use["id"], sample[kind])
                    ],
                }
            )

        # 知识轮单独计入改写和重排请求；按实际生产请求参数估算。
        if index in knowledge_turns:
            from mcp.tool_manager import build_rerank_prompt, build_rewrite_prompt

            rewrite_prompt = build_rewrite_prompt(message, 3)
            rewrite_payload = {
                "model": agent._model,
                "max_tokens": REWRITE_MAX_TOKENS,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": rewrite_prompt}],
            }
            rerank_prompt = build_rerank_prompt(
                message, sample["knowledge"]["results"]
            )
            rerank_payload = {
                "model": agent._model,
                "max_tokens": RERANK_MAX_TOKENS,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": rerank_prompt}],
            }
            for stage, payload, plan_output in (
                ("rewrite", rewrite_payload, PLAN_REWRITE_OUTPUT_TOKENS),
                ("rerank", rerank_payload, PLAN_RERANK_OUTPUT_TOKENS),
            ):
                hard = _hard_tokens(payload)
                plan = _plan_tokens(payload)
                requests.append(
                    {
                        "turn": index,
                        "request": len(requests) + 1,
                        "stage": stage,
                        "plan_input_tokens": plan,
                        "hard_input_tokens": hard,
                        "plan_cost_cny": _request_cost(
                            plan,
                            plan_output,
                            prices["input_price_cny_per_mtok"],
                            prices["output_price_cny_per_mtok"],
                        ),
                        "hard_cost_cny": _request_cost(
                            hard,
                            int(payload["max_tokens"]),
                            prices["input_price_cny_per_mtok"],
                            prices["output_price_cny_per_mtok"],
                        ),
                    }
                )
    plan_total = sum(item["plan_cost_cny"] for item in requests)
    hard_max = max((item["hard_cost_cny"] for item in requests), default=0.0)
    per_turn_count: Dict[int, int] = {}
    for item in requests:
        per_turn_count[item["turn"]] = per_turn_count.get(item["turn"], 0) + 1
    return {
        "case_id": case["id"],
        "channel": case.get("channel"),
        "planned_requests": len(requests),
        "declared_max_requests": int(case.get("max_requests") or 0),
        "requests_per_turn": per_turn_count,
        "planned_cost_cny": plan_total,
        "hard_upper_max_cny": hard_max,
        "requests": requests,
    }


def run_preflight(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(override=False)

    cases = _load_json(CASES_PATH)
    prices = _pricing()
    pricing = prices["data"]
    assets = _asset_record()
    services = _service_status()
    ledger = _ledger_state(float(config["budget_cny"]))
    model = str(pricing.get("model") or "")

    config_checks = {
        "model_matches_env": str(os.getenv("LLM_MODEL", "")) == model,
        "base_url_matches_pricing": str(
            os.getenv("ANTHROPIC_BASE_URL", "")
        ).rstrip("/")
        == str(pricing.get("endpoint") or "").rstrip("/"),
        "api_key_present": bool(os.getenv("ANTHROPIC_API_KEY", "").strip()),
        "pricing_currency_cny": str(pricing.get("currency")) == "CNY",
        "pricing_sources_present": bool(
            pricing.get("source_zh") and pricing.get("source_en")
        ),
        "case_count": len(cases.get("cases") or []) == 8,
        "budget_matches": float(cases.get("budget_cny")) == float(
            config["budget_cny"]
        ),
        "ledger_price_matches": float(cases.get("ledger_input_cny_per_mtok"))
        == prices["input_price_cny_per_mtok"]
        and float(cases.get("ledger_output_cny_per_mtok"))
        == prices["output_price_cny_per_mtok"],
        "ledger_not_overrun": not ledger.get("cost_stop_reason"),
        "ledger_remaining_positive": float(ledger.get("remaining_cny") or 0) > 0,
        "services_available": bool(
            services["redis"]["ok"]
            and services["chroma"]["ok"]
            and services["embedding_cache"]["ok"]
        ),
        "assets_hashes_ok": all(
            assets[name]["ok"] for name in ("seed", "policy", "knowledge", "license")
        ),
        "knowledge_count_ok": bool(assets["knowledge_count"]["ok"]),
        "seed_counts_ok": bool(assets["seed_counts_ok"]),
    }

    work_dir = run_dir / "preflight"
    work_dir.mkdir(parents=True, exist_ok=True)
    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    agent, store = _build_preflight_agent(work_dir, model, policy_text)
    sample = _sample_payloads(
        store,
        str(cases["customer"]["email"]),
        "#W8171054",
        ["9747045638"],
        "credit_card_1565124",
    )
    plans = [
        _simulate_case_plan(
            agent,
            case,
            sample,
            str(cases["customer"]["user_id"]),
            prices,
        )
        for case in cases.get("cases") or []
    ]
    planned_total = sum(item["planned_cost_cny"] for item in plans)
    hard_max = max((item["hard_upper_max_cny"] for item in plans), default=0.0)
    remaining = float(ledger.get("remaining_cny") or 0.0)
    gate = {
        "planned_total_cny": planned_total,
        "planned_limit_cny": remaining * 0.9,
        "planned_ok": planned_total <= remaining * 0.9,
        "hard_upper_max_cny": hard_max,
        "hard_limit_cny": remaining * 0.5,
        "hard_ok": hard_max <= remaining * 0.5,
    }
    gate["passed"] = bool(
        gate["planned_ok"]
        and gate["hard_ok"]
        and all(config_checks.values())
    )

    result = {
        "mode": "preflight",
        "generated_at": _now(),
        "run_id": RUN_ID,
        "run_dir": str(run_dir),
        "cases_path": str(CASES_PATH.relative_to(ROOT)),
        "pricing_path": str(PRICING_PATH.relative_to(ROOT)),
        "pricing": pricing,
        "model": model,
        "config_checks": config_checks,
        "assets": assets,
        "services": services,
        "ledger": ledger,
        "gate": gate,
        "plans": [
            {
                **plan,
                "requests": plan["requests"],
            }
            for plan in plans
        ],
        "code_hashes": _code_hashes(),
        "git": _git_state(),
        "passed": gate["passed"],
    }
    _write_json(run_dir / "preflight.json", result)
    shutil.rmtree(work_dir, ignore_errors=True)
    return result


def _event_to_turn(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "turn": int(event.get("turn") or 0),
        "message": str(event.get("user_message") or ""),
        "response": str(event.get("assistant_message") or ""),
        "tools_used": list(event.get("tools_used") or []),
        "tool_traces": list(event.get("tool_traces") or []),
        "knowledge_hits": int(event.get("knowledge_hits") or 0),
        "escalated": bool(event.get("escalated")),
        "agent_success": bool(event.get("agent_success", False)),
        "latency_ms": float(event.get("latency_ms") or 0.0),
        "request_id": str(event.get("request_id") or ""),
        "store_events": list(event.get("store_events") or []),
        "pending_after": list(event.get("pending_after") or []),
        "budget": dict(event.get("budget") or {}),
    }


def _run_api_case(
    case: Dict[str, Any],
    case_dir: Path,
    env: Dict[str, str],
    before_ids: set,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import api.main as api_main
    from fastapi.testclient import TestClient

    previous = _apply_env(env)
    turns: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "channel": "api",
        "error": "",
        "health": {},
        "monitor": {},
    }
    started = time.monotonic()
    previous_budget_requests = len(before_ids)
    try:
        with TestClient(api_main.app) as client:
            health = client.get("/health")
            meta["health"] = {
                "status_code": health.status_code,
                "body": health.json() if health.headers.get("content-type", "").startswith("application/json") else health.text[:500],
            }
            if health.status_code != 200:
                raise RuntimeError(f"/health 返回 {health.status_code}")
            for index, message in enumerate(case["turns"], start=1):
                if index > int(case["max_turns"]):
                    break
                response = client.post(
                    "/chat",
                    json={
                        "message": message,
                        "user_id": env["RETAIL_CLIENT_USER"],
                        "conv_id": env["RETAIL_CONV_ID"],
                    },
                )
                if response.status_code != 200:
                    meta["error"] = f"/chat 返回 {response.status_code}: {response.text[:500]}"
                    break
                body = response.json()
                trace = None
                try:
                    trace_response = client.get(
                        f"/trace/tool/{body.get('request_id', '')}"
                    )
                    if trace_response.status_code == 200:
                        trace = trace_response.json()
                except Exception as ex:
                    trace = {"error": f"{type(ex).__name__}: {ex}"}
                events = _turn_events(case_dir / "events.jsonl")
                event = events[-1] if len(events) >= index else {}
                budget = dict(event.get("budget") or {})
                total_requests = int(budget.get("requests") or 0)
                requests_in_turn = max(
                    0, total_requests - previous_budget_requests
                )
                previous_budget_requests = total_requests or previous_budget_requests
                turn = {
                    "turn": index,
                    "message": message,
                    "response": str(body.get("response") or ""),
                    "tools_used": list(body.get("tools_used") or []),
                    "knowledge_hits": 1 if body.get("knowledge_used") else 0,
                    "escalated": bool(body.get("escalated")),
                    "agent_success": bool(event.get("agent_success", True)),
                    "latency_ms": float(body.get("latency_ms") or 0.0),
                    "request_id": str(body.get("request_id") or ""),
                    "http_status": response.status_code,
                    "conv_id": body.get("conv_id"),
                    "trace": trace,
                    "event": event,
                    "store_events": list(event.get("store_events") or []),
                    "pending_after": list(event.get("pending_after") or []),
                    "budget": budget,
                    "requests_in_turn": requests_in_turn,
                }
                turns.append(turn)
                rows = [row for row in _ledger_rows() if row["request_id"] not in before_ids]
                if len(rows) > int(case["max_requests"]):
                    meta["error"] = "请求数超过该用例上限"
                    break
                if not turn["agent_success"]:
                    meta["error"] = "模型回合失败"
                    break
                ledger = _ledger_state(float(config["budget_cny"]))
                if ledger.get("cost_stop_reason"):
                    meta["error"] = "账本永久停止: " + str(ledger["cost_stop_reason"])
                    break
                if float(ledger.get("remaining_cny") or 0) <= 0.01:
                    meta["error"] = "预算不足，停止后续请求"
                    break
            try:
                monitor = client.get("/monitor")
                meta["monitor"] = (
                    monitor.json()
                    if monitor.status_code == 200
                    and monitor.headers.get("content-type", "").startswith("application/json")
                    else {"status_code": monitor.status_code}
                )
            except Exception as ex:
                meta["monitor"] = {"error": f"{type(ex).__name__}: {ex}"}
    except Exception as ex:
        meta["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        _restore_env(previous)
        events = _turn_events(case_dir / "events.jsonl")
        rows = [row for row in _ledger_rows() if row["request_id"] not in before_ids]
        meta["request_count"] = len(rows)
        meta["tool_count"] = sum(len(turn.get("tools_used") or []) for turn in turns)
        meta["case_cost_cny"] = sum(float(row.get("actual_cny") or 0.0) for row in rows)
        meta["duration_ms"] = (time.monotonic() - started) * 1000
        meta["turn_event_count"] = len(events)
    return turns, meta


def _run_cli_case(
    case: Dict[str, Any],
    case_dir: Path,
    env: Dict[str, str],
    before_ids: set,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "channel": "cli",
        "error": "",
        "returncode": None,
        "timed_out": False,
    }
    started = time.monotonic()
    user_input = "\n".join(list(case["turns"]) + ["退出"]) + "\n"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "api.main", "--cli"],
            cwd=str(ROOT),
            env=env,
            input=user_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
        )
        meta["returncode"] = completed.returncode
        meta["stdout_tail"] = (completed.stdout or "")[-8000:]
        meta["stderr_tail"] = (completed.stderr or "")[-4000:]
        if completed.returncode != 0:
            meta["error"] = f"CLI 退出码 {completed.returncode}"
    except subprocess.TimeoutExpired as ex:
        meta["timed_out"] = True
        meta["error"] = "CLI 超时"
        meta["stdout_tail"] = (ex.stdout or "")[-8000:] if ex.stdout else ""
        meta["stderr_tail"] = (ex.stderr or "")[-4000:] if ex.stderr else ""
    finally:
        events = _turn_events(case_dir / "events.jsonl")
        turns = [_event_to_turn(event) for event in events]
        previous_budget_requests = len(before_ids)
        for turn, event in zip(turns, events):
            budget = turn.get("budget") or {}
            total_requests = int(budget.get("requests") or 0)
            turn["requests_in_turn"] = max(
                0, total_requests - previous_budget_requests
            )
            previous_budget_requests = (
                total_requests or previous_budget_requests
            )
        rows = [row for row in _ledger_rows() if row["request_id"] not in before_ids]
        meta["request_count"] = len(rows)
        meta["tool_count"] = sum(len(turn.get("tools_used") or []) for turn in turns)
        meta["case_cost_cny"] = sum(float(row.get("actual_cny") or 0.0) for row in rows)
        meta["duration_ms"] = (time.monotonic() - started) * 1000
        meta["turn_event_count"] = len(events)
        if not turns and not meta["error"]:
            meta["error"] = "CLI 未产生回合事件"
    return turns, meta


def _collection_name(item: Any) -> str:
    if isinstance(item, str):
        return item
    return str(getattr(item, "name", item))


def _clean_case_assets(assets: Dict[str, Any]) -> Dict[str, Any]:
    """只删除本轮用例创建的 Redis 键和 Chroma 集合，并验证删除结果。"""
    import redis
    from chromadb.config import Settings
    import chromadb

    collections = [assets["collection"], *assets["memory_collections"]]
    result: Dict[str, Any] = {
        "redis_prefix": assets["redis_prefix"],
        "collections": collections,
        "deleted_keys": 0,
        "remaining_keys": 0,
        "deleted_collections": [],
        "remaining_collections": [],
        "errors": [],
    }
    try:
        client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip(),
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )
        keys = list(client.scan_iter(match=assets["redis_prefix"] + "*", count=500))
        if keys:
            result["deleted_keys"] = int(client.delete(*keys))
        result["remaining_keys"] = len(
            list(client.scan_iter(match=assets["redis_prefix"] + "*", count=500))
        )
    except Exception as ex:
        result["errors"].append(f"redis: {type(ex).__name__}: {ex}")
    try:
        chroma = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "127.0.0.1").strip(),
            port=int(os.getenv("CHROMA_PORT", "8000")),
            settings=Settings(anonymized_telemetry=False),
        )
        names = {_collection_name(item) for item in chroma.list_collections()}
        for name in collections:
            if name in names:
                chroma.delete_collection(name)
                result["deleted_collections"].append(name)
        remaining = {_collection_name(item) for item in chroma.list_collections()}
        result["remaining_collections"] = [
            name for name in collections if name in remaining
        ]
    except Exception as ex:
        result["errors"].append(f"chroma: {type(ex).__name__}: {ex}")
    return result


def run_live(
    run_dir: Path,
    config: Dict[str, Any],
    run_short: str,
) -> Dict[str, Any]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(override=False)

    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        preflight = run_preflight(run_dir, config)
    else:
        preflight = _load_json(preflight_path)
    if not preflight.get("passed"):
        return {
            "mode": "live",
            "generated_at": _now(),
            "status": "REFUSED",
            "reason": "预检未通过，未发起任何付费请求",
            "preflight": str(preflight_path),
        }

    cases = _load_json(CASES_PATH)
    prices = _pricing()
    baseline = _seed_snapshot()
    results: List[Dict[str, Any]] = []
    stopped = ""
    for case in cases.get("cases") or []:
        case_id = str(case["id"])
        case_dir = run_dir / case_id
        if case_dir.exists() and any(case_dir.iterdir()):
            raise RuntimeError(f"用例目录已存在且非空，拒绝覆盖: {case_dir}")
        case_dir.mkdir(parents=True, exist_ok=True)
        assets = _case_assets(case, run_short)
        env = _base_env(case, case_dir, assets, prices, config)
        before_ids = {row["request_id"] for row in _ledger_rows()}
        _write_json(
            case_dir / "config.json",
            {
                "case": case,
                "assets": assets,
                "env": {
                    key: value
                    for key, value in env.items()
                    if key.startswith("RETAIL_")
                    or key in {"ECHOMIND_SKILLS_DIR", "PYTHONIOENCODING", "PYTHONUNBUFFERED"}
                },
            },
        )
        if stopped:
            result = {
                "case_id": case_id,
                "status": "NOT_RUN",
                "reason": stopped,
                "passed": False,
            }
            results.append(result)
            _write_json(case_dir / "result.json", result)
            continue

        started = time.monotonic()
        try:
            if str(case.get("channel")) == "cli":
                turns, meta = _run_cli_case(
                    case, case_dir, env, before_ids, config
                )
            else:
                turns, meta = _run_api_case(
                    case, case_dir, env, before_ids, config
                )
        except Exception as ex:
            turns, meta = [], {
                "channel": str(case.get("channel")),
                "error": f"{type(ex).__name__}: {ex}",
                "request_count": 0,
                "tool_count": 0,
                "case_cost_cny": 0.0,
                "duration_ms": (time.monotonic() - started) * 1000,
            }

        if case_dir.joinpath("retail.db").is_file():
            from mcp.retail_store import RetailStore

            final_store = RetailStore(case_dir / "retail.db", SEED_PATH)
            final = final_store.business_snapshot()
            db_stats = final_store.stats()
        else:
            final = None
            db_stats = {}
        checks = _check_case(
            case,
            turns,
            meta,
            case_dir / "retail.db",
            baseline,
            final,
            assets,
            config,
        )
        cleanup = _clean_case_assets(assets)
        ledger_state = _ledger_state(float(config["budget_cny"]))
        if meta.get("error"):
            status = "FAIL"
        elif checks["passed"]:
            status = "PASS"
        else:
            status = "FAIL"
        if ledger_state.get("cost_stop_reason"):
            stopped = "账本永久停止: " + str(ledger_state["cost_stop_reason"])
        elif float(ledger_state.get("remaining_cny") or 0) <= 0.01:
            stopped = "预算不足，后续用例 NOT_RUN"

        result = {
            "case_id": case_id,
            "title": case.get("title"),
            "channel": case.get("channel"),
            "status": status,
            "passed": bool(checks["passed"]) and not meta.get("error"),
            "error": meta.get("error", ""),
            "checks": checks["checks"],
            "checks_passed": checks["passed"],
            "requests": int(meta.get("request_count") or 0),
            "tools": int(meta.get("tool_count") or 0),
            "cost_cny": float(meta.get("case_cost_cny") or 0.0),
            "duration_ms": float(meta.get("duration_ms") or 0.0),
            "ledger": ledger_state,
            "cleanup": cleanup,
            "db_stats": db_stats,
            "turns": turns,
            "meta": {
                key: value
                for key, value in meta.items()
                if key
                not in {
                    "stdout_tail",
                    "stderr_tail",
                }
            },
        }
        _write_json(case_dir / "result.json", result)
        _write_json(
            case_dir / "db_diff.json",
            {
                "baseline_counts": {
                    table: len(baseline.get(table) or {})
                    for table in ("users", "orders", "products")
                },
                "final_counts": {
                    table: len((final or {}).get(table) or {})
                    for table in ("users", "orders", "products")
                },
                "diff": _order_diff(baseline, final or {}),
            },
        )
        results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"turns"}
            }
        )
        if meta.get("error") and not stopped:
            # 运行或预算层面的失败不再重试；检查项失败仍继续其余用例。
            stopped = f"{case_id} 运行失败，停止后续付费请求"

    summary = {
        "mode": "live",
        "generated_at": _now(),
        "run_id": RUN_ID,
        "run_short": run_short,
        "run_dir": str(run_dir),
        "budget_cny": float(config["budget_cny"]),
        "ledger": _ledger_state(float(config["budget_cny"])),
        "results": results,
        "passed": bool(results)
        and all(item.get("status") == "PASS" for item in results),
        "status": (
            "PASS"
            if results and all(item.get("status") == "PASS" for item in results)
            else ("PARTIAL" if any(item.get("status") == "PASS" for item in results) else "FAIL")
        ),
        "code_hashes": _code_hashes(),
        "git": _git_state(),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _new_run_dir() -> Tuple[Path, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_short = stamp.replace("T", "").replace("Z", "")[-8:] + uuid.uuid4().hex[:2]
    run_dir = RUNS_DIR / f"{stamp}-{run_short}"
    if run_dir.exists():
        raise RuntimeError(f"运行目录已存在: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir, run_short


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="T13 零售沙箱真实运行器")
    parser.add_argument(
        "--mode",
        choices=("preflight", "live", "all"),
        default="preflight",
        help="preflight 只估算；live/all 会发起真实付费请求",
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="已有运行目录；为空时新建 data/retail/t13_live/runs/<stamp>",
    )
    args = parser.parse_args(argv)

    config = {
        "budget_cny": float(os.getenv("RETAIL_BUDGET_CNY", "5.0")),
    }
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"运行目录不存在: {run_dir}")
        run_short = run_dir.name.split("-")[-1][-10:]
    else:
        run_dir, run_short = _new_run_dir()

    if args.mode == "preflight":
        result = run_preflight(run_dir, config)
        print(
            json.dumps(
                {
                    "passed": result["passed"],
                    "gate": result["gate"],
                    "config_checks": result["config_checks"],
                    "run_dir": str(run_dir),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result["passed"] else 2

    result = run_live(run_dir, config, run_short)
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "passed": result.get("passed"),
                "ledger": result.get("ledger"),
                "results": [
                    {
                        "case_id": item.get("case_id"),
                        "status": item.get("status"),
                        "cost_cny": item.get("cost_cny"),
                        "error": item.get("error"),
                    }
                    for item in result.get("results") or []
                ],
                "run_dir": str(run_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
