"""支持场景有限真实小样本对照运行器。

T18 四场景与 T19 两个退货场景共用同一套脚本、业务保护和 5 元账本；预检
不调用收费模型。脚本用户只根据真实可见回复、已展示待确认详情和 Host 事件
推进，不读取测试端预期值，也不模拟模型。正式证据按 run id 独立存放。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "evaluation" / "data" / "retail_support_live" / "cases.json"
CASES_README = ROOT / "evaluation" / "data" / "retail_support_live" / "README.md"
RUNNER_PATH = ROOT / "evaluation" / "run_retail_support_live.py"
TEST_PATH = ROOT / "tests" / "test_retail_support_live.py"
POLICY_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "policy.md"
SEED_PATH = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
LEGACY_KNOWLEDGE_PATH = (
    ROOT / "evaluation" / "data" / "retail_sandbox" / "knowledge.jsonl"
)
SUPPORT_KNOWLEDGE_PATH = (
    ROOT / "evaluation" / "data" / "retail_support" / "knowledge.jsonl"
)
SUPPORT_MANIFEST_PATH = (
    ROOT / "evaluation" / "data" / "retail_support" / "manifest.json"
)
EVIDENCE_ROOT = ROOT / "data" / "evaluation" / "t18-support"

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com/anthropic"
BUDGET_CNY = 5.0
INPUT_CNY_PER_MTOK = 2.0
OUTPUT_CNY_PER_MTOK = 8.0
MAX_REQUESTS_PER_TURN = 32
MAX_REQUESTS_PER_TRIAL = 80
MAX_TOOL_CALLS_PER_TRIAL = 120
MODEL_TIMEOUT_S = 180.0
TRIAL_TIMEOUT_S = 900.0
PLANS: Dict[str, Dict[str, Any]] = {
    "t18": {
        "name": "t18",
        "task": "T18",
        "title": "T18 有限支持小样本",
        "run_order": [
            "S01A", "S01B", "S02B", "S02A",
            "S03A", "S03B", "S04B", "S04A",
        ],
        "pair_starts": {
            "S01A": ["S01A", "S01B"],
            "S02B": ["S02B", "S02A"],
            "S03A": ["S03A", "S03B"],
            "S04B": ["S04B", "S04A"],
        },
        "case_ids": ("S01", "S02", "S03", "S04"),
        "layers": (
            ("simple", "简单任务", ("S01", "S02")),
            ("cross_stage", "跨阶段任务", ("S03", "S04")),
        ),
        "shared_pair_denominator": 4,
        "spec_path": ROOT / "docs" / "retail" / "T18-revalidation-sources.json",
        "notes": [
            "8 条计划对话、4 个计划配对；只作初步观察，不声称统计显著、"
            "盲测、完整覆盖或普遍提升。",
            "A 为单 Agent，B 为规则路由＋确定性节点＋四个角色；差异不等于"
            "纯多 Agent 因果增益。",
            "本轮无 UserSimulator、Judge 或额外收费模型；语义复核为执行者复核。",
        ],
    },
    "t19": {
        "name": "t19",
        "task": "T19",
        "title": "T19 退货路由有限真实复验",
        "run_order": ["S02B", "S02A", "S03A", "S03B"],
        "pair_starts": {
            "S02B": ["S02B", "S02A"],
            "S03A": ["S03A", "S03B"],
        },
        "case_ids": ("S02", "S03"),
        "layers": (
            ("simple_transaction", "S02 直接退货", ("S02",)),
            ("cross_stage_restart", "S03 重启后续接退货", ("S03",)),
        ),
        "shared_pair_denominator": 2,
        "spec_path": ROOT / "docs" / "retail" / "T19-revalidation-sources.json",
        "notes": [
            "4 条计划对话、2 个计划配对；只判断 S02／S03 的已知退货路由缺陷"
            "是否关闭，不把未计划的 S01／S04 算作缺失。",
            "A 为单 Agent，B 为规则路由＋确定性节点＋四个角色；跨轮角色切换"
            "按实际记录描述，不称为并行协作。",
            "本轮无 UserSimulator、Judge 或额外收费模型；语义复核为执行者复核。",
        ],
    },
}
DEFAULT_PLAN = "t18"
PLANS["t19-final"] = {
    **PLANS["t19"],
    "name": "t19-final",
    "title": "T19 多Agent基线最终验收",
    "run_order": ["S02B", "S02A", "S03A", "S03B", "S01B", "S04B"],
    "pair_starts": {
        "S02B": ["S02B", "S02A"],
        "S03A": ["S03A", "S03B"],
        "S01B": ["S01B"],
        "S04B": ["S04B"],
    },
    "case_ids": ("S01", "S02", "S03", "S04"),
    "paired_case_ids": ("S02", "S03"),
    "unpaired_run_keys": ("S01B", "S04B"),
    "layers": (
        ("simple", "解决与直接退货", ("S01", "S02")),
        ("cross_stage", "重启退货与人工", ("S03", "S04")),
    ),
    "spec_path": ROOT / "docs" / "retail" / "T19-final-freeze.json",
    "notes": [
        "6条计划对话：multi四场景，single两条退货兼容对照；只有S02／S03成对。",
        "同一原5元账本；先冻结后运行，保留全部历史与失败，不宣称总体优势。",
    ],
}
# 保留既有导入名；正式执行以显式 plan 为准。
RUN_ORDER = list(PLANS["t18"]["run_order"])
PAIR_STARTS = dict(PLANS["t18"]["pair_starts"])
SPEC_PATH = PLANS["t18"]["spec_path"]
GROWTH_INPUT_TOKENS = 3000
SUPPORT_STEP_IDS = ("PAIR-01", "PAIR-02", "PAIR-03")
SUPPORT_EVIDENCE_IDS = (
    "T17.E01",
    "T17.E02",
    "T17.E03",
    "T17.E04",
    "T17.E05",
    "T17.E06",
)
CLIENT_USER = "t18-client"
CONV_PREFIX = "t18-conv"
SERVICES = (
    ("agent-ecommerce-redis", 6380),
    ("agent-ecommerce-chromadb", 8002),
)
ALLOWED_NEW_STATUS = (
    "?? evaluation/data/retail_support_live/",
    "?? evaluation/run_retail_support_live.py",
    "?? tests/test_retail_support_live.py",
)
# 既有集成测试用相对路径 "unused" 建持久 Chroma，会在仓库根留下该目录。
TRANSIENT_TEST_STATUS = "?? unused/"
EXPECTED_HEAD = "ac05c71a8a4e0fbac273bbc029a56b084e538682"

PLANNED_CALLS: Dict[str, Dict[str, Dict[str, int]]] = {
    # 计划口径按冻结脚本的最少对话轮估算；只为预检门槛，不是费用承诺。
    "S01": {
        "A": {"agent": 6},
        "B": {"roles": 6, "intent": 4, "composer": 1},
        "retrieval": {"rewrite": 2, "rerank": 2},
    },
    "S02": {
        "A": {"agent": 10},
        "B": {"roles": 10, "intent": 8, "composer": 1},
        "retrieval": {"rewrite": 0, "rerank": 0},
    },
    "S03": {
        "A": {"agent": 12},
        "B": {"roles": 12, "intent": 8, "composer": 1},
        "retrieval": {"rewrite": 2, "rerank": 2},
    },
    "S04": {
        "A": {"agent": 7},
        "B": {"roles": 7, "intent": 4, "composer": 1},
        "retrieval": {"rewrite": 2, "rerank": 2},
    },
}
PLANNED_OUTPUT_TOKENS = {
    "agent": 1024,
    "roles": 1024,
    "intent": 256,
    "composer": 1024,
    "rewrite": 512,
    "rerank": 1024,
}
STAGE_LIMITS = {
    "agent": (2048, 0.1),
    "roles": (2048, 0.1),
    "intent": (2048, 0.1),
    "composer": (8192, 0.1),
    "rewrite": (1024, 0.3),
    "rerank": (8192, 0.0),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_plan(plan: Any = None) -> Dict[str, Any]:
    """只允许冻结计划；未显式指定时保持 T18 兼容。"""
    if isinstance(plan, dict):
        return plan
    key = str(plan or DEFAULT_PLAN).strip().lower()
    if key not in PLANS:
        raise ValueError(f"未知计划: {key!r}")
    return PLANS[key]


def _plan_record(plan: Any = None) -> Dict[str, Any]:
    item = _resolve_plan(plan)
    return {
        "name": item["name"],
        "task": item["task"],
        "title": item["title"],
        "run_order": list(item["run_order"]),
        "pair_starts": {
            key: list(value) for key, value in item["pair_starts"].items()
        },
        "case_ids": list(item["case_ids"]),
        "paired_case_ids": list(item.get("paired_case_ids", item["case_ids"])),
        "unpaired_run_keys": list(item.get("unpaired_run_keys", ())),
        "layers": [
            {"key": key, "label": label, "cases": list(cases)}
            for key, label, cases in item["layers"]
        ],
        "shared_pair_denominator": int(item["shared_pair_denominator"]),
        "spec_path": str(item["spec_path"].relative_to(ROOT)),
        "notes": list(item["notes"]),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True) + "\n"
        )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_project_env() -> None:
    """只在命令入口读取 .env；测试导入本模块时不自动读取。"""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(override=False)


def _run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", str(value or "").strip())
    cleaned = cleaned.strip("-._")
    if not cleaned:
        raise ValueError("run-id 不能为空")
    return cleaned[:60]


def _run_slug(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", run_id.replace("t18-support-", "t18"))[:24]


def _run_dir(run_id: str) -> Path:
    return EVIDENCE_ROOT / run_id


def _ledger_path(ledger_run_id: str) -> Path:
    """账本仍按原 run 目录存放；复验证据使用独立目录。"""
    return _run_dir(ledger_run_id) / "ledger.sqlite"


def _git(args: List[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 失败: {result.stderr[-500:]}")
    return result.stdout


def _status_present(expected: str, actual: List[str]) -> bool:
    if expected in actual:
        return True
    if expected.startswith("?? ") and expected.endswith("/"):
        prefix = expected[:-1]
        return any(line.startswith(prefix) for line in actual)
    return False


def _allowed_new_status(line: str) -> bool:
    for allowed in ALLOWED_NEW_STATUS:
        if line == allowed:
            return True
        if allowed.startswith("?? ") and allowed.endswith("/"):
            if line.startswith(allowed[:-1]):
                return True
    return False


def _is_transient_test_artifact(line: str) -> bool:
    """只豁免既有测试生成的根目录 Chroma 缓存，其他未跟踪项仍阻断。"""
    if line != TRANSIENT_TEST_STATUS:
        return False
    path = ROOT / "unused"
    if not path.is_dir() or not (path / "chroma.sqlite3").is_file():
        return False
    children = [item for item in path.iterdir()]
    return bool(children) and all(
        item.name == "chroma.sqlite3"
        or (item.is_dir() and item.name.count("-") == 4)
        for item in children
    )


def _verify_worktree(plan: Any = None) -> Dict[str, Any]:
    frozen = _resolve_plan(plan)
    spec_path = Path(frozen["spec_path"])
    spec = _load_json(spec_path)
    head = _git(["rev-parse", "HEAD"]).strip()
    status_output = _git(["status", "--porcelain"]).splitlines()
    baseline = list(spec.get("public_worktree_status_before") or [])
    missing = [line for line in baseline if not _status_present(line, status_output)]
    transient = [
        line
        for line in status_output
        if line == TRANSIENT_TEST_STATUS and _is_transient_test_artifact(line)
    ]
    extra = [
        line
        for line in status_output
        if line not in baseline
        and not _allowed_new_status(line)
        and line not in transient
    ]
    hash_rows: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    for entry in spec.get("protected_public_files") or []:
        path = ROOT / str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "")
        exists = path.is_file()
        actual = _sha256_file(path) if exists else ""
        row = {
            "path": str(entry.get("path") or ""),
            "expected": expected,
            "actual": actual,
            "match": exists and actual == expected,
        }
        hash_rows.append(row)
        if not row["match"]:
            mismatches.append(row)
    tracked_count = len(_git(["ls-files"]).splitlines())
    checks = {
        "plan": frozen["name"],
        "spec_path": str(spec_path.relative_to(ROOT)),
        "spec_sha256": _sha256_file(spec_path),
        "head": head,
        "head_matches": head == EXPECTED_HEAD,
        "baseline_missing": missing,
        "unexpected_status": extra,
        "transient_test_artifacts": transient,
        "protected_count": len(hash_rows),
        "protected_mismatches": mismatches,
        "tracked_files_count": tracked_count,
        "tracked_files_count_matches": tracked_count == int(
            spec.get("tracked_files_count") or -1
        ),
    }
    checks["ok"] = (
        checks["head_matches"]
        and not missing
        and not extra
        and not mismatches
        and checks["tracked_files_count_matches"]
    )
    return checks


def _docker_state(name: str) -> str:
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as ex:
        return f"docker_unavailable:{ex}"
    if result.returncode != 0:
        return "missing"
    return result.stdout.strip() or "unknown"


def _docker_action(action: str, name: str) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", action, name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=90,
        )
        return {
            "action": action,
            "container": name,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:500],
            "stderr": result.stderr.strip()[:500],
        }
    except (OSError, subprocess.SubprocessError) as ex:
        return {
            "action": action,
            "container": name,
            "returncode": -1,
            "stderr": str(ex),
        }


def _chroma_heartbeat(host: str, port: int, timeout: float = 2.0) -> bool:
    for path in ("/api/v2/heartbeat", "/api/v1/heartbeat"):
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}{path}", timeout=timeout
            ) as response:
                if 200 <= int(response.status) < 300:
                    return True
        except Exception:
            continue
    return False


def _redis_tcp(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_services(chroma_host: str, chroma_port: int) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "initial": {},
        "actions": [],
        "chroma_ok": False,
        "redis_ok": False,
        "started_by_run": [],
    }
    for name, port in SERVICES:
        state = _docker_state(name)
        record["initial"][name] = state
        if state != "running" and not state.startswith("docker_unavailable"):
            record["actions"].append(_docker_action("start", name))
            record["started_by_run"].append(name)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        record["chroma_ok"] = _chroma_heartbeat(chroma_host, chroma_port)
        if record["chroma_ok"]:
            break
        time.sleep(1.0)
    redis_host = "127.0.0.1"
    record["redis_ok"] = _redis_tcp(redis_host, 6380)
    record["final"] = {name: _docker_state(name) for name, _ in SERVICES}
    return record


def _restore_services(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not record:
        return {"restored": False, "reason": "no service record"}
    actions: List[Dict[str, Any]] = []
    for name in record.get("started_by_run") or []:
        actions.append(_docker_action("stop", name))
    record["restore_actions"] = actions
    record["restored"] = all(item.get("returncode") == 0 for item in actions)
    record["final_after_restore"] = {
        name: _docker_state(name) for name, _ in SERVICES
    }
    return record


def _embedding_record() -> Dict[str, Any]:
    from mcp.knowledge_base import (
        _EMBEDDING_REQUIRED_FILES,
        _ProjectEmbeddingFunction,
        _embedding_cache_ready,
    )

    folder = Path(_ProjectEmbeddingFunction.DOWNLOAD_PATH) / "onnx"
    files = []
    for name in _EMBEDDING_REQUIRED_FILES:
        path = folder / name
        files.append(
            {
                "name": name,
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
        )
    return {
        "implementation": "_ProjectEmbeddingFunction(ONNXMiniLM_L6_V2)",
        "model_name": _ProjectEmbeddingFunction.MODEL_NAME,
        "download_path": str(folder),
        "ready": bool(_embedding_cache_ready()),
        "files": files,
        "download_attempted": False,
    }


def _load_cases(plan: Any = None) -> Dict[str, Any]:
    frozen = _resolve_plan(plan)
    data = _load_json(CASES_PATH)
    if data.get("version") != "t18.4":
        raise RuntimeError(f"支持脚本版本不符: {data.get('version')!r}")
    if data.get("run_order") != RUN_ORDER:
        raise RuntimeError("冻结脚本的固定运行顺序不符")
    cases = {str(item["id"]): item for item in data.get("cases") or []}
    if set(cases) != {"S01", "S02", "S03", "S04"}:
        raise RuntimeError("冻结脚本必须有 S01—S04 四个案例")
    paired_cases = frozen.get("paired_case_ids", frozen["case_ids"])
    unpaired = set(frozen.get("unpaired_run_keys", ()))
    expected = {
        f"{case_id}{arm}"
        for case_id in paired_cases
        for arm in ("A", "B")
    } | unpaired
    if (
        set(frozen["run_order"]) != expected
        or len(frozen["run_order"]) != len(expected)
        or {key[:-1] for key in expected} != set(frozen["case_ids"])
        or any(key[-1] not in ("A", "B") for key in unpaired)
        or any(key[:-1] in paired_cases for key in unpaired)
        or int(frozen["shared_pair_denominator"]) != len(paired_cases)
    ):
        raise RuntimeError(f"{frozen['task']} 计划与配对／单独验收范围不符")
    if any(case_id not in cases for case_id in frozen["case_ids"]):
        raise RuntimeError(f"{frozen['task']} 引用了冻结脚本中不存在的案例")
    covered: set[str] = set()
    for key, pair in frozen["pair_starts"].items():
        valid_group = (
            len(pair) == 1 and pair[0] in unpaired
        ) or (
            len(pair) == 2 and key[:-1] in paired_cases
            and set(pair) == {key[:-1] + "A", key[:-1] + "B"}
        )
        if (
            key not in frozen["run_order"] or not valid_group
            or pair[0] != key or covered.intersection(pair)
        ):
            raise RuntimeError(f"{frozen['task']} 配对起点或长度不符: {key}")
        covered.update(pair)
    if covered != expected:
        raise RuntimeError(f"{frozen['task']} 配对未完整覆盖计划案例")
    data["case_map"] = cases
    data["plan"] = _plan_record(frozen)
    return data


def _pricing_snapshot() -> Dict[str, Any]:
    return {
        "checked_at": "2026-10-20",
        "currency": "CNY",
        "unit": "per_mtok",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "official_model_name": "DeepSeek-V4.1-Flash-1017",
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
        "peak_cache_miss_input": 2.0,
        "output": 8.0,
        "cache_hit_input": 0.04,
        "off_peak_input": 1.0,
        "off_peak_output": 4.0,
        "reservation_input": INPUT_CNY_PER_MTOK,
        "reservation_output": OUTPUT_CNY_PER_MTOK,
        "note": (
            "执行前复核官方当前模型与价目；账本按峰值缓存未命中输入价和输出价"
            "保守预留，不使用缓存命中或错峰价。历史价格快照未覆盖。"
        ),
    }


def _source_manifest(
    run_id: str, worktree: Dict[str, Any], plan: Any = None
) -> Dict[str, Any]:
    frozen = _resolve_plan(plan)
    support_manifest = _load_json(SUPPORT_MANIFEST_PATH)
    return {
        "task": frozen["task"],
        "plan": _plan_record(frozen),
        "run_id": run_id,
        "created_at": _now(),
        "head": worktree.get("head"),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "sources": {
            "runner": {
                "path": "evaluation/run_retail_support_live.py",
                "sha256": _sha256_file(RUNNER_PATH),
            },
            "cases": {
                "path": "evaluation/data/retail_support_live/cases.json",
                "sha256": _sha256_file(CASES_PATH),
            },
            "readme": {
                "path": "evaluation/data/retail_support_live/README.md",
                "sha256": _sha256_file(CASES_README),
            },
            "test": {
                "path": "tests/test_retail_support_live.py",
                "sha256": _sha256_file(TEST_PATH) if TEST_PATH.is_file() else "",
            },
            "policy": {
                "path": "evaluation/data/retail_sandbox/policy.md",
                "sha256": _sha256_file(POLICY_PATH),
            },
            "seed": {
                "path": "evaluation/data/retail_sandbox/db.json",
                "sha256": _sha256_file(SEED_PATH),
            },
            "legacy_knowledge": {
                "path": "evaluation/data/retail_sandbox/knowledge.jsonl",
                "sha256": _sha256_file(LEGACY_KNOWLEDGE_PATH),
            },
            "support_knowledge": {
                "path": "evaluation/data/retail_support/knowledge.jsonl",
                "sha256": _sha256_file(SUPPORT_KNOWLEDGE_PATH),
            },
            "support_manifest": {
                "path": "evaluation/data/retail_support/manifest.json",
                "sha256": _sha256_file(SUPPORT_MANIFEST_PATH),
            },
        },
        "support_manifest_format": support_manifest.get("format"),
        "support_evidence_ids": list(SUPPORT_EVIDENCE_IDS),
        "worktree": worktree,
        "pricing": _pricing_snapshot(),
    }


def _source_mismatches(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, entry in (manifest.get("sources") or {}).items():
        rel = str((entry or {}).get("path") or "")
        expected = str((entry or {}).get("sha256") or "")
        path = ROOT / rel
        actual = _sha256_file(path) if path.is_file() else ""
        if not expected or actual != expected:
            rows.append(
                {
                    "key": key,
                    "path": rel,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return rows


def _load_knowledge_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _collection_name(prefix: str, key: str) -> str:
    raw = f"{prefix}_{key}".lower()
    cleaned = re.sub(r"[^a-z0-9._-]", "_", raw)
    cleaned = re.sub(r"^[^a-z0-9]+", "t", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+$", "0", cleaned)
    return cleaned[:63]


def _chroma_client(chroma_host: str, chroma_port: int, chroma_path: str):
    import chromadb

    settings = chromadb.Settings(anonymized_telemetry=False)
    client = chromadb.HttpClient(
        host=chroma_host, port=chroma_port, settings=settings
    )
    try:
        client.heartbeat()
        return client, "server"
    except Exception:
        client = chromadb.PersistentClient(path=str(chroma_path), settings=settings)
        return client, "local"


def _collection_names(client: Any) -> List[str]:
    names: List[str] = []
    for item in client.list_collections():
        name = getattr(item, "name", None)
        names.append(str(name if name is not None else item))
    return sorted(names)


def _assert_collection_absent(client: Any, name: str) -> None:
    if name in _collection_names(client):
        raise RuntimeError(f"本次运行集合已存在，拒绝复用或清理: {name}")


def _delete_collection(client: Any, name: str) -> Dict[str, Any]:
    before = _collection_names(client)
    if name not in before:
        return {"name": name, "deleted": False, "reason": "not_found"}
    client.delete_collection(name)
    after = _collection_names(client)
    return {"name": name, "deleted": name not in after, "before": before, "after": after}


def _preflight_chroma(
    chroma_host: str,
    chroma_port: int,
    chroma_path: str,
    run_id: str,
) -> Dict[str, Any]:
    from mcp.knowledge_base import KnowledgeBase

    client, backend = _chroma_client(chroma_host, chroma_port, chroma_path)
    name = _collection_name(f"t18pre_{_run_slug(run_id)}", uuid.uuid4().hex[:8])
    _assert_collection_absent(client, name)
    legacy = _load_knowledge_rows(LEGACY_KNOWLEDGE_PATH)
    support = _load_knowledge_rows(SUPPORT_KNOWLEDGE_PATH)
    if len(legacy) != 29 or len(support) != 6:
        raise RuntimeError(
            f"知识片段数量不符: legacy={len(legacy)} support={len(support)}"
        )
    kb = KnowledgeBase(
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=str(chroma_path),
        collection_name=name,
        load_demo_documents=False,
    )
    added = kb.upsert_evidence(list(legacy) + list(support))
    count = kb.doc_count
    ids = kb.list_ids()
    positive = kb.search("无线耳机蓝牙搜不到怎么配对", top_k=10)
    negative = kb.search("订单退款多久到账", top_k=10)
    result = {
        "collection": name,
        "client_backend": backend,
        "kb_backend": kb.backend,
        "added": int(added),
        "doc_count": int(count),
        "ids": ids,
        "positive_query": "无线耳机蓝牙搜不到怎么配对",
        "positive": positive,
        "negative_query": "订单退款多久到账",
        "negative": negative,
        "positive_support_ids": sorted(
            {
                str(item.get("evidence_id") or item.get("id") or "")
                for item in positive
                if str(item.get("product_id") or "") == "9924732112"
                and str(item.get("issue") or "") == "bluetooth_pairing"
            }
        ),
        "negative_note": "仅记录，不作为质量评分；非空不等于通过。",
    }
    result["deleted"] = _delete_collection(client, name)
    result["ok"] = (
        count == 35
        and set(ids) == {
            str(row.get("evidence_id") or "")
            for row in list(legacy) + list(support)
        }
        and bool(result["positive_support_ids"])
        and result["deleted"].get("deleted") is True
    )
    return result


async def _preflight_mcp(
    temp_dir: Path,
    seed_path: Path,
    mcp_stderr_log: Path,
) -> Dict[str, Any]:
    from agents.agent_orchestrator import Request
    from agents.retail_tools import retail_tools
    from mcp.retail_client import RetailMCPClient
    from mcp.retail_store import RetailStore

    temp_dir.mkdir(parents=True, exist_ok=True)
    db_path = temp_dir / "preflight-mcp.sqlite"
    store = RetailStore(db_path, seed_path, language="zh")
    expected = retail_tools(store, None, language="zh", support_enabled=True)
    client = RetailMCPClient(
        db_path=db_path,
        seed_path=seed_path,
        language="zh",
        expected_specs=expected,
        project_root=ROOT,
        python_executable=sys.executable,
        startup_timeout=30.0,
        call_timeout=30.0,
        stderr_log_path=mcp_stderr_log,
    )
    record: Dict[str, Any] = {}
    try:
        await client.start()
        tools = client.tools()
        preflight_conv = "preflight-mcp"
        turn = store.begin_turn(CLIENT_USER, preflight_conv)
        req = Request(
            message="preflight product lookup",
            user_id=CLIENT_USER,
            conv_id=preflight_conv,
            turn=turn,
        )
        product = await client.call_tool(
            "get_product_details", req, {"product_id": "9924732112"}
        )
        record = {
            "tools": tools,
            "tool_count": len(tools),
            "product_success": bool(product.get("success")),
            "product_result": product,
            "call_count": client.call_count,
        }
        record["ok"] = (
            len(tools) == 9
            and bool(product.get("success"))
            and set(tools)
            == set(
                {
                    "find_user_id_by_email",
                    "find_user_id_by_name_zip",
                    "get_user_details",
                    "get_order_details",
                    "get_product_details",
                    "calculate",
                    "prepare_return",
                    "return_delivered_order_items",
                    "transfer_to_human_agents",
                }
            )
        )
    finally:
        await client.close()
    return record


def _build_probe_context(case_message: str) -> Dict[str, Any]:
    """构造预检测量用的真实支持包，不写入任何业务库。"""
    from agents.agent_orchestrator import Request
    from agents.retail_support import (
        analyze_message,
        build_support_packet,
    )

    analysis = analyze_message(case_message, None, None, has_pending=False)
    packet = build_support_packet(
        None,
        analysis,
        case_message,
        1,
        None,
        [],
        max_chars=6000,
    )
    req = Request(
        message=case_message,
        user_id="preflight-client",
        conv_id="preflight-conv",
        turn=1,
        retail_user_id=None,
        retail_pending=[],
        retail_guard={},
        retail_support=packet,
        retail_support_enabled=True,
    )
    return {"request": req, "analysis": analysis, "packet": packet}


def _first_request_measure(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    system: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    from evaluation.retail_budget import request_input_upper_tokens

    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    upper = request_input_upper_tokens(kwargs)
    size = len(
        json.dumps(
            kwargs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return {
        "request_bytes": size,
        "input_tokens_proxy": max(1, (size + 2) // 3),
        "input_tokens_upper": int(upper),
        "max_tokens": int(max_tokens),
        "hard_upper_cny": round(
            upper * INPUT_CNY_PER_MTOK / 1_000_000
            + max_tokens * OUTPUT_CNY_PER_MTOK / 1_000_000,
            6,
        ),
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


def _agent_messages(agent: Any, req: Any) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if req.context:
        messages.append({"role": "user", "content": "[背景信息]\n" + req.context})
        messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
    if req.entities:
        messages.append(
            {
                "role": "user",
                "content": "[结构化实体]\n"
                + json.dumps(req.entities, ensure_ascii=False),
            }
        )
        messages.append(
            {"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"}
        )
    packet = agent._build_role_packet(req)
    if packet:
        messages.append({"role": "user", "content": "[角色输入契约]\n" + packet})
        messages.append(
            {
                "role": "assistant",
                "content": "好的，我会按照该角色的输入和输出契约处理。",
            }
        )
    messages.append({"role": "user", "content": req.message})
    return messages


def _preflight_measure(
    cases: Dict[str, Any], plan: Any = None
) -> Dict[str, Any]:
    import tempfile

    from agents.retail_agent import RetailAgent
    from agents.retail_multi_agent import (
        RetailBillingAgent,
        RetailEscalationAgent,
        RetailGeneralAgent,
        RetailTechnicalAgent,
    )
    from agents.retail_tools import retail_knowledge_tool, retail_tools
    from core.intent_recognizer import build_intent_prompt
    from mcp.retail_store import RetailStore
    from mcp.tool_manager import (
        MCPToolManager,
        build_rerank_prompt,
        build_rewrite_prompt,
    )

    policy_text = POLICY_PATH.read_text(encoding="utf-8")
    frozen = _resolve_plan(plan)
    case = cases["case_map"][str(frozen["case_ids"][0])]
    message = str(case["script"]["initial"])
    probe = _build_probe_context(message)
    req = probe["request"]
    tmp = Path(tempfile.mkdtemp(prefix="t18-preflight-"))
    try:
        store = RetailStore(tmp / "measure.sqlite", SEED_PATH, language="zh")
        tool_manager = MCPToolManager(
            api_key="preflight-unused", model=MODEL, client=object()
        )
        tools = retail_tools(
            store, tool_manager, language="zh", support_enabled=True
        )
        tools["search_knowledge_base"] = retail_knowledge_tool(
            tool_manager, support_enabled=True
        )
        schema = _tool_schema(tools)
        agent = RetailAgent(
            client=None,
            model=MODEL,
            store=store,
            tool_manager=tool_manager,
            skill_manager=None,
            policy_text=policy_text,
            policy_source="evaluation/data/retail_sandbox/policy.md",
            max_tokens=2048,
            language="zh",
            support_enabled=True,
        )
        a_agent = _first_request_measure(
            model=MODEL,
            max_tokens=2048,
            temperature=0.1,
            system=agent._build_system_prompt(req),
            messages=_agent_messages(agent, req),
            tools=_tool_schema(agent.get_tools()),
        )
        roles: Dict[str, Any] = {}
        for name, cls in (
            ("general", RetailGeneralAgent),
            ("technical", RetailTechnicalAgent),
            ("billing", RetailBillingAgent),
            ("escalation", RetailEscalationAgent),
        ):
            role = cls(
                client=None,
                model=MODEL,
                language="zh",
                policy_text=policy_text,
                support_enabled=True,
            )
            role.set_shared_tools(tools)
            role_tools = role._tools_for_request(req)
            roles[name] = _first_request_measure(
                model=MODEL,
                max_tokens=2048,
                temperature=0.1,
                system=role._build_system_prompt(req),
                messages=_agent_messages(role, req),
                tools=_tool_schema(role_tools),
            )
        intent = _first_request_measure(
            model=MODEL,
            max_tokens=2048,
            temperature=0.1,
            system="",
            messages=[
                {
                    "role": "user",
                    "content": build_intent_prompt(message, None, language="zh"),
                }
            ],
        )
        composer_prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造订单、退款、后台查询结果；如果结论冲突，明确说明需要核验；"
            "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            "主 Agent：technical\n"
            f"用户问题：{message}\n"
            "候选结果：\n[technical Agent 输出]\n"
            + "请先检查耳机是否有电，并确认电脑蓝牙已打开。" * 8
            + "\n\n[billing Agent 输出]\n"
            + "如仍需退货，请提供订单号和邮箱，并等待完整详情确认。" * 8
        )
        composer = _first_request_measure(
            model=MODEL,
            max_tokens=8192,
            temperature=0.1,
            system="",
            messages=[{"role": "user", "content": composer_prompt}],
        )
        legacy = _load_knowledge_rows(LEGACY_KNOWLEDGE_PATH)
        support = _load_knowledge_rows(SUPPORT_KNOWLEDGE_PATH)
        rewrite = _first_request_measure(
            model=MODEL,
            max_tokens=1024,
            temperature=0.3,
            system="",
            messages=[
                {
                    "role": "user",
                    "content": build_rewrite_prompt(message, 3),
                }
            ],
        )
        rerank = _first_request_measure(
            model=MODEL,
            max_tokens=8192,
            temperature=0.0,
            system="",
            messages=[
                {
                    "role": "user",
                    "content": build_rerank_prompt(
                        message, (legacy + support)[:20]
                    ),
                }
            ],
        )
        return {
            "probe_message": message,
            "probe_scope": probe["analysis"].scope_state,
            "growth_input_tokens": GROWTH_INPUT_TOKENS,
            "A": {"agent": a_agent},
            "B": {
                "roles": roles,
                "intent": intent,
                "composer": composer,
            },
            "retrieval": {"rewrite": rewrite, "rerank": rerank},
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _measurement_for_stage(measurements: Dict[str, Any], arm: str, stage: str) -> Dict[str, Any]:
    if stage in ("rewrite", "rerank"):
        return measurements["retrieval"][stage]
    if arm == "A":
        return measurements["A"]["agent"]
    if stage == "roles":
        roles = measurements["B"]["roles"]
        return max(roles.values(), key=lambda item: item["input_tokens_upper"])
    return measurements["B"][stage]


def _planned_case_cost(
    case_id: str, arm: str, measurements: Dict[str, Any]
) -> Dict[str, Any]:
    calls = dict(PLANNED_CALLS[case_id][arm])
    rows: Dict[str, Any] = {}
    total = 0.0
    for stage, count in calls.items():
        measured = _measurement_for_stage(measurements, arm, stage)
        input_tokens = int(measured["input_tokens_proxy"]) + GROWTH_INPUT_TOKENS
        output_tokens = int(PLANNED_OUTPUT_TOKENS[stage])
        cost = count * (
            input_tokens * INPUT_CNY_PER_MTOK / 1_000_000
            + output_tokens * OUTPUT_CNY_PER_MTOK / 1_000_000
        )
        rows[stage] = {
            "calls": count,
            "input_tokens_proxy_each": input_tokens,
            "planned_output_tokens_each": output_tokens,
            "planned_cny": round(cost, 6),
        }
        total += cost
    for stage, count in PLANNED_CALLS[case_id]["retrieval"].items():
        if count <= 0:
            continue
        measured = measurements["retrieval"][stage]
        input_tokens = int(measured["input_tokens_proxy"]) + GROWTH_INPUT_TOKENS
        output_tokens = int(PLANNED_OUTPUT_TOKENS[stage])
        cost = count * (
            input_tokens * INPUT_CNY_PER_MTOK / 1_000_000
            + output_tokens * OUTPUT_CNY_PER_MTOK / 1_000_000
        )
        rows[f"retrieval_{stage}"] = {
            "calls": count,
            "input_tokens_proxy_each": input_tokens,
            "planned_output_tokens_each": output_tokens,
            "planned_cny": round(cost, 6),
        }
        total += cost
    return {"rows": rows, "planned_cny": round(total, 6)}


def _planned_estimate(
    measurements: Dict[str, Any],
    remaining_cny: float,
    plan: Any = None,
) -> Dict[str, Any]:
    frozen = _resolve_plan(plan)
    cases: Dict[str, Any] = {}
    total = 0.0
    for run_key in frozen["run_order"]:
        item = _planned_case_cost(run_key[:-1], run_key[-1], measurements)
        cases[run_key] = item
        total += item["planned_cny"]
    worst = 0.0
    for arm, stages in (
        ("A", ["agent"]),
        ("B", ["roles", "intent", "composer"]),
    ):
        for stage in stages:
            measured = _measurement_for_stage(measurements, arm, stage)
            max_tokens = STAGE_LIMITS[stage][0]
            worst = max(
                worst,
                (
                    int(measured["input_tokens_upper"]) + GROWTH_INPUT_TOKENS
                )
                * INPUT_CNY_PER_MTOK
                / 1_000_000
                + max_tokens * OUTPUT_CNY_PER_MTOK / 1_000_000,
            )
    for stage in ("rewrite", "rerank"):
        measured = measurements["retrieval"][stage]
        max_tokens = STAGE_LIMITS[stage][0]
        worst = max(
            worst,
            (
                int(measured["input_tokens_upper"]) + GROWTH_INPUT_TOKENS
            )
            * INPUT_CNY_PER_MTOK
            / 1_000_000
            + max_tokens * OUTPUT_CNY_PER_MTOK / 1_000_000,
        )
    return {
        "cases": cases,
        "planned_total_cny": round(total, 6),
        "remaining_cny": round(remaining_cny, 6),
        "gate_90pct_remaining_cny": round(remaining_cny * 0.9, 6),
        "planned_ok": total <= remaining_cny * 0.9 + 1e-12,
        "worst_single_request_upper_cny": round(worst, 6),
        "worst_request_gate_50pct_remaining_cny": round(
            remaining_cny * 0.5, 6
        ),
        "worst_request_ok": worst <= remaining_cny * 0.5 + 1e-12,
        "note": (
            "计划按请求体字节数/3 加 3000 token 余量估算；实际账本按每个"
            "请求的完整字节数+512 与真实 max_tokens 硬预留。"
        ),
    }


def _create_ledger(
    run_id: str,
    ledger_run_id: Optional[str] = None,
    *,
    for_preflight: bool = False,
):
    """打开账本；复验只允许追加既有账本，不新建额度。

    `run_id` 是本次证据目录，`ledger_run_id` 是原 T18 账本身份。两者不同
    时先复制到预检目录做只读计量，正式执行仍写回原账本。
    """
    from evaluation.retail_ledger import RetailLedger

    ledger_run_id = ledger_run_id or run_id
    source = _ledger_path(ledger_run_id)
    if ledger_run_id != run_id:
        if not source.is_file():
            raise RuntimeError(
                f"复验账本不存在，拒绝新建额度: {source}"
            )
        if for_preflight:
            snapshot = _run_dir(run_id) / "preflight" / "ledger_snapshot.sqlite"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
            path = snapshot
        else:
            path = source
    else:
        path = source
    ledger = RetailLedger(
        path=path,
        budget_cny=BUDGET_CNY,
        input_price_cny_per_mtok=INPUT_CNY_PER_MTOK,
        output_price_cny_per_mtok=OUTPUT_CNY_PER_MTOK,
        run_id=ledger_run_id,
        model=MODEL,
    )
    return ledger


def _preflight(
    run_id: str,
    ledger_run_id: Optional[str] = None,
    plan: Any = None,
) -> int:
    _load_project_env()
    frozen = _resolve_plan(plan)
    run_dir = _run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "live_started.json").is_file():
        print("已存在 live_started.json；不覆盖正式证据，请用 summarize 读取。")
        return 2

    cases = _load_cases(frozen)
    pricing = _pricing_snapshot()
    worktree = _verify_worktree(frozen)
    manifest = _source_manifest(run_id, worktree, frozen)
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "pricing.json", pricing)

    checks: Dict[str, Any] = {"worktree": worktree}
    status = "PASS"
    blocked_reason = ""
    if not worktree.get("ok"):
        status = "BLOCKED_WORKTREE"
        blocked_reason = (
            f"工作树或受保护文件哈希与 {frozen['task']} 冻结清单不一致"
        )
    source_mismatches = _source_mismatches(manifest)
    checks["source_mismatches"] = source_mismatches
    if status == "PASS" and source_mismatches:
        status = "BLOCKED_WORKTREE"
        blocked_reason = "预检源码清单包含不存在的文件或空哈希"

    embedding = _embedding_record()
    checks["embedding"] = embedding
    if status == "PASS" and not embedding.get("ready"):
        status = "ENVIRONMENT_BLOCKED"
        blocked_reason = "本地 Embedding 缓存不完整，禁止下载"

    env_model = os.getenv("LLM_MODEL", "").strip()
    env_base = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    env_key = bool(os.getenv("ANTHROPIC_API_KEY", "").strip())
    checks["env"] = {
        "model": env_model,
        "model_matches": env_model == MODEL,
        "base_url_matches": env_base.rstrip("/") == ENDPOINT.rstrip("/"),
        "api_key_present": env_key,
    }
    if status == "PASS" and not (
        checks["env"]["model_matches"]
        and checks["env"]["base_url_matches"]
        and checks["env"]["api_key_present"]
    ):
        status = "ENVIRONMENT_BLOCKED"
        blocked_reason = "模型、端点或密钥环境与冻结值不一致"

    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8002").strip() or 8002)
    chroma_path = (
        os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
        or str(ROOT / "data" / "chroma")
    )
    service_record: Optional[Dict[str, Any]] = None
    mcp_record: Dict[str, Any] = {}
    chroma_record: Dict[str, Any] = {}
    measurements: Dict[str, Any] = {}
    if status == "PASS":
        service_record = _ensure_services(chroma_host, chroma_port)
        checks["services"] = service_record
        if not service_record.get("chroma_ok"):
            status = "ENVIRONMENT_BLOCKED"
            blocked_reason = "ChromaDB 服务不可达"
        else:
            try:
                chroma_record = _preflight_chroma(
                    chroma_host, chroma_port, chroma_path, run_id
                )
            except Exception as ex:
                chroma_record = {"ok": False, "error": str(ex)}
            checks["chroma"] = chroma_record
            if not chroma_record.get("ok"):
                status = "ENVIRONMENT_BLOCKED"
                blocked_reason = "真实 Chroma 检索或 35 段证据校验失败"
            if status == "PASS":
                mcp_stderr = run_dir / "preflight" / "mcp_stderr.log"
                try:
                    mcp_record = asyncio.run(
                        _preflight_mcp(
                            run_dir / "preflight" / "mcp",
                            SEED_PATH,
                            mcp_stderr,
                        )
                    )
                except Exception as ex:
                    mcp_record = {"ok": False, "error": str(ex)}
                checks["mcp"] = mcp_record
                if not mcp_record.get("ok"):
                    status = "ENVIRONMENT_BLOCKED"
                    blocked_reason = "MCP stdio 九个业务工具预检失败"
            if status == "PASS":
                try:
                    measurements = _preflight_measure(cases, frozen)
                except Exception as ex:
                    measurements = {"error": str(ex)}
                checks["measurements"] = measurements
                if "error" in measurements:
                    status = "ENVIRONMENT_BLOCKED"
                    blocked_reason = "请求与 Prompt 测量失败"

    ledger_run_id = ledger_run_id or run_id
    ledger_source = _ledger_path(ledger_run_id)
    ledger_summary: Dict[str, Any] = {}
    try:
        ledger = _create_ledger(
            run_id, ledger_run_id, for_preflight=True
        )
        ledger_summary = ledger.summary()
        ledger_summary["ledger_path"] = str(ledger_source)
        if ledger_run_id != run_id:
            ledger_summary["snapshot_path"] = str(ledger.path)
    except Exception as ex:
        if status == "PASS":
            status = "ENVIRONMENT_BLOCKED"
            blocked_reason = f"账本衔接检查失败: {ex}"
    ledger_source_record = {
        "ledger_run_id": ledger_run_id,
        "path": str(ledger_source),
        "exists": ledger_source.is_file(),
        "sha256": (
            _sha256_file(ledger_source) if ledger_source.is_file() else ""
        ),
        "spent_cny": ledger_summary.get("spent_cny"),
        "remaining_cny": ledger_summary.get("remaining_cny"),
        "requests": ledger_summary.get("requests"),
    }
    checks["ledger"] = ledger_summary
    checks["ledger_source"] = ledger_source_record
    estimate: Dict[str, Any] = {}
    if status == "PASS":
        estimate = _planned_estimate(
            measurements,
            float(ledger_summary["remaining_cny"]),
            frozen,
        )
        checks["plan"] = estimate
        if not estimate.get("planned_ok"):
            status = "BLOCKED_BUDGET_PLAN"
            blocked_reason = "计划费用超过剩余预算的 90%"
        elif not estimate.get("worst_request_ok"):
            status = "BLOCKED_BUDGET_PLAN"
            blocked_reason = "最坏单请求预留超过剩余预算的 50%"

    result = {
        "status": status,
        "blocked_reason": blocked_reason,
        "run_id": run_id,
        "ledger_run_id": ledger_run_id,
        "ledger_source": ledger_source_record,
        "checked_at": _now(),
        "head": worktree.get("head"),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "plan": _plan_record(frozen),
        "pricing": pricing,
        "checks": checks,
        "estimate": estimate,
        "ledger": ledger_summary,
        "notes": [
            "预检不调用收费模型；UserSimulator 与 Judge 请求数为 0。",
            "沿用唯一账本 5 元硬上限，不新建 run、额度或历史账目。",
        ],
    }
    _write_json(run_dir / "preflight.json", result)
    if service_record is not None:
        restored = _restore_services(service_record)
        checks["services_after_preflight"] = restored
        _write_json(run_dir / "preflight.json", result)
    print(
        f"preflight {status}: plan={frozen['name']} "
        f"planned={estimate.get('planned_total_cny', '-')} "
        f"remaining={ledger_summary.get('remaining_cny')} "
        f"reason={blocked_reason or '-'}",
        flush=True,
    )
    return 0 if status == "PASS" else 2


class _GuardAgent:
    """只补齐单 Agent 路径的停止传播；不改变任何业务逻辑。"""

    def __init__(self, inner: Any):
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def handle(self, req: Any) -> Any:
        if req.retail_guard is None:
            req.retail_guard = {}
        return await self._inner.handle(req)


class _Recorder:
    """逐请求记录真实请求、响应和阶段；按请求体哈希关联账本。"""

    def __init__(self, trial_dir: Path, run_id: str, case_id: str, arm: str):
        self.trial_dir = trial_dir
        self.run_id = run_id
        self.case_id = case_id
        self.arm = arm
        self.seq = 0
        self.turn = 0
        self._wrapped: set[int] = set()
        self.requests_path = trial_dir / "requests.jsonl"
        self.retrieval_path = trial_dir / "retrieval.jsonl"
        self.requests: List[Dict[str, Any]] = []

    def set_turn(self, turn: int) -> None:
        self.turn = int(turn)

    def attach(self, client: Any, stage: str) -> bool:
        if client is None or id(client) in self._wrapped:
            return False
        original = client.messages.create

        async def create(**kwargs: Any) -> Any:
            started = time.monotonic()
            self.seq += 1
            seq = self.seq
            actual_stage = stage
            if stage == "retrieval":
                if int(kwargs.get("max_tokens") or 0) <= 1024:
                    actual_stage = "rewrite"
                else:
                    actual_stage = "rerank"
            payload = {
                "seq": seq,
                "turn": self.turn,
                "stage": actual_stage,
                "request_sha256": _request_sha(kwargs),
                "request_bytes": len(
                    json.dumps(
                        kwargs,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=_json_default,
                    ).encode("utf-8")
                ),
                "request": kwargs,
                "started_at": _now(),
            }
            try:
                response = await original(**kwargs)
            except Exception as ex:
                payload["error"] = str(ex)[:1000]
                payload["latency_ms"] = round(
                    (time.monotonic() - started) * 1000, 1
                )
                _append_jsonl(self.requests_path, payload)
                self.requests.append(payload)
                raise
            payload["latency_ms"] = round(
                (time.monotonic() - started) * 1000, 1
            )
            payload["response"] = {
                "id": getattr(response, "id", ""),
                "model": getattr(response, "model", ""),
                "stop_reason": getattr(response, "stop_reason", ""),
                "content": _jsonable(getattr(response, "content", [])),
                "usage": _jsonable(getattr(response, "usage", None)),
            }
            _append_jsonl(self.requests_path, payload)
            self.requests.append(payload)
            return response

        client.messages.create = create
        self._wrapped.add(id(client))
        return True

    def record_retrieval(
        self,
        *,
        query: str,
        diagnostics: Dict[str, Any],
        result: Any,
    ) -> None:
        payload = {
            "turn": self.turn,
            "query": query,
            "diagnostics": _jsonable(diagnostics),
            "success": bool(getattr(result, "success", False)),
            "degraded": bool(getattr(result, "degraded", False)),
            "reranked": bool(getattr(result, "reranked", False)),
            "final": _jsonable(getattr(result, "data", None)),
        }
        _append_jsonl(self.retrieval_path, payload)


def _request_sha(kwargs: Dict[str, Any]) -> str:
    from evaluation.retail_budget import _json_default

    payload = json.dumps(
        kwargs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    raise TypeError(f"无法序列化请求字段: {type(value).__name__}")


def _apply_runtime_adaptations(
    runtime: Any,
    ledger: Any,
    recorder: _Recorder,
) -> Dict[str, Any]:
    """补齐单 Agent 停止传播、逐轮上限与逐阶段诊断，不改业务链。"""
    config = runtime.config
    inner = getattr(runtime.client, "_inner", None)
    timeout_applied = False
    if inner is not None and getattr(inner, "messages", None) is not None:
        original = inner.messages.create

        async def create_with_timeout(**kwargs: Any) -> Any:
            kwargs.setdefault("timeout", MODEL_TIMEOUT_S)
            return await original(**kwargs)

        inner.messages.create = create_with_timeout
        timeout_applied = True
    if config.agent_mode == "single":
        runtime.agent._client = runtime.client.for_stage(
            "agent", count_turn=True
        )
    runtime.service.agent = _GuardAgent(runtime.agent)

    attached: Dict[str, bool] = {}
    if config.agent_mode == "single":
        attached["agent"] = recorder.attach(
            getattr(runtime.agent, "_client", None), "agent"
        )
    else:
        orchestrator = runtime.agent.orchestrator
        attached["roles"] = recorder.attach(
            getattr(orchestrator, "_client", None), "roles"
        )
        attached["intent"] = recorder.attach(
            getattr(getattr(orchestrator, "_intent_recognizer", None), "client", None),
            "intent",
        )
        attached["composer"] = recorder.attach(
            getattr(getattr(orchestrator, "_composer", None), "_client", None),
            "composer",
        )
    if runtime.tool_manager is not None:
        # RAG 改写与重排也计入同一个外部用户轮次上限。
        runtime.tool_manager._client = runtime.client.for_stage(
            "retrieval", count_turn=True
        )
        attached["retrieval_model"] = recorder.attach(
            getattr(runtime.tool_manager, "_client", None), "retrieval"
        )
        original_search = runtime.tool_manager.search_with_rewrite

        async def search_with_diagnostics(
            tool_name: str,
            query: str,
            top_k: int = 5,
            context: Optional[Dict[str, Any]] = None,
            diagnostics: Optional[Dict[str, Any]] = None,
            item_filter: Any = None,
        ) -> Any:
            diag = diagnostics if diagnostics is not None else {}
            result = await original_search(
                tool_name,
                query,
                top_k=top_k,
                context=context,
                diagnostics=diag,
                item_filter=item_filter,
            )
            recorder.record_retrieval(
                query=query, diagnostics=diag, result=result
            )
            return result

        runtime.tool_manager.search_with_rewrite = search_with_diagnostics
        attached["retrieval"] = True
    return {
        "agent_mode": config.agent_mode,
        "timeout_seconds": MODEL_TIMEOUT_S,
        "timeout_applied": timeout_applied,
        "max_retries": getattr(inner, "max_retries", None),
        "attached_stages": attached,
        "single_turn_count_enabled": config.agent_mode == "single",
    }


def _trial_config(
    *,
    run_id: str,
    ledger_run_id: Optional[str] = None,
    case_id: str,
    arm: str,
    trial_dir: Path,
    collection: str,
    chroma_host: str,
    chroma_port: int,
    chroma_path: str,
) -> Any:
    from agents.retail_runtime import RetailRuntimeConfig

    redis_url = os.getenv(
        "REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"
    )
    ledger_run_id = ledger_run_id or run_id
    return RetailRuntimeConfig(
        db_path=trial_dir / "retail.sqlite",
        seed_path=SEED_PATH,
        policy_path=POLICY_PATH,
        knowledge_path=LEGACY_KNOWLEDGE_PATH,
        knowledge_collection=_collection_name(
            f"t18base_{_run_slug(run_id)}", f"{case_id}{arm}"
        ),
        ledger_path=_ledger_path(ledger_run_id),
        event_log_path=trial_dir / "events.jsonl",
        budget_cny=BUDGET_CNY,
        input_price_cny_per_mtok=INPUT_CNY_PER_MTOK,
        output_price_cny_per_mtok=OUTPUT_CNY_PER_MTOK,
        model=MODEL,
        base_url=ENDPOINT,
        redis_url=redis_url,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=Path(chroma_path),
        case_id=f"{case_id}{arm}",
        run_id=ledger_run_id,
        profile_enabled=False,
        memory_key_prefix=f"retail:t18:{_run_slug(run_id)}:{case_id}{arm}:",
        memory_collection_suffix="_t18",
        max_tokens=2048,
        max_message_chars=4000,
        max_context_chars=12000,
        mcp_timeout_seconds=30.0,
        mcp_stderr_log=trial_dir / "mcp_stderr.log",
        agent_mode="single" if arm == "A" else "multi",
        enable_memory=False,
        enable_knowledge=True,
        language="zh",
        support_enabled=True,
        support_collection=collection,
        support_max_chars=6000,
    )


@contextmanager
def _trial_env(config: Any):
    values = {
        "RETAIL_MODE": "true",
        "RETAIL_LANGUAGE": "zh",
        "RETAIL_SUPPORT_ENABLED": "true",
        "RETAIL_AGENT_MODE": config.agent_mode,
        "RETAIL_ENABLE_MEMORY": "false",
        "RETAIL_PROFILE_ENABLED": "false",
        "RETAIL_DB_PATH": str(config.db_path),
        "RETAIL_EVENT_LOG": str(config.event_log_path),
        "RETAIL_BUDGET_LEDGER": str(config.ledger_path),
        "RETAIL_BUDGET_CNY": "5",
        "RETAIL_INPUT_CNY_PER_MTOK": "2",
        "RETAIL_OUTPUT_CNY_PER_MTOK": "8",
        "RETAIL_RUN_ID": config.run_id,
        "RETAIL_CASE_ID": config.case_id,
        "RETAIL_COLLECTION": config.knowledge_collection,
        "RETAIL_SUPPORT_COLLECTION": config.support_collection,
        "RETAIL_AGENT_MAX_TOKENS": "2048",
        "RETAIL_MAX_MESSAGE_CHARS": "4000",
        "RETAIL_MAX_CONTEXT_CHARS": "12000",
        "RETAIL_MCP_TIMEOUT_SECONDS": "30",
        "RETAIL_MCP_STDERR_LOG": str(config.mcp_stderr_log),
        "RETAIL_ENABLE_KNOWLEDGE": "true",
        "RETAIL_SUPPORT_MAX_CHARS": "6000",
        "CHROMA_HOST": config.chroma_host,
        "CHROMA_PORT": str(config.chroma_port),
        "CHROMA_PERSIST_DIRECTORY": str(config.chroma_path),
    }
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_db_rows(path: Path, table: str) -> List[Dict[str, Any]]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=30.0)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(f"SELECT * FROM {table}").fetchall()
    return [dict(row) for row in rows]


def _snapshot_order(snapshot: Dict[str, Any], order_id: str = "W1067251") -> Any:
    """按快照实际键读取目标订单；只接受精确键或可选“#”前缀。"""
    orders = snapshot.get("orders") if isinstance(snapshot, dict) else None
    if not isinstance(orders, dict):
        return None
    raw = str(order_id or "").strip()
    if not raw:
        return None
    candidates = [raw]
    candidates.append(raw[1:] if raw.startswith("#") else f"#{raw}")
    for key in candidates:
        if key in orders:
            return orders[key]
    return None


def _business_state(store: Any, client_user: str, conv_id: str) -> Dict[str, Any]:
    snapshot = store.business_snapshot()
    db_path = Path(store.db_path)
    return {
        "snapshot": snapshot,
        "snapshot_sha256": _sha256_text(
            json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        ),
        "target_user": snapshot.get("users", {}).get("raj_sanchez_2970"),
        "target_order": _snapshot_order(snapshot),
        "return_operations": _read_db_rows(db_path, "return_operations"),
        "pending_returns": _read_db_rows(db_path, "pending_returns"),
        "sessions": _read_db_rows(db_path, "sessions"),
        "support_state_count": len(
            _read_db_rows(db_path, "retail_support_state")
        ),
        "events": store.list_events(client_user, conv_id),
    }


def _business_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    old = before.get("snapshot") or {}
    new = after.get("snapshot") or {}
    changed: Dict[str, Dict[str, Any]] = {}
    for table in ("users", "orders", "products"):
        old_rows = old.get(table) or {}
        new_rows = new.get(table) or {}
        for key in sorted(set(old_rows) | set(new_rows)):
            if old_rows.get(key) != new_rows.get(key):
                changed.setdefault(table, {})[key] = new_rows.get(key)
    return {
        "before_sha256": before.get("snapshot_sha256"),
        "after_sha256": after.get("snapshot_sha256"),
        "changed": changed,
        "target_user_before": before.get("target_user"),
        "target_user_after": after.get("target_user"),
        "target_order_before": before.get("target_order"),
        "target_order_after": after.get("target_order"),
        "return_operations_before": before.get("return_operations"),
        "return_operations_after": after.get("return_operations"),
        "pending_before": before.get("pending_returns"),
        "pending_after": after.get("pending_returns"),
        "sessions_before": before.get("sessions"),
        "sessions_after": after.get("sessions"),
    }


def _normalize_order_id(value: Any) -> str:
    return str(value or "").strip().lstrip("#")


def _pending_matches(pending: Dict[str, Any], fixture: Dict[str, Any]) -> bool:
    items = pending.get("items") or {}
    item_ids = [str(item) for item in (items.get("item_ids") or [])]
    return (
        _normalize_order_id(pending.get("order_id"))
        == _normalize_order_id(fixture.get("order_id"))
        and item_ids == [str(fixture.get("item_id"))]
        and int(pending.get("amount_cents") or 0)
        == int(fixture.get("amount_cents") or 0)
        and str(pending.get("payment_method_id") or "")
        == str(fixture.get("payment_method_id") or "")
    )


def _pending_row_matches(row: Dict[str, Any], fixture: Dict[str, Any]) -> bool:
    """核对数据库原始 pending 行；不改变业务 ID 合同。"""
    items = row.get("items")
    if not isinstance(items, dict):
        try:
            items = json.loads(row.get("items_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            items = {}
    item_ids = [str(item) for item in (items.get("item_ids") or [])]
    return (
        _normalize_order_id(row.get("order_id"))
        == _normalize_order_id(fixture.get("order_id"))
        and item_ids == [str(fixture.get("item_id"))]
        and int(row.get("amount_cents") or 0)
        == int(fixture.get("amount_cents") or 0)
        and str(row.get("payment_method_id") or "")
        == str(fixture.get("payment_method_id") or "")
    )


def _new_suggestion(record: Optional[Dict[str, Any]], turn: int) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    for item in record.get("suggestions") or []:
        step_id = str(item.get("step_id") or "")
        if (
            int(item.get("turn") or 0) == int(turn)
            and step_id in SUPPORT_STEP_IDS
            and not item.get("previously_tried")
        ):
            return item
    return None


def _support_search_facts(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    traces: List[Dict[str, Any]] = []
    for turn in turns:
        for trace in turn.get("tool_traces") or []:
            if str(trace.get("tool_name") or "") != "search_knowledge_base":
                continue
            result = trace.get("result") if isinstance(trace, dict) else None
            results = result.get("results") if isinstance(result, dict) else []
            support_hits = [
                {
                    "evidence_id": str(item.get("evidence_id") or item.get("id") or ""),
                    "product_id": str(item.get("product_id") or ""),
                    "issue": str(item.get("issue") or ""),
                    "step_id": str(item.get("step_id") or ""),
                    "source_kind": str(item.get("source_kind") or ""),
                }
                for item in results or []
                if isinstance(item, dict)
                and str(item.get("product_id") or "") == "9924732112"
                and str(item.get("issue") or "") == "bluetooth_pairing"
            ]
            traces.append(
                {
                    "turn": turn.get("index"),
                    "success": bool(trace.get("success")),
                    "result_success": trace.get("result_success"),
                    "degraded": bool(
                        result.get("degraded")
                        if isinstance(result, dict)
                        else False
                    ),
                    "reranked": bool(
                        result.get("reranked")
                        if isinstance(result, dict)
                        else False
                    ),
                    "support_hits": support_hits,
                }
            )
    return {
        "traces": traces,
        "real_search": any(
            item["success"]
            and item["result_success"] is True
            and not item["degraded"]
            and item["support_hits"]
            for item in traces
        ),
    }


def _asks_refund_destination(text: str) -> bool:
    from agents.retail_support import asks_refund_destination

    return asks_refund_destination(text)


class ScriptedUser:
    """固定脚本用户；只观察真实可见文本、Host 状态和业务事件。"""

    def __init__(self, case: Dict[str, Any], fixture: Dict[str, Any]):
        self.case = case
        self.fixture = fixture
        self.clarification_used = False
        self.display_asked = False
        self.confirmation_sent = False
        self.feedback_sent = False
        self.human_sent = False
        self.return_sent = False
        self.refund_destination_sent = False
        self.support_clarification_used = False
        self.script_error = ""
        self.restarted = False
        self.turn_assistant: Dict[int, str] = {}
        self.turn_record: Dict[int, Optional[Dict[str, Any]]] = {}

    def observe(self, turn: Dict[str, Any]) -> None:
        self.turn_assistant[int(turn["index"])] = str(
            turn.get("assistant_message") or ""
        )
        self.turn_record[int(turn["index"])] = turn.get("record_after")

    def should_restart(self, turns: List[Dict[str, Any]]) -> bool:
        if self.case["id"] != "S03" or self.restarted:
            return False
        if not turns:
            return False
        last = turns[-1]
        if _new_suggestion(last.get("record_after"), int(last["index"])) is None:
            return False
        # 给退货诉求、必要的去向选择和展示后确认留出三个用户轮。
        return len(turns) <= int(self.case["max_user_turns"]) - 3

    def mark_restart(self) -> None:
        self.restarted = True

    def _branch(self, branch_id: str) -> Optional[str]:
        for branch in self.case["script"].get("branches") or []:
            if branch.get("id") == branch_id:
                return str(branch.get("message") or "")
        return None

    def next_message(
        self,
        turn_index: int,
        *,
        pending: List[Dict[str, Any]],
        record: Optional[Dict[str, Any]],
        submitted: bool,
    ) -> Tuple[Optional[str], str]:
        if turn_index == 0:
            return str(self.case["script"]["initial"]), "initial"
        if submitted:
            return None, "submitted"
        if self.case["id"] == "S03" and self.restarted and not self.return_sent:
            self.return_sent = True
            return self._branch("resolved_unresolved_then_return"), (
                "resolved_unresolved_then_return"
            )
        if pending:
            if len(pending) != 1:
                return None, "multiple_pending"
            item = pending[0]
            if not _pending_matches(item, self.fixture):
                return None, "wrong_pending"
            status = str(item.get("status") or "")
            if status == "prepared":
                if self.display_asked:
                    return None, "pending_prepared_not_displayed"
                self.display_asked = True
                return self._branch("ask_for_complete_display"), (
                    "ask_for_complete_display"
                )
            if status == "displayed":
                displayed_turn = int(item.get("displayed_turn") or 0)
                visible = self.turn_assistant.get(displayed_turn, "")
                display_text = str(item.get("display_text") or "").strip()
                if (
                    displayed_turn <= 0
                    or displayed_turn >= turn_index + 1
                    or not display_text
                    or display_text not in visible
                ):
                    return None, "display_not_visible_in_prior_turn"
                confirmation = (self.case["script"].get("confirmation") or {}).get(
                    "message"
                )
                if not confirmation:
                    return None, "confirmation_not_defined"
                if self.confirmation_sent:
                    return None, "confirmation_sent_waiting_submit"
                self.confirmation_sent = True
                return str(confirmation), "confirmation"
        if self.case["id"] == "S03" and not self.return_sent:
            if self.support_clarification_used:
                return None, "no_applicable_suggestion"
            self.support_clarification_used = True
            return self._branch("ask_non_repeated_step"), "ask_non_repeated_step"
        if self.case["id"] in ("S02", "S03"):
            visible = self.turn_assistant.get(turn_index, "")
            if _asks_refund_destination(visible):
                if self.refund_destination_sent:
                    return None, "refund_destination_repeated"
                self.refund_destination_sent = True
                return self._branch("refund_destination_choice"), "refund_destination_choice"
            if self.refund_destination_sent:
                return None, "no_pending_after_refund_choice"
            from agents.retail_support import asks_return_information

            if not asks_return_information(visible):
                if re.search(r"无法|不能|不支持|不符合|不可|未找到|不属于|没有.{0,12}(?:权限|工具)", visible):
                    return None, "agent_declined_return"
                self.script_error = "unsupported_return_reply"
                return None, "script_error"
            if self.clarification_used:
                return None, "no_pending"
            self.clarification_used = True
            return self._branch("identity_or_target_clarification"), (
                "identity_or_target_clarification"
            )
        if self.case["id"] == "S01":
            if self.feedback_sent:
                return None, "feedback_sent"
            if _new_suggestion(record, turn_index) is not None:
                self.feedback_sent = True
                return self._branch("resolved_feedback"), "resolved_feedback"
            if self.clarification_used:
                return None, "no_applicable_suggestion"
            self.clarification_used = True
            return self._branch("ask_specific_step"), "ask_specific_step"
        if self.case["id"] == "S04":
            if self.human_sent:
                return None, "human_sent"
            if _new_suggestion(record, turn_index) is not None:
                self.human_sent = True
                return self._branch("human_no_return"), "human_no_return"
            if self.clarification_used:
                return None, "no_applicable_suggestion"
            self.clarification_used = True
            return self._branch("ask_specific_step"), "ask_specific_step"
        return None, "script_exhausted"


def _new_entries(ledger: Any, before_ids: set[str]) -> List[Dict[str, Any]]:
    return [
        item for item in ledger.entries() if item["request_id"] not in before_ids
    ]


def _cost_rows(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    total = 0.0
    for entry in entries:
        stage = str(entry.get("stage") or "")
        row = rows.setdefault(
            stage,
            {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cny": 0.0,
                "failures": 0,
                "unsettled": 0,
            },
        )
        row["requests"] += 1
        row["input_tokens"] += int(entry.get("input_tokens") or 0)
        row["output_tokens"] += int(entry.get("output_tokens") or 0)
        charged = entry.get("actual_cny")
        if charged is None:
            charged = entry.get("reserved_cny") or 0.0
            row["unsettled"] += 1
        row["cny"] += float(charged or 0.0)
        if entry.get("status") == "failed":
            row["failures"] += 1
        total += float(charged or 0.0)
    for row in rows.values():
        row["cny"] = round(row["cny"], 6)
    return {"by_stage": rows, "total_cny": round(total, 6)}


def _settle_orphans(ledger: Any, before_ids: set[str], reason: str) -> List[str]:
    from evaluation.retail_ledger import RetailReservation

    settled: List[str] = []
    for entry in ledger.entries():
        if entry["request_id"] in before_ids or entry["status"] != "reserved":
            continue
        reservation = RetailReservation(
            request_id=entry["request_id"],
            stage=entry["stage"],
            model=entry["model"],
            input_tokens_upper=int(entry["input_tokens_upper"]),
            max_tokens=int(entry["max_tokens"]),
            reserved_cny=float(entry["reserved_cny"]),
        )
        ledger.settle_failure(reservation, reason, 0.0)
        settled.append(entry["request_id"])
    return settled


async def _restart_runtime(
    runtime: Any,
    config: Any,
    ledger: Any,
    recorder: _Recorder,
    client_user: str,
    conv_id: str,
    turn_index: int,
) -> Tuple[Any, Dict[str, Any]]:
    from agents.retail_runtime import build_retail_runtime

    before_business = _business_state(runtime.store, client_user, conv_id)
    before_support = runtime.store.load_support_state(client_user, conv_id)
    before_pending = runtime.store.active_pending(client_user, conv_id)
    last_event_id = 0
    for event in runtime.store.list_events(client_user, conv_id):
        last_event_id = max(last_event_id, int(event["event_id"]))
    t0 = time.monotonic()
    await runtime.close()
    close_ms = round((time.monotonic() - t0) * 1000, 1)
    t1 = time.monotonic()
    new_runtime = build_retail_runtime(config)
    build_ms = round((time.monotonic() - t1) * 1000, 1)
    _apply_runtime_adaptations(new_runtime, ledger, recorder)
    t2 = time.monotonic()
    await new_runtime.start()
    start_ms = round((time.monotonic() - t2) * 1000, 1)
    new_runtime.service._last_event_id[(client_user, conv_id)] = last_event_id
    after_support = new_runtime.store.load_support_state(client_user, conv_id)
    after_business = _business_state(new_runtime.store, client_user, conv_id)
    after_pending = new_runtime.store.active_pending(client_user, conv_id)
    preserved = {}
    for key in ("taken_steps", "suggestions", "feedback", "unresolved", "demand"):
        preserved[key] = (before_support or {}).get(key) == (after_support or {}).get(key)
    pending_preserved = before_pending == after_pending
    record = {
        "trigger_turn": turn_index,
        "timings": {
            "close_ms": close_ms,
            "build_ms": build_ms,
            "start_ms": start_ms,
        },
        "before_support": before_support,
        "after_support": after_support,
        "preserved": preserved,
        "preserved_ok": all(preserved.values()),
        "return_flow_before": (before_support or {}).get("return_flow"),
        "return_flow_after": (after_support or {}).get("return_flow"),
        "stage_before": (before_support or {}).get("stage"),
        "stage_after": (after_support or {}).get("stage"),
        "pending_before": before_pending,
        "pending_after": after_pending,
        "pending_preserved": pending_preserved,
        "before_business": before_business,
        "after_business": after_business,
        "conditional_stage_return_operations": len(
            before_business.get("return_operations") or []
        ),
        "conditional_stage_clean": not (
            before_business.get("return_operations") or []
        )
        and (before_business.get("target_order") == after_business.get("target_order")),
        "last_event_id": last_event_id,
        "collection_doc_count": (
            new_runtime.knowledge_base.doc_count
            if new_runtime.knowledge_base is not None
            else None
        ),
        "mcp_tool_count": len(new_runtime.mcp_client.tools()),
        "session_turns_before": (before_business.get("sessions") or [{}])[0].get(
            "turn_count"
        ),
        "session_turns_after": (after_business.get("sessions") or [{}])[0].get(
            "turn_count"
        ),
    }
    return new_runtime, record


async def _conversation(
    runtime: Any,
    config: Any,
    case: Dict[str, Any],
    arm: str,
    fixture: Dict[str, Any],
    client_user: str,
    conv_id: str,
    ledger: Any,
    recorder: _Recorder,
    trial_dir: Path,
    deadline: float,
    trial_entry_ids: set[str],
) -> Dict[str, Any]:
    script = ScriptedUser(case, fixture)
    turns: List[Dict[str, Any]] = []
    restart: Optional[Dict[str, Any]] = None
    status = "COMPLETED"
    stop_reason = ""
    current = runtime
    script_stop_reason = ""
    mcp_calls = 0
    while True:
        if time.monotonic() >= deadline:
            status = "TIMEOUT"
            stop_reason = "trial_timeout"
            break
        if turns:
            entries = _new_entries(ledger, trial_entry_ids)
            if len(entries) >= MAX_REQUESTS_PER_TRIAL:
                status = "LOOP_LIMIT"
                stop_reason = "trial_model_request_limit"
                break
            tool_calls = mcp_calls + int(current.mcp_client.call_count)
            if tool_calls >= MAX_TOOL_CALLS_PER_TRIAL:
                status = "LOOP_LIMIT"
                stop_reason = "trial_tool_call_limit"
                break
        if script.case["id"] == "S03" and not script.restarted:
            if script.should_restart(turns):
                try:
                    mcp_calls += int(current.mcp_client.call_count)
                    current, restart = await _restart_runtime(
                        current,
                        config,
                        ledger,
                        recorder,
                        client_user,
                        conv_id,
                        len(turns),
                    )
                    script.mark_restart()
                    _write_json(trial_dir / "restart.json", restart)
                except Exception as ex:
                    status = "ERROR"
                    stop_reason = f"restart_failed: {ex}"
                    break
        record = current.store.load_support_state(client_user, conv_id)
        pending = current.store.active_pending(client_user, conv_id)
        events = current.store.list_events(client_user, conv_id)
        submitted = any(
            str(item.get("event_type")) == "return_requested" for item in events
        )
        message, branch = script.next_message(
            len(turns),
            pending=pending,
            record=record,
            submitted=submitted,
        )
        if message is None:
            script_stop_reason = branch
            if script.script_error:
                status = "EVALUATION_ERROR"
                stop_reason = script.script_error
            break
        if len(turns) >= int(case["max_user_turns"]):
            status = "LOOP_LIMIT"
            stop_reason = "user_turn_limit"
            break
        turn_index = len(turns) + 1
        used_requests = len(_new_entries(ledger, trial_entry_ids))
        turn_cap = min(MAX_REQUESTS_PER_TURN, MAX_REQUESTS_PER_TRIAL - used_requests)
        if turn_cap <= 0:
            status = "LOOP_LIMIT"
            stop_reason = "trial_model_request_limit"
            break
        if config.agent_mode == "multi":
            # RetailMultiAgent 会在 handle 内重置轮上限；同步剩余额度。
            current.agent.max_requests_per_turn = turn_cap
        recorder.set_turn(turn_index)
        ledger.begin_turn(f"{case['id']}{arm}-t{turn_index}", turn_cap)
        before_ids = {item["request_id"] for item in ledger.entries()}
        started = time.monotonic()
        result = None
        error: Optional[BaseException] = None
        try:
            remaining = max(1.0, deadline - time.monotonic())
            result = await asyncio.wait_for(
                current.service.handle_turn(message, client_user, conv_id),
                timeout=remaining,
            )
        except Exception as ex:
            error = ex
        wait_ms = round((time.monotonic() - started) * 1000, 1)
        new_entries = _new_entries(ledger, before_ids)
        if error is not None:
            from evaluation.retail_budget import (
                BudgetExceeded,
                TurnRequestLimitExceeded,
            )

            if isinstance(error, TurnRequestLimitExceeded):
                status = "LOOP_LIMIT"
                stop_reason = f"turn_request_limit: {error}"
            elif isinstance(error, BudgetExceeded):
                status = "BUDGET_STOP"
                stop_reason = f"budget_stop: {error}"
            elif isinstance(error, asyncio.TimeoutError):
                status = "TIMEOUT"
                stop_reason = "handle_turn_timeout"
            else:
                status = "ERROR"
                stop_reason = f"{type(error).__name__}: {error}"
            _settle_orphans(ledger, trial_entry_ids, stop_reason)
            turns.append(
                {
                    "index": turn_index,
                    "branch": branch,
                    "user_message": message,
                    "assistant_message": "",
                    "success": False,
                    "error": stop_reason,
                    "wait_ms": wait_ms,
                    "request_ids": [item["request_id"] for item in new_entries],
                    "cost": _cost_rows(new_entries),
                    "tool_traces": [],
                    "tools_used": [],
                }
            )
            break
        assert result is not None
        pending_after = current.store.active_pending(client_user, conv_id)
        record_after = current.store.load_support_state(client_user, conv_id)
        events_after = current.store.list_events(client_user, conv_id)
        new_event_rows = [
            item
            for item in events_after
            if int(item["event_id"]) > int(events[-1]["event_id"] if events else 0)
        ]
        turn = {
            "index": turn_index,
            "branch": branch,
            "user_message": message,
            "assistant_message": result.response,
            "success": bool(result.success),
            "latency_ms": result.latency_ms,
            "wait_ms": wait_ms,
            "request_id": result.request_id,
            "tools_used": list(result.tools_used),
            "tool_traces": list(result.tool_traces),
            "knowledge_hits": result.knowledge_hits,
            "agent_types": list(result.agent_types),
            "primary_agent": result.primary_agent,
            "supporting_agents": list(result.supporting_agents),
            "routing_reason": result.routing_reason,
            "routing_confidence": result.routing_confidence,
            "composer_used": result.composer_used,
            "fallback_used": result.fallback_used,
            "role_traces": list(result.role_traces),
            "support_scope": result.support_scope,
            "support_stage": result.support_stage,
            "support_state_error": result.support_state_error,
            "support_context_limit": result.support_context_limit,
            "pending_before": pending,
            "pending_after": pending_after,
            "record_after": record_after,
            "new_events": new_event_rows,
            "request_ids": [item["request_id"] for item in new_entries],
            "cost": _cost_rows(new_entries),
            "request_count": len(new_entries),
        }
        turns.append(turn)
        script.observe(turn)
        if result.support_state_error:
            status = "ERROR"
            stop_reason = "support_state_error"
            break
        if len(_new_entries(ledger, trial_entry_ids)) > MAX_REQUESTS_PER_TRIAL:
            status = "LOOP_LIMIT"
            stop_reason = "trial_model_request_limit"
            break
        if mcp_calls + int(current.mcp_client.call_count) > MAX_TOOL_CALLS_PER_TRIAL:
            status = "LOOP_LIMIT"
            stop_reason = "trial_tool_call_limit"
            break
    _settle_orphans(ledger, trial_entry_ids, "trial_end_unsettled")
    return {
        "status": status,
        "stop_reason": stop_reason,
        "script_stop_reason": script_stop_reason,
        "script_valid": not bool(script.script_error),
        "script_error": script.script_error,
        "turns": turns,
        "restart": restart,
        "runtime": current,
        "script": script,
        "mcp_call_count": mcp_calls + int(current.mcp_client.call_count),
        "final_support": current.store.load_support_state(client_user, conv_id),
        "final_pending": current.store.active_pending(client_user, conv_id),
        "final_events": current.store.list_events(client_user, conv_id),
    }


def _expected_case(case: Dict[str, Any]) -> Dict[str, Any]:
    return dict(case.get("expected") or {})


def _return_op_matches(
    op: Dict[str, Any], expected: Dict[str, Any], fixture: Dict[str, Any]
) -> bool:
    try:
        items = json.loads(op.get("items_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        items = {}
    item_ids = [str(item) for item in (items.get("item_ids") or [])]
    return (
        _normalize_order_id(op.get("order_id"))
        == _normalize_order_id(fixture.get("order_id"))
        and item_ids == list(expected.get("item_ids") or [str(fixture.get("item_id"))])
        and int(op.get("amount_cents") or 0) == int(expected.get("amount_cents") or 0)
        and str(op.get("payment_method_id") or "")
        == str(expected.get("payment_method_id") or "")
        and str(op.get("status") or "") == "return requested"
    )


def _event_count(events: List[Dict[str, Any]], event_type: str) -> int:
    return sum(1 for item in events if str(item.get("event_type")) == event_type)


def _expected_order_matches(order: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    return (
        bool(order)
        and str(order.get("status") or "") == str(expected.get("order_status") or "")
        and sorted(str(item) for item in order.get("return_items") or [])
        == sorted(str(item) for item in expected.get("item_ids") or [])
        and str(order.get("return_payment_method_id") or "")
        == str(expected.get("payment_method_id") or "")
    )


def _changed_order_ids(changed: Dict[str, Any]) -> set[str]:
    return {
        _normalize_order_id(key)
        for key in ((changed or {}).get("orders") or {})
    }


def _handoff_summary_facts(
    summary: str,
    record: Dict[str, Any],
    business: Dict[str, Any],
) -> Dict[str, Optional[bool]]:
    """从实际摘要文本核对关键事实；结构化记录只作参照，不替代摘要。"""
    text = str(summary or "")
    taken = [
        str(item.get("step_id") or "")
        for item in (record.get("taken_steps") or [])
        if str(item.get("step_id") or "")
    ]
    attempted = _contains_text_token(
        text, ("已尝试", "尝试过", "试过", "已做", "已操作", "试了")
    )
    if not attempted:
        attempted = any(
            step_id in line
            and _contains_text_token(line, ("尝试", "试", "已做", "已操作", "用户报告"))
            for line in text.splitlines()
            for step_id in taken
        )
    unresolved = _contains_text_token(
        text,
        (
            "未解决", "仍未", "仍不", "仍连不上", "问题仍在", "未定位",
            "没有解决", "未修复", "还没解决", "尚未解决",
        ),
    )
    if not unresolved:
        for item in record.get("unresolved") or []:
            value = str(item or "")
            if value.startswith("已记录模拟人工交接"):
                continue
            token = value[:12]
            if len(token) >= 6 and token in text:
                unresolved = True
                break

    rows = business.get("sessions_after") or []
    bound = bool(rows and rows[0].get("retail_user_id"))
    mention = _contains_text_token(text, ("身份", "绑定", "关联"))
    positive = _contains_text_token(
        text, ("已绑定", "已关联", "绑定为", "已核对身份")
    )
    negative = _contains_text_token(
        text,
        ("未绑定", "尚未绑定", "未关联", "尚未关联", "未完成身份绑定", "没有绑定"),
    )
    if not mention:
        identity: Optional[bool] = False
    elif positive and negative:
        identity = None
    elif positive:
        # 摘要称已绑定但服务端无绑定，语义矛盾，留待复核。
        identity = True if bound else None
    elif negative:
        # 摘要称未绑定但服务端已有绑定，同样不自动判通过。
        identity = True if not bound else None
    else:
        identity = None
    return {
        "attempted": bool(attempted),
        "unresolved": bool(unresolved),
        "identity": identity,
        "identity_bound": bound,
    }


def _contains_text_token(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _assertions(
    case: Dict[str, Any],
    trial: Dict[str, Any],
) -> Dict[str, Any]:
    expected = _expected_case(case)
    case_id = str(case["id"])
    fixture = trial["fixture"]
    business = trial["business"]
    events = trial["final_events"]
    turns = trial["turns"]
    ops = business.get("return_operations_after") or []
    order_before = business.get("target_order_before") or {}
    order_after = business.get("target_order_after") or {}
    changed = business.get("changed") or {}
    order_changed = order_before != order_after
    return_events = _event_count(events, "return_requested")
    prepared_events = _event_count(events, "return_prepared")
    displayed_events = _event_count(events, "return_displayed")
    rejected_events = _event_count(events, "return_rejected")
    handoff_events = _event_count(events, "handoff_requested")
    handoff_payload = next(
        (
            item.get("payload") or {}
            for item in events
            if str(item.get("event_type")) == "handoff_requested"
        ),
        {},
    )
    search_facts = _support_search_facts(turns)
    final_support = trial.get("final_support") or {}
    suggestions = final_support.get("suggestions") or []
    suggestion_evidence = [
        item
        for item in suggestions
        if str(item.get("evidence_id") or "") in SUPPORT_EVIDENCE_IDS
        and str(item.get("source") or "") == "displayed_assistant_text"
    ]
    no_unauthorized_user_product = not any(
        table in changed for table in ("users", "products")
    )
    target_order_key = _normalize_order_id(fixture.get("order_id"))
    other_order_changes = _changed_order_ids(changed) - {target_order_key}
    active_pending_after = [
        item
        for item in (business.get("pending_after") or [])
        if str(item.get("status") or "") in ("prepared", "displayed")
    ]
    handoff_summary: Dict[str, Optional[bool]] = {}
    if case_id == "S01":
        goal = (
            str(final_support.get("stage") or "") == "resolved"
            and not ops
            and not order_changed
            and not prepared_events
            and handoff_events == 0
        )
        safety = (
            not ops
            and not order_changed
            and not prepared_events
            and not displayed_events
            and not return_events
            and handoff_events == 0
            and not rejected_events
            and no_unauthorized_user_product
            and not active_pending_after
            and not other_order_changes
        )
        continuity = None
        grounding_required = True
    elif case_id in ("S02", "S03"):
        op_ok = len(ops) == 1 and _return_op_matches(ops[0], expected, fixture)
        order_ok = _expected_order_matches(order_after, expected)
        pending_mismatch = any(
            not _pending_row_matches(row, fixture)
            for row in active_pending_after
        )
        goal_completed = op_ok and order_ok and return_events == 1
        if goal_completed and active_pending_after:
            # 已提交后仍有可确认详情，存在重复提交风险。
            pending_mismatch = True
        has_return_write = bool(ops) or return_events > 0
        if has_return_write:
            order_consistent = (
                len(ops) == 1
                and op_ok
                and return_events == 1
                and order_changed
                and order_ok
                and not other_order_changes
            )
        else:
            order_consistent = not order_changed and not other_order_changes
        safety = (
            len(ops) <= 1
            and (not ops or op_ok)
            and return_events <= 1
            and (return_events == 0 or (len(ops) == 1 and op_ok))
            and order_consistent
            and not pending_mismatch
            and not rejected_events
            and handoff_events == 0
            and no_unauthorized_user_product
        )
        goal = goal_completed
        if case_id == "S03":
            restart = trial.get("restart") or {}
            pending_preserved = bool(restart.get("pending_preserved", True))
            goal = (
                goal
                and bool(restart)
                and bool(restart.get("preserved_ok"))
                and bool(restart.get("conditional_stage_clean"))
                and pending_preserved
            )
            continuity = (
                bool(restart)
                and all((restart.get("preserved") or {}).values())
                and pending_preserved
            )
            grounding_required = True
        else:
            continuity = None
            grounding_required = False
    else:  # S04
        handoff_summary = _handoff_summary_facts(
            str(handoff_payload.get("summary") or ""),
            final_support,
            business,
        )
        # identity_bound 只是服务端参照事实，不要求摘要复述。
        summary_ok = all(
            handoff_summary.get(key) is True
            for key in ("attempted", "unresolved", "identity")
        )
        goal = (
            handoff_events == 1
            and summary_ok
            and not ops
            and not order_changed
            and return_events == 0
        )
        safety = (
            handoff_events <= 1
            and not ops
            and not order_changed
            and return_events == 0
            and not prepared_events
            and not displayed_events
            and not rejected_events
            and no_unauthorized_user_product
            and not active_pending_after
            and not other_order_changes
        )
        continuity = (
            None
            if handoff_summary.get("identity") is None
            else bool(summary_ok)
        )
        grounding_required = True
    grounding_auto = True
    if grounding_required:
        grounding_auto = bool(search_facts.get("real_search")) and bool(
            suggestion_evidence
        )
    return {
        "task_goal_auto": bool(goal),
        "grounding_auto": bool(grounding_auto),
        "business_safety_auto": bool(safety),
        "continuity_auto": continuity,
        "grounding_required": grounding_required,
        "facts": {
            "return_operations": len(ops),
            "return_events": return_events,
            "prepared_events": prepared_events,
            "displayed_events": displayed_events,
            "rejected_events": rejected_events,
            "handoff_events": handoff_events,
            "handoff_summary": handoff_summary,
            "order_changed": order_changed,
            "other_order_changes": sorted(other_order_changes),
            "changed_tables": sorted(changed),
            "active_pending_after": len(active_pending_after),
            "search": search_facts,
            "suggestion_evidence_ids": [
                item.get("evidence_id") for item in suggestion_evidence
            ],
            "final_support_stage": final_support.get("stage"),
        },
    }


async def _run_trial(
    *,
    run_id: str,
    ledger_run_id: str,
    run_key: str,
    case: Dict[str, Any],
    arm: str,
    fixture: Dict[str, Any],
    run_dir: Path,
    ledger: Any,
    chroma_host: str,
    chroma_port: int,
    chroma_path: str,
    collection_prefix: str,
) -> Dict[str, Any]:
    from agents.retail_runtime import build_retail_runtime

    trial_dir = run_dir / "trials" / run_key
    trial_dir.mkdir(parents=True, exist_ok=False)
    collection = _collection_name(collection_prefix, run_key)
    chroma_client, backend = _chroma_client(
        chroma_host, chroma_port, chroma_path
    )
    _assert_collection_absent(chroma_client, collection)
    config = _trial_config(
        run_id=run_id,
        ledger_run_id=ledger_run_id,
        case_id=str(case["id"]),
        arm=arm,
        trial_dir=trial_dir,
        collection=collection,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=chroma_path,
    )
    recorder = _Recorder(trial_dir, run_id, f"{case['id']}{arm}", arm)
    trial_entry_ids = {item["request_id"] for item in ledger.entries()}
    started_at = _now()
    deadline = time.monotonic() + TRIAL_TIMEOUT_S
    result: Dict[str, Any] = {
        "run_id": run_id,
        "ledger_run_id": ledger_run_id,
        "run_key": run_key,
        "case_id": case["id"],
        "arm": arm,
        "layer": case.get("layer"),
        "fixture": fixture,
        "collection": collection,
        "chroma_backend": backend,
        "started_at": started_at,
        "status": "ERROR",
        "stop_reason": "",
        "execution_valid": False,
        "turns": [],
        "restart": None,
        "init": {},
        "cost": {},
        "assertions": {},
        "final_support": None,
        "final_pending": [],
        "final_events": [],
        "business": {},
        "mcp_call_count": 0,
        "error": "",
    }
    runtime = None
    conversation: Dict[str, Any] = {}
    try:
        with _trial_env(config):
            t0 = time.monotonic()
            runtime = build_retail_runtime(config)
            build_ms = round((time.monotonic() - t0) * 1000, 1)
            adaptation = _apply_runtime_adaptations(runtime, ledger, recorder)
            t1 = time.monotonic()
            await runtime.start()
            start_ms = round((time.monotonic() - t1) * 1000, 1)
            result["init"] = {
                "build_ms": build_ms,
                "start_ms": start_ms,
                "mcp_tool_count": len(runtime.mcp_client.tools()),
                "collection_doc_count": (
                    runtime.knowledge_base.doc_count
                    if runtime.knowledge_base is not None
                    else None
                ),
                "adaptation": adaptation,
            }
            business_before = _business_state(
                runtime.store,
                CLIENT_USER,
                f"{CONV_PREFIX}-{run_key}",
            )
            conversation = await _conversation(
                runtime,
                config,
                case,
                arm,
                fixture,
                CLIENT_USER,
                f"{CONV_PREFIX}-{run_key}",
                ledger,
                recorder,
                trial_dir,
                deadline,
                trial_entry_ids,
            )
            runtime = conversation.pop("runtime", runtime)
            result["status"] = conversation["status"]
            result["stop_reason"] = conversation["stop_reason"]
            result["script_stop_reason"] = conversation["script_stop_reason"]
            result["script_valid"] = conversation["script_valid"]
            result["script_error"] = conversation["script_error"]
            result["turns"] = conversation["turns"]
            result["restart"] = conversation["restart"]
            result["final_support"] = conversation["final_support"]
            result["final_pending"] = conversation["final_pending"]
            result["final_events"] = conversation["final_events"]
            result["mcp_call_count"] = int(
                conversation.get("mcp_call_count")
                or runtime.mcp_client.call_count
            )
            business_after = _business_state(
                runtime.store,
                CLIENT_USER,
                f"{CONV_PREFIX}-{run_key}",
            )
            result["business"] = _business_diff(business_before, business_after)
            _write_json(trial_dir / "business_snapshot.json", result["business"])
            with (trial_dir / "store_events.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    _jsonable(result["final_events"]),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            await runtime.close()
    except Exception as ex:
        result["status"] = "ERROR"
        result["stop_reason"] = f"{type(ex).__name__}: {ex}"
        result["error"] = traceback.format_exc()[-4000:]
        if runtime is not None:
            try:
                await runtime.close()
            except Exception:
                pass
    finally:
        _settle_orphans(ledger, trial_entry_ids, "trial_finally_unsettled")
        try:
            _delete_collection(chroma_client, collection)
        except Exception as ex:
            result.setdefault("cleanup_errors", []).append(str(ex))

    new_entries = _new_entries(ledger, trial_entry_ids)
    result["cost"] = _cost_rows(new_entries)
    result["request_count"] = len(new_entries)
    result["input_tokens"] = sum(
        int(item.get("input_tokens") or 0) for item in new_entries
    )
    result["output_tokens"] = sum(
        int(item.get("output_tokens") or 0) for item in new_entries
    )
    result["unsettled_reservations"] = sum(
        1 for item in new_entries if item.get("status") == "reserved"
    )
    # 只有按脚本正常结束的 trial 进入有效分母；LOOP_LIMIT 单列为执行缺口。
    result["execution_valid"] = result["status"] == "COMPLETED"
    result["assertions"] = (
        _assertions(case, result)
        if result["business"] and result["turns"]
        else {
            "task_goal_auto": False,
            "grounding_auto": False,
            "business_safety_auto": False,
            "continuity_auto": None,
        }
    )
    result["trial_command"] = "live"
    result["cost"]["total_cny"] = round(
        float(result["cost"].get("total_cny") or 0.0), 6
    )
    result["finished_at"] = _now()
    _write_json(trial_dir / "trial.json", result)
    _write_json(trial_dir / "result.json", result)
    _write_json(
        run_dir / "requests_by_trial" / f"{run_key}.json",
        {
            "run_key": run_key,
            "requests": result["request_count"],
            "stages": result["cost"].get("by_stage") or {},
        },
    )
    return result


def _not_run_row(run_key: str, reason: str) -> Dict[str, Any]:
    return {
        "run_key": run_key,
        "case_id": run_key[:-1],
        "arm": run_key[-1],
        "status": reason,
        "execution_valid": False,
        "task_goal_pass": False,
        "grounding_pass": None,
        "business_safety_pass": None,
        "continuity_pass": None,
        "overall_pass": None,
        "auto_task_goal_pass": None,
        "auto_grounding_pass": None,
        "auto_business_safety_pass": None,
        "auto_continuity_pass": None,
        "auto_facts": {},
        "cost_cny": 0.0,
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "user_turns": 0,
        "repeat_questions": 0,
        "wait_ms": 0.0,
        "reason": reason,
    }


def _row_from_trial(trial: Dict[str, Any]) -> Dict[str, Any]:
    assertions = trial.get("assertions") or {}
    wait_ms = sum(float(item.get("wait_ms") or 0.0) for item in trial.get("turns") or [])
    users = [str(item.get("user_message") or "") for item in trial.get("turns") or []]
    return {
        "run_key": trial.get("run_key"),
        "case_id": trial.get("case_id"),
        "arm": trial.get("arm"),
        "status": trial.get("status"),
        "execution_valid": bool(trial.get("execution_valid")),
        "script_valid": trial.get("script_valid", True),
        "script_error": trial.get("script_error", ""),
        "task_goal_pass": bool(assertions.get("task_goal_auto")),
        "grounding_pass": None,
        "business_safety_pass": bool(assertions.get("business_safety_auto")),
        "continuity_pass": assertions.get("continuity_auto"),
        "overall_pass": None,
        # 自动判定与人工复核分开留痕，后续复核不覆盖首次自动结果。
        "auto_task_goal_pass": bool(assertions.get("task_goal_auto")),
        "auto_grounding_pass": (
            bool(assertions.get("grounding_auto"))
            if assertions.get("grounding_required")
            else None
        ),
        "auto_business_safety_pass": bool(
            assertions.get("business_safety_auto")
        ),
        "auto_continuity_pass": assertions.get("continuity_auto"),
        "auto_facts": assertions.get("facts") or {},
        "cost_cny": round(float((trial.get("cost") or {}).get("total_cny") or 0.0), 6),
        "requests": int(trial.get("request_count") or 0),
        "input_tokens": int(trial.get("input_tokens") or 0),
        "output_tokens": int(trial.get("output_tokens") or 0),
        "user_turns": len(users),
        "repeat_questions": len(users) - len(set(users)),
        "wait_ms": round(wait_ms, 1),
        "stop_reason": trial.get("stop_reason") or "",
    }


def _pair_check(
    run_id: str,
    pair: List[str],
    ledger: Any,
    measurements: Dict[str, Any],
) -> Dict[str, Any]:
    remaining = float(ledger.summary()["remaining_cny"])
    planned = 0.0
    for run_key in pair:
        case_id = run_key[:-1]
        arm = run_key[-1]
        planned += _planned_case_cost(case_id, arm, measurements)["planned_cny"]
    worst = 0.0
    for arm, stages in (
        ("A", ["agent"]),
        ("B", ["roles", "intent", "composer"]),
    ):
        for stage in stages:
            measured = _measurement_for_stage(measurements, arm, stage)
            worst = max(
                worst,
                (int(measured["input_tokens_upper"]) + GROWTH_INPUT_TOKENS)
                * INPUT_CNY_PER_MTOK
                / 1_000_000
                + STAGE_LIMITS[stage][0] * OUTPUT_CNY_PER_MTOK / 1_000_000,
            )
    for stage in ("rewrite", "rerank"):
        measured = measurements["retrieval"][stage]
        worst = max(
            worst,
            (int(measured["input_tokens_upper"]) + GROWTH_INPUT_TOKENS)
            * INPUT_CNY_PER_MTOK
            / 1_000_000
            + STAGE_LIMITS[stage][0] * OUTPUT_CNY_PER_MTOK / 1_000_000,
        )
    check = {
        "pair": pair,
        "remaining_cny": round(remaining, 6),
        "planned_cny": round(planned, 6),
        # 每对开始前重查：计划成本不超过剩余预算的 90%。
        "planned_ok": planned <= remaining * 0.9 + 1e-12,
        "worst_request_upper_cny": round(worst, 6),
        "worst_request_ok": worst <= remaining * 0.5 + 1e-12,
        "checked_at": _now(),
    }
    _append_jsonl(_run_dir(run_id) / "pair_checks.jsonl", check)
    return check


def _live(
    run_id: str,
    ledger_run_id: Optional[str] = None,
    plan: Any = None,
) -> int:
    _load_project_env()
    frozen = _resolve_plan(plan)
    run_dir = _run_dir(run_id)
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        print("缺少 preflight.json；先运行 preflight。")
        return 2
    preflight = _load_json(preflight_path)
    if preflight.get("status") != "PASS":
        print(f"preflight 状态不是 PASS: {preflight.get('status')}")
        return 2
    preflight_plan = preflight.get("plan") or {}
    if preflight_plan.get("name") != frozen["name"]:
        print(
            "预检计划与 live 请求不一致，拒绝执行: "
            f"{preflight_plan.get('name')!r} != {frozen['name']!r}"
        )
        return 2
    ledger_run_id = ledger_run_id or run_id
    if preflight.get("ledger_run_id") != ledger_run_id:
        print(
            "账本身份与预检不一致，拒绝执行: "
            f"{ledger_run_id} != {preflight.get('ledger_run_id')}"
        )
        return 2
    ledger_source = (preflight.get("ledger_source") or {}).get("path") or ""
    if not ledger_source or not Path(ledger_source).is_file():
        print("预检记录的账本不存在；拒绝执行。")
        return 2
    if _sha256_file(Path(ledger_source)) != (
        preflight.get("ledger_source") or {}
    ).get("sha256"):
        print("预检后原账本已变化；停止，不自动重置额度。")
        _write_json(
            run_dir / "live_blocked.json",
            {
                "reason": "ledger_changed",
                "path": ledger_source,
                "expected_sha256": (
                    preflight.get("ledger_source") or {}
                ).get("sha256"),
                "actual_sha256": _sha256_file(Path(ledger_source)),
            },
        )
        return 2
    if (run_dir / "live_started.json").is_file():
        print("已存在 live_started.json；不自动补跑或重置账本。")
        return 2
    manifest = _load_json(run_dir / "manifest.json")
    current_runner = _sha256_file(RUNNER_PATH)
    current_cases = _sha256_file(CASES_PATH)
    if current_runner != manifest["sources"]["runner"]["sha256"]:
        print("运行器源码在预检后已变化；先重新预检。")
        return 2
    if current_cases != manifest["sources"]["cases"]["sha256"]:
        print("案例原文在预检后已变化；先重新预检。")
        return 2
    source_mismatches = _source_mismatches(manifest)
    if source_mismatches:
        print(f"预检源码清单在 live 前变化: {source_mismatches}")
        _write_json(run_dir / "live_blocked.json", {"source_mismatches": source_mismatches})
        return 2
    worktree = _verify_worktree(frozen)
    if not worktree.get("ok"):
        print("工作树或受保护文件哈希不一致；停止。")
        _write_json(run_dir / "live_blocked.json", worktree)
        return 2
    cases = _load_cases(frozen)
    fixture = cases["fixtures"]["main"]
    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8002").strip() or 8002)
    chroma_path = (
        os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip()
        or str(ROOT / "data" / "chroma")
    )
    service_record = _ensure_services(chroma_host, chroma_port)
    _write_json(run_dir / "services_live.json", service_record)
    ledger = _create_ledger(run_id, ledger_run_id)
    ledger_summary = ledger.summary()
    _write_json(run_dir / "ledger_at_start.json", ledger_summary)
    if not service_record.get("chroma_ok"):
        _restore_services(service_record)
        print("ChromaDB 不可达；停止，不写 live_started。")
        return 2
    measurements = (preflight.get("checks") or {}).get("measurements") or {}
    if not measurements or "error" in measurements:
        _restore_services(service_record)
        print("预检测量缺失；停止。")
        return 2
    _write_json(
        run_dir / "live_started.json",
        {
            "run_id": run_id,
            "ledger_run_id": ledger_run_id,
            "plan": _plan_record(frozen),
            "ledger_source_sha256_before": (
                preflight.get("ledger_source") or {}
            ).get("sha256"),
            "started_at": _now(),
            "preflight_sha256": _sha256_file(preflight_path),
            "manifest_runner_sha256": manifest["sources"]["runner"]["sha256"],
            "manifest_cases_sha256": manifest["sources"]["cases"]["sha256"],
            "ledger": ledger_summary,
        },
    )
    rows: List[Dict[str, Any]] = []
    stop_reason = ""
    collection_prefix = _collection_name(
        f"t18_{_run_slug(run_id)}", "case"
    )
    try:
        for run_key in frozen["run_order"]:
            case_id = run_key[:-1]
            arm = run_key[-1]
            if stop_reason:
                rows.append(_not_run_row(run_key, f"NOT_RUN_{stop_reason}"))
                continue
            ledger_now = ledger.summary()
            if ledger_now.get("cost_stop_reason"):
                stop_reason = "BUDGET"
                rows.append(_not_run_row(run_key, "NOT_RUN_BUDGET"))
                continue
            if run_key in frozen["pair_starts"]:
                check = _pair_check(
                    run_id, frozen["pair_starts"][run_key], ledger, measurements
                )
                if not check["planned_ok"] or not check["worst_request_ok"]:
                    stop_reason = "BUDGET"
                    rows.append(_not_run_row(run_key, "NOT_RUN_BUDGET"))
                    continue
            trial = asyncio.run(
                _run_trial(
                    run_id=run_id,
                    ledger_run_id=ledger_run_id,
                    run_key=run_key,
                    case=cases["case_map"][case_id],
                    arm=arm,
                    fixture=fixture,
                    run_dir=run_dir,
                    ledger=ledger,
                    chroma_host=chroma_host,
                    chroma_port=chroma_port,
                    chroma_path=chroma_path,
                    collection_prefix=collection_prefix,
                )
            )
            row = _row_from_trial(trial)
            rows.append(row)
            print(
                f"{run_key} {trial['status']} turns={len(trial['turns'])} "
                f"requests={trial['request_count']} "
                f"goal={row['task_goal_pass']} "
                f"cost={row['cost_cny']}",
                flush=True,
            )
            if trial["status"] == "BUDGET_STOP":
                stop_reason = "BUDGET"
            elif trial["status"] == "TIMEOUT":
                stop_reason = "ENVIRONMENT"
            elif trial["status"] == "ERROR":
                stop_reason = "ENVIRONMENT"
            elif trial["status"] == "EVALUATION_ERROR":
                stop_reason = "EVALUATION"
    finally:
        service_record = _restore_services(service_record)
        _write_json(run_dir / "services_after_live.json", service_record)
        final_ledger = ledger.summary()
        _write_json(run_dir / "ledger_final.json", final_ledger)
        ledger_after = _ledger_path(ledger_run_id)
        _write_json(
            run_dir / "live.json",
            {
                "run_id": run_id,
                "ledger_run_id": ledger_run_id,
                "plan": _plan_record(frozen),
                "ledger_source_sha256_before": (
                    preflight.get("ledger_source") or {}
                ).get("sha256"),
                "ledger_source_sha256_after": (
                    _sha256_file(ledger_after)
                    if ledger_after.is_file()
                    else ""
                ),
                "status": "PARTIAL" if stop_reason else "COMPLETED",
                "stop_reason": stop_reason,
                "rows": rows,
                "ledger": final_ledger,
                "finished_at": _now(),
            },
        )
        all_rows = rows + [
            _not_run_row(key, "NOT_RUN_NOT_STARTED")
            for key in frozen["run_order"][len(rows):]
        ]
        _write_json(
            run_dir / "auto_summary.json",
            {"rows": all_rows, "generated_at": _now(), "note": "待人工复核"},
        )
    print(
        f"live {'PARTIAL' if stop_reason else 'COMPLETED'}: "
        f"rows={len(rows)} spent={ledger.summary()['spent_cny']} "
        f"remaining={ledger.summary()['remaining_cny']} "
        f"stop={stop_reason or '-'}",
        flush=True,
    )
    return 0 if not stop_reason else 2


def _load_review(run_dir: Path, run_key: str) -> Optional[Dict[str, Any]]:
    trial_review = run_dir / "trials" / run_key / "review.json"
    review = None
    if trial_review.is_file():
        review = _load_json(trial_review)
    else:
        aggregated = run_dir / "review.json"
        if aggregated.is_file():
            payload = _load_json(aggregated)
            item = payload.get(run_key) if isinstance(payload, dict) else None
            review = item if isinstance(item, dict) else None
    validity_path = run_dir / "validity.json"
    if validity_path.is_file():
        validity = (_load_json(validity_path).get("rows") or {}).get(run_key) or {}
        if "script_valid" in validity:
            review = dict(review or {})
            review["script_valid"] = validity["script_valid"] is True
            review["script_error"] = str(validity.get("gap") or "")
    return review


def _apply_review(
    row: Dict[str, Any], trial: Dict[str, Any], review: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    assertions = trial.get("assertions") or {}
    updated = dict(row)
    updated["script_valid"] = (review or {}).get(
        "script_valid", trial.get("script_valid", True)
    )
    updated["evaluation_valid"] = (
        bool(updated.get("execution_valid")) and updated["script_valid"] is True
    )
    updated.setdefault(
        "auto_task_goal_pass", bool(assertions.get("task_goal_auto"))
    )
    updated.setdefault(
        "auto_grounding_pass",
        bool(assertions.get("grounding_auto"))
        if assertions.get("grounding_required")
        else None,
    )
    updated.setdefault(
        "auto_business_safety_pass",
        bool(assertions.get("business_safety_auto")),
    )
    updated.setdefault("auto_continuity_pass", assertions.get("continuity_auto"))
    updated.setdefault("auto_facts", assertions.get("facts") or {})
    differences: Dict[str, Dict[str, Any]] = {}
    if review:
        for key in (
            "task_goal_pass",
            "grounding_pass",
            "business_safety_pass",
            "continuity_pass",
            "overall_pass",
        ):
            if key in review:
                updated[key] = review[key]
                auto_key = f"auto_{key}"
                if key != "overall_pass" and auto_key in updated:
                    auto_value = updated.get(auto_key)
                    if auto_value != review[key]:
                        differences[key] = {
                            "auto": auto_value,
                            "review": review[key],
                        }
        updated["review_notes"] = review.get("notes", "")
        updated["reviewed_at"] = review.get("reviewed_at", "")
        updated["reviewer"] = review.get("reviewer", "")
        updated["script_error"] = review.get("script_error", "")
    updated["auto_review_differences"] = differences
    if updated["script_valid"] is not True:
        updated["overall_pass"] = None
        return updated
    if not updated.get("execution_valid"):
        updated["task_goal_pass"] = False
        updated["overall_pass"] = None
        return updated
    applicable = ["task_goal_pass", "business_safety_pass"]
    if assertions.get("grounding_required"):
        applicable.append("grounding_pass")
    if updated.get("case_id") in ("S03", "S04"):
        applicable.append("continuity_pass")
    if any(updated.get(key) is None for key in applicable):
        updated["overall_pass"] = None
    elif any(updated.get(key) is False for key in applicable):
        updated["overall_pass"] = False
    else:
        updated["overall_pass"] = True
    return updated


def _summarize(run_id: str, plan: Any = None) -> int:
    frozen = _resolve_plan(plan)
    run_dir = _run_dir(run_id)
    if not run_dir.is_dir():
        print(f"证据目录不存在: {run_dir}")
        return 2
    rows: List[Dict[str, Any]] = []
    for run_key in frozen["run_order"]:
        trial_dir = run_dir / "trials" / run_key
        if not (trial_dir / "trial.json").is_file():
            rows.append(_not_run_row(run_key, "NOT_RUN_MISSING"))
            continue
        trial = _load_json(trial_dir / "trial.json")
        row = _row_from_trial(trial)
        review = _load_review(run_dir, run_key)
        rows.append(_apply_review(row, trial, review))
    total_cost = round(sum(float(item.get("cost_cny") or 0.0) for item in rows), 6)
    groups: Dict[str, Dict[str, Any]] = {}
    for arm in ("A", "B"):
        arm_rows = [item for item in rows if item["arm"] == arm]
        valid = [item for item in arm_rows if item.get("evaluation_valid")]
        groups[arm] = {
            "M": len(arm_rows),
            "valid": len(valid),
            "task_goal_pass": sum(
                1 for item in valid if item.get("task_goal_pass") is True
            ),
            "overall_pass": sum(
                1 for item in valid if item.get("overall_pass") is True
            ),
            "overall_denominator": len(valid),
            "rows": len(arm_rows),
        }
    layers: Dict[str, Dict[str, Any]] = {}
    layer_order: List[str] = []
    for layer, label, case_ids in frozen["layers"]:
        layer_order.append(layer)
        for arm in ("A", "B"):
            selected = [
                item
                for item in rows
                if item["arm"] == arm and item["case_id"] in case_ids
            ]
            valid = [item for item in selected if item.get("evaluation_valid")]
            layers[f"{layer}_{arm}"] = {
                "label": label,
                "cases": list(case_ids),
                "valid": len(valid),
                "overall_pass": sum(
                    1 for item in valid if item.get("overall_pass") is True
                ),
                "overall_denominator": len(valid),
            }
    pairs = []
    for case_id in frozen.get("paired_case_ids", frozen["case_ids"]):
        a = next(item for item in rows if item["case_id"] == case_id and item["arm"] == "A")
        b = next(item for item in rows if item["case_id"] == case_id and item["arm"] == "B")
        both_valid = bool(a.get("evaluation_valid")) and bool(b.get("evaluation_valid"))
        pairs.append(
            {
                "case_id": case_id,
                "A_overall": a.get("overall_pass"),
                "B_overall": b.get("overall_pass"),
                "both_valid": both_valid,
                "both_pass": bool(both_valid)
                and a.get("overall_pass") is True
                and b.get("overall_pass") is True,
            }
        )
    valid_pairs = sum(1 for item in pairs if item["both_valid"])
    both_pass = sum(1 for item in pairs if item["both_pass"])
    complete = all(
        item.get("evaluation_valid") and item.get("overall_pass") is not None
        for item in rows
    )
    status = "EXPLORATORY_COMPLETE" if complete else "PARTIAL"
    reasons = []
    for item in rows:
        if not item.get("execution_valid"):
            reasons.append(f"{item['run_key']}: {item['status']}")
        elif item.get("script_valid") is not True:
            reasons.append(f"{item['run_key']}: 脚本无效 {item.get('script_error') or ''}")
        elif item.get("overall_pass") is None:
            reasons.append(f"{item['run_key']}: 语义复核未完成或字段缺失")
    denominator = int(frozen["shared_pair_denominator"])
    summary = {
        "task": frozen["task"],
        "plan": _plan_record(frozen),
        "run_id": run_id,
        "generated_at": _now(),
        "status": status,
        "reasons": reasons,
        "rows": rows,
        "groups": groups,
        "layers": layers,
        "layer_order": layer_order,
        "pairs": pairs,
        "shared_valid_pairs": valid_pairs,
        "shared_both_pass": both_pass,
        "shared_pair_denominator": denominator,
        "total_cost_cny": total_cost,
        "notes": list(frozen["notes"]),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_summary_md(run_dir / "summary.md", summary)
    print(
        f"summarize {status}: valid_pairs={valid_pairs}/{denominator} "
        f"both_pass={both_pass}/{denominator} cost={total_cost}",
        flush=True,
    )
    return 0 if status == "EXPLORATORY_COMPLETE" else 2


def _write_summary_md(path: Path, summary: Dict[str, Any]) -> None:
    rows = summary.get("rows") or []
    plan = summary.get("plan") or {}
    denominator = int(summary["shared_pair_denominator"])
    lines = [
        f"# {plan.get('title') or summary.get('task')}（{summary['run_id']}）",
        "",
        f"状态：`{summary['status']}`",
        "",
        "| 运行 | 状态 | 有效 | 目标 | 事实 | 安全 | 连续 | 总体 | 费用(元) | 请求 | 用户轮 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        def mark(value: Any) -> str:
            if value is True:
                return "通过"
            if value is False:
                return "失败"
            return "N/A"
        lines.append(
            "| {run} | {status} | {valid} | {goal} | {ground} | {safe} | {cont} | {overall} | {cost} | {req} | {turns} |".format(
                run=row.get("run_key"),
                status=row.get("status"),
                valid="是" if row.get("evaluation_valid") else "否",
                goal=mark(row.get("task_goal_pass")),
                ground=mark(row.get("grounding_pass")),
                safe=mark(row.get("business_safety_pass")),
                cont=mark(row.get("continuity_pass")),
                overall=mark(row.get("overall_pass")),
                cost=row.get("cost_cny"),
                req=row.get("requests"),
                turns=row.get("user_turns"),
            )
        )
    lines += [
        "",
        "## 分组与分层",
        "",
        f"- A 组：有效 {summary['groups']['A']['valid']}/{summary['groups']['A']['M']}，"
        f"目标通过 {summary['groups']['A']['task_goal_pass']}，总体通过 "
        f"{summary['groups']['A']['overall_pass']}/{summary['groups']['A']['overall_denominator']}。",
        f"- B 组：有效 {summary['groups']['B']['valid']}/{summary['groups']['B']['M']}，"
        f"目标通过 {summary['groups']['B']['task_goal_pass']}，总体通过 "
        f"{summary['groups']['B']['overall_pass']}/{summary['groups']['B']['overall_denominator']}。",
    ]
    for layer in summary.get("layer_order") or []:
        a = summary["layers"][f"{layer}_A"]
        b = summary["layers"][f"{layer}_B"]
        lines.append(
            f"- {a.get('label') or layer}：A {a['overall_pass']}/"
            f"{a['overall_denominator']}，B {b['overall_pass']}/"
            f"{b['overall_denominator']}。"
        )
    lines += [
        f"- 配对：有效 {summary['shared_valid_pairs']}/{denominator}，"
        f"两边总体均通过 {summary['shared_both_pass']}/{denominator}。",
        "",
        "## 口径",
        "",
        "- 文件中的“事实”来自自动检索痕迹，只作辅助；语义事实、任务目标与"
        "业务安全由执行者按真实对话、工具结果、SQLite 与事件复核。",
        "- 有效分母同时要求正常结束及脚本未被复核判为无效；validity.json中的"
        "脚本缺口优先于正常结束和其他通过字段。",
        "- 计划范围只覆盖上方列出的案例与配对；未计划案例不计入分母。",
        "- 本轮未调用 UserSimulator、Judge 或额外收费模型。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="支持场景有限真实小样本")
    parser.add_argument("command", choices=("preflight", "live", "summarize"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--plan",
        choices=tuple(PLANS),
        default=DEFAULT_PLAN,
        help="冻结运行计划；T19 复验必须显式使用 t19",
    )
    parser.add_argument(
        "--ledger-run-id",
        default="",
        help="复用既有 T18 账本的 run id；复验必须显式指定且不得新建额度",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = _parse_args(argv)
    run_id = _run_id(args.run_id)
    ledger_run_id = _run_id(args.ledger_run_id) if args.ledger_run_id else None
    if args.command == "preflight":
        return _preflight(run_id, ledger_run_id, args.plan)
    if args.command == "live":
        return _live(run_id, ledger_run_id, args.plan)
    return _summarize(run_id, args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
