"""对外部私有记录做离线汇总，不调用模型，不分发数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.retail_metrics import PRIMARY_INTENTS, score_intents


ROLES = {"general", "technical", "billing", "escalation"}


def score(rows: list[dict], kind: str, planned: int) -> dict:
    if planned <= 0 or len(rows) != planned:
        raise ValueError("每个计划槽位必须有一行，包括错误、无效和未运行记录")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(key, str) or not key.strip() for key in ids) or len(set(ids)) != planned:
        raise ValueError("槽位id必须非空且唯一")
    if kind == "intent":
        if any(row.get("gold") not in PRIMARY_INTENTS for row in rows):
            raise ValueError("gold必须使用六类主意图标签")
        result = score_intents(rows)
        result.pop("cases", None)
        return result
    if kind == "route":
        if any(row.get("gold") not in ROLES for row in rows):
            raise ValueError("gold必须使用四角色标签")
        correct = sum(not row.get("is_error") and row.get("prediction") == row["gold"] for row in rows)
        return {"planned": planned, "correct": correct, "accuracy": correct / planned}
    if kind == "recovery":
        if any(type(row.get("passed")) is not bool for row in rows):
            raise ValueError("每个恢复场景必须记录布尔passed")
        passed = sum(row["passed"] for row in rows)
        return {"recovery_passed": passed, "recovery_total": planned, "pass_rate": passed / planned}
    if kind != "task":
        raise ValueError("未知指标类型")
    if any(row.get("status") not in {"valid", "invalid", "not_run"} or type(row.get("success")) is not bool
           or (row["status"] != "valid" and row["success"]) for row in rows):
        raise ValueError("任务必须记录status和布尔success；无效／未运行不能成功")
    valid = sum(row["status"] == "valid" for row in rows)
    success = sum(row["success"] for row in rows)
    return {"planned": planned, "effective": valid, "successful": success,
            "invalid": sum(row["status"] == "invalid" for row in rows),
            "not_run": sum(row["status"] == "not_run" for row in rows),
            "success_effective": success / valid if valid else None,
            "effective_planned": valid / planned, "success_planned": success / planned}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("intent", "route", "recovery", "task"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--planned", required=True, type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(score(rows, args.kind, args.planned), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
