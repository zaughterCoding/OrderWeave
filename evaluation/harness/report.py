"""汇总评测结果：成功率、pass^k、置信区间、安全、成本与延迟。

计划会话缺结果按未运行计入分母；无效会话不算成功，另报有效子集比例。
"""
from __future__ import annotations

import json
import random
from math import comb
from pathlib import Path
from statistics import median
from typing import Any, Dict, List


def pass_hat_k(successes: Dict[str, int], trials: int, k: int) -> float:
    """τ 官方定义：每题 C(c,k)/C(n,k) 的平均。"""
    return sum(comb(c, k) / comb(trials, k) for c in successes.values()) / len(successes)


def bootstrap_ci(values: List[float], rounds: int = 2000, seed: int = 7) -> List[float]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(rounds))
    return [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds) - 1], 4)]


def percentile(values: List[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(run_dir: Path) -> Dict[str, Any]:
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    trials = int(plan["trials"])
    rows: List[Dict[str, Any]] = []
    for task_id in plan["task_ids"]:
        for trial in range(1, trials + 1):
            path = run_dir / "sessions" / task_id / f"trial-{trial}" / "result.json"
            rows.append(json.loads(path.read_text(encoding="utf-8")) if path.exists()
                        else {"task_id": task_id, "trial": trial, "not_run": True})
    ran = [r for r in rows if not r.get("not_run")]
    valid = [r for r in ran if r.get("valid")]
    success = [r for r in valid if r.get("success")]
    per_task = {t: sum(1 for r in rows if r["task_id"] == t and r.get("success")) for t in plan["task_ids"]}
    task_rates = [per_task[t] / trials for t in plan["task_ids"]]
    costs = [r["cost"].get("total_cny", 0) for r in ran if r.get("cost")]
    system_costs = [r["cost"].get("system_cny", 0) for r in ran if r.get("cost")]
    latencies = [x for r in ran for x in r.get("turn_latency_ms", [])]
    terminations: Dict[str, int] = {}
    for r in ran:
        terminations[r["termination"]] = terminations.get(r["termination"], 0) + 1
    safety = {"writes": sum(r["safety"]["writes"] for r in ran),
              "unconfirmed_writes": sum(len(r["safety"]["unconfirmed_writes"]) for r in ran),
              "duplicate_writes": sum(len(r["safety"]["duplicate_writes"]) for r in ran)}
    return {
        "run_id": plan["run_id"], "suite": plan["suite"], "profile": plan["profile"], "model": plan["model"],
        "git_commit": plan.get("git_commit"), "tasks": len(plan["task_ids"]), "trials": trials,
        "planned": len(rows), "ran": len(ran), "valid": len(valid), "success": len(success),
        "success_over_planned": round(len(success) / len(rows), 4),
        "success_over_valid": round(len(success) / len(valid), 4) if valid else None,
        "pass_hat_k": {k: round(pass_hat_k(per_task, trials, k), 4) for k in range(1, trials + 1)},
        "pass1_ci95": bootstrap_ci(task_rates),
        "safety": safety, "terminations": terminations,
        "cost_cny": {"total": round(sum(costs), 6), "system": round(sum(system_costs), 6),
                     "per_session_mean": round(sum(costs) / len(costs), 6) if costs else None,
                     "system_per_success": round(sum(system_costs) / len(success), 6) if success else None},
        "turn_latency_ms": {"p50": median(latencies) if latencies else None, "p95": percentile(latencies, 0.95)},
        "per_task": {t: f"{per_task[t]}/{trials}" for t in plan["task_ids"]},
        "invalid": [{"task_id": r["task_id"], "trial": r["trial"], "termination": r["termination"],
                     "detail": str(r.get("termination_detail") or r.get("eval_error"))[:200]}
                    for r in ran if not r.get("valid")],
    }


def write_report(run_dir: Path) -> Dict[str, Any]:
    summary = summarize(run_dir)
    (run_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# {summary['run_id']}", "",
             f"- 套件 {summary['suite']}，profile {summary['profile']}，模型 {summary['model']}，提交 {summary['git_commit']}",
             f"- 计划 {summary['planned']}，运行 {summary['ran']}，有效 {summary['valid']}，成功 {summary['success']}",
             f"- 成功／计划 {summary['success_over_planned']}，成功／有效 {summary['success_over_valid']}",
             f"- pass^k {summary['pass_hat_k']}，pass^1 95%区间 {summary['pass1_ci95']}",
             f"- 安全 {summary['safety']}",
             f"- 费用 {summary['cost_cny']}",
             f"- 轮延迟 {summary['turn_latency_ms']}",
             f"- 终止 {summary['terminations']}", "", "| 任务 | 成功 |", "| --- | --- |"]
    lines += [f"| {t} | {v} |" for t, v in summary["per_task"].items()]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
