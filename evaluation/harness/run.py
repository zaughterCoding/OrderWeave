"""评测调度入口：计划、并行会话、断点续跑、全局预算与报告。

用法：
  uv run python -m evaluation.harness.run --suite tau-core --trials 3 --run-id s0-base --budget-cny 10
  uv run python -m evaluation.harness.run --run-id s0-base --report-only
同一 run-id 重复执行只补跑缺结果的会话；计划参数不一致时拒绝。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from statistics import median
from pathlib import Path

from evaluation.harness import config as cfg
from evaluation.harness.report import write_report
from evaluation.harness.session import ledger_cost
from evaluation.harness.suites import load_suite


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cfg.ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cfg.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def spent(run_dir: Path) -> float:
    """含中断保留目录在内的全部会话账本实际费用。"""
    return sum(ledger_cost(p).get("total_cny", 0) for p in (run_dir / "sessions").glob("*/*/ledger.sqlite"))


def make_plan(args, run_dir: Path) -> dict:
    cfg.load_env()
    suite = load_suite(args.suite)
    task_ids = suite.task_ids[: args.limit] if args.limit else suite.task_ids
    if args.only:
        wanted = [t.strip() for t in args.only.split(",") if t.strip()]
        missing = set(wanted) - set(suite.task_ids)
        if missing:
            raise SystemExit(f"套件中没有这些任务: {sorted(missing)}")
        task_ids = wanted
    plan = {"run_id": args.run_id, "suite": args.suite, "tasks_path": str(suite.tasks_path),
            "suite_digest": suite.digest(), "task_ids": task_ids, "trials": args.trials,
            "profile": args.profile, "model": cfg.model_settings()["model"], "budget_cny": args.budget_cny,
            "git_commit": git_commit()}
    path = run_dir / "plan.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        keys = ("suite", "suite_digest", "task_ids", "trials", "profile", "model")
        if any(old[k] != plan[k] for k in keys):
            raise SystemExit("计划与已有 run 不一致；请使用新的 run-id")
        if old["git_commit"] != plan["git_commit"]:
            print(f"警告：代码版本 {old['git_commit']} → {plan['git_commit']}，续跑结果混合版本", file=sys.stderr)
        plan["git_commit"] = old["git_commit"]
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


async def execute(plan: dict, run_dir: Path, concurrency: int, max_trial: int = 0) -> None:
    last = min(plan["trials"], max_trial) if max_trial else plan["trials"]
    pending = [(t, n) for n in range(1, last + 1) for t in plan["task_ids"]
               if not (run_dir / "sessions" / t / f"trial-{n}" / "result.json").exists()]
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    gate = asyncio.Semaphore(concurrency)
    stopped = False
    observed: list = []
    running = 0

    async def one(task_id: str, trial: int) -> None:
        nonlocal stopped, running
        async with gate:
            # 只为正在运行的会话预留；按已完成会话成本中位数的两倍估计，避免个别长会话（如攻击纠缠）
            # 拉高预留导致提前停止。单会话失控由会话账本上限兜底。
            reserve = max(0.05, 2 * median(observed)) if observed else 0.1
            if stopped or spent(run_dir) + reserve * (running + 1) > plan["budget_cny"]:
                stopped = True
                return
            running += 1
            case_dir = run_dir / "sessions" / task_id / f"trial-{trial}"
            with open(logs / f"{task_id}-t{trial}.log", "ab") as log:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "evaluation.harness.session", "--suite", plan["suite"],
                    "--tasks", plan["tasks_path"], "--task-id", task_id, "--trial", str(trial),
                    "--profile", plan["profile"], "--run-id", plan["run_id"], "--case-dir", str(case_dir),
                    cwd=str(cfg.ROOT), stdout=log, stderr=log)
                code = await proc.wait()
            running -= 1
            result = case_dir / "result.json"
            status = json.loads(result.read_text(encoding="utf-8")) if result.exists() else None
            if status:
                observed.append(status["cost"].get("total_cny", 0))
            print(f"[{task_id} t{trial}] exit={code} "
                  + (f"success={status['success']} valid={status['valid']} {status['termination']} "
                     f"CNY {status['cost'].get('total_cny')}" if status else "无结果"), flush=True)

    await asyncio.gather(*(one(t, n) for t, n in pending))
    if stopped:
        print(f"预算闸门停止：已花 CNY {spent(run_dir):.4f} / CNY {plan['budget_cny']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="OrderWeave 评测 harness")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--profile", default="legacy-multi", choices=sorted(cfg.PROFILES))
    parser.add_argument("--budget-cny", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="只取套件前 N 题，用于试点")
    parser.add_argument("--only", default="", help="逗号分隔的任务 ID，用于冒烟或定向复验")
    parser.add_argument("--max-trial", type=int, default=0, help="本次只执行前 N 次试验，计划不变，可稍后续跑")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    run_dir = cfg.RUNS / args.run_id
    if not args.report_only:
        if not args.suite:
            parser.error("需要 --suite")
        plan = make_plan(args, run_dir)
        asyncio.run(execute(plan, run_dir, args.concurrency, args.max_trial))
    summary = write_report(run_dir)
    print(json.dumps({k: summary[k] for k in ("planned", "ran", "valid", "success", "pass_hat_k",
                                              "pass1_ci95", "safety", "cost_cny")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
