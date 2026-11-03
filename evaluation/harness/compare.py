"""两次运行的配对比较：同任务同试验号逐一配对，按任务 bootstrap 差值区间。

用法：
  uv run python -m evaluation.harness.compare --before s1-before-tau --after s1-after-tau
未运行或无效会话按失败计入（与报告的“成功／计划”口径一致）；只比较两边共有的任务与试验号，
因此减半套件可直接与上一阶段的完整运行配对（paired 中 before_rate／after_rate 为共有部分的成功率）。
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from evaluation.harness import config as cfg
from evaluation.harness.report import summarize


def _success(run_id: str) -> Dict[str, Dict[int, bool]]:
    run_dir = cfg.RUNS / run_id
    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    out: Dict[str, Dict[int, bool]] = {}
    for task_id in plan["task_ids"]:
        for trial in range(1, int(plan["trials"]) + 1):
            path = run_dir / "sessions" / task_id / f"trial-{trial}" / "result.json"
            ok = bool(json.loads(path.read_text(encoding="utf-8")).get("success")) if path.exists() else False
            out.setdefault(task_id, {})[trial] = ok
    return out


def paired_delta(before: Dict[str, Dict[int, bool]], after: Dict[str, Dict[int, bool]],
                 rounds: int = 4000, seed: int = 11) -> Dict[str, Any]:
    tasks = sorted(set(before) & set(after), key=lambda t: (len(t), t))
    diffs: List[float] = []
    for t in tasks:
        trials = sorted(set(before[t]) & set(after[t]))
        if trials:
            diffs.append(sum(after[t][n] - before[t][n] for n in trials) / len(trials))
    if not diffs:
        return {"tasks": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    flips = {"fixed": sum(1 for t in tasks for n in before[t] if n in after[t] and after[t][n] and not before[t][n]),
             "broken": sum(1 for t in tasks for n in before[t] if n in after[t] and before[t][n] and not after[t][n])}
    rate = lambda side: round(sum(side[t][n] for t in tasks for n in side[t] if n in before[t] and n in after[t])  # noqa: E731
                              / max(1, sum(1 for t in tasks for n in before[t] if n in after[t])), 4)
    return {"tasks": len(diffs), "before_rate": rate(before), "after_rate": rate(after),
            "delta": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds) - 1], 4)], **flips}


def main() -> int:
    parser = argparse.ArgumentParser(description="配对比较两次评测运行")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    args = parser.parse_args()
    b, a = summarize(cfg.RUNS / args.before), summarize(cfg.RUNS / args.after)
    keys = ("success_over_planned", "pass_hat_k", "safety", "cost_cny", "tokens", "turn_latency_ms")
    print(json.dumps({"before": {k: b.get(k) for k in keys}, "after": {k: a.get(k) for k in keys},
                      "paired": paired_delta(_success(args.before), _success(args.after))},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
