"""Judge 校准：抽样导出自然语言断言判定供人工标注，再计算一致率。

  uv run python -m evaluation.harness.calibrate export --runs s0-base-b s0-base-c --n 50
  （在 calibration.csv 的 human 列填 1／0 后）
  uv run python -m evaluation.harness.calibrate score
人工标注只看对话本身判断断言是否成立，不参考 judge 列。
"""
from __future__ import annotations

import argparse
import csv
import json
import random

from evaluation.harness import config as cfg

OUT = cfg.RUNS / "calibration.csv"


def export(runs, n: int, seed: int) -> None:
    rows = []
    for run in runs:
        for path in sorted((cfg.RUNS / run / "sessions").glob("*/trial-*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            for index, check in enumerate(result.get("nl_checks") or []):
                rows.append({"run": run, "task_id": result["task_id"], "trial": result["trial"], "index": index,
                             "assertion": check.get("nl_assertion"), "judge": int(bool(check.get("met"))),
                             "dialogue": str(path.parent / "dialogue.json"), "human": ""})
    random.Random(seed).shuffle(rows)
    if OUT.exists():
        raise SystemExit(f"{OUT} 已存在，避免覆盖已有标注")
    with OUT.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows[:n])
    print(f"导出 {min(n, len(rows))}/{len(rows)} 条到 {OUT}")


def score() -> None:
    with OUT.open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["human"].strip() in {"0", "1"}]
    agree = sum(r["human"].strip() == r["judge"] for r in rows)
    pos = [r for r in rows if r["human"].strip() == "1"]
    neg = [r for r in rows if r["human"].strip() == "0"]
    print(json.dumps({"labeled": len(rows), "agreement": round(agree / len(rows), 4) if rows else None,
                      "judge_false_negative": sum(r["judge"] == "0" for r in pos),
                      "judge_false_positive": sum(r["judge"] == "1" for r in neg)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--seed", type=int, default=11)
    sub.add_parser("score")
    args = parser.parse_args()
    export(args.runs, args.n, args.seed) if args.cmd == "export" else score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
