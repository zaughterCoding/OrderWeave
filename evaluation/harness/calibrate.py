"""Judge 校准：抽样导出自然语言断言判定供人工标注，再计算一致率。

  uv run python -m evaluation.harness.calibrate export --runs s0-base-b s0-base-c --n 50
  （在 calibration.csv 的 human 列填 1／0 后）
  uv run python -m evaluation.harness.calibrate score
  或导出盲标文件交独立标注者：calibrate blind → 填 label → calibrate score --labels <文件>
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


BLIND = cfg.RUNS / "calibration-blind.csv"


def blind() -> None:
    """导出不含 judge 列的盲标文件，供独立标注者（人或另一模型家族）填写 label。"""
    with OUT.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if BLIND.exists():
        raise SystemExit(f"{BLIND} 已存在，避免覆盖已有标注")
    with BLIND.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row", "assertion", "dialogue", "label", "reason"])
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({"row": i, "assertion": r["assertion"], "dialogue": r["dialogue"], "label": "", "reason": ""})
    print(f"导出 {len(rows)} 条到 {BLIND}")


def score(labels: str = "") -> None:
    with OUT.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if labels:
        # 盲标文件按 row 序号对回原行，标注写入 human 口径参与计算。
        with open(labels, encoding="utf-8-sig") as stream:
            marks = {int(m["row"]): m["label"].strip() for m in csv.DictReader(stream)}
        for i, r in enumerate(rows):
            r["human"] = marks.get(i, "")
    rows = [r for r in rows if r["human"].strip() in {"0", "1"}]
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
    sub.add_parser("blind")
    sc = sub.add_parser("score")
    sc.add_argument("--labels", default="", help="盲标文件路径（含 row 与 label 列）")
    args = parser.parse_args()
    if args.cmd == "export":
        export(args.runs, args.n, args.seed)
    elif args.cmd == "blind":
        blind()
    else:
        score(args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
