"""受控自进化：由开发集失败轨迹反思，生成 Skill 操作要点的修改候选。

  uv run python -m evaluation.harness.evolve collect --runs <开发集运行...> --out <digest.json>
  uv run python -m evaluation.harness.evolve propose --digest <digest.json> --out-dir <候选根目录> --n 3
  uv run python -m evaluation.harness.evolve diff --candidate <候选目录>

只允许修改 Skill 的 description 与操作要点正文；政策原文章节、绑定工具、名称与授权逻辑不可变。
候选必须通过不变量校验（字段未变、长度受限、不含具体订单号／商品号／人名等案例原文），
随后在开发集上评估、经留出集闸门与维护者批准才发布。只读取开发集运行，不读留出集。
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

from evaluation.harness import config as cfg

SKILLS = cfg.ROOT / "skills"
_CASE_TEXT = re.compile(r"#?W\d{7}|\b\d{10}\b|[\w.+-]+@[\w-]+\.\w+|\b(?:gift_card|credit_card|paypal)_\d+\b", re.I)


def _skill_files(root: Path) -> Dict[str, Path]:
    return {p.parent.name: p for p in sorted(root.glob("retail-*/SKILL.md"))}


def _split(text: str):
    parts = text.split("---", 2)
    return parts[1], parts[2]


def collect(runs: List[str], max_turns: int = 14) -> List[Dict[str, Any]]:
    """把开发集失败会话压缩为反思输入：用户与 Agent 原话、工具调用摘要、评分失败点。"""
    out = []
    for run in runs:
        for path in sorted((cfg.RUNS / run / "sessions").glob("*/trial-*/result.json")):
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("success") or not result.get("valid"):
                continue
            dialogue = json.loads((path.parent / "dialogue.json").read_text(encoding="utf-8"))
            turns = []
            for turn in dialogue["turns"][-max_turns:]:
                tools = [f"{t.get('tool_name')}({json.dumps(t.get('input'), ensure_ascii=False)[:160]})"
                         f"->{'ok' if t.get('result_success') else 'fail'}" for t in turn.get("tool_traces") or []]
                turns.append({"user": turn["user"][:600], "tools": tools, "agent": turn["assistant"][:900]})
            out.append({"run": run, "task": result["task_id"], "termination": result["termination"],
                        "db_ok": result.get("db_reward") == 1.0, "changed": result.get("changed"),
                        "failed_assertions": [c.get("nl_assertion") for c in result.get("nl_checks") or []
                                              if not c.get("met")], "turns": turns})
    return out


_SYSTEM = """You improve the procedure text of retail customer-service skills from failed conversations.
Each skill has fixed policy text (you cannot change it) and a short procedure ("playbook") the agent reads before acting.
Study the failures, find recurring causes that a better procedure would prevent, and rewrite the playbooks.

Rules:
- Write general procedures that would help any customer. Never include order ids, item ids, product names from the
  failures, customer names, emails, payment ids, amounts or quotes from the conversations.
- Do not weaken any safety rule: identity verification, showing details and waiting for explicit confirmation in a
  later message, one action at a time, and the policy limits stay as they are.
- Do not add steps that the tools or policy cannot support. Keep each playbook under 1800 characters.
- Change only skills whose failures you can explain; return the full new playbook for each changed skill.

Return only JSON: {"analysis": "short summary of failure causes",
"changes": [{"skill": "<directory name>", "description": "<one-line description>", "playbook": "<full new playbook markdown>",
"rationale": "why this prevents the failures"}]}"""


async def _propose_one(client, model: str, skills: Dict[str, str], failures: List[Dict[str, Any]], focus: str) -> Dict[str, Any]:
    from core.llm_utils import extract_text_content

    payload = {"focus": focus, "skills": skills, "failures": failures}
    resp = await client.messages.create(model=model, max_tokens=6000, temperature=0.4, system=_SYSTEM,
                                        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                        extra_body={"thinking": {"type": "disabled"}})
    text = extract_text_content(resp.content)
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0) if match else text)


def validate(candidate: Path) -> List[str]:
    """不变量：技能集合、名称、绑定工具、政策章节、启用标记不变；正文受限且不含案例原文。"""
    problems = []
    base, cand = _skill_files(SKILLS), _skill_files(candidate)
    if set(base) != set(cand):
        return ["技能集合变化"]
    for name, path in base.items():
        old_meta, _ = _split(path.read_text(encoding="utf-8"))
        new_meta, new_body = _split(cand[name].read_text(encoding="utf-8"))
        keep = lambda meta: [l for l in meta.strip().splitlines() if not l.startswith("description:")]  # noqa: E731
        if keep(old_meta) != keep(new_meta):
            problems.append(f"{name}: 除 description 外的字段被修改")
        if re.search(r"(?m)^---\s*$", new_body):
            problems.append(f"{name}: 正文含 front matter")
        if len(new_body) > 2400:
            problems.append(f"{name}: 正文过长")
        if _CASE_TEXT.search(new_body + new_meta):
            problems.append(f"{name}: 含具体标识（疑似案例原文）")
    return problems


def write_candidate(change_set: Dict[str, Any], out: Path) -> List[str]:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SKILLS, out, ignore=lambda d, names: [n for n in names if Path(d) == SKILLS and not n.startswith("retail-")])
    for change in change_set.get("changes") or []:
        path = out / str(change.get("skill")) / "SKILL.md"
        if not path.exists():
            continue
        meta, _ = _split(path.read_text(encoding="utf-8"))
        meta = re.sub(r"(?m)^description:.*$", "description: " + str(change.get("description", "")).replace("\n", " ")[:200], meta)
        playbook = str(change.get("playbook", "")).strip()
        # 模型有时连同 front matter 一起返回整份文件，只保留正文。
        if playbook.startswith("---") and playbook.count("---") >= 2:
            playbook = playbook.split("---", 2)[2].strip()
        path.write_text("---" + meta + "---\n\n" + playbook + "\n", encoding="utf-8")
    (out / "CHANGES.json").write_text(json.dumps(change_set, ensure_ascii=False, indent=2), encoding="utf-8")
    return validate(out)


def diff(candidate: Path) -> str:
    lines = []
    for name, path in _skill_files(SKILLS).items():
        new = candidate / name / "SKILL.md"
        lines += difflib.unified_diff(path.read_text(encoding="utf-8").splitlines(), new.read_text(encoding="utf-8").splitlines(),
                                      f"skills/{name}/SKILL.md", f"candidate/{name}/SKILL.md", lineterm="")
    return "\n".join(lines)


async def propose(digest: Path, out_dir: Path, n: int) -> None:
    cfg.load_env()
    from anthropic import AsyncAnthropic

    failures = json.loads(digest.read_text(encoding="utf-8"))
    skills = {name: path.read_text(encoding="utf-8") for name, path in _skill_files(SKILLS).items()}
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url=os.environ.get("ANTHROPIC_BASE_URL") or None)
    model = cfg.model_settings()["model"]
    focuses = ["all failures", "failures where the database ended in the wrong state",
               "failures where the agent's wording or summary was judged wrong"][:n]
    results = await asyncio.gather(*(_propose_one(client, model, skills, failures, f) for f in focuses), return_exceptions=True)
    for index, result in enumerate(results, start=1):
        out = out_dir / f"cand-{index}"
        if isinstance(result, Exception):
            print(f"cand-{index}: 生成失败 {result}")
            continue
        problems = write_candidate(result, out)
        print(f"cand-{index}: {len(result.get('changes') or [])} 处修改，校验{'通过' if not problems else '失败: ' + '; '.join(problems)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="受控自进化：反思失败、生成并校验 Skill 修改候选")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--digest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=3)
    d = sub.add_parser("diff")
    d.add_argument("--candidate", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--candidate", required=True)
    args = parser.parse_args()
    if args.cmd == "collect":
        data = collect(args.runs)
        Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(data)} 个失败会话")
    elif args.cmd == "propose":
        asyncio.run(propose(Path(args.digest), Path(args.out_dir), args.n))
    elif args.cmd == "diff":
        print(diff(Path(args.candidate)))
    else:
        print(validate(Path(args.candidate)) or "通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
