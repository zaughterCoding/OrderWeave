"""用本项目两阶段协议回放 τ² 参考写操作，并与官方环境的数据库哈希逐题比对。

每个写操作依次经过：准备 → 标记已展示 → 下一轮用户确认 → 同参数再次调用执行。
须用 τ 环境解释器运行（本项目业务库只依赖标准库）：
  .venv-tau/Scripts/python.exe evaluation/harness/replay_reference.py <tau_data_dir> [tasks.json]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    data_dir = Path(sys.argv[1]).resolve()
    tasks_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else data_dir / "tau2/domains/retail/tasks.json"
    os.environ["TAU2_DATA_DIR"] = str(data_dir)
    from loguru import logger

    logger.remove()
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.environment import get_environment

    from mcp_tools.retail_domain import WRITE_KINDS
    from mcp_tools.retail_store import RetailStore

    seed = data_dir / "tau2/domains/retail/db.json"
    base = json.loads(seed.read_text(encoding="utf-8"))
    rows, mismatches = [], []
    for raw in json.loads(tasks_path.read_text(encoding="utf-8")):
        task = Task.model_validate(raw)
        actions = task.evaluation_criteria.actions or []
        writes = [a for a in actions if a.name in WRITE_KINDS]
        if not writes:
            continue
        gold = get_environment()
        for action in actions:
            try:
                gold.make_tool_call(action.name, requestor="assistant", **action.arguments)
            except Exception:
                pass
        first = writes[0].arguments
        owner = first.get("user_id") or base["orders"].get(first.get("order_id"), {}).get("user_id")
        with tempfile.TemporaryDirectory() as tmp:
            store = RetailStore(Path(tmp) / "replay.db", seed, language="en")
            client, conv = "replay", f"task-{task.id}"
            store.begin_turn(client, conv)
            bound = store.find_user_by_email(client, conv, base["users"][owner]["email"]) if owner else {}
            outcomes = []
            for action in writes:
                turn = store.current_turn(client, conv)
                prepared = store.write_action(client, conv, action.name, dict(action.arguments), "")
                if not prepared.get("success"):
                    outcomes.append({"action": action.name, "stage": "prepare", "error": prepared.get("error")})
                    store.begin_turn(client, conv)
                    continue
                store.mark_pending_displayed(client, conv, prepared["op_id"], turn, False)
                store.begin_turn(client, conv)
                done = store.write_action(client, conv, action.name, dict(action.arguments), "Yes, please proceed.")
                outcomes.append({"action": action.name, "stage": "execute", "success": done.get("success"),
                                 "error": done.get("error")})
                store.begin_turn(client, conv)
            ours = get_environment(db=RetailDB.model_validate(store.business_snapshot())).get_db_hash()
        match = ours == gold.get_db_hash()
        rows.append({"id": task.id, "match": match, "bound": bool(bound.get("success")), "outcomes": outcomes})
        if not match:
            mismatches.append(task.id)
    print(json.dumps({"write_tasks": len(rows), "mismatches": mismatches,
                      "details": [r for r in rows if not r["match"]]}, ensure_ascii=False, indent=1))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
