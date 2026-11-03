"""在 τ² 官方环境中回放任务的参考写操作，确认 gold 可执行。

须用独立 τ 环境的解释器运行：.venv-tau/Scripts/python.exe evaluation/harness/validate_tasks.py <tasks.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    tasks_path = Path(sys.argv[1]).resolve()
    data_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (
        Path(__file__).resolve().parents[2] / "evaluation" / "data" / "tau_retail" / "upstream_data")
    os.environ["TAU2_DATA_DIR"] = str(data_dir)
    from loguru import logger

    logger.remove()
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment

    write_tools = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
                   "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
                   "return_delivered_order_items"}
    failures = []
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    for raw in tasks:
        task = Task.model_validate(raw)
        env = get_environment()
        before = env.get_db_hash()
        actions = task.evaluation_criteria.actions or []
        # 与官方回放一致：单个动作报错不中断后续动作；写操作报错才影响目标状态。
        for action in actions:
            try:
                env.make_tool_call(action.name, requestor="assistant", **action.arguments)
            except Exception as exc:
                failures.append({"id": task.id, "action": action.name, "kind": "write" if action.name in write_tools else "read",
                                 "error": f"{type(exc).__name__}: {exc}"})
        if any(a.name in write_tools for a in actions) and env.get_db_hash() == before:
            failures.append({"id": task.id, "kind": "write", "error": "参考写操作没有改变数据库"})
    print(json.dumps({"tasks": len(tasks), "failures": failures}, ensure_ascii=False, indent=1))
    return 1 if any(f["kind"] == "write" for f in failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
