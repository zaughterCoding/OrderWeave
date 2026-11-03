"""评测套件：τ² 官方任务与本项目生成的 τ 格式数据集。

自定义数据集目录含 tasks.json（官方 Task 格式）与 manifest.json（划分与标签），
套件名写作 ``数据集`` 或 ``数据集:标签[+标签]``，多个标签取交集，用于选择划分或能力兼容子集。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from evaluation.harness import config as cfg

# 现有退货／交接能力内的 τ 候选（任务 46 参考订单无效已排除）。
TAU_CORE = ("11", "12", "13", "14", "47", "48", "51", "53", "73", "82", "83", "84", "92", "108")
TAU_MINI_SIZE = 40
TAU_MINI_SEED = 20261103
_WRITE_TOOLS = {
    "return_delivered_order_items", "exchange_delivered_order_items", "cancel_pending_order",
    "modify_pending_order_items", "modify_pending_order_address", "modify_pending_order_payment",
    "modify_user_address",
}


def tau_mini(tasks: list) -> List[str]:
    """按参考写操作组合分层、固定种子抽样；只依赖任务定义，不看任何运行结果。"""
    import random

    groups: dict = {}
    for task in tasks:
        actions = (task.get("evaluation_criteria") or {}).get("actions") or []
        key = tuple(sorted({a["name"] for a in actions if a["name"] in _WRITE_TOOLS}))
        groups.setdefault(key, []).append(str(task["id"]))
    rng = random.Random(TAU_MINI_SEED)
    chosen: List[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key], key=int)
        rng.shuffle(ids)
        chosen += ids[: round(len(ids) * TAU_MINI_SIZE / len(tasks))]
    # 每类写操作至少一题，保证小样本能发现任一业务回退。
    for kind in sorted(_WRITE_TOOLS):
        if not any(kind in k and set(groups[k]) & set(chosen) for k in groups):
            key = min((k for k in groups if kind in k), key=len)
            chosen.append(sorted(groups[key], key=int)[0])
    rest = sorted((str(t["id"]) for t in tasks if str(t["id"]) not in chosen), key=int)
    rng.shuffle(rest)
    chosen += rest[: max(0, TAU_MINI_SIZE - len(chosen))]
    return sorted(chosen, key=int)


@dataclass(frozen=True)
class Suite:
    name: str
    tasks_path: Path
    task_ids: List[str]

    def digest(self) -> str:
        data = self.tasks_path.read_bytes() + json.dumps(self.task_ids).encode()
        return hashlib.sha256(data).hexdigest()


_GENERIC_TAGS = {"b", "c", "dev", "holdout", "compatible"}
HALF_SEED = "20261103"


def half(ids: List[str], tags: dict) -> List[str]:
    """按场景类型分层、固定哈希排序，每组取一半（向上取整）；只依赖任务定义。"""
    groups: dict = {}
    for task_id in ids:
        # 按首个非通用标签（扰动类型／攻击类型）分层，组数少、每组取半后总量接近一半。
        key = next((t for t in tags.get(task_id, []) if t not in _GENERIC_TAGS), "")
        groups.setdefault(key, []).append(task_id)
    chosen: List[str] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda t: hashlib.sha256((HALF_SEED + t).encode()).hexdigest())
        chosen += ranked[: (len(ranked) + 1) // 2]
    return [t for t in ids if t in set(chosen)]


def load_suite(name: str) -> Suite:
    """套件名可加 ``@half`` 后缀，取分层固定的一半任务（减半评测）。"""
    if name.endswith("@half"):
        suite = load_suite(name[: -len("@half")])
        tags = {}
        manifest = suite.tasks_path.with_name("manifest.json")
        if manifest.exists():
            tags = {row["id"]: row.get("tags", []) for row in json.loads(manifest.read_text(encoding="utf-8"))["tasks"]}
        return Suite(name, suite.tasks_path, half(suite.task_ids, tags))
    if name in {"tau-core", "tau-base", "tau-mini"}:
        tasks = json.loads(cfg.TAU_TASKS.read_text(encoding="utf-8"))
        ids = [str(t["id"]) for t in tasks]
        if name == "tau-core":
            missing = set(TAU_CORE) - set(ids)
            if missing:
                raise ValueError(f"τ任务缺失: {sorted(missing)}")
            ids = list(TAU_CORE)
        elif name == "tau-mini":
            ids = tau_mini(tasks)
        return Suite(name, cfg.TAU_TASKS, ids)
    dataset, _, tag = name.partition(":")
    folder = cfg.DATASETS / dataset
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    wanted = {t for t in tag.split("+") if t}
    ids = [row["id"] for row in manifest["tasks"] if wanted <= set(row.get("tags", []))]
    if not ids:
        raise ValueError(f"套件 {name} 没有任务")
    return Suite(name, folder / "tasks.json", ids)
