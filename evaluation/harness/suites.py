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


@dataclass(frozen=True)
class Suite:
    name: str
    tasks_path: Path
    task_ids: List[str]

    def digest(self) -> str:
        data = self.tasks_path.read_bytes() + json.dumps(self.task_ids).encode()
        return hashlib.sha256(data).hexdigest()


def load_suite(name: str) -> Suite:
    if name in {"tau-core", "tau-base"}:
        tasks = json.loads(cfg.TAU_TASKS.read_text(encoding="utf-8"))
        ids = [str(t["id"]) for t in tasks]
        if name == "tau-core":
            missing = set(TAU_CORE) - set(ids)
            if missing:
                raise ValueError(f"τ任务缺失: {sorted(missing)}")
            ids = list(TAU_CORE)
        return Suite(name, cfg.TAU_TASKS, ids)
    dataset, _, tag = name.partition(":")
    folder = cfg.DATASETS / dataset
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    wanted = {t for t in tag.split("+") if t}
    ids = [row["id"] for row in manifest["tasks"] if wanted <= set(row.get("tags", []))]
    if not ids:
        raise ValueError(f"套件 {name} 没有任务")
    return Suite(name, folder / "tasks.json", ids)
