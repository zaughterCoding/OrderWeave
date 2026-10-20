"""T14.5 原评分适配的离线等价与严格 Judge 校验。

测试在独立 `.venv-tau` 子进程中调用固定官方评分组件；Fake Judge 不发网络
请求，只验证接线、一一对应和空结果风险。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from evaluation.tau_eval_bridge import EvalBridge

ROOT = Path(__file__).resolve().parents[1]
TAU_PY = ROOT / ".venv-tau" / "Scripts" / "python.exe"
TAU_DATA = ROOT / "evaluation" / "data" / "tau_retail" / "upstream_data"
TASKS = TAU_DATA / "tau2" / "domains" / "retail" / "tasks.json"
EVAL_WORKER = ROOT / "evaluation" / "tau_eval_worker.py"

pytestmark = pytest.mark.skipif(
    not TAU_PY.is_file(), reason="缺少独立 .venv-tau 评测环境"
)


def _official_snapshot(tmp_path: Path, actions: list[dict], name: str) -> dict:
    """在独立官方环境中执行给定 actions，导出官方 RetailDB 快照。"""
    out = tmp_path / f"{name}.json"
    script = r"""
import json, os, sys
os.environ["TAU2_DATA_DIR"] = sys.argv[1]
from loguru import logger
logger.remove()
from tau2.data_model.message import ToolCall
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import RETAIL_DB_PATH, get_environment
db = RetailDB.load(RETAIL_DB_PATH)
env = get_environment(db=db)
for i, action in enumerate(json.loads(sys.argv[2])):
    env.get_response(ToolCall(
        id=f"gold-{i}", name=action["name"],
        arguments=action["arguments"],
        requestor=action.get("requestor", "assistant"),
    ))
with open(sys.argv[3], "w", encoding="utf-8") as fh:
    json.dump(env.tools.db.model_dump(mode="json"), fh)
"""
    result = subprocess.run(
        [str(TAU_PY), "-c", script, str(TAU_DATA), json.dumps(actions), str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return json.loads(out.read_text(encoding="utf-8"))


def _official_trajectory_baseline(
    tmp_path: Path, task_id: str, name: str
) -> dict:
    """用官方环境生成一条工具轨迹，并直接调用原 EnvironmentEvaluator。"""
    out = tmp_path / f"{name}.json"
    script = r"""
import json, os, sys
os.environ["TAU2_DATA_DIR"] = sys.argv[1]
from loguru import logger
logger.remove()
from tau2.data_model.message import (
    AssistantMessage, ToolCall, UserMessage,
)
from tau2.data_model.tasks import Task
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.environment import RETAIL_DB_PATH, get_environment
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

raw_tasks = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
raw = next(item for item in raw_tasks if str(item.get("id")) == sys.argv[3])
task = Task.model_validate(raw)

live = get_environment(db=RetailDB.load(RETAIL_DB_PATH))
trajectory = [
    AssistantMessage(
        role="assistant", content="Hi! How can I help you today?", cost=0.0
    ),
    UserMessage(role="user", content="Please process my return."),
]
for index, action in enumerate(task.evaluation_criteria.actions or []):
    call = ToolCall(
        id=f"toolu-gold-{index}",
        name=action.name,
        arguments=action.arguments,
        requestor="assistant",
    )
    response = live.get_response(call)
    trajectory.append(
        AssistantMessage(
            role="assistant", content=None, tool_calls=[call], cost=0.0
        )
    )
    trajectory.append(response)
trajectory.append(
    AssistantMessage(
        role="assistant",
        content="I have submitted the return as confirmed.",
        cost=0.0,
    )
)
trajectory.append(UserMessage(role="user", content="###STOP###"))

predicted = get_environment(db=RetailDB.load(RETAIL_DB_PATH))
gold = get_environment(db=RetailDB.load(RETAIL_DB_PATH))
calls = []
def factory(**kwargs):
    calls.append(kwargs)
    return predicted if len(calls) == 1 else gold

reward = EnvironmentEvaluator.calculate_reward(
    environment_constructor=factory,
    task=task,
    full_trajectory=trajectory,
    solo_mode=False,
    env_kwargs={},
)
with open(sys.argv[4], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "trajectory": [item.model_dump(mode="json") for item in trajectory],
            "reward": reward.model_dump(mode="json"),
            "gold_snapshot": gold.tools.db.model_dump(mode="json"),
            "factory_calls": calls,
        },
        handle,
        ensure_ascii=False,
    )
"""
    result = subprocess.run(
        [str(TAU_PY), "-c", script, str(TAU_DATA), str(TASKS), task_id, str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-1500:]
    return json.loads(out.read_text(encoding="utf-8"))


def _task(task_id: str) -> dict:
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    return next(item for item in tasks if str(item.get("id")) == task_id)


def _task_actions(task_id: str) -> list[dict]:
    return list((_task(task_id).get("evaluation_criteria") or {}).get("actions") or [])


def _task_assertions(task_id: str) -> list[str]:
    return list(
        (_task(task_id).get("evaluation_criteria") or {}).get("nl_assertions") or []
    )


def _run_eval(
    tmp_path: Path,
    task_id: str,
    snapshot: dict,
    judge_handler,
    trajectory: list[dict] | None = None,
) -> dict:
    snapshot_path = tmp_path / f"snapshot-{task_id}.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    trajectory_path = tmp_path / f"trajectory-{task_id}.json"
    trajectory_path.write_text(json.dumps(trajectory or []), encoding="utf-8")
    bridge = EvalBridge(
        python_exe=TAU_PY,
        worker_path=EVAL_WORKER,
        data_dir=TAU_DATA,
        tasks_path=TASKS,
        task_id=task_id,
        snapshot_path=snapshot_path,
        trajectory_path=trajectory_path,
        judge_model="deepseek-flash",
        judge_max_tokens=1024,
        judge_temperature=0.0,
        judge_seed=300,
        stderr_path=tmp_path / f"eval-{task_id}.log",
        timeout_s=120.0,
        judge_handler=judge_handler,
    )
    return asyncio.run(bridge.run())


@pytest.fixture(scope="module")
def gold11(tmp_path_factory) -> dict:
    return _official_snapshot(
        tmp_path_factory.mktemp("gold11"), _task_actions("11"), "gold11"
    )


@pytest.fixture(scope="module")
def seed46(tmp_path_factory) -> dict:
    return _official_snapshot(tmp_path_factory.mktemp("seed46"), [], "seed46")


def test_snapshot_evaluator_matches_official_gold_state(tmp_path, gold11):
    """实际完整快照接入原 EnvironmentEvaluator 后与 gold 状态一致。"""

    async def no_judge(event):
        raise AssertionError("DB 评分不应调用 Judge")

    result = _run_eval(tmp_path, "11", gold11, no_judge)
    db = result["db"]

    assert db["status"] == "OK", db
    assert db["db_match"] is True
    assert db["predicted_hash"] == db["gold_hash"] == db["snapshot_hash"]
    assert db["official_reward"]["reward"] == 1.0
    assert db["official_reward"]["db_check"]["db_match"] is True
    assert len(db["factory_calls"]) == 2
    assert db["factory_calls"][0]["kwargs"] == {"solo_mode": False}
    assert db["factory_calls"][1]["kwargs"] == {}
    assert db["predicted_set_state_calls"] == [
        {"replay_messages": 0, "strict": True}
    ]
    assert db["gold_errors"] == []
    assert result["nl"]["status"] == "NOT_APPLICABLE"
    assert result["nl"]["judge_requests"] == 0


def test_snapshot_adapter_matches_official_replay_on_tool_trajectory(tmp_path):
    """同一条官方兼容工具轨迹：快照适配与原回放评分结果相等。"""
    baseline = _official_trajectory_baseline(tmp_path, "11", "baseline11")
    assert baseline["reward"]["reward"] == 1.0
    assert baseline["factory_calls"] == [{"solo_mode": False}, {}]

    async def no_judge(event):
        raise AssertionError("DB 评分不应调用 Judge")

    result = _run_eval(
        tmp_path,
        "11",
        baseline["gold_snapshot"],
        no_judge,
        trajectory=baseline["trajectory"],
    )
    assert result["db"]["status"] == "OK", result["db"]
    assert result["db"]["official_reward"]["reward"] == 1.0
    assert result["db"]["db_match"] is True
    assert result["nl"]["status"] == "NOT_APPLICABLE"
    # 工具轨迹被完整读入官方评分器。
    assert result["db"]["trajectory_messages"] == len(baseline["trajectory"])


def test_snapshot_evaluator_detects_single_field_and_other_user_changes(
    tmp_path, gold11
):
    """单字段、非目标用户改动以及未执行目标写入都必须判 0。"""

    async def no_judge(event):
        raise AssertionError("DB 评分不应调用 Judge")

    missing_write = _official_snapshot(tmp_path, [], "seed")
    no_write = _run_eval(tmp_path, "11", missing_write, no_judge)["db"]
    assert no_write["official_reward"]["reward"] == 0.0
    assert no_write["db_match"] is False

    single_field = json.loads(json.dumps(gold11))
    single_field["orders"]["#W5490111"]["status"] = "delivered"
    changed = _run_eval(tmp_path, "11", single_field, no_judge)["db"]
    assert changed["official_reward"]["reward"] == 0.0
    assert changed["db_match"] is False

    other_user = json.loads(json.dumps(gold11))
    other_user["users"]["mia_garcia_4516"]["name"]["first_name"] = "Tampered"
    changed_user = _run_eval(tmp_path, "11", other_user, no_judge)["db"]
    assert changed_user["official_reward"]["reward"] == 0.0
    assert changed_user["db_match"] is False

    other_order = json.loads(json.dumps(gold11))
    untouched_order = next(
        key
        for key in other_order["orders"]
        if key not in {"#W5490111", "#W7387996"}
    )
    other_order["orders"][untouched_order]["status"] = "return requested"
    changed_order = _run_eval(tmp_path, "11", other_order, no_judge)["db"]
    assert changed_order["official_reward"]["reward"] == 0.0
    assert changed_order["db_match"] is False

    other_product = json.loads(json.dumps(gold11))
    untouched_product = next(iter(other_product["products"]))
    other_product["products"][untouched_product]["name"] = "Tampered"
    changed_product = _run_eval(
        tmp_path, "11", other_product, no_judge
    )["db"]
    assert changed_product["official_reward"]["reward"] == 0.0
    assert changed_product["db_match"] is False


def _judge_payload(assertions: list[str], met=True) -> str:
    return json.dumps(
        {
            "results": [
                {
                    "expectedOutcome": item,
                    "reasoning": "fake",
                    "metExpectation": met,
                }
                for item in assertions
            ]
        }
    )


def test_nl_judge_is_wired_to_official_prompt_and_strict_result(
    tmp_path, seed46
):
    assertions = _task_assertions("46")
    assert len(assertions) == 2
    captured = {}

    async def judge(event):
        captured.update(event["request"])
        return {
            "content": _judge_payload(assertions, met=True),
            "usage": {"prompt_tokens": 200, "completion_tokens": 60},
            "raw_data": {"fake": True},
        }

    result = _run_eval(tmp_path, "46", seed46, judge)
    nl = result["nl"]

    assert nl["status"] == "OK", nl
    assert nl["reward"] == 1.0
    assert [check["met"] for check in nl["checks"]] == [True, True]
    assert nl["judge_requests"] == 1
    assert captured["model"] == "deepseek-flash"
    assert captured["max_tokens"] == 1024
    assert captured["num_retries"] == 0
    combined = "\n".join(item["content"] for item in captured["messages"])
    assert "expectedOutcomes" in combined
    assert assertions[0] in combined
    assert assertions[1] in combined


def test_nl_empty_or_incomplete_results_are_judge_errors(tmp_path, seed46):
    assertions = _task_assertions("46")

    async def empty(event):
        return {
            "content": json.dumps({"results": []}),
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
        }

    empty_nl = _run_eval(tmp_path, "46", seed46, empty)["nl"]
    assert empty_nl["status"] == "JUDGE_ERROR"
    assert empty_nl["reward"] is None
    assert any("缺少断言结果" in item for item in empty_nl["errors"])
    assert json.loads(empty_nl["raw_output"]) == {"results": []}

    async def missing(event):
        return {
            "content": _judge_payload(assertions[:1]),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    missing_nl = _run_eval(tmp_path, "46", seed46, missing)["nl"]
    assert missing_nl["status"] == "JUDGE_ERROR"
    assert any("缺少断言结果" in item for item in missing_nl["errors"])

    async def wrong_type(event):
        return {
            "content": json.dumps(
                {
                    "results": [
                        {
                            "expectedOutcome": item,
                            "reasoning": "fake",
                            "metExpectation": "true",
                        }
                        for item in assertions
                    ]
                }
            ),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }

    typed_nl = _run_eval(tmp_path, "46", seed46, wrong_type)["nl"]
    assert typed_nl["status"] == "JUDGE_ERROR"
    assert any("不是布尔值" in item for item in typed_nl["errors"])
    assert typed_nl["reward"] is None
