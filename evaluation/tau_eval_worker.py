"""固定 τ² 评分组件的独立进程适配。

本文件在 `.venv-tau` 中运行，复用官方 `EnvironmentEvaluator` 与
`NLAssertionsEvaluator`。预测侧只把官方 `set_state` 换成“使用实际快照、
不回放消息”的最小适配；参考侧仍是官方种子环境并按原 gold actions 执行。
所有 Judge 请求通过 JSON 行交给主进程，由同一账本预留和结算。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

JUDGE_REQUEST_LIMIT = 1


def _write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read() -> Dict[str, Any]:
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def _message_from_dict(item: Dict[str, Any], message_types: Dict[str, Any]):
    role = str(item.get("role") or "")
    if role == "assistant":
        tool_calls = []
        for call in item.get("tool_calls") or []:
            tool_calls.append(
                message_types["ToolCall"](
                    id=str(call.get("id") or ""),
                    name=str(call.get("name") or ""),
                    arguments=dict(call.get("arguments") or {}),
                    requestor=str(call.get("requestor") or "assistant"),
                )
            )
        return message_types["AssistantMessage"](
            id=item.get("id"),
            role="assistant",
            content=item.get("content"),
            tool_calls=tool_calls or None,
            cost=0.0,
        )
    if role == "user":
        return message_types["UserMessage"](
            id=item.get("id"),
            role="user",
            content=item.get("content") or "",
            cost=0.0,
        )
    if role == "tool":
        return message_types["ToolMessage"](
            id=str(item.get("id") or ""),
            role="tool",
            content=item.get("content") or "",
            requestor=str(item.get("requestor") or "assistant"),
            error=bool(item.get("error")),
        )
    raise RuntimeError(f"不支持的轨迹角色: {role}")


def _strict_nl_result(
    raw_content: str,
    assertions: List[str],
    official_reward: Any,
) -> Dict[str, Any]:
    """校验 Judge 输出与每条原断言一一对应，空列表不算通过。"""
    errors: List[str] = []
    try:
        parsed = json.loads(raw_content)
    except Exception as exc:
        return {
            "status": "JUDGE_ERROR",
            "errors": [f"Judge 输出不是 JSON: {type(exc).__name__}: {exc}"],
            "raw_output": raw_content,
            "checks": [],
            "reward": None,
            "official_reward": official_reward,
        }

    results = parsed.get("results") if isinstance(parsed, dict) else None
    if not isinstance(results, list):
        errors.append("Judge 输出缺少 results 列表")
        results = []

    matched: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(results):
        if not isinstance(item, dict):
            errors.append(f"results[{index}] 不是对象")
            continue
        outcome = item.get("expectedOutcome")
        if not isinstance(outcome, str):
            errors.append(f"results[{index}].expectedOutcome 不是字符串")
            continue
        key = outcome.strip()
        if key in matched:
            errors.append(f"断言重复: {key!r}")
            continue
        matched[key] = item
        met = item.get("metExpectation")
        if type(met) is not bool:
            errors.append(
                f"results[{index}].metExpectation 不是布尔值: {type(met).__name__}"
            )

    checks: List[Dict[str, Any]] = []
    for assertion in assertions:
        key = assertion.strip()
        item = matched.pop(key, None)
        if item is None:
            errors.append(f"缺少断言结果: {assertion!r}")
            continue
        checks.append(
            {
                "nl_assertion": assertion,
                "met": item.get("metExpectation"),
                "reasoning": item.get("reasoning"),
            }
        )
    for leftover in matched:
        errors.append(f"出现未知断言结果: {leftover!r}")

    if errors:
        return {
            "status": "JUDGE_ERROR",
            "errors": errors,
            "raw_output": raw_content,
            "checks": checks,
            "reward": None,
            "official_reward": official_reward,
        }

    reward = 1.0 if all(check["met"] is True for check in checks) else 0.0
    official_value = None
    if isinstance(official_reward, dict):
        official_value = official_reward.get("reward")
    if official_value is not None and float(official_value) != reward:
        errors.append(
            "严格校验后的 NL 奖励与原组件不一致: "
            f"{reward} != {official_value}"
        )
        return {
            "status": "JUDGE_ERROR",
            "errors": errors,
            "raw_output": raw_content,
            "checks": checks,
            "reward": None,
            "official_reward": official_reward,
        }
    return {
        "status": "OK",
        "errors": [],
        "raw_output": raw_content,
        "checks": checks,
        "reward": reward,
        "official_reward": official_reward,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-max-tokens", type=int, required=True)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-seed", type=int, default=300)
    parser.add_argument("--skip-nl", action="store_true")
    args = parser.parse_args()

    import os

    # tau2 在导入时解析数据目录，必须先设置；子进程日志只保留错误级。
    os.environ["TAU2_DATA_DIR"] = str(args.data_dir)
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    import tau2.utils.llm_utils as llm_utils
    import tau2.evaluator.evaluator_nl_assertions as nl_module
    from tau2.data_model.message import (
        AssistantMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.data_model import RetailDB
    from tau2.domains.retail.environment import RETAIL_DB_PATH, get_environment
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
    from tau2.utils.llm_utils import to_litellm_messages

    message_types = {
        "AssistantMessage": AssistantMessage,
        "UserMessage": UserMessage,
        "ToolMessage": ToolMessage,
        "ToolCall": ToolCall,
        "SystemMessage": SystemMessage,
    }

    raw_tasks = json.loads(open(args.tasks, "r", encoding="utf-8").read())
    raw_task = next(
        item for item in raw_tasks if str(item.get("id")) == str(args.task_id)
    )
    task = Task.model_validate(raw_task)
    snapshot = json.loads(open(args.snapshot, "r", encoding="utf-8").read())
    raw_trajectory = json.loads(
        open(args.trajectory, "r", encoding="utf-8").read()
    )
    trajectory = [
        _message_from_dict(item, message_types) for item in raw_trajectory
    ]

    predicted_db = RetailDB.model_validate(snapshot)
    predicted_env = get_environment(db=predicted_db)
    set_state_calls: List[Dict[str, Any]] = []

    def predicted_set_state(
        *,
        initialization_data: Any,
        initialization_actions: Any,
        message_history: Any,
        strict: bool = True,
    ) -> None:
        # 实际快照已经包含业务终态；这里只记录官方回放输入，不再改写状态。
        if initialization_data is not None or initialization_actions:
            raise RuntimeError(
                "预测侧快照适配只支持 initial_state 为空的任务"
            )
        set_state_calls.append(
            {
                "replay_messages": len(message_history or []),
                "strict": bool(strict),
            }
        )

    predicted_env.set_state = predicted_set_state

    gold_db = RetailDB.load(RETAIL_DB_PATH)
    gold_env = get_environment(db=gold_db)
    gold_errors: List[Dict[str, Any]] = []
    original_make_tool_call = gold_env.make_tool_call

    def recording_make_tool_call(
        tool_name: str, requestor: str = "assistant", **kwargs: Any
    ) -> Any:
        try:
            return original_make_tool_call(
                tool_name=tool_name, requestor=requestor, **kwargs
            )
        except Exception as exc:
            gold_errors.append(
                {
                    "tool_name": tool_name,
                    "requestor": requestor,
                    "arguments": kwargs,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

    gold_env.make_tool_call = recording_make_tool_call

    factory_calls: List[Dict[str, Any]] = []

    def factory(**kwargs: Any) -> Any:
        index = len(factory_calls)
        factory_calls.append({"index": index, "kwargs": dict(kwargs)})
        if index == 0:
            if kwargs.get("solo_mode") is not False:
                raise RuntimeError("官方首次工厂调用必须显式 solo_mode=False")
            return predicted_env
        if index == 1:
            if "solo_mode" in kwargs:
                raise RuntimeError("官方参考侧工厂调用不应传入 solo_mode")
            return gold_env
        raise RuntimeError("EnvironmentEvaluator 工厂调用次数超过预期")

    db_status = "OK"
    reward_info: Any = None
    evaluation_error = ""
    try:
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=factory,
            task=task,
            full_trajectory=trajectory,
            solo_mode=False,
            env_kwargs={},
        )
    except Exception as exc:
        db_status = "DB_ERROR"
        evaluation_error = f"{type(exc).__name__}: {exc}"

    if len(factory_calls) != 2:
        db_status = "DB_ERROR"
        evaluation_error = (
            f"官方评分工厂调用次数异常: {len(factory_calls)} != 2; "
            f"{evaluation_error}"
        ).strip("; ")

    reward_payload = (
        reward_info.model_dump(mode="json") if reward_info is not None else None
    )
    predicted_hash = predicted_env.get_db_hash()
    gold_hash = gold_env.get_db_hash()
    db_match = predicted_hash == gold_hash
    if gold_errors:
        db_status = "INVALID_REFERENCE"

    _write(
        {
            "type": "db_result",
            "status": db_status,
            "error": evaluation_error,
            "gold_errors": gold_errors,
            "official_reward": reward_payload,
            "predicted_hash": predicted_hash,
            "gold_hash": gold_hash,
            "snapshot_hash": predicted_db.get_hash(),
            "db_match": db_match,
            "factory_calls": factory_calls,
            "predicted_set_state_calls": set_state_calls,
            "trajectory_messages": len(trajectory),
            "task_id": str(task.id),
        }
    )

    evaluation_criteria = task.evaluation_criteria
    assertions = (
        list(evaluation_criteria.nl_assertions)
        if evaluation_criteria is not None
        and evaluation_criteria.nl_assertions
        else []
    )
    if not assertions:
        _write(
            {
                "type": "nl_result",
                "status": "NOT_APPLICABLE",
                "assertions": [],
                "checks": [],
                "reward": None,
                "raw_output": "",
                "judge_requests": 0,
                "note": "原任务无 NL 断言；沿用原实现恒等分量，不调用 Judge",
            }
        )
        _write({"type": "done"})
        return 0

    if args.skip_nl:
        # 预算已停止时保留免费 DB 结果，但绝不把未执行的 Judge 当作通过。
        _write(
            {
                "type": "nl_result",
                "status": "JUDGE_ERROR",
                "assertions": assertions,
                "checks": [],
                "reward": None,
                "raw_output": "",
                "judge_requests": 0,
                "errors": ["预算已停止，未调用 Judge"],
            }
        )
        _write({"type": "done"})
        return 0

    judge_capture: Dict[str, Any] = {}
    judge_requests = 0

    def bridge_generate(
        model: str,
        messages: List[Any],
        tools: Any = None,
        tool_choice: Any = None,
        call_name: Any = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        nonlocal judge_requests
        if tools:
            raise RuntimeError("NL Judge 不应使用工具")
        if judge_requests >= JUDGE_REQUEST_LIMIT:
            raise RuntimeError("单次评分只允许一个 Judge 请求")
        judge_requests += 1
        kwargs.pop("num_retries", None)
        request_id = f"judge-{task.id}-{judge_requests}"
        request = {
            "model": model,
            "messages": to_litellm_messages(messages),
            "max_tokens": int(kwargs.get("max_tokens") or args.judge_max_tokens),
            "temperature": float(
                kwargs.get("temperature", args.judge_temperature)
            ),
            "seed": args.judge_seed,
            "num_retries": 0,
        }
        _write(
            {
                "type": "judge_request",
                "id": request_id,
                "call_name": call_name,
                "request": request,
            }
        )
        response = _read()
        if response.get("type") != "judge_response":
            raise RuntimeError(
                f"预期 judge_response，收到 {response.get('type')}"
            )
        if response.get("id") != request_id:
            raise RuntimeError(
                f"Judge 响应 ID 不匹配: {response.get('id')} != {request_id}"
            )
        if response.get("error"):
            raise RuntimeError(f"Judge 请求失败: {response['error']}")
        judge_capture["content"] = response.get("content") or ""
        judge_capture["raw_data"] = response.get("raw_data") or {}
        usage = response.get("usage") or {}
        return AssistantMessage(
            role="assistant",
            content=judge_capture["content"],
            cost=0.0,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
            raw_data=judge_capture["raw_data"],
            generation_time_seconds=float(
                response.get("generation_time_seconds") or 0.0
            ),
        )

    nl_module.DEFAULT_LLM_NL_ASSERTIONS = args.judge_model
    nl_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
        "temperature": args.judge_temperature,
        "max_tokens": args.judge_max_tokens,
    }
    nl_module.generate = bridge_generate
    llm_utils.generate = bridge_generate

    official_nl: Any = None
    nl_error = ""
    try:
        official_nl = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
        )
        official_nl = official_nl.model_dump(mode="json")
    except Exception as exc:
        nl_error = f"{type(exc).__name__}: {exc}"

    raw_output = judge_capture.get("content") or ""
    if nl_error:
        result = {
            "status": "JUDGE_ERROR",
            "errors": [nl_error],
            "raw_output": raw_output,
            "checks": [],
            "reward": None,
            "official_reward": official_nl,
        }
    else:
        result = _strict_nl_result(raw_output, assertions, official_nl)
    result.update(
        {
            "type": "nl_result",
            "assertions": assertions,
            "judge_requests": judge_requests,
            "judge_raw_data": judge_capture.get("raw_data") or {},
        }
    )
    _write(result)
    _write({"type": "done"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
