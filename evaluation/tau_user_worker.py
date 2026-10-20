"""固定 τ² UserSimulator 的半双工子进程桥。

本文件在独立 `.venv-tau` 环境中运行，只负责官方模拟用户的角色翻转、
逐轮生成和停止标记。模型请求通过 JSON 行交给主进程，由主进程的同一
RetailLedger 预留、发送并结算，不在子进程内直接联网。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List

STOP = "###STOP###"
TRANSFER = "###TRANSFER###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


def _write(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _read() -> Dict[str, Any]:
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def _stop_kind(content: str) -> str:
    for marker, name in (
        (STOP, "STOP"),
        (TRANSFER, "TRANSFER"),
        (OUT_OF_SCOPE, "OUT_OF_SCOPE"),
    ):
        if marker in content:
            return name
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=300)
    args = parser.parse_args()

    # DATA_DIR 在 tau2 导入时解析，必须先设置；日志只保留错误级。
    os.environ["TAU2_DATA_DIR"] = str(args.data_dir)
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.tasks import Task
    from tau2.user.user_simulator import UserSimulator
    from tau2.utils.llm_utils import to_litellm_messages
    import tau2.user.user_simulator as user_simulator_module
    import tau2.utils.llm_utils as llm_utils

    raw_tasks = json.loads(open(args.tasks, "r", encoding="utf-8").read())
    raw_task = next(
        item for item in raw_tasks if str(item.get("id")) == str(args.task_id)
    )
    task = Task.model_validate(raw_task)
    instructions = str(task.user_scenario)

    def bridge_generate(
        model: str,
        messages: List[Any],
        tools: Any = None,
        tool_choice: Any = None,
        call_name: Any = None,
        **kwargs: Any,
    ) -> AssistantMessage:
        """把官方 generate 的同一请求转成主进程 RPC，不调用 LiteLLM。"""
        if tools:
            raise RuntimeError("本桥只支持无工具的纯文字模拟用户")
        # 官方默认会填入 num_retries=3；这里明确剔除并要求主进程不重试。
        kwargs.pop("num_retries", None)
        litellm_messages = to_litellm_messages(messages)
        request_id = f"user-{int(time.time() * 1000)}-{os.getpid()}"
        _write(
            {
                "type": "llm_request",
                "id": request_id,
                "call_name": call_name,
                "request": {
                    "model": model,
                    "messages": litellm_messages,
                    "max_tokens": int(kwargs.get("max_tokens") or args.max_tokens),
                    "temperature": float(
                        kwargs.get("temperature", args.temperature)
                    ),
                    "seed": kwargs.get("seed", args.seed),
                    "num_retries": 0,
                },
            }
        )
        response = _read()
        if response.get("type") != "llm_response":
            raise RuntimeError(f"预期 llm_response，收到 {response.get('type')}")
        if response.get("id") != request_id:
            raise RuntimeError(
                f"LLM 响应 ID 不匹配: {response.get('id')} != {request_id}"
            )
        if response.get("error"):
            raise RuntimeError(f"模型请求失败: {response['error']}")
        usage = response.get("usage") or {}
        return AssistantMessage(
            role="assistant",
            content=response.get("content") or "",
            cost=0.0,
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
            raw_data=response.get("raw_data") or {},
            generation_time_seconds=float(
                response.get("generation_time_seconds") or 0.0
            ),
        )

    # UserSimulator 模块以 from ... import generate 绑定了函数，两处都要替换。
    llm_utils.generate = bridge_generate
    user_simulator_module.generate = bridge_generate

    simulator = UserSimulator(
        llm=args.llm,
        instructions=instructions,
        tools=None,
        llm_args={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        persona_config=None,
    )
    simulator.set_seed(args.seed)
    state = simulator.get_init_state()

    _write(
        {
            "type": "ready",
            "task_id": str(task.id),
            "instruction_sha256": hashlib.sha256(
                instructions.encode("utf-8")
            ).hexdigest(),
            "stop_markers": [STOP, TRANSFER, OUT_OF_SCOPE],
        }
    )

    while True:
        command = _read()
        kind = command.get("type")
        if kind == "stop":
            _write({"type": "stopped"})
            return 0
        if kind != "message":
            raise RuntimeError(f"未知命令: {kind}")

        content = str(command.get("content") or "")
        if not content.strip():
            raise RuntimeError("Agent 回复为空，无法交给模拟用户")
        assistant_message = AssistantMessage(role="assistant", content=content)
        try:
            user_message, state = simulator.generate_next_message(
                assistant_message, state
            )
        except Exception as exc:
            _write(
                {
                    "type": "error",
                    "id": command.get("id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            return 3
        text = user_message.content or ""
        _write(
            {
                "type": "user",
                "id": command.get("id"),
                "content": text,
                "is_stop": simulator.is_stop(user_message),
                "stop_kind": _stop_kind(text),
                "usage": user_message.usage,
                "cost": user_message.cost,
            }
        )


if __name__ == "__main__":
    raise SystemExit(main())
