"""主进程侧的最小 JSON 行桥，连接独立环境中的官方 UserSimulator。

桥只转发完整可见的 agent／user 文本；模型请求由 `llm_handler` 在主进程
处理，从而让 agent、user、judge 共用同一个持久化预算账本。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from evaluation.retail_budget import BudgetExceeded


class BridgeError(RuntimeError):
    """子进程协议、超时或模型传输失败。"""


LlmHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


def to_anthropic_kwargs(request: Dict[str, Any]) -> Dict[str, Any]:
    """把官方翻转后的 OpenAI 风格消息转为 Anthropic 请求字段。

    官方文字模拟用户不使用工具，因此这里只接受 system/user/assistant
    文本消息；`seed` 与 `num_retries` 只留在协议日志中，不传给 SDK。
    """
    messages = request.get("messages") or []
    system_parts: List[str] = []
    chat: List[Dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "")
        content = item.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise BridgeError(f"模拟用户消息内容不是文本: {type(content)}")
        if item.get("tool_calls"):
            raise BridgeError("纯文字模拟用户不应产生工具调用")
        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            chat.append({"role": role, "content": content})
        else:
            raise BridgeError(f"模拟用户消息角色不支持: {role}")
    if not chat:
        raise BridgeError("模拟用户请求缺少对话消息")
    if chat[0]["role"] != "user":
        raise BridgeError("模拟用户请求的首条对话消息必须是 user")
    expected = "user"
    for item in chat:
        if item["role"] != expected:
            raise BridgeError(
                "模拟用户请求角色未严格交替: "
                f"期望 {expected}，收到 {item['role']}"
            )
        expected = "assistant" if expected == "user" else "user"
    kwargs: Dict[str, Any] = {
        "model": str(request.get("model") or ""),
        "max_tokens": int(request.get("max_tokens") or 0),
        "temperature": float(request.get("temperature") or 0.0),
        "messages": chat,
    }
    if system_parts:
        kwargs["system"] = "\n\n".join(system_parts)
    return kwargs


class UserSimulatorBridge:
    """启动一个官方模拟用户子进程，并在半双工对话中逐轮转发。"""

    def __init__(
        self,
        *,
        python_exe: str | Path,
        worker_path: str | Path,
        data_dir: str | Path,
        tasks_path: str | Path,
        task_id: str,
        llm: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 300,
        stderr_path: str | Path,
        timeout_s: float = 180.0,
        llm_handler: Optional[LlmHandler] = None,
    ):
        self.python_exe = str(python_exe)
        self.worker_path = str(worker_path)
        self.data_dir = str(Path(data_dir).resolve())
        self.tasks_path = str(Path(tasks_path).resolve())
        self.task_id = str(task_id)
        self.llm = llm
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.stderr_path = Path(stderr_path)
        self.timeout_s = float(timeout_s)
        self.llm_handler = llm_handler
        self.proc: Optional[subprocess.Popen] = None
        self.ready: Dict[str, Any] = {}
        self.protocol_log: List[Dict[str, Any]] = []
        self._stderr_handle = None
        self._turn = 0

    async def start(self) -> Dict[str, Any]:
        if self.llm_handler is None:
            raise BridgeError("未配置模型请求处理函数")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = open(self.stderr_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        self.proc = subprocess.Popen(
            [
                self.python_exe,
                "-u",
                self.worker_path,
                "--data-dir",
                self.data_dir,
                "--tasks",
                self.tasks_path,
                "--task-id",
                self.task_id,
                "--llm",
                self.llm,
                "--max-tokens",
                str(self.max_tokens),
                "--temperature",
                str(self.temperature),
                "--seed",
                str(self.seed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        self.ready = await self._read_event()
        if self.ready.get("type") != "ready":
            raise BridgeError(f"模拟用户未就绪: {self.ready}")
        self._log({"type": "ready", **self.ready})
        return self.ready

    async def send_agent_message(self, content: str) -> Dict[str, Any]:
        """发送一条完整可见的 agent 回复，返回官方模拟用户的回复。"""
        if self.proc is None:
            raise BridgeError("模拟用户子进程未启动")
        self._turn += 1
        turn_id = f"turn-{self._turn}"
        self._write(
            {"type": "message", "id": turn_id, "content": str(content)}
        )
        self._log({"type": "agent_text", "id": turn_id, "content": content})
        while True:
            event = await self._read_event()
            kind = event.get("type")
            if kind == "llm_request":
                await self._handle_llm_request(event)
                continue
            if kind == "user":
                if event.get("id") != turn_id:
                    raise BridgeError(
                        f"模拟用户回复 ID 不匹配: {event.get('id')} != {turn_id}"
                    )
                self._log(event)
                return event
            if kind == "error":
                raise BridgeError(str(event.get("error") or "模拟用户失败"))
            raise BridgeError(f"模拟用户返回未知事件: {kind}")

    async def _handle_llm_request(self, event: Dict[str, Any]) -> None:
        request_id = str(event.get("id") or "")
        started = time.monotonic()
        try:
            result = await self.llm_handler(event)
            payload = {
                "type": "llm_response",
                "id": request_id,
                **result,
            }
            status = "success"
        except BudgetExceeded:
            # 子进程正阻塞等待本条响应；预算拒绝后直接终止，不能继续对话。
            self.kill()
            raise
        except Exception as exc:
            payload = {
                "type": "llm_response",
                "id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "failed"
        payload["generation_time_seconds"] = round(
            time.monotonic() - started, 3
        )
        self._log(
            {
                "type": "llm_exchange",
                "id": request_id,
                "status": status,
                "request": event.get("request"),
                "response": payload,
            }
        )
        self._write(payload)

    async def _read_event(self) -> Dict[str, Any]:
        if self.proc is None or self.proc.stdout is None:
            raise BridgeError("模拟用户子进程未启动")
        try:
            line = await asyncio.wait_for(
                asyncio.to_thread(self.proc.stdout.readline),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            self.kill()
            raise BridgeError(
                f"等待模拟用户超时（{self.timeout_s:.0f}s）"
            ) from exc
        if not line:
            code = await asyncio.to_thread(self.proc.wait)
            raise BridgeError(
                f"模拟用户子进程提前退出，code={code}; "
                f"{self.stderr_tail()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"模拟用户输出不是 JSON 行: {line[:400]}") from exc

    def _write(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise BridgeError("模拟用户子进程未启动")
        try:
            self.proc.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            self.proc.stdin.flush()
        except Exception as exc:
            raise BridgeError(f"写入模拟用户失败: {exc}") from exc

    def _log(self, payload: Dict[str, Any]) -> None:
        self.protocol_log.append({"timestamp": time.time(), **payload})

    def stderr_tail(self, limit: int = 1200) -> str:
        try:
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-limit:].strip()

    def kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass

    async def close(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self._write({"type": "stop"})
                await asyncio.wait_for(
                    asyncio.to_thread(self.proc.wait), timeout=5.0
                )
            except Exception:
                self.kill()
                await asyncio.to_thread(self.proc.wait)
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        self.proc = None
