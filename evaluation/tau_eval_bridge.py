"""主进程侧评分桥：驱动独立进程中的官方 DB／NL 评分组件。

DB 评分不产生模型请求；NL 评分只产生一个 Judge 请求，由主进程用与
Agent／模拟用户相同的账本预留、发送和结算。
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from evaluation.retail_budget import BudgetExceeded

JudgeHandler = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class EvalBridgeError(RuntimeError):
    """评分子进程协议、超时或模型传输失败。"""


class EvalBridge:
    def __init__(
        self,
        *,
        python_exe: str | Path,
        worker_path: str | Path,
        data_dir: str | Path,
        tasks_path: str | Path,
        task_id: str,
        snapshot_path: str | Path,
        trajectory_path: str | Path,
        judge_model: str,
        judge_max_tokens: int,
        judge_temperature: float = 0.0,
        judge_seed: int = 300,
        stderr_path: str | Path,
        timeout_s: float = 180.0,
        judge_handler: Optional[JudgeHandler] = None,
        skip_nl: bool = False,
    ):
        self.python_exe = str(python_exe)
        self.worker_path = str(worker_path)
        self.data_dir = str(Path(data_dir).resolve())
        self.tasks_path = str(Path(tasks_path).resolve())
        self.task_id = str(task_id)
        self.snapshot_path = str(Path(snapshot_path).resolve())
        self.trajectory_path = str(Path(trajectory_path).resolve())
        self.judge_model = judge_model
        self.judge_max_tokens = int(judge_max_tokens)
        self.judge_temperature = float(judge_temperature)
        self.judge_seed = int(judge_seed)
        self.stderr_path = Path(stderr_path)
        self.timeout_s = float(timeout_s)
        self.judge_handler = judge_handler
        self.skip_nl = bool(skip_nl)
        self.proc: Optional[subprocess.Popen] = None
        self.protocol_log: list[Dict[str, Any]] = []
        self._stderr_handle = None
        # 即使后续 Judge 或进程失败，已得到的 DB 结果仍可供报告。
        self.db_result: Optional[Dict[str, Any]] = None
        self.nl_result: Optional[Dict[str, Any]] = None

    async def run(self) -> Dict[str, Any]:
        if self.judge_handler is None:
            raise EvalBridgeError("未配置 Judge 请求处理函数")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_handle = open(self.stderr_path, "w", encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        command = [
            self.python_exe,
            "-u",
            self.worker_path,
            "--data-dir",
            self.data_dir,
            "--tasks",
            self.tasks_path,
            "--task-id",
            self.task_id,
            "--snapshot",
            self.snapshot_path,
            "--trajectory",
            self.trajectory_path,
            "--judge-model",
            self.judge_model,
            "--judge-max-tokens",
            str(self.judge_max_tokens),
            "--judge-temperature",
            str(self.judge_temperature),
            "--judge-seed",
            str(self.judge_seed),
        ]
        if self.skip_nl:
            command.append("--skip-nl")
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        db_result: Optional[Dict[str, Any]] = None
        nl_result: Optional[Dict[str, Any]] = None
        try:
            while True:
                event = await self._read_event()
                kind = event.get("type")
                if kind == "db_result":
                    db_result = event
                    self.db_result = event
                    self._log(event)
                    continue
                if kind == "judge_request":
                    await self._handle_judge(event)
                    continue
                if kind == "nl_result":
                    nl_result = event
                    self.nl_result = event
                    self._log(event)
                    continue
                if kind == "done":
                    self._log({"type": "done"})
                    break
                if kind == "error":
                    raise EvalBridgeError(str(event.get("error") or "评分进程失败"))
                raise EvalBridgeError(f"评分进程返回未知事件: {kind}")
        finally:
            await self.close()
        if db_result is None:
            raise EvalBridgeError("评分进程未返回数据库结果")
        return {"db": db_result, "nl": nl_result, "protocol_log": self.protocol_log}

    async def _handle_judge(self, event: Dict[str, Any]) -> None:
        request_id = str(event.get("id") or "")
        started = time.monotonic()
        try:
            result = await self.judge_handler(event)
            payload = {"type": "judge_response", "id": request_id, **result}
            status = "success"
        except BudgetExceeded:
            # Judge 端预算拒绝后子进程会等待响应，必须先终止。
            self.kill()
            raise
        except Exception as exc:
            payload = {
                "type": "judge_response",
                "id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
            status = "failed"
        payload["generation_time_seconds"] = round(
            time.monotonic() - started, 3
        )
        self._log(
            {
                "type": "judge_exchange",
                "id": request_id,
                "status": status,
                "request": event.get("request"),
                "response": payload,
            }
        )
        self._write(payload)

    async def _read_event(self) -> Dict[str, Any]:
        if self.proc is None or self.proc.stdout is None:
            raise EvalBridgeError("评分子进程未启动")
        try:
            line = await asyncio.wait_for(
                asyncio.to_thread(self.proc.stdout.readline),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError as exc:
            self.kill()
            raise EvalBridgeError(
                f"等待评分进程超时（{self.timeout_s:.0f}s）"
            ) from exc
        if not line:
            code = await asyncio.to_thread(self.proc.wait)
            raise EvalBridgeError(
                f"评分子进程提前退出，code={code}; {self.stderr_tail()}"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalBridgeError(
                f"评分进程输出不是 JSON 行: {line[:400]}"
            ) from exc

    def _write(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise EvalBridgeError("评分子进程未启动")
        try:
            self.proc.stdin.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )
            self.proc.stdin.flush()
        except Exception as exc:
            raise EvalBridgeError(f"写入评分进程失败: {exc}") from exc

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
            self.kill()
            await asyncio.to_thread(self.proc.wait)
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self._stderr_handle is not None:
            self._stderr_handle.close()
        self.proc = None
