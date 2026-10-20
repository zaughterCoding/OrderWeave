"""本地零售 MCP 客户端。

主程序在同一个事件循环内启动一个工具子进程，初始化后核对九工具白名单，
再把 tools/call 包装成 AgentToolSpec。业务失败按原对象返回；超时、进程
退出或结果格式不符抛出传输／协议错误，不重试、不改走本地直调。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import anyio
import mcp.types as types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import McpError

from agents.tools import AgentToolSpec
from mcp.retail_protocol import build_context_meta

logger = logging.getLogger(__name__)

PROTOCOL_ERROR_TIMEOUT = 408


class RetailMCPError(RuntimeError):
    """零售 MCP 客户端错误的基类。"""


class RetailMCPTransportError(RetailMCPError):
    """进程、管道或超时导致的调用失败。"""


class RetailMCPProtocolError(RetailMCPError):
    """工具发现或结果合同不符合约定。"""

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code


class RetailMCPClient:
    """一个运行时复用一个工具子进程和一个 MCP 会话。"""

    def __init__(
        self,
        *,
        db_path: str | Path,
        seed_path: str | Path,
        language: str,
        expected_specs: Dict[str, AgentToolSpec],
        project_root: str | Path,
        python_executable: Optional[str] = None,
        startup_timeout: float = 30.0,
        call_timeout: float = 30.0,
        stderr_log_path: Optional[str | Path] = None,
    ):
        self.db_path = Path(db_path).expanduser().resolve()
        self.seed_path = Path(seed_path).expanduser().resolve()
        self.language = str(language or "zh").lower()
        self.expected_specs = dict(expected_specs)
        self.project_root = Path(project_root).expanduser().resolve()
        self.python_executable = python_executable or sys.executable
        self.startup_timeout = max(1.0, float(startup_timeout))
        self.call_timeout = max(1.0, float(call_timeout))
        self.stderr_log_path = (
            Path(stderr_log_path).expanduser().resolve()
            if stderr_log_path is not None
            else None
        )
        self.call_count = 0
        self.protocol_version = ""
        self.server_name = ""
        self.server_version = ""

        self._session: Optional[ClientSession] = None
        self._session_cm: Any = None
        self._stdio_cm: Any = None
        self._stderr_handle: Optional[TextIO] = None
        self._tools: Dict[str, AgentToolSpec] = {}
        self._started = False
        self._closed = False

    def _server_env(self) -> Dict[str, str]:
        """只传运行所需配置，不继承模型密钥或其他服务地址。"""
        return {
            "RETAIL_DB_PATH": str(self.db_path),
            "RETAIL_SEED_PATH": str(self.seed_path),
            "RETAIL_LANGUAGE": self.language,
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }

    async def start(self) -> None:
        """启动子进程并完成初始化与工具发现；失败时清理已开资源。"""
        if self._started:
            return
        if self._closed:
            raise RetailMCPTransportError("MCP 客户端已关闭，不能重新启动")

        params = StdioServerParameters(
            command=self.python_executable,
            args=["-m", "mcp.retail_server"],
            env=self._server_env(),
            cwd=str(self.project_root),
            encoding="utf-8",
        )
        # 会话上下文内部持有 anyio 任务组，不能用外层取消作用域包裹其
        # 进入过程；初始化与发现的超时由 ClientSession 的读取超时负责。
        try:
            if self.stderr_log_path is not None:
                self.stderr_log_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                self._stderr_handle = self.stderr_log_path.open(
                    "a", encoding="utf-8"
                )
            self._stdio_cm = stdio_client(
                params, errlog=self._stderr_handle or sys.stderr
            )
            read_stream, write_stream = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.startup_timeout),
            )
            self._session = await self._session_cm.__aenter__()
            initialized = await self._session.initialize()
            listed = await self._session.list_tools()
            self._tools = self._verify_tools(listed)
        except TimeoutError as ex:
            await self._close_transport()
            raise RetailMCPTransportError("MCP 工具服务启动超时") from ex
        except McpError as ex:
            await self._close_transport()
            code = getattr(ex.error, "code", None)
            if code == PROTOCOL_ERROR_TIMEOUT:
                raise RetailMCPTransportError(
                    "MCP 工具服务启动超时"
                ) from ex
            raise RetailMCPProtocolError(
                f"MCP 工具服务初始化失败: {ex}", code=code
            ) from ex
        except RetailMCPError:
            await self._close_transport()
            raise
        except Exception as ex:
            await self._close_transport()
            raise RetailMCPTransportError(
                f"MCP 工具服务启动失败: {type(ex).__name__}: {ex}"
            ) from ex

        self.protocol_version = str(initialized.protocolVersion)
        self.server_name = str(initialized.serverInfo.name)
        self.server_version = str(initialized.serverInfo.version or "")
        self._started = True

    def _verify_tools(
        self, listed: types.ListToolsResult
    ) -> Dict[str, AgentToolSpec]:
        """核对固定白名单、描述与 Schema 后映射回 AgentToolSpec。"""
        found = {tool.name: tool for tool in listed.tools}
        expected_names = set(self.expected_specs)
        if set(found) != expected_names:
            raise RetailMCPProtocolError(
                "MCP 工具白名单不一致: "
                f"{sorted(found)} != {sorted(expected_names)}"
            )

        specs: Dict[str, AgentToolSpec] = {}
        for name, expected in self.expected_specs.items():
            tool = found[name]
            if (tool.description or "") != expected.description:
                raise RetailMCPProtocolError(
                    f"MCP 工具 {name} 的描述与本地定义不一致"
                )
            if tool.inputSchema != expected.input_schema:
                raise RetailMCPProtocolError(
                    f"MCP 工具 {name} 的 Schema 与本地定义不一致"
                )

            async def handler(
                req: Any,
                args: Dict[str, Any],
                _name: str = name,
            ) -> Dict[str, Any]:
                return await self.call_tool(_name, req, args)

            specs[name] = AgentToolSpec(
                name=name,
                description=expected.description,
                input_schema=expected.input_schema,
                handler=handler,
            )
        return specs

    def tools(self) -> Dict[str, AgentToolSpec]:
        """返回经协议发现并核对的工具定义。"""
        return dict(self._tools)

    async def call_tool(
        self,
        name: str,
        req: Any,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """经真实 tools/call 执行一次业务工具；不做自动重试。"""
        if not self._started or self._session is None:
            raise RetailMCPTransportError("MCP 工具服务未启动")
        if name not in self._tools:
            raise RetailMCPProtocolError(f"工具不在 MCP 白名单: {name}")

        meta = build_context_meta(
            client_user_id=req.user_id,
            conv_id=req.conv_id,
            request_id=req.request_id,
            turn=req.turn,
            message=req.message,
            language=self.language,
        )
        request = types.ClientRequest(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name=name,
                    arguments=args,
                    _meta=meta,
                ),
            )
        )
        self.call_count += 1
        try:
            result = await self._session.send_request(
                request,
                types.CallToolResult,
                request_read_timeout_seconds=timedelta(
                    seconds=self.call_timeout
                ),
            )
        except McpError as ex:
            code = getattr(ex.error, "code", None)
            if code == PROTOCOL_ERROR_TIMEOUT:
                raise RetailMCPTransportError(
                    f"工具调用超时: {name}"
                ) from ex
            raise RetailMCPProtocolError(
                f"工具协议错误: {ex}", code=code
            ) from ex
        except (
            anyio.ClosedResourceError,
            anyio.BrokenResourceError,
            anyio.EndOfStream,
        ) as ex:
            raise RetailMCPTransportError(
                f"工具连接已失效: {type(ex).__name__}"
            ) from ex
        except RetailMCPError:
            raise
        except Exception as ex:
            raise RetailMCPTransportError(
                f"工具调用失败: {type(ex).__name__}: {ex}"
            ) from ex

        return self._validate_result(name, result)

    @staticmethod
    def _validate_result(
        name: str, result: types.CallToolResult
    ) -> Dict[str, Any]:
        """结构化结果、JSON 文本与 isError 必须互相一致。"""
        payload = result.structuredContent
        if not isinstance(payload, dict):
            raise RetailMCPProtocolError(
                f"工具 {name} 缺少结构化结果"
            )
        texts = [
            block.text
            for block in result.content
            if isinstance(block, types.TextContent)
        ]
        if len(texts) != 1:
            raise RetailMCPProtocolError(
                f"工具 {name} 的 JSON 文本数量不符"
            )
        try:
            parsed = json.loads(texts[0])
        except json.JSONDecodeError as ex:
            raise RetailMCPProtocolError(
                f"工具 {name} 的 JSON 文本无法解析"
            ) from ex
        if parsed != payload:
            raise RetailMCPProtocolError(
                f"工具 {name} 的结构化结果与 JSON 文本不一致"
            )
        business_failure = payload.get("success") is False
        if bool(result.isError) != business_failure:
            raise RetailMCPProtocolError(
                f"工具 {name} 的 isError 与业务结果不一致"
            )
        return payload

    async def _close_transport(self) -> list[str]:
        errors: list[str] = []
        if self._session_cm is not None:
            session_cm = self._session_cm
            self._session_cm = None
            self._session = None
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception as ex:
                errors.append(f"关闭 MCP 会话失败: {type(ex).__name__}: {ex}")
        if self._stdio_cm is not None:
            stdio_cm = self._stdio_cm
            self._stdio_cm = None
            try:
                await stdio_cm.__aexit__(None, None, None)
            except Exception as ex:
                errors.append(f"关闭 MCP 子进程失败: {type(ex).__name__}: {ex}")
        if self._stderr_handle is not None:
            handle = self._stderr_handle
            self._stderr_handle = None
            try:
                handle.close()
            except Exception as ex:
                errors.append(f"关闭 MCP 日志失败: {type(ex).__name__}: {ex}")
        return errors

    async def close(self) -> None:
        """关闭会话与子进程；重复调用无副作用。"""
        if self._closed:
            return
        self._closed = True
        self._started = False
        errors = await self._close_transport()
        self._tools = {}
        if errors:
            logger.warning("MCP 客户端关闭出现问题: %s", "；".join(errors))
