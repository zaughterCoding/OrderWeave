"""本地零售 MCP 工具服务。

仅由本项目受信任主程序通过 stdio 启动。服务只注册九个非 RAG 零售工具，
业务状态全部来自与主程序共用绝对路径的 SQLite；stdout 只写协议消息，
诊断信息写入 stderr。
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anyio
import mcp.types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

from agents.agent_orchestrator import Request
from agents.retail_tools import retail_tool_scope, retail_tools
from agents.tools import AgentToolSpec, validate_tool_input
from mcp.retail_protocol import RetailContextError, parse_context_meta
from mcp.retail_store import RetailStore, RetailStoreError

logger = logging.getLogger("retail_mcp")

SERVER_NAME = "agentecommerce-retail"
SERVER_VERSION = "0.1.0"
EXPECTED_TOOL_NAMES = tuple(retail_tool_scope(has_knowledge=False))


def _configure_logging() -> None:
    """协议独占 stdout，所有诊断只走 stderr。"""
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="retail-mcp %(levelname)s %(message)s",
    )
    logger.setLevel(logging.INFO)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"未配置 {name}")
    return value


def _message(language: str, zh: str, en: str) -> str:
    return en if language == "en" else zh


def _error(code: int, message: str) -> McpError:
    return McpError(types.ErrorData(code=code, message=message))


def _build_server() -> Tuple[
    Server, Dict[str, AgentToolSpec], RetailStore, str
]:
    """打开共用业务库并注册固定工具白名单。"""
    db_path = Path(_required_env("RETAIL_DB_PATH")).expanduser().resolve()
    seed_path = Path(_required_env("RETAIL_SEED_PATH")).expanduser().resolve()
    language = os.getenv("RETAIL_LANGUAGE", "zh").strip().lower() or "zh"
    if language not in ("zh", "en"):
        raise RuntimeError(f"不支持的 RETAIL_LANGUAGE: {language!r}")

    store = RetailStore(db_path, seed_path, language=language)
    specs = retail_tools(store, None, language=language)
    if set(specs) != set(EXPECTED_TOOL_NAMES):
        raise RuntimeError(
            "零售 MCP 工具白名单不一致: "
            f"{sorted(specs)} != {sorted(EXPECTED_TOOL_NAMES)}"
        )

    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=spec.name,
                description=spec.description,
                inputSchema=spec.input_schema,
            )
            for spec in specs.values()
        ]

    async def call_tool(req: types.CallToolRequest) -> types.ServerResult:
        return await _call_tool(
            req, specs, store, language, server.request_context.meta
        )

    server.request_handlers[types.CallToolRequest] = call_tool
    return server, specs, store, language


async def _call_tool(
    req: types.CallToolRequest,
    specs: Dict[str, AgentToolSpec],
    store: RetailStore,
    language: str,
    request_meta: Any,
) -> types.ServerResult:
    name = str(req.params.name)
    spec = specs.get(name)
    if spec is None:
        logger.info("rejected tool=%s reason=unknown pid=%s", name, os.getpid())
        raise _error(types.METHOD_NOT_FOUND, f"未知工具: {name}")

    try:
        context = parse_context_meta(
            request_meta, expected_language=language
        )
    except RetailContextError as ex:
        logger.info("rejected tool=%s reason=context pid=%s", name, os.getpid())
        raise _error(types.INVALID_PARAMS, str(ex)) from ex

    current_turn = store.current_turn(context.client_user_id, context.conv_id)
    if current_turn != context.turn:
        logger.info(
            "rejected tool=%s reason=stale_turn request_id=%s "
            "turn=%s current=%s pid=%s",
            name,
            context.request_id,
            context.turn,
            current_turn,
            os.getpid(),
        )
        raise _error(
            types.INVALID_PARAMS,
            _message(
                language,
                "调用上下文与当前会话轮次不一致，请拒绝执行",
                "The call context does not match the current session turn",
            ),
        )

    args = req.params.arguments or {}
    try:
        validate_tool_input(spec, args, language=language)
    except ValueError as ex:
        logger.info(
            "rejected tool=%s reason=arguments request_id=%s pid=%s",
            name,
            context.request_id,
            os.getpid(),
        )
        raise _error(types.INVALID_PARAMS, str(ex)) from ex

    request = Request(
        message=context.message,
        user_id=context.client_user_id,
        conv_id=context.conv_id,
        turn=context.turn,
        retail_user_id=store.bound_user_id(
            context.client_user_id, context.conv_id
        ),
        retail_pending=store.active_pending(
            context.client_user_id, context.conv_id
        ),
    )

    try:
        result = spec.handler(request, args)
        if inspect.isawaitable(result):
            result = await result
    except RetailStoreError as ex:
        result = {"success": False, "error": str(ex)}
    except Exception as ex:
        logger.exception(
            "failed tool=%s request_id=%s pid=%s",
            name,
            context.request_id,
            os.getpid(),
        )
        raise _error(
            types.INTERNAL_ERROR,
            _message(language, "工具执行失败", f"Tool execution failed: {name}"),
        ) from ex

    if not isinstance(result, dict):
        logger.error(
            "invalid_result tool=%s request_id=%s pid=%s",
            name,
            context.request_id,
            os.getpid(),
        )
        raise _error(
            types.INTERNAL_ERROR,
            _message(language, "工具返回格式无效", "Invalid tool result"),
        )

    is_error = result.get("success") is False
    logger.info(
        "call tool=%s request_id=%s turn=%s success=%s pid=%s",
        name,
        context.request_id,
        context.turn,
        not is_error,
        os.getpid(),
    )
    return types.ServerResult(
        types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, sort_keys=True),
                )
            ],
            structuredContent=result,
            isError=is_error,
        )
    )


async def _run() -> None:
    server, specs, store, language = _build_server()
    logger.info(
        "started pid=%s language=%s tools=%s",
        os.getpid(),
        language,
        len(specs),
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    _configure_logging()
    try:
        anyio.run(_run)
    except Exception:
        logger.exception("stopped with error pid=%s", os.getpid())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
