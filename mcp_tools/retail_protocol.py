"""零售工具 MCP 传输的共享上下文合同。

主程序把本次真实用户轮次写入 tools/call 的私有 _meta；工具端只从这里
恢复会话与消息，不接受模型通过工具参数提交的同类字段。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

META_CLIENT_USER_ID = "client_user_id"
META_CONV_ID = "conv_id"
META_REQUEST_ID = "request_id"
META_TURN = "turn"
META_MESSAGE = "message"
META_LANGUAGE = "language"

SUPPORTED_LANGUAGES = ("zh", "en")


class RetailContextError(ValueError):
    """tools/call 上下文缺失、类型错误或与数据库轮次不符。"""


@dataclass(frozen=True)
class RetailContext:
    """从 _meta 还原的单次调用上下文。"""

    client_user_id: str
    conv_id: str
    request_id: str
    turn: int
    message: str
    language: str


def build_context_meta(
    client_user_id: str,
    conv_id: str,
    request_id: str,
    turn: int,
    message: str,
    language: str,
) -> dict[str, Any]:
    """由主程序构造可信 _meta；字段不进入模型可填写的工具 Schema。"""
    return {
        META_CLIENT_USER_ID: client_user_id,
        META_CONV_ID: conv_id,
        META_REQUEST_ID: request_id,
        META_TURN: int(turn),
        META_MESSAGE: message,
        META_LANGUAGE: language,
    }


def _meta_mapping(meta: Any) -> Mapping[str, Any]:
    if meta is None:
        return {}
    if isinstance(meta, Mapping):
        return meta
    dump = getattr(meta, "model_dump", None)
    if callable(dump):
        value = dump()
        if isinstance(value, Mapping):
            return value
    raise RetailContextError("_meta 格式无效")


def parse_context_meta(
    meta: Any,
    expected_language: str,
) -> RetailContext:
    """严格解析 _meta；缺字段、类型错误或语言不符时拒绝调用。"""
    english = str(expected_language).lower() == "en"

    def fail(zh: str, en: str) -> None:
        raise RetailContextError(en if english else zh)

    data = _meta_mapping(meta)
    if not data:
        fail(
            "缺少受信任的调用上下文（_meta）",
            "The trusted call context (_meta) is missing",
        )

    values: dict[str, str] = {}
    for key, zh, en in (
        (META_CLIENT_USER_ID, "client_user_id", "client_user_id"),
        (META_CONV_ID, "conv_id", "conv_id"),
        (META_REQUEST_ID, "request_id", "request_id"),
        (META_MESSAGE, "message", "message"),
    ):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            fail(
                f"_meta 字段 {zh} 缺失或不是非空字符串",
                f"_meta field {en} is missing or not a non-empty string",
            )
        values[key] = value

    turn = data.get(META_TURN)
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        fail(
            "_meta 字段 turn 必须是正整数",
            "_meta field turn must be a positive integer",
        )

    language = data.get(META_LANGUAGE)
    if not isinstance(language, str) or language.lower() not in SUPPORTED_LANGUAGES:
        fail(
            "_meta 字段 language 必须是 zh 或 en",
            "_meta field language must be zh or en",
        )
    language = language.lower()
    if language != str(expected_language).lower():
        fail(
            "_meta 语言与工具服务配置不一致",
            "_meta language does not match the tool server configuration",
        )

    return RetailContext(
        client_user_id=values[META_CLIENT_USER_ID],
        conv_id=values[META_CONV_ID],
        request_id=values[META_REQUEST_ID],
        turn=turn,
        message=values[META_MESSAGE],
        language=language,
    )
