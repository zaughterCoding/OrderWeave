"""单轮执行的实时事件流，供 Web 控制台展示链路。

只做旁路观测：未注册接收方时所有调用都是空操作；接收方出错只记日志，
绝不影响业务结果。父子关系由 ContextVar 维护，asyncio.gather 等并发子任务
各自继承当前父节点，因此并行步骤会挂在同一父节点下。
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, Optional

logger = logging.getLogger(__name__)

Sink = Callable[[Dict[str, Any]], None]

_sink: ContextVar[Optional[Sink]] = ContextVar("live_event_sink", default=None)
_parent: ContextVar[Optional[str]] = ContextVar("live_event_parent", default=None)
_ids = itertools.count(1)

# 单个字段的文本上限；完整 prompt 由前端折叠，过长时截断以免拖垮事件流。
MAX_TEXT = 60000


def active() -> bool:
    return _sink.get() is not None


@contextmanager
def capture(sink: Sink) -> Iterator[None]:
    """在当前上下文内把事件交给 sink；退出后恢复。"""
    token = _sink.set(sink)
    parent = _parent.set(None)
    try:
        yield
    finally:
        _parent.reset(parent)
        _sink.reset(token)


def clip(value: Any, limit: int = MAX_TEXT) -> Any:
    """把任意值转为可 JSON 序列化的结构，并限制文本长度。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = json.dumps(str(value), ensure_ascii=False)
    if len(text) <= limit:
        return json.loads(text)
    return {"truncated": True, "chars": len(text), "head": text[:limit]}


def _send(event: Dict[str, Any]) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        logger.exception("实时事件接收方出错，已忽略")


def emit(kind: str, name: str, **data: Any) -> None:
    """发出一个挂在当前父节点下的瞬时事件。"""
    if _sink.get() is None:
        return
    _send({"type": "point", "id": f"e{next(_ids)}", "parent": _parent.get(), "kind": kind,
           "name": name, "ts": time.time(), "data": clip(data)})


class SpanHandle:
    """span 结束时附带的数据；调用方可在 with 块内补充。"""

    def __init__(self, span_id: Optional[str]):
        self.id = span_id
        self.data: Dict[str, Any] = {}
        self.status = "ok"

    def set(self, **data: Any) -> None:
        self.data.update(data)

    def fail(self, reason: str) -> None:
        self.status = "error"
        self.data["error"] = reason


@contextmanager
def span(kind: str, name: str, **data: Any) -> Iterator[SpanHandle]:
    """有起止的步骤；块内发出的事件都以它为父节点。"""
    if _sink.get() is None:
        yield SpanHandle(None)
        return
    span_id = f"s{next(_ids)}"
    handle = SpanHandle(span_id)
    started = time.time()
    _send({"type": "start", "id": span_id, "parent": _parent.get(), "kind": kind, "name": name,
           "ts": started, "data": clip(data)})
    token = _parent.set(span_id)
    try:
        yield handle
    except BaseException as ex:
        # 调用方可预先标记为等待等非错误状态（例如工作图中断等待用户）。
        if handle.status == "ok":
            handle.status = "error"
            handle.data.setdefault("error", f"{type(ex).__name__}: {ex}"[:500])
        raise
    finally:
        _parent.reset(token)
        _send({"type": "end", "id": span_id, "kind": kind, "name": name, "ts": time.time(),
               "status": handle.status, "ms": round((time.time() - started) * 1000, 1),
               "data": clip(handle.data)})
