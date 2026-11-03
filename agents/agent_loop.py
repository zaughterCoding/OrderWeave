"""所有角色共用的 Agent 工具循环与钩子。

循环只负责“调模型 → 执行工具 → 回填结果”，安全、权限、停止条件等策略
以钩子插件挂入，单 Agent 与多 Agent 角色走同一条路径。每次模型调用和工具
调用都记录为追踪片段（span），并把已完成的工具往返写入请求级 transcript，
供服务层持久化为完整消息日志。
"""
from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from agents.tools import validate_tool_input

logger = logging.getLogger(__name__)


class LoopHook:
    """钩子基类；默认全部为空操作，子类只覆盖需要的阶段。"""

    name = "hook"

    def pre_model(self, ctx: "LoopContext", request: Dict[str, Any]) -> None:
        """模型调用前，可调整本次请求参数（例如追加临时上下文）。"""

    def pre_tool(self, ctx: "LoopContext", name: str, args: Dict[str, Any]) -> Optional[str]:
        """工具执行前；返回非空字符串即拒绝执行，并作为工具错误回填给模型。"""
        return None

    def post_tool(self, ctx: "LoopContext", name: str, args: Dict[str, Any], result: Any) -> None:
        """工具实际执行后。"""

    def on_tool_error(self, ctx: "LoopContext", name: str, args: Dict[str, Any], error: BaseException) -> None:
        """工具抛错、结果不明时。"""

    def stop_response(self, ctx: "LoopContext") -> Optional[str]:
        """一批工具结果回填后；返回非空字符串即结束本角色调用并以此作为回复。"""
        return None


@dataclass
class LoopContext:
    """一次角色调用的运行上下文；state 在同一真实请求的多个角色间共享。"""

    req: Any
    agent: Any
    state: Dict[str, Any]
    spans: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LoopResult:
    content: str
    tools_used: List[str]
    tool_traces: List[Dict[str, Any]]
    knowledge_hits: int
    spans: List[Dict[str, Any]]


class LoopError(RuntimeError):
    """循环失败时携带已发生的工具轨迹，避免上层误判为“什么都没做”。"""

    def __init__(self, message: str, result: LoopResult):
        super().__init__(message)
        self.result = result


def block_type(block: Any) -> Optional[str]:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def block_value(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


def plain_blocks(content: Any) -> List[Dict[str, Any]]:
    """把 SDK 内容块转为可持久化、可原样回放的字典（只保留文本与工具调用）。"""
    blocks: List[Dict[str, Any]] = []
    for block in content or []:
        kind = block_type(block)
        if kind == "text":
            text = str(block_value(block, "text") or "")
            if text:
                blocks.append({"type": "text", "text": text})
        elif kind == "tool_use":
            blocks.append({"type": "tool_use", "id": block_value(block, "id"),
                           "name": block_value(block, "name"), "input": block_value(block, "input") or {}})
    return blocks


def _dump_request(request: Dict[str, Any], agent_name: str) -> None:
    """调试缓存命中：设置 ORDERWEAVE_DUMP_REQUESTS 目录时逐次落盘实际请求。"""
    import os

    folder = os.getenv("ORDERWEAVE_DUMP_REQUESTS", "").strip()
    if not folder:
        return
    from pathlib import Path

    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    index = len(list(path.glob("*.json"))) + 1
    (path / f"{index:04d}-{agent_name}.json").write_text(
        json.dumps(request, ensure_ascii=False, default=str, indent=1), encoding="utf-8")


def _usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    read = lambda key: int(getattr(usage, key, 0) or 0) if usage is not None else 0  # noqa: E731
    return {"input": read("input_tokens"), "cache_read": read("cache_read_input_tokens"),
            "cache_write": read("cache_creation_input_tokens"), "output": read("output_tokens")}


async def run_tool_loop(
    agent: Any,
    req: Any,
    messages: List[Dict[str, Any]],
    tools: Dict[str, Any],
    hooks: Sequence[LoopHook],
    *,
    system: str,
    stop_check: Any,
) -> LoopResult:
    """执行工具循环；stop_check(req, ex) 为真时异常直接上抛（预算等全局停止）。"""
    english = getattr(agent, "language", "zh") == "en"
    state = req.retail_guard if getattr(req, "retail_guard", None) is not None else {}
    shared_spans = getattr(req, "spans", None)
    ctx = LoopContext(req=req, agent=agent, state=state,
                      spans=shared_spans if shared_spans is not None else [])
    transcript = getattr(req, "transcript", None)
    agent_name = agent.agent_type.value
    tools_used: List[str] = []
    tool_traces: List[Dict[str, Any]] = []
    knowledge_hits = 0
    max_rounds = max(1, int(getattr(agent, "max_tool_rounds", 3) or 3))
    per_response = getattr(agent, "max_tool_uses_per_response", None)
    serial_read = set(getattr(agent, "serial_read_tools", ()) or ())

    def result(content: str) -> LoopResult:
        return LoopResult(content, list(tools_used), list(tool_traces), knowledge_hits, ctx.spans)

    try:
        for _ in range(max_rounds):
            request: Dict[str, Any] = {
                "model": agent._model,
                "max_tokens": agent.profile.max_tokens,
                "temperature": agent.profile.temperature,
                "system": system,
                "messages": list(messages),
            }
            if tools:
                request["tools"] = [{"name": s.name, "description": s.description,
                                     "input_schema": s.input_schema} for s in tools.values()]
            for hook in hooks:
                hook.pre_model(ctx, request)
            _dump_request(request, agent_name)
            t0 = time.monotonic()
            resp = await agent._client.messages.create(**request)
            ctx.spans.append({"kind": "model", "agent": agent_name, "usage": _usage(resp),
                              "request_chars": len(json.dumps([request["system"], request["messages"],
                                                               request.get("tools", [])],
                                                              ensure_ascii=False, default=str)),
                              "stop_reason": getattr(resp, "stop_reason", None),
                              "latency_ms": round((time.monotonic() - t0) * 1000, 1)})
            tool_uses = [b for b in (resp.content or []) if block_type(b) == "tool_use"]
            if not tool_uses:
                from core.llm_utils import extract_text_content
                return result(extract_text_content(resp.content))

            assistant_blocks = plain_blocks(resp.content)
            messages.append({"role": "assistant", "content": assistant_blocks})
            tool_results = []
            # 供应商可能忽略单工具参数；仅纯只读批次可串行消费，写入仍逐次限制。
            read_batch = all(block_value(b, "name") in serial_read for b in tool_uses)
            for index, block in enumerate(tool_uses):
                name = block_value(block, "name")
                tool_use_id = block_value(block, "id")
                args = block_value(block, "input") or {}
                spec = tools.get(name)
                tool_t0 = time.monotonic()
                call_success, result_success, error_text = True, None, ""
                skipped, blocked_by = False, ""
                if per_response is not None and index >= per_response and not read_batch:
                    call_success, result_success, skipped = False, False, True
                    error_text = ("At most one tool call may be executed per model response; this call was not executed."
                                  if english else "每次模型响应最多执行一个工具调用；本次调用未执行。")
                    value: Any = {"success": False, "error": error_text}
                elif spec is None:
                    call_success = False
                    error_text = (f"Tool is not in the {agent_name} agent allowlist"
                                  if english else f"工具不在 {agent_name} Agent 白名单中")
                    value = {"success": False, "error": error_text}
                else:
                    reason = ""
                    for hook in hooks:
                        reason = hook.pre_tool(ctx, name, args) or ""
                        if reason:
                            blocked_by = hook.name
                            break
                    if reason:
                        call_success, result_success, error_text = False, False, reason
                        value = {"success": False, "error": reason}
                    else:
                        try:
                            validate_tool_input(spec, args, language=agent.language)
                            value = spec.handler(req, args)
                            if inspect.isawaitable(value):
                                value = await value
                            tools_used.append(name)
                            if isinstance(value, dict) and "success" in value:
                                result_success = bool(value.get("success"))
                            for hook in hooks:
                                hook.post_tool(ctx, name, args, value)
                        except Exception as ex:
                            if stop_check(req, ex):
                                raise
                            call_success = False
                            logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                            error_text = str(ex)
                            value = {"success": False, "error": error_text}
                            # 结果不明的工具调用交由钩子登记，避免重放。
                            for hook in hooks:
                                hook.on_tool_error(ctx, name, args, ex)
                latency = round((time.monotonic() - tool_t0) * 1000, 1)
                if not error_text and isinstance(value, dict):
                    error_text = str(value.get("error", "") or "")
                knowledge_hits += agent._knowledge_hit_count(name, value)
                trace: Dict[str, Any] = {
                    "agent_type": agent_name, "tool_name": name, "tool_use_id": tool_use_id,
                    "input": dict(args), "success": call_success, "result_success": result_success,
                    "latency_ms": latency,
                    "cached": bool(value.get("cached")) if isinstance(value, dict) else False,
                    "reranked": bool(value.get("reranked")) if isinstance(value, dict) else False,
                    "error": error_text,
                }
                if skipped:
                    trace["skipped"] = True
                if blocked_by:
                    trace["blocked_by_guard"] = True
                    trace["blocked_by"] = blocked_by
                if getattr(agent, "capture_tool_results", False):
                    trace["result"] = value
                tool_traces.append(trace)
                ctx.spans.append({"kind": "tool", "agent": agent_name, "tool": name, "success": call_success,
                                  "result_success": result_success, "blocked_by": blocked_by,
                                  "latency_ms": latency})
                tool_results.append({"type": "tool_result", "tool_use_id": tool_use_id,
                                     "content": json.dumps(value, ensure_ascii=False, default=str)})
            messages.append({"role": "user", "content": tool_results})
            if transcript is not None:
                # 只记录完整的“调用＋结果”往返，保证回放时消息成对出现。
                transcript.append({"agent": agent_name, "assistant": assistant_blocks, "results": tool_results})
            for hook in hooks:
                stop = hook.stop_response(ctx)
                if stop:
                    return result(stop)
        raise LoopError(f"{agent_name} 工具调用超过最大轮数", result(""))
    except LoopError:
        raise
    except Exception as ex:
        if stop_check(req, ex):
            raise
        raise LoopError(str(ex), result("")) from ex


class MethodHook(LoopHook):
    """把旧式子类方法（_guard_tool_call 等）适配为钩子，旧客服角色行为不变。"""

    name = "agent_methods"

    def pre_tool(self, ctx, name, args):
        return ctx.agent._guard_tool_call(ctx.req, name, args, ctx.state)

    def post_tool(self, ctx, name, args, result):
        ctx.agent._on_tool_result(ctx.req, name, args, result, ctx.state)

    def on_tool_error(self, ctx, name, args, error):
        ctx.agent._on_tool_error(ctx.req, name, args, error, ctx.state)

    def stop_response(self, ctx):
        return ctx.state.get("stop_response")
