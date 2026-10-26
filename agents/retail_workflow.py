"""零售退货的持久化工作图；业务授权始终由 RetailStore 核验。"""
from __future__ import annotations

import hashlib
import asyncio
import json
import re
import time
import uuid
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langsmith import tracing_context

from agents.agent_orchestrator import AgentResponse, AgentType, OrchestratorResult
from agents.retail_handoff import RoleHandoff, RoleOutcome, build_role_handoff, role_outcome
from agents.retail_support import analyze_message
from mcp.retail_store import evaluate_confirmation


class RetailGraphState(TypedDict, total=False):
    current_task: str
    phase: str
    waiting_reason: str
    current_input: str
    turn: int
    pending_refs: list[str]
    route: str
    result: dict[str, Any]
    node_path: list[str]
    active_role: str
    next_role: str
    role_queue: list[str]
    role_result: RoleOutcome
    handoff: RoleHandoff


class RetailWorkflow:
    """只持久化有界状态；Request、模型上下文与工具结果留在当前调用。"""

    def __init__(self, service, checkpoint_path=None):
        self.service = service
        self.path = Path(checkpoint_path or service.store.db_path.with_suffix(".checkpoints.sqlite")).resolve()
        if self.path == service.store.db_path:
            raise ValueError("checkpoint 文件不能与业务数据库相同")
        self._call: ContextVar[dict] = ContextVar("retail_graph_call")
        self._connection = None
        self._graph = None
        self._start_lock = asyncio.Lock()

    def config(self, client_user_id, conv_id):
        # 包含业务库边界，不能由外部传入任意 checkpoint id。
        identity = json.dumps([str(self.service.store.db_path), client_user_id, conv_id], ensure_ascii=False)
        return {"configurable": {"thread_id": hashlib.sha256(identity.encode()).hexdigest()}}

    async def start(self):
        async with self._start_lock:
            await self._start()

    async def _start(self):
        if self._graph is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(str(self.path))
        saver = AsyncSqliteSaver(self._connection)
        builder = StateGraph(RetailGraphState)
        for name in ("prepare_state", "route_agent", "run_agent", "leave_return", "cancel_return", "update_result", "wait_user", "resume_input", "recover_result"):
            builder.add_node(name, getattr(self, "_" + name))
        for name in ("select_role", "handoff_role", "compose_roles"):
            builder.add_node(name, getattr(self, "_" + name))
        for role in ("general", "technical", "billing", "escalation"):
            builder.add_node(role, partial(self._run_role, role=role))
            builder.add_edge(role, "handoff_role")
        builder.add_conditional_edges(START, lambda s: "recover_result" if s.get("phase") == "recovering" else "prepare_state")
        builder.add_edge("prepare_state", "route_agent")
        builder.add_conditional_edges("route_agent", self._dispatch)
        builder.add_conditional_edges("leave_return", lambda s: "select_role" if self._uses_roles() else "run_agent")
        builder.add_conditional_edges("select_role", lambda s: s["next_role"])
        builder.add_conditional_edges("handoff_role", lambda s: s["next_role"])
        builder.add_edge("compose_roles", "update_result")
        builder.add_edge("cancel_return", "update_result")
        builder.add_edge("run_agent", "update_result")
        builder.add_conditional_edges("update_result", lambda s: "wait_user" if s["waiting_reason"] else END)
        builder.add_edge("wait_user", "resume_input")
        builder.add_edge("resume_input", "prepare_state")
        builder.add_edge("recover_result", END)
        self._graph = builder.compile(checkpointer=saver)

    @staticmethod
    def _step(state, name, **values):
        return {"node_path": [*state.get("node_path", []), name], **values}

    def _uses_roles(self):
        agent = self.service.agent
        return hasattr(agent, "begin_turn") and hasattr(agent, "orchestrator")

    def _dispatch(self, state):
        if state["route"] == "cancel":
            return "cancel_return"
        if state["route"] == "human":
            return "leave_return"
        if self._call.get()["context"]["support_blocked"]:
            return "run_agent"
        return "select_role" if self._uses_roles() else "run_agent"

    @staticmethod
    def _reset_roles(call):
        return {"active_role": "", "next_role": "", "role_queue": [], "handoff": {},
                "role_result": call.get("previous_role_result", {})}

    async def _select_role(self, state):
        call = self._call.get()
        agent = self.service.agent
        req = call["context"]["request"]
        started = time.perf_counter()
        call["agent_started"] = agent.begin_turn(req)
        try:
            prepared = await agent.orchestrator.prepare_route(req)
            if isinstance(prepared, OrchestratorResult):
                call["response"] = agent.finish_turn(req, prepared, call["agent_started"])
                return self._step(state, "select_role", next_role="update_result")
            supporting = [r for r in prepared.supporting_agents if r != prepared.primary_agent][:2]
            call.update(decision=prepared, supporting=supporting, responses=[],
                        previous_role_result=call.get("previous_role_result", {}))
            queue = [role.value for role in supporting] + [prepared.primary_agent.value]
            return self._step(state, "select_role", next_role=queue[0], role_queue=queue, phase="role_routed")
        except Exception as ex:
            call["role_error"] = True
            call["response"] = agent.fail_turn(ex, call["agent_started"])
            return self._step(state, "select_role", next_role="update_result", phase="role_failed")
        finally:
            call["business_ms"] += (time.perf_counter() - started) * 1000

    async def _run_role(self, state, *, role):
        call = self._call.get()
        req = call["context"]["request"]
        handoff = build_role_handoff(self.service.store, req, call.get("previous_role_result"), role)
        req.retail_handoff = handoff
        call.setdefault("handoffs", []).append({key: handoff[key] for key in
            ("from_role", "to_role", "source_turn", "target_turn")})
        started = time.perf_counter()
        try:
            response = await self.service.agent.orchestrator.execute_role(req, AgentType(role), allow_missing_fallback=False)
            call["role_response"] = response
            outcome = role_outcome(req, response)
            call["previous_role_result"] = outcome
            return self._step(state, role, active_role=role, handoff=handoff, role_result=outcome, phase="role_executed")
        except Exception as ex:
            call["role_error"] = True
            call["response"] = self.service.agent.fail_turn(ex, call["agent_started"])
            return self._step(state, role, active_role=role, handoff=handoff, phase="role_failed")
        finally:
            call["business_ms"] += (time.perf_counter() - started) * 1000

    async def _handoff_role(self, state):
        call = self._call.get()
        if call.get("role_error"):
            return self._step(state, "handoff_role", next_role="update_result", role_queue=[])
        req = call["context"]["request"]
        orchestrator = self.service.agent.orchestrator
        response = call["role_response"]
        if call.get("failed_response") is not None:
            response = orchestrator.merge_fallback(call.pop("failed_response"), response)
        elif orchestrator.should_fallback(req, AgentType(state["active_role"]), response) and orchestrator._best_agent(AgentType.GENERAL):
            call["failed_response"] = response
            return self._step(state, "handoff_role", next_role="general", phase="role_fallback")
        if len(call["responses"]) < len(call["supporting"]) and getattr(orchestrator, "retail_support_enabled", False):
            from agents.retail_support import merge_role_handoff
            req.retail_support = merge_role_handoff(req.retail_support, state["active_role"],
                call["decision"].primary_agent.value, req.turn, response)
        call["responses"].append(response)
        queue = state["role_queue"][1:]
        return self._step(state, "handoff_role", role_queue=queue,
                          next_role=queue[0] if queue else "compose_roles", phase="role_handoff")

    async def _compose_roles(self, state):
        call = self._call.get()
        agent = self.service.agent
        req = call["context"]["request"]
        decision = call["decision"]
        responses = call["responses"]
        started = time.perf_counter()
        try:
            if decision.multi_agent:
                result = await agent.orchestrator.finish_parallel(req, decision, responses,
                    [responses[-1]] + responses[:-1], responses[-1], call["agent_started"])
            else:
                result = agent.orchestrator.finish_single(req, decision, responses[-1], call["agent_started"])
            call["response"] = agent.finish_turn(req, result, call["agent_started"])
            return self._step(state, "compose_roles", phase="executed")
        except Exception as ex:
            call["role_error"] = True
            call["response"] = agent.fail_turn(ex, call["agent_started"])
            return self._step(state, "compose_roles", phase="role_failed")
        finally:
            call["business_ms"] += (time.perf_counter() - started) * 1000

    async def _prepare_state(self, state):
        call = self._call.get()
        started = time.perf_counter()
        context = await self.service._prepare_turn(call["message"], call["user"], call["conv"])
        call["business_ms"] = (time.perf_counter() - started) * 1000
        call["context"] = context
        refs = [str(p["op_id"]) for p in context["pending_before"]]
        return self._step(state, "prepare_state", phase="prepared", turn=context["turn"],
                          current_task="return" if refs else "retail", pending_refs=refs)

    async def _route_agent(self, state):
        context = self._call.get()["context"]
        pending = context["pending_before"] or self._call.get()["pending_at_entry"]
        text = context["request"].message
        route = "normal"
        if pending:
            analysis = context.get("analysis")
            # 支持未启用时 Host 分析没有动作，仍需按待办消息本身判定退出。
            action = (
                analysis.return_flow_action
                if analysis is not None and analysis.return_flow_action != "none"
                else analyze_message(text, has_pending=True).return_flow_action
            )
            if action == "cancel" or re.fullmatch(r"(?:cancel|cancel (?:the )?return)[.!]?", text, re.I):
                route = "cancel"
            elif action == "human" or re.search(r"\b(?:human|live agent)\b", text, re.I):
                route = "human"
            elif any(evaluate_confirmation(text, p, pending, language=self.service.language).approved for p in pending):
                route = "confirm"
            else:
                route = "revise_or_question"
        # 此分支只用于编排；提交工具仍以真实 Request.message 独立核验授权。
        return self._step(state, "route_agent", route=route, phase="routed")

    async def _leave_return(self, state):
        call = self._call.get()
        self.service.store.cancel_pending(call["user"], call["conv"], "workflow_human")
        call["context"]["request"].retail_pending = []
        return self._step(state, "leave_return", pending_refs=[])

    async def _cancel_return(self, state):
        call = self._call.get()
        self.service.store.cancel_pending(call["user"], call["conv"], "workflow_cancel")
        call["response"] = AgentResponse(agent_type=AgentType.RETAIL, content=(
            "The pending return was cancelled." if self.service.language == "en" else "已取消待确认退货，旧详情不能再用于确认。"
        ), success=True, latency_ms=0)
        return self._step(state, "cancel_return", pending_refs=[], phase="cancelled")

    async def _run_agent(self, state):
        call = self._call.get()
        started = time.perf_counter()
        call["response"] = await self.service._execute_turn(call["context"])
        call["business_ms"] += (time.perf_counter() - started) * 1000
        return self._step(state, "run_agent", phase="executed")

    async def _update_result(self, state):
        call = self._call.get()
        started = time.perf_counter()
        result = await self.service._finish_turn(call["context"], call["response"])
        call["business_ms"] += (time.perf_counter() - started) * 1000
        call["result"] = result
        pending = self.service.store.active_pending(call["user"], call["conv"])
        waiting = any(p["status"] == "displayed" for p in pending)
        guard = call["context"]["request"].retail_guard or {}
        unknown = ("unknown_action_result" in result.response_guard.get("findings", []) or
                   (bool(call.get("role_error")) and bool(guard.get("state_op_entered") or guard.get("unknown_result"))))
        extra = {}
        if state.get("handoff"):
            old = state["handoff"]
            extra["handoff"] = build_role_handoff(self.service.store, call["context"]["request"],
                                                 old.get("previous_role"), old["to_role"])
        return self._step(state, "update_result", **extra, phase="waiting" if waiting else "complete",
                          current_task="return" if pending or result.prepared_op_ids or state.get("current_task") == "return" else "retail",
                          waiting_reason="return_confirmation" if waiting else "",
                          pending_refs=[str(p["op_id"]) for p in pending],
                          result={"request_id": result.request_id, "success": result.success, "unknown": unknown,
                                  "tools_used": result.tools_used, "escalated": result.escalated})

    async def _wait_user(self, state):
        # 恢复会重跑节点；interrupt 前不得调用工具或写入业务。
        message = interrupt({"reason": state["waiting_reason"], "pending_refs": state["pending_refs"]})
        call = self._call.get()
        if not isinstance(message, str) or message != call["message"]:
            raise ValueError("恢复值必须来自本轮真实用户文本")
        return {"current_input": message, "node_path": ["wait_user"]}

    async def _resume_input(self, state):
        roles = self._reset_roles(self._call.get()) if self._uses_roles() else {}
        return self._step(state, "resume_input", **roles, phase="resuming", waiting_reason="", result={})

    async def _recover_result(self, state):
        from agents.retail_agent import RetailTurnResult
        call = self._call.get()
        # 非等待点崩溃可能已完成写入：只读权威事件，不重跑 Agent 或工具。
        events = self.service.store.list_events(call["user"], call["conv"])
        known = [e for e in events if e["event_type"] == "return_requested"]
        pending = self.service.store.active_pending(call["user"], call["conv"])
        text = ("Execution was interrupted. Business records were checked; no tools were replayed. "
                if self.service.language == "en" else "上次执行中断，已核对业务记录，本轮未重放工具。")
        text += (f" Verified return operations: {len(known)}. Please query the order before continuing."
                 if self.service.language == "en" else f"本会话已记录退货申请 {len(known)} 笔；请先查询订单核实后再继续。")
        call["result"] = RetailTurnResult(response=text, success=False, agent_type=AgentType.RETAIL,
                                        request_id=str(uuid.uuid4()),
                                        turn=self.service.store.current_turn(call["user"], call["conv"]), latency_ms=0)
        return self._step(state, "recover_result", phase="reconciled", waiting_reason="",
                          pending_refs=[str(p["op_id"]) for p in pending], result={"verified_operations": len(known)})

    async def handle_turn(self, message, client_user_id, conv_id):
        await self.start()
        config = self.config(client_user_id, conv_id)
        call = {"message": message, "user": client_user_id, "conv": conv_id,
                "pending_at_entry": self.service.store.active_pending(client_user_id, conv_id)}
        token = self._call.set(call)
        started = time.perf_counter()
        summary = self.service.ledger_summary
        requests_before = summary()["requests"] if summary else None
        try:
            # 依赖带有可选 LangSmith SDK；这里显式禁用网络追踪。
            with tracing_context(enabled=False):
                snapshot = await self._graph.aget_state(config)
                call["previous_role_result"] = snapshot.values.get("role_result", {})
                unknown = (snapshot.values.get("result") or {}).get("unknown", False)
                has_interrupt = any(task.interrupts for task in snapshot.tasks)
                resumed = snapshot.next == ("wait_user",) and has_interrupt and not unknown
                recovering = (bool(snapshot.next) and not resumed) or unknown
                if recovering:
                    # 结束未完成的调度位置，转只读核实；不执行其悬挂节点。
                    await self._graph.aupdate_state(config, {}, as_node="recover_result")
                value = Command(resume=message) if resumed else {
                    "current_input": message, "phase": "recovering" if recovering else "input",
                    "node_path": [], "waiting_reason": "", "result": {}, "pending_refs": [],
                    **(self._reset_roles(call) if self._uses_roles() else {}),
                }
                output = await self._graph.ainvoke(value, config)
                snapshot = await self._graph.aget_state(config)
            result = call["result"]
            elapsed = (time.perf_counter() - started) * 1000
            result.workflow = {"mode": "langgraph", "node_path": output["node_path"],
                               "phase": output["phase"], "waiting_reason": output["waiting_reason"],
                               "resumed": resumed, "recovered": recovering,
                               "elapsed_ms": round(elapsed, 3),
                               "overhead_ms": round(max(0, elapsed - call.get("business_ms", 0)), 3),
                               "tool_calls": len(result.tool_traces),
                               "model_calls": summary()["requests"] - requests_before if summary else None,
                               "next": list(snapshot.next)}
            if self._uses_roles():
                result.workflow.update(active_role=output.get("active_role", ""),
                    handoffs=call.get("handoffs", []), role_handoff=output.get("handoff", {}))
            self.service._record_trace(result, client_user_id, conv_id)
            if "context" in call:
                self.service._record_turn_event(**call["context"]["event"])
            else:
                self.service._record_turn_event(result=result, message=message, client_user_id=client_user_id,
                    conv_id=conv_id, bound_user_id=self.service.store.bound_user_id(client_user_id, conv_id),
                    pending_before=call["pending_at_entry"], new_events=[], display_failed=[])
            return result
        finally:
            self._call.reset(token)

    async def close(self):
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._graph = None
