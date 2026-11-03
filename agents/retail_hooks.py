"""零售业务挂入工具循环的钩子插件。

每个钩子只负责一条不变量；单 Agent 与多 Agent 角色按需组合同一组插件，
不再在各角色类里分散实现。钩子只读写同一真实请求的共享 state，不能授权写入。
"""
from __future__ import annotations

from typing import Optional

from agents.agent_loop import LoopHook
from agents.retail_tools import RETAIL_STATE_TOOLS, RETAIL_WRITE_TOOLS, is_execution_attempt


def _en(ctx) -> bool:
    return getattr(ctx.agent, "language", "zh") == "en"


class UnknownResultHook(LoopHook):
    """状态工具结果不明后，本请求不再调用任何工具，防止重放未知写入。"""

    name = "unknown_result"

    def pre_tool(self, ctx, name, args) -> Optional[str]:
        if not ctx.state.get("unknown_result"):
            return None
        return ("A previous tool result is unknown in this request; do not call more tools. "
                "Report the uncertainty and the safe next step." if _en(ctx) else
                "本次请求此前已有结果不明的工具调用，不能继续调用工具；请说明不确定性并给出安全的下一步。")

    def on_tool_error(self, ctx, name, args, error) -> None:
        if name in RETAIL_STATE_TOOLS:
            ctx.state["state_op_entered"] = True
            ctx.state["unknown_result"] = True


class SameTurnExecutionHook(LoopHook):
    """同一轮已生成待确认详情后，禁止执行任何已展示动作（展示与执行必须跨轮）。"""

    name = "same_turn_execution"

    def pre_tool(self, ctx, name, args) -> Optional[str]:
        if not (ctx.state.get("prepared") and is_execution_attempt(name, args, ctx.req.retail_pending)):
            return None
        return ("Pending-confirmation details were generated in this turn; do not submit or perform other "
                "business actions in the same turn. Show the complete details and wait for the user's "
                "explicit confirmation." if _en(ctx) else
                "本轮已生成待确认详情，不能在同一轮提交或执行其他业务操作；请展示完整详情并等待用户明确确认。")

    def post_tool(self, ctx, name, args, result) -> None:
        if (name in RETAIL_WRITE_TOOLS and isinstance(result, dict) and result.get("success") is True
                and result.get("action_status") == "pending_confirmation"):
            ctx.state["prepared"] = True


class PrimaryOnlyHook(LoopHook):
    """多 Agent 下只有本请求主执行角色可以绑定身份、写入或记录交接。"""

    name = "primary_only"

    def pre_tool(self, ctx, name, args) -> Optional[str]:
        agent = ctx.agent
        if agent._is_primary(ctx.req) or name in agent._supporting_read_only():
            return None
        return ("Only the primary agent of this request may bind identity, prepare or submit a return, "
                "or record a handoff." if _en(ctx) else
                "只有本次请求的主执行 Agent 才能绑定身份、准备／提交退货或记录人工交接。")


class ConfirmationStopHook(LoopHook):
    """确认被服务端拒绝后立即结束本角色：本轮用户消息不变，继续调用不能取得新授权。"""

    name = "confirmation_stop"

    def post_tool(self, ctx, name, args, result) -> None:
        if (name in RETAIL_WRITE_TOOLS and isinstance(result, dict)
                and result.get("error_code") == "confirmation_required"):
            ctx.state["stop_response"] = result["confirmation_hint"]

    def stop_response(self, ctx) -> Optional[str]:
        return ctx.state.get("stop_response")


class WriteStateHook(LoopHook):
    """记录本请求是否进入状态操作、已执行写入或已交接，供编排降级判断。"""

    name = "write_state"

    def post_tool(self, ctx, name, args, result) -> None:
        if name in RETAIL_STATE_TOOLS:
            ctx.state["state_op_entered"] = True
        success = isinstance(result, dict) and result.get("success") is True
        if name in RETAIL_WRITE_TOOLS and success and result.get("action_status") != "pending_confirmation":
            ctx.state["submitted"] = True
        elif name == "transfer_to_human_agents" and success:
            ctx.state["handoff"] = True


class SkillGateHook(LoopHook):
    """写工具绑定 Skill：本会话尚未加载对应 Skill 时拒绝调用，确保执行前读过该动作的政策原文。"""

    name = "skill_gate"

    def pre_tool(self, ctx, name, args) -> Optional[str]:
        library = getattr(ctx.req, "retail_skills", None)
        if library is None:
            return None
        bound = library.bound_skills(name)
        loaded = set(getattr(ctx.req, "loaded_skills", None) or ()) | set(ctx.state.get("loaded_skills") or ())
        if not bound or loaded & set(bound):
            return None
        from core.retail_skills import SKILL_TOOL

        return (f"Call {SKILL_TOOL} with name '{bound[0]}' first to read this action's policy and procedure; "
                "nothing was changed." if _en(ctx) else
                f"请先调用 {SKILL_TOOL}（name='{bound[0]}'）阅读该动作的政策与流程；本次未做任何修改。")

    def post_tool(self, ctx, name, args, result) -> None:
        from core.retail_skills import SKILL_TOOL

        if name == SKILL_TOOL and isinstance(result, dict) and result.get("success"):
            loaded = ctx.state.setdefault("loaded_skills", [])
            if result["skill"] not in loaded:
                loaded.append(result["skill"])


def single_agent_hooks():
    return [SkillGateHook(), SameTurnExecutionHook(), ConfirmationStopHook()]


def role_hooks():
    # 顺序即拒绝优先级：未知结果最先，其次跨轮执行，最后角色权限。
    return [UnknownResultHook(), SameTurnExecutionHook(), PrimaryOnlyHook(), SkillGateHook(),
            ConfirmationStopHook(), WriteStateHook()]


__all__ = ["UnknownResultHook", "SameTurnExecutionHook", "PrimaryOnlyHook", "ConfirmationStopHook",
           "WriteStateHook", "SkillGateHook", "single_agent_hooks", "role_hooks"]
