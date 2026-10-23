"""多 Agent 路由与编排。

路由分三层：
  1. 意图路由：把 IntentCategory 映射到专属 Agent。
  2. 性能路由：按成功率和延迟选择最佳实例。
  3. 降级路由：专属 Agent 不可用时退回 GeneralAgent。

复杂请求可并行分派给多个 Agent。升级请求直接产生确定性的交接回复，
不再调用一次 LLM。
"""
import asyncio
import inspect
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    billing_tools,
    escalation_tools,
    general_tools,
    technical_tools,
    validate_tool_input,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


def _stop_exception(ex: BaseException) -> bool:
    """预算停止与单轮请求上限必须向上传播，不能被包装成普通业务失败。"""
    from evaluation.retail_budget import (
        BudgetExceeded,
        TurnRequestLimitExceeded,
    )

    return isinstance(ex, (BudgetExceeded, TurnRequestLimitExceeded))


def _request_stop(req: "Request", ex: BaseException) -> bool:
    """只有零售编排请求才穿透业务层；单 Agent 评测保留原有失败语义。"""
    return getattr(req, "retail_guard", None) is not None and _stop_exception(ex)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    BILLING   = "billing"    # 账单与退款
    ESCALATION = "escalation" # 人工升级与交接
    RETAIL    = "retail"     # 零售沙箱单 Agent


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；非法值不应阻塞启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；非法值不应阻塞启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
    """供监控和路由决策使用的运行时统计。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """成功率越高、延迟越低，路由得分越高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_hits: int = 0
    # 零售编排需要如实保留实际执行路径；旧模式保持默认值。
    fallback_used: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCallResult:
    """单次 LLM 调用的请求级结果；共享 Agent 上不保存状态。"""
    content: str
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_hits: int = 0


class AgentCallError(RuntimeError):
    """保留失败前已执行的工具调用，使请求追踪保持准确。"""

    def __init__(
        self,
        message: str,
        tools_used: List[str],
        tool_traces: List[Dict[str, Any]],
        knowledge_hits: int = 0,
    ):
        super().__init__(message)
        self.tools_used = list(tools_used)
        self.tool_traces = list(tool_traces)
        self.knowledge_hits = knowledge_hits


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # MemoryManager 格式化后的上下文
    history:     Optional[List[Dict[str, str]]] = None  # 用于意图识别的对话历史
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    # 零售模式使用；旧模式保持默认值，不参与原有路由。
    turn: int = 0
    retail_user_id: Optional[str] = None
    retail_pending: List[Dict[str, Any]] = field(default_factory=list)
    # 零售编排在同一真实请求内共享的写操作状态；由 RetailMultiAgent 创建。
    retail_primary: Optional[str] = None
    retail_guard: Optional[Dict[str, Any]] = None
    # 有限支持上下文由 Host 在角色调用前写入，不是模型自报进度。
    retail_support: Optional[Dict[str, Any]] = None
    retail_support_enabled: bool = False
    retail_support_blocked: bool = False
    retail_handoff: Optional[Dict[str, Any]] = None
    retail_task: Dict[str, Any] = field(default_factory=dict)
    retail_routing: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_hits: int = 0
    routing_reason: str = ""
    routing_confidence: float = 0.0
    # 零售对照需要的真实结果与协作轨迹；旧调用方不依赖这些字段。
    success: bool = True
    fallback_used: bool = False
    composer_used: bool = False
    role_traces: List[Dict[str, Any]] = field(default_factory=list)
    composer_inputs: List[Dict[str, Any]] = field(default_factory=list)
    composer_output: str = ""


@dataclass
class RoutingDecision:
    """单个请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 共用的 LLM 调用、角色契约与统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile
    # 旧 Agent 保持原有 3 轮、每轮执行全部 tool_use；零售 Agent 在子类收窄。
    max_tool_rounds: int = 3
    max_tool_uses_per_response: Optional[int] = None
    capture_tool_results: bool = False
    language: str = "zh"

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
        language: Optional[str] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        if language:
            self.language = str(language).lower()

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色可用的工具白名单。"""
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            call_result = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(call_result.content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=call_result.content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=call_result.tools_used,
                tool_traces=call_result.tool_traces,
                knowledge_hits=call_result.knowledge_hits,
            )
        except Exception as ex:
            if _request_stop(req, ex):
                raise
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            tools_used = ex.tools_used if isinstance(ex, AgentCallError) else []
            tool_traces = ex.tool_traces if isinstance(ex, AgentCallError) else []
            knowledge_hits = (
                ex.knowledge_hits if isinstance(ex, AgentCallError) else 0
            )
            return AgentResponse(
                agent_type=self.agent_type,
                content=(
                    "Sorry, something went wrong while handling your request. "
                    "Please try again shortly."
                    if self.language == "en"
                    else "抱歉，处理您的请求时出现问题，请稍后重试。"
                ),
                success=False,
                latency_ms=ms,
                tools_used=list(tools_used),
                tool_traces=list(tool_traces),
                knowledge_hits=knowledge_hits,
            )

    async def _call_llm(self, req: Request) -> AgentCallResult:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        english = self.language == "en"
        messages = []
        if req.context:
            label = "[Background]" if english else "[背景信息]"
            ack = (
                "Understood, I have reviewed the background information."
                if english
                else "好的，我已了解背景信息。"
            )
            messages.append({"role": "user", "content": f"{label}\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": ack})
        if req.entities:
            entities_text = json.dumps(req.entities, ensure_ascii=False)
            label = "[Structured entities]" if english else "[结构化实体]"
            ack = (
                "Understood, I will use these structured entities."
                if english
                else "好的，我会结合这些结构化实体处理。"
            )
            messages.append({"role": "user", "content": f"{label}\n{_clean(entities_text)}"})
            messages.append({"role": "assistant", "content": ack})
        role_packet = self._build_role_packet(req)
        if role_packet:
            label = "[Role input contract]" if english else "[角色输入契约]"
            ack = (
                "Understood, I will follow this role's input and output contract."
                if english
                else "好的，我会按照该角色的输入和输出契约处理。"
            )
            messages.append({"role": "user", "content": f"{label}\n{_clean(role_packet)}"})
            messages.append({"role": "assistant", "content": ack})
        messages.append({"role": "user", "content": _clean(req.message)})

        tools = self._tools_for_request(req)
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        knowledge_hits = 0
        # 普通请求只在本次调用内保存状态；零售编排由 Request 提供同一请求共享状态。
        guard_state: Dict[str, Any] = (
            req.retail_guard if req.retail_guard is not None else {}
        )
        try:
            for _ in range(max(1, int(self.max_tool_rounds or 3))):
                request_kwargs: Dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": self.profile.max_tokens,
                    "temperature": self.profile.temperature,
                    "system": self._build_system_prompt(req),
                    "messages": messages,
                }
                if tools:
                    request_kwargs["tools"] = [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "input_schema": spec.input_schema,
                        }
                        for spec in tools.values()
                    ]
                resp = await self._client.messages.create(**request_kwargs)
                tool_uses = [block for block in (resp.content or []) if self._block_type(block) == "tool_use"]
                if not tool_uses:
                    return AgentCallResult(
                        content=extract_text_content(resp.content),
                        tools_used=list(tools_used),
                        tool_traces=list(tool_traces),
                        knowledge_hits=knowledge_hits,
                    )

                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                for index, block in enumerate(tool_uses):
                    name = self._block_value(block, "name")
                    tool_use_id = self._block_value(block, "id")
                    args = self._block_value(block, "input") or {}
                    spec = tools.get(name)
                    tool_t0 = time.monotonic()
                    call_success = True
                    result_success: Optional[bool] = None
                    error_text = ""
                    skipped = False
                    guard_reason = ""
                    if (
                        self.max_tool_uses_per_response is not None
                        and index >= self.max_tool_uses_per_response
                    ):
                        call_success = False
                        result_success = False
                        skipped = True
                        result: Any = {
                            "success": False,
                            "error": (
                                "At most one tool call may be executed per model "
                                "response; this call was not executed."
                                if english
                                else "每次模型响应最多执行一个工具调用；本次调用未执行。"
                            ),
                        }
                        error_text = result["error"]
                    elif spec is None:
                        call_success = False
                        result = {
                            "success": False,
                            "error": (
                                f"Tool is not in the {self.agent_type.value} "
                                "agent allowlist"
                                if english
                                else f"工具不在 {self.agent_type.value} Agent 白名单中"
                            ),
                        }
                        error_text = result["error"]
                    else:
                        guard_reason = self._guard_tool_call(
                            req, name, args, guard_state
                        ) or ""
                        if guard_reason:
                            call_success = False
                            result_success = False
                            result = {"success": False, "error": guard_reason}
                            error_text = guard_reason
                        else:
                            try:
                                validate_tool_input(
                                    spec, args, language=self.language
                                )
                                result = spec.handler(req, args)
                                if inspect.isawaitable(result):
                                    result = await result
                                tools_used.append(name)
                                if isinstance(result, dict) and "success" in result:
                                    result_success = bool(result.get("success"))
                                self._on_tool_result(
                                    req, name, args, result, guard_state
                                )
                            except Exception as ex:
                                if _request_stop(req, ex):
                                    raise
                                call_success = False
                                logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                                error_text = str(ex)
                                result = {"success": False, "error": error_text}
                                # 结果不明的工具调用交由业务角色登记，避免重放。
                                self._on_tool_error(
                                    req, name, args, ex, guard_state
                                )
                    tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                    if not error_text and isinstance(result, dict):
                        error_text = str(result.get("error", "") or "")
                    knowledge_hits += self._knowledge_hit_count(name, result)
                    trace: Dict[str, Any] = {
                        "agent_type": self.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
                        "latency_ms": round(tool_latency_ms, 1),
                        "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                        "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                        "error": error_text,
                    }
                    if skipped:
                        trace["skipped"] = True
                    if guard_reason:
                        trace["blocked_by_guard"] = True
                    if self.capture_tool_results:
                        trace["result"] = result
                    tool_traces.append(trace)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                messages.append({"role": "user", "content": tool_results})

            raise AgentCallError(
                f"{self.agent_type.value} 工具调用超过最大轮数",
                tools_used,
                tool_traces,
                knowledge_hits,
            )
        except AgentCallError:
            raise
        except Exception as ex:
            if _request_stop(req, ex):
                raise
            raise AgentCallError(
                str(ex), tools_used, tool_traces, knowledge_hits
            ) from ex

    @staticmethod
    def _knowledge_hit_count(name: str, result: Any) -> int:
        """统计可用的知识片段数，忽略失败或空结果的调用。"""
        if name != "search_knowledge_base" or not isinstance(result, dict):
            return 0
        if result.get("success") is not True:
            return 0
        fragments = result.get("results")
        if not isinstance(fragments, list):
            return 0
        return sum(
            1
            for item in fragments
            if isinstance(item, dict)
            and str(item.get("content") or "").strip()
        )

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(
        spec: AgentToolSpec,
        args: Any,
        language: str = "zh",
    ) -> None:
        """保留原入口；校验实现已提取到 agents.tools 供 MCP 工具端共用。"""
        validate_tool_input(spec, args, language=language)

    def _tools_for_request(self, req: Request) -> Dict[str, AgentToolSpec]:
        """按真实请求返回工具白名单；零售角色据此区分主执行与只读辅助。"""
        return self.get_tools()

    def _guard_tool_call(
        self,
        req: Request,
        name: str,
        args: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Optional[str]:
        """业务 Agent 可据此阻止本轮不该执行的工具；默认不限制。"""
        return None

    def _on_tool_result(
        self,
        req: Request,
        name: str,
        args: Dict[str, Any],
        result: Any,
        state: Dict[str, Any],
    ) -> None:
        """工具实际执行后的钩子；默认不保存状态。"""
        return None

    def _on_tool_error(
        self,
        req: Request,
        name: str,
        args: Dict[str, Any],
        error: BaseException,
        state: Dict[str, Any],
    ) -> None:
        """工具抛出异常、结果不明时的钩子；默认不记录。"""
        return None

    def _build_system_prompt(self, req: Request) -> str:
        """把角色契约与动态 Skills 合并进 system prompt。"""
        if self.language == "en":
            profile_prompt = (
                "\n\n[Role contract]\n"
                f"Role: {self.profile.role}\n"
                f"Mission: {self.profile.mission}\n"
                f"Workflow: {' -> '.join(self.profile.workflow)}\n"
                f"Available input: {'; '.join(self.profile.input_contract)}\n"
                f"Output requirements: {'; '.join(self.profile.output_contract)}\n"
                "Handoff conditions: "
                f"{'; '.join(self.profile.handoff_conditions) or 'none; follow the general support rules'}\n"
                "Allowed data/tools: "
                f"{', '.join(self.profile.tool_scope) or 'only the current request context'}\n"
                "Do not claim to have performed a query, modification, or "
                "refund that was not actually performed; if evidence is "
                "missing, say that verification is needed."
            )
        else:
            profile_prompt = (
                f"\n\n[角色契约]\n"
                f"角色：{self.profile.role}\n"
                f"职责：{self.profile.mission}\n"
                f"处理流程：{' -> '.join(self.profile.workflow)}\n"
                f"可用输入：{'；'.join(self.profile.input_contract)}\n"
                f"输出要求：{'；'.join(self.profile.output_contract)}\n"
                f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
                f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
                "不要声称执行了未提供的查询、修改或退款操作；缺少证据时明确说明需要核验。"
            )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可追加领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """用简单关键词匹配检测升级建议。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="通用客服分诊与首轮接待",
        mission="快速回答基础问题，澄清不完整需求，并识别是否需要专业 Agent 或人工处理。",
        workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及权限、资金、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        # 真实模型会把推理写入输出，900 会在答复中途截断。
        max_tokens=1600,
    )
    system_prompt = (
        "你是 EchoMind 智能客服。友好、简洁地回答用户问题。"
        "如果问题超出你的能力范围，明确说明并建议转接专业客服。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["technical", "billing", "escalation"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    profile = AgentProfile(
        role="技术故障诊断与排障",
        mission="基于错误码、环境和复现信息缩小根因范围，给出低风险、可验证的排查步骤。",
        workflow=("确认现象", "判断影响范围", "按网络/权限/配置/依赖排查", "给出验证方式", "判断升级条件"),
        input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
        handoff_conditions=("生产大面积不可用", "数据丢失或权限异常", "需要后台日志、数据库或人工操作"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        # 排查步骤较长，1200 会在答复中途截断。
        max_tokens=2000,
    )
    system_prompt = (
        "你是技术支持专家。专注于：故障排查、错误诊断、系统配置。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["diagnostic_fields"] = {
            "error_codes": req.entities.get("error_code", []),
            "environment_hint": "请从用户消息和背景中确认设备、系统、版本、网络",
            "risk_boundary": "不得要求密码、验证码、完整密钥；不得建议破坏性操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(technical_tools())
        return tools


class BillingAgent(BaseAgent):
    agent_type    = AgentType.BILLING
    profile = AgentProfile(
        role="账单核验与售后处理",
        mission="区分扣款、退款、发票、订阅等资金场景，解释可判断事实，并明确核验和人工审核边界。",
        workflow=("确认账单场景", "收集必要核验字段", "区分订单/实付/退款金额", "说明处理路径与时效", "判断是否升级"),
        input_contract=("订单号", "金额与币种", "支付时间", "支付渠道", "用户期望", "知识库上下文"),
        output_contract=("需要核验的信息", "当前可判断内容", "下一步处理路径", "时效边界"),
        handoff_conditions=("实际退款或补偿", "重复扣款或支付成功但订单未生效", "发票作废/重开", "企业合同或大额订单"),
        tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts"),
        temperature=0.0,
        # 核验字段与退款边界说明较长，1100 会在答复中途截断。
        max_tokens=2000,
    )
    system_prompt = (
        "你是账单服务专家。专注于：账单查询、退款申请、发票问题、订阅管理。"
        "对财务问题保持准确和专业。涉及实际退款操作时，说明需要人工审核。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["verification_fields"] = {
            "order_id": req.entities.get("order_id", []),
            "amount": req.entities.get("amount", []),
            "date": req.entities.get("date", []),
            "missing_fields": [
                field for field, values in (
                    ("订单号或交易号", req.entities.get("order_id", [])),
                    ("支付金额", req.entities.get("amount", [])),
                ) if not values
            ],
            "risk_boundary": "不得承诺退款成功、立即到账或直接修改账单",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(billing_tools())
        return tools


class EscalationAgent(BaseAgent):
    """人工升级节点。

    升级不是普通的问答提示：它产出标准化的交接内容，并阻止其他 Agent
    凭空编造答案。生产环境可在此接入工单系统、人工队列或 webhook。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="人工升级与交接",
        mission="确认升级原因，整理已知上下文，告知用户下一步，不执行未经授权的业务操作。",
        workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("search_knowledge_base", "create_handoff_summary"),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责客服人工升级交接，不要继续模拟已完成的后台操作。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools())
        return tools

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        intent = req.intent.value if req.intent else "unknown"
        urgency = req.urgency.name if req.urgency else "UNKNOWN"
        entities = json.dumps(req.entities or {}, ensure_ascii=False)
        content = (
            "我已将这个问题标记为人工升级处理。\n\n"
            f"升级原因：意图={intent}，紧急度={urgency}\n"
            f"已记录信息：{entities}\n"
            "请不要发送密码、短信验证码或完整支付凭证；人工客服会根据会话记录继续核验。"
        )
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=[],
        )


@dataclass
class ComposerResult:
    content: str
    used_llm: bool
    inputs: List[Dict[str, Any]] = field(default_factory=list)


# 出现这些工具的轮次已经改动或即将改动业务状态，Composer 不得重写回复。
_COMPOSER_STATE_TOOLS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "prepare_return",
    "return_delivered_order_items",
    "transfer_to_human_agents",
}


class ResponseComposer:
    """把多个 Agent 的响应合并为一条主次分明的回复。"""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        language: Optional[str] = None,
        retail_mode: bool = False,
    ):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager
        self.language = str(language or "zh").lower()
        self.retail_mode = bool(retail_mode)

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        """保留原有字符串返回入口。"""
        return (await self.compose_result(req, responses)).content

    async def compose_result(
        self,
        req: Request,
        responses: List[AgentResponse],
    ) -> ComposerResult:
        successful = [
            response
            for response in responses
            if response.success and response.content.strip()
        ]
        if not successful:
            return ComposerResult(
                content=(
                    "Sorry, all agents failed to handle the request."
                    if self.language == "en"
                    else "抱歉，所有 Agent 均处理失败。"
                ),
                used_llm=False,
            )
        inputs = [
            {
                "agent_type": response.agent_type.value,
                "content": response.content,
                "success": response.success,
                "tools_used": list(response.tools_used),
            }
            for response in successful
        ]
        if len(successful) == 1:
            return ComposerResult(
                content=successful[0].content, used_llm=False, inputs=inputs
            )
        # 零售模式下，涉及身份、退货准备／提交或交接的轮次不做二次改写，
        # 直接按主次拼接，避免订单事实、确认内容和交接事实被改述。
        if self.retail_mode and self._retail_must_skip(req, successful):
            return ComposerResult(
                content=self._deterministic_merge(successful),
                used_llm=False,
                inputs=inputs,
            )

        if self.language == "en":
            evidence = "\n\n".join(
                f"[{response.agent_type.value} Agent output]\n{response.content}"
                for response in successful
            )
            prompt = (
                "You are the customer-service Response Composer. Merge the "
                "specialist agents' results into one final reply for the user.\n"
                "Requirements: lead with the primary agent's conclusion and "
                "organise the rest by the user's priorities; remove duplicate "
                "or conflicting wording; do not invent orders, returns, backend "
                "lookups, refunds, or handoffs; if conclusions conflict, say "
                "that verification is needed; preserve needed troubleshooting "
                "steps, verification fields, and escalation boundaries. Do not "
                "add, alter, or omit any order number, item id, quantity, "
                "amount, payment method, return status, or handoff fact. "
                "Output only the English reply for the user and do not mention "
                "agents.\n\n"
                f"Primary agent: {successful[0].agent_type.value}\n"
                f"User message: {req.message}\n"
                f"Candidate results:\n{evidence}"
            )
        else:
            evidence = "\n\n".join(
                f"[{response.agent_type.value} Agent 输出]\n{response.content}"
                for response in successful
            )
            prompt = (
                "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
                "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
                "不能补造订单、退款、后台查询结果；如果结论冲突，明确说明需要核验；"
                "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
                f"主 Agent：{successful[0].agent_type.value}\n"
                f"用户问题：{req.message}\n"
                f"候选结果：\n{evidence}"
            )
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, "general")
            if skill:
                prompt += f"\n\n[通用客服输出边界]\n{skill}"
        try:
            if self.retail_mode:
                # 零售对照冻结温度与输出预算，不使用环境变量覆盖。
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=8192,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                response = await self._client.messages.create(
                    model=self._model,
                    # 两个完整 Agent 答复较长时，模型会先输出推理，3000 仍可能只产生 reasoning。
                    max_tokens=_env_int("ECHOMIND_COMPOSER_MAX_TOKENS", 8192),
                    temperature=_env_float("ECHOMIND_COMPOSER_TEMPERATURE", 0.1),
                    messages=[{"role": "user", "content": prompt}],
                )
            content = extract_text_content(response.content).strip()
            if content:
                return ComposerResult(
                    content=content, used_llm=True, inputs=inputs
                )
        except Exception as ex:
            if _stop_exception(ex):
                raise
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        return ComposerResult(
            content=self._deterministic_merge(successful),
            used_llm=False,
            inputs=inputs,
        )

    @staticmethod
    def _retail_must_skip(
        req: Request,
        successful: List[AgentResponse],
    ) -> bool:
        """状态操作或待确认详情存在时，禁止 Composer 改述。"""
        guard = req.retail_guard or {}
        if guard.get("state_op_entered") or guard.get("unknown_result"):
            return True
        for response in successful:
            for trace in response.tool_traces:
                if str(trace.get("tool_name") or "") in _COMPOSER_STATE_TOOLS:
                    return True
            content = response.content
            if "PENDING CONFIRMATION" in content or "待确认" in content:
                return True
        return False

    def _deterministic_merge(self, successful: List[AgentResponse]) -> str:
        """Composer 不可用时保留主 Agent 与补充 Agent 的标注。"""
        extra = (
            "Additional information:"
            if self.language == "en"
            else "补充说明："
        )
        return "\n\n".join(
            f"{response.content}"
            if index == 0
            else f"{extra}\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """多 Agent 编排器。

    路由层次：
      1. 意图到 Agent 类型的映射。
      2. 同一类型有多个实例时，按 routing_score() 选最佳实例。
      3. 专属 Agent 失败或不可用时退回 GeneralAgent。
    """

    # 静态的意图到 Agent 路由表。
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
        IntentCategory.BILLING:    AgentType.BILLING,
        IntentCategory.REFUND:     AgentType.BILLING,
        IntentCategory.INVOICE:    AgentType.BILLING,
        IntentCategory.PAYMENT_ISSUE: AgentType.BILLING,
        IntentCategory.ACCOUNT:    AgentType.BILLING,
        IntentCategory.ACCOUNT_SECURITY: AgentType.BILLING,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
        # 其余意图默认走 GENERAL。
    }

    # 专属 Agent 的监控惩罚达到该值时，新请求退回 GeneralAgent。
    # 默认值与监控模块的严重惩罚上限一致。
    MONITOR_FALLBACK_PENALTY = 0.5

    def __init__(
        self,
        api_key:  Optional[str] = None,
        base_url: Optional[str] = None,
        model:    Optional[str] = None,
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
        *,
        client: Optional[Any] = None,
        intent_recognizer: Optional[IntentRecognizer] = None,
        composer: Optional[ResponseComposer] = None,
        language: Optional[str] = None,
        retail_mode: bool = False,
        policy_text: str = "",
        retail_role_classes: Optional[Dict[str, type]] = None,
        retail_role_clients: Optional[Dict[str, Any]] = None,
        retail_support_enabled: bool = False,
        retail_semantic_router: Optional[Any] = None,
    ):
        if not model:
            raise ValueError("model is required")
        owns_client = client is None
        if client is None:
            if not api_key:
                raise ValueError("api_key or client is required")
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = AsyncAnthropic(**kwargs)

        self._client = client
        self._owns_client = owns_client
        self._model = model
        self.language = str(language or "zh").lower()
        self.retail_mode = bool(retail_mode)
        self.retail_support_enabled = bool(retail_support_enabled and retail_mode)
        self._retail_semantic_router = retail_semantic_router if retail_mode else None
        self.policy_text = policy_text

        if intent_recognizer is not None:
            self._intent_recognizer = intent_recognizer
        elif self.retail_mode:
            # 零售对照注入同一预算客户端与语言，避免自建未受控连接。
            self._intent_recognizer = IntentRecognizer(
                model=model, client=client, language=self.language
            )
        else:
            self._intent_recognizer = IntentRecognizer(
                api_key=api_key, base_url=base_url, model=model
            )
        self._skill_manager = None if self.retail_mode else skill_manager
        if composer is not None:
            self._composer = composer
        elif self.retail_mode:
            self._composer = ResponseComposer(
                client,
                model,
                None,
                language=self.language,
                retail_mode=True,
            )
        else:
            self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("ECHOMIND_TOOL_TRACE_MAX", 200))

        if self.retail_mode:
            classes = dict(retail_role_classes or {})
            required = ("general", "technical", "billing", "escalation")
            missing = [name for name in required if name not in classes]
            if missing:
                raise ValueError(
                    "零售多 Agent 模式缺少角色类: " + ", ".join(missing)
                )
            self._pool = {}
            for name, agent_type in (
                ("general", AgentType.GENERAL),
                ("technical", AgentType.TECHNICAL),
                ("billing", AgentType.BILLING),
                ("escalation", AgentType.ESCALATION),
            ):
                agent_cls = classes[name]
                profile = replace(
                    agent_cls.profile,
                    model=None,
                    temperature=0.1,
                    max_tokens=2048,
                )
                role_client = (retail_role_clients or {}).get(name, client)
                self._pool[agent_type] = [
                    agent_cls(
                        role_client,
                        model,
                        profile=profile,
                        language=self.language,
                        policy_text=self.policy_text,
                        support_enabled=self.retail_support_enabled,
                    )
                ]
            # 零售模式的工具由运行时经 MCP 发现后统一注入。
            self.set_shared_tools({})
            return

        # Agent 池：每种类型可包含多个实例以支持横向扩展。
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager)],
            AgentType.TECHNICAL: [self._make_agent(TechnicalAgent, client, model, skill_manager)],
            AgentType.BILLING: [self._make_agent(BillingAgent, client, model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, client, model, skill_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """创建角色 Agent，允许通过环境变量覆盖其模型。

        可用更强的模型服务专属角色，而通用接待使用更快的模型；
        升级节点不调用 LLM。
        """
        profile = agent_cls.profile
        env_name = f"ECHOMIND_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """运行时重载后替换 SkillManager 引用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共用的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    @property
    def shared_tools(self) -> Dict[str, AgentToolSpec]:
        """返回当前共享工具的快照，供外层包装器报告能力范围。"""
        return dict(self._shared_tools)

    async def aclose(self) -> None:
        """只关闭编排器自己创建的客户端；注入连接由调用方负责。"""
        if not self._owns_client:
            return
        closer = getattr(self._client, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """暴露意图识别，供 API 判断是否需要 RAG。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "escalated": result.escalated,
            "success": bool(result.success),
            "fallback_used": bool(result.fallback_used),
            "composer_used": bool(result.composer_used),
            "latency_ms": round(result.latency_ms, 1),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """处理单个请求，复用可供工作图调用的路由与收口边界。"""
        t0 = time.monotonic()
        prepared = await self.prepare_route(req)
        if isinstance(prepared, OrchestratorResult):
            return prepared
        if prepared.multi_agent:
            return await self.run_parallel(req, prepared)
        response = await self._execute(req, prepared.primary_agent)
        return self.finish_single(req, prepared, response, t0)

    async def prepare_route(self, req: Request):
        """识别并选择原有角色；澄清结果不执行角色。"""
        t0 = time.monotonic()
        retail_mode = bool(getattr(self, "retail_mode", False))
        language = str(getattr(self, "language", "zh"))
        # 1. 调用方未识别过意图时，这里识别。
        if req.intent is None:
            if retail_mode and req.retail_pending:
                # 待确认详情后的回复固定交给账单角色；是否授权仍由服务端
                # 根据真实用户消息判断，模型不能自行确认。
                req.intent = IntentCategory.REFUND
                req.intent_group = "billing"
                req.intent_confidence = 1.0
                req.retail_routing = {"path": "rule", "reason": "pending", "encoder_calls": 0}
            else:
                intent_result = await self._retail_selective_intent(req) if retail_mode else None
                if intent_result is None:
                    recognition_history = req.history
                    if retail_mode and getattr(self, "_retail_semantic_router", None) is not None and language == "zh":
                        # 状态标明服务端来源并进入原识别缓存键，不能冒作用户确认。
                        recognition_history = list((req.history or [])[-2:]) + [{
                            "role": "system",
                            "content": "服务端调度语境，仅供意图消歧，不提供身份或交易授权：" +
                                       json.dumps(req.retail_task, ensure_ascii=False, sort_keys=True),
                        }]
                    intent_result = await self._intent_recognizer.recognize(
                        req.message, history=recognition_history
                    )
                req.intent  = intent_result.intent
                req.intent_group = intent_result.intent_group
                req.urgency = intent_result.urgency
                req.intent_confidence = intent_result.confidence
                # 调用方显式传入的实体可信度更高，识别出的字段只用于补缺。
                recognized_entities = dict(intent_result.entities or {})
                if req.entities:
                    recognized_entities.update(req.entities)
                req.entities = recognized_entities

        if self._needs_clarification(req):
            if retail_mode and self._support_routing(req) is not None:
                # 支持范围的澄清由 General 结合 Host 上下文完成，避免通用
                # 低置信度分支先截断用户已说明的耳机问题。
                pass
            else:
                result = OrchestratorResult(
                    request_id=req.request_id,
                    response=(
                        "I cannot tell which issue you want to handle yet. Please "
                        "clarify whether it is about an order or delivery, a return "
                        "or billing issue, account details, or a technical problem."
                        if retail_mode and language == "en"
                        else "我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？"
                    ),
                    agent_type=AgentType.GENERAL,
                    intent=req.intent,
                    escalated=False,
                    latency_ms=(time.monotonic() - t0) * 1000,
                    agent_types=[AgentType.GENERAL],
                    primary_agent=AgentType.GENERAL,
                    routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                    routing_confidence=req.intent_confidence,
                )
                self._record_tool_trace(result)
                return result

        # 复杂请求可能涉及多个 Agent，例如同一条消息里既有登录失败
        # 又有重复扣款。
        support_decision = self._support_routing(req)
        local_role = req.retail_routing.get("role") if retail_mode and req.retail_routing.get("path") == "local" else None
        local_decision = (RoutingDecision(primary_agent=AgentType(local_role), supporting_agents=[],
                                          reason="selective_local_semantic_route", confidence=req.intent_confidence)
                          if local_role and req.urgency != UrgencyLevel.CRITICAL else None)
        decision = support_decision or local_decision or self._route_decision(req)
        flow_exit_reasons = {
            "support_return_flow_cancelled",
            "support_return_flow_switched_to_troubleshooting",
            "support_user_requested_human_handoff",
        }
        if (
            retail_mode
            and req.retail_pending
            and req.intent not in (
                IntentCategory.ESCALATION,
                IntentCategory.HUMAN_HANDOFF,
            )
            and not (
                support_decision is not None
                and support_decision.reason in flow_exit_reasons
            )
            and decision.primary_agent != AgentType.ESCALATION
            and self._pool.get(AgentType.BILLING)
        ):
            # 已展示待确认详情时，只允许账单角色处理，且不再并行辅助角色。
            decision = RoutingDecision(
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                reason="pending_confirmation_routed_to_billing",
                confidence=1.0,
            )
        if retail_mode:
            req.retail_primary = decision.primary_agent.value
        return decision

    async def _retail_selective_intent(self, req):
        """状态分支和本地分类先于唯一一次原识别器回退。"""
        router = getattr(self, "_retail_semantic_router", None)
        if router is None or self.language != "zh":
            return None
        from core.intent_recognizer import IntentResult
        support = self._support_routing(req)
        human = bool(re.fullmatch(r"(?:请|麻烦|帮我)?(?:转(?:接)?|联系|找)(?:人工(?:客服)?|真人(?:客服)?)[。！!？?\s]*", req.message))
        if human or support is not None:
            role = "escalation" if human else support.primary_agent.value
            label = {"general": "query", "billing": "refund", "technical": "technical",
                     "escalation": "human_handoff"}[role]
            req.retail_routing = {"path": "rule", "reason": "human" if human else support.reason,
                                  "role": role, "intent": label, "encoder_calls": 0}
        else:
            req.retail_routing = await asyncio.to_thread(router.classify, req.message, req.history, req.retail_task)
            if req.retail_routing.get("path") != "local":
                return None
            label = req.retail_routing["intent"]
        intent = IntentCategory(label)
        recognizer = self._intent_recognizer
        return IntentResult(intent=intent, confidence=min(1.0, max(0.0, req.retail_routing.get("score", 1.0))),
                            urgency=recognizer._urgency(req.message, intent),
                            intent_group=recognizer._intent_group(intent), entities=recognizer._extract_entities(req.message),
                            reasoning="选择性零售路由", latency_ms=req.retail_routing.get("total_ms", 0.0),
                            diagnostics=dict(req.retail_routing))

    def finish_single(self, req, decision, response, t0):
        """单角色结果沿用原升级核验与轨迹。"""
        retail_mode = bool(getattr(self, "retail_mode", False))
        # 3. 检查升级。零售模式只认实际记录成功的 MCP 交接。
        if retail_mode:
            escalated = self._escalation_recorded([response])
        else:
            escalated = False
            if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
                IntentCategory.ESCALATION,
                IntentCategory.HUMAN_HANDOFF,
            ):
                escalated = True
                logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
                # 生产环境：在此创建工单并通知人工客服。

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            knowledge_hits=response.knowledge_hits if response.success else 0,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            success=bool(response.success),
            fallback_used=bool(response.fallback_used),
            role_traces=[self._role_trace(response)],
        )
        self._record_tool_trace(result)
        return result

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """分派给多个 Agent 并合并响应。"""
        t0 = time.monotonic()
        agent_types = decision.agent_types
        retail_mode = bool(getattr(self, "retail_mode", False))
        if retail_mode:
            # 零售模式串行执行：辅助角色只读并先完成，主执行者最后执行，
            # 避免同一请求内的写入与辅助读取并行交互。
            supporting = [
                at for at in decision.supporting_agents if at != decision.primary_agent
            ][:2]
            req.retail_primary = decision.primary_agent.value
            supporting_responses = [
                await self._execute(req, at) for at in supporting
            ]
            if getattr(self, "retail_support_enabled", False):
                from agents.retail_support import merge_role_handoff

                for agent_type, response in zip(supporting, supporting_responses):
                    req.retail_support = merge_role_handoff(
                        req.retail_support,
                        agent_type.value,
                        decision.primary_agent.value,
                        req.turn,
                        response,
                    )
            primary_response = await self._execute(req, decision.primary_agent)
            valid_responses = supporting_responses + [primary_response]
            # Composer 以主角色结论为主，辅助内容附后。
            ordered_responses = [primary_response] + supporting_responses
        else:
            tasks = [self._execute(req, at) for at in agent_types]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
            ordered_responses = valid_responses
            primary_response = next(
                (
                    r for r in valid_responses
                    if r.agent_type == decision.primary_agent
                ),
                None,
            )

        return await self.finish_parallel(
            req, decision, valid_responses, ordered_responses, primary_response, t0
        )

    async def finish_parallel(self, req, decision, valid_responses, ordered_responses, primary_response, t0):
        """共享主辅结果合成；角色工作图与原调度保持同一口径。"""
        retail_mode = bool(getattr(self, "retail_mode", False))
        agent_types = decision.agent_types
        composer_result = await self._compose(req, ordered_responses)
        combined = composer_result.content
        if retail_mode:
            escalated = self._escalation_recorded(valid_responses)
        else:
            escalated = any(
                isinstance(r, AgentResponse) and r.escalate
                for r in valid_responses
            )
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        # 任一参与 Agent 的有效命中都算作本次请求有知识库依据。
        knowledge_hits = sum(
            response.knowledge_hits
            for response in valid_responses
            if response.success
        )
        actual_agent_types = [
            r.agent_type for r in valid_responses if r.success
        ] or agent_types
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=actual_agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,
            tool_traces=tool_traces,
            knowledge_hits=knowledge_hits,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            success=(
                bool(primary_response.success)
                if retail_mode and primary_response is not None
                else True
            ),
            fallback_used=any(r.fallback_used for r in valid_responses),
            composer_used=bool(composer_result.used_llm),
            role_traces=[self._role_trace(r) for r in valid_responses],
            composer_inputs=list(composer_result.inputs),
            composer_output=(
                composer_result.content if composer_result.used_llm else ""
            ),
        )
        self._record_tool_trace(result)
        return result

    async def _compose(
        self,
        req: Request,
        responses: List[AgentResponse],
    ) -> ComposerResult:
        """兼容只实现 compose() 的旧 Composer 和测试替身。"""
        compose_result = getattr(self._composer, "compose_result", None)
        if compose_result is not None:
            return await compose_result(req, responses)
        content = await self._composer.compose(req, responses)
        return ComposerResult(content=content, used_llm=False)

    @staticmethod
    def _escalation_recorded(responses: List[AgentResponse]) -> bool:
        """只有实际成功调用 transfer_to_human_agents 才算升级交接。"""
        return any(
            trace.get("tool_name") == "transfer_to_human_agents"
            and trace.get("result_success") is True
            for response in responses
            for trace in response.tool_traces
        )

    @staticmethod
    def _role_trace(response: AgentResponse) -> Dict[str, Any]:
        """记录每个实际参与角色的成功、工具与降级情况。"""
        return {
            "agent_type": response.agent_type.value,
            "success": bool(response.success),
            "fallback_used": bool(response.fallback_used),
            "latency_ms": round(float(response.latency_ms or 0.0), 1),
            "tools_used": list(response.tools_used),
            "tool_traces": list(response.tool_traces),
            "knowledge_hits": int(response.knowledge_hits or 0),
            "escalate": bool(response.escalate),
            "meta": dict(response.meta or {}),
        }

    # ── 路由 ──────────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """三层路由：意图映射、紧急度覆盖，最后回退到 GENERAL。"""
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 仅当目标类型存在可用实例时才使用它。
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _support_routing(self, req: Request) -> Optional[RoutingDecision]:
        """有限支持模式由 Host 按当前诉求选择下一步，不扩成固定流水线。"""
        if not getattr(self, "retail_support_enabled", False):
            return None
        packet = getattr(req, "retail_support", None)
        if not isinstance(packet, dict) or packet.get("enabled") is not True:
            return None
        scope = str(packet.get("scope_state") or "")
        if scope == "ambiguous":
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                supporting_agents=[],
                reason="support_scope_requires_clarification",
                confidence=0.95,
            )
        if scope == "out_of_scope":
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                supporting_agents=[],
                reason="support_request_out_of_scope",
                confidence=0.95,
            )
        if scope != "active":
            return None
        demand = packet.get("current_demand") or {}
        demand_kind = str(demand.get("kind") or "none")
        record = packet.get("work_record") or {}
        stage = str(record.get("stage") or "")
        flow = packet.get("return_flow") or {}
        flow_status = str(flow.get("status") or "idle")
        flow_action = str(flow.get("action") or "none")
        flow_active = flow_status in ("active", "awaiting_confirmation")
        if demand_kind == "human" or flow_action == "human":
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                supporting_agents=[],
                reason="support_user_requested_human_handoff",
                confidence=1.0,
            )
        if flow_action == "switch_to_troubleshooting":
            return RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[],
                reason="support_return_flow_switched_to_troubleshooting",
                confidence=0.95,
            )
        if flow_action == "cancel":
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                supporting_agents=[],
                reason="support_return_flow_cancelled",
                confidence=0.95,
            )
        if demand_kind == "return":
            return RoutingDecision(
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                reason="support_user_currently_chose_return",
                confidence=0.95,
            )
        if (
            flow_active
            and flow_action == "related_question"
            and self._pool.get(AgentType.TECHNICAL)
        ):
            # 暂时询问相关排查问题不结束退货流程；状态仍保留在支持记录中。
            return RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[],
                reason="support_return_flow_related_question",
                confidence=0.85,
            )
        if flow_active and flow_action != "conditional_return":
            # 退货流程已开始且未明确退出时，补充信息、确认或追问都继续由
            # 账单角色处理；本轮的“当前诉求”不能覆盖进行中的任务。
            return RoutingDecision(
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                reason="support_return_flow_continuation",
                confidence=0.95,
            )
        if demand_kind == "conditional_return":
            return RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[],
                reason="support_conditional_return_requires_troubleshooting_first",
                confidence=0.9,
            )
        if demand_kind == "resolved":
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                supporting_agents=[],
                reason="support_issue_user_reported_resolved",
                confidence=0.9,
            )
        if stage == "return_requested":
            return RoutingDecision(
                primary_agent=AgentType.BILLING,
                supporting_agents=[],
                reason="support_return_flow_continuation",
                confidence=0.9,
            )
        if demand_kind == "none" and stage in ("resolved", "handoff_recorded"):
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                supporting_agents=[],
                reason="support_record_already_closed_or_handed_off",
                confidence=0.8,
            )
        if self._pool.get(AgentType.TECHNICAL):
            return RoutingDecision(
                primary_agent=AgentType.TECHNICAL,
                supporting_agents=[],
                reason="support_limited_troubleshooting",
                confidence=0.9,
            )
        return RoutingDecision(
            primary_agent=AgentType.GENERAL,
            supporting_agents=[],
            reason="support_no_technical_role_available",
            confidence=0.4,
        )

    def _route_decision(self, req: Request) -> RoutingDecision:
        """构造结构化路由决策。

        先处理紧急和交接请求。随后由领域得分选出主 Agent 和可选的
        协作 Agent，表达“主处理加辅助诊断”的关系，避免把命中的每个
        关键词都不分优先级地拼在一起。
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
                confidence=max(req.intent_confidence, 0.8),
            )

        scores = self._domain_scores(req)
        degraded_agents = {
            agent_type
            for agent_type in (AgentType.TECHNICAL, AgentType.BILLING)
            if scores.get(agent_type, 0.0) > 0 and self._monitor_degraded(agent_type)
        }
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL
            or (self._pool.get(agent_type) and agent_type not in degraded_agents)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self._collaboration_targets(req)
        supporting_agents = [
            agent_type
            for agent_type in collaboration_targets
            if agent_type != primary_agent and agent_type in available_scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                and score >= 0.45
                and score >= primary_score * 0.55
            ]

        if getattr(self, "retail_mode", False):
            # 零售对照固定主 Agent 加至多两个只读辅助。
            supporting_agents = supporting_agents[:2]
        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)
        if degraded_agents:
            degraded_text = ", ".join(sorted(agent.value for agent in degraded_agents))
            reason += f", monitor_fallback=[{degraded_text}]"
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """根据意图、关键词和实体为各领域 Agent 打分。"""
        msg = req.message.lower()
        scores = {
            AgentType.GENERAL: 0.1,
            AgentType.TECHNICAL: 0.0,
            AgentType.BILLING: 0.0,
        }

        if req.intent in (
            IntentCategory.QUERY,
            IntentCategory.ORDER_STATUS,
            IntentCategory.LOGISTICS,
            IntentCategory.REQUEST,
            IntentCategory.COMPLAINT,
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.OTHER,
        ):
            scores[AgentType.GENERAL] += 0.55

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
        ):
            scores[AgentType.TECHNICAL] += 0.75

        if req.intent in (
            IntentCategory.BILLING,
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.REFUND,
            IntentCategory.INVOICE,
            IntentCategory.PAYMENT_ISSUE,
        ):
            scores[AgentType.BILLING] += 0.75

        technical_kws = [
            "崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500",
            "401", "验证码", "bug", "technical", "login",
        ]
        billing_kws = [
            "退款", "退货", "扣款", "发票", "账单", "支付", "订阅", "refund",
            "invoice", "多扣", "return", "charged", "payment", "billing",
            "price", "amount", "gift card",
        ]
        general_kws = [
            "订单", "物流", "快递", "配送", "会员", "积分", "咨询", "帮助",
            "order", "delivery", "shipping", "tracking", "product", "policy",
            "help",
        ]

        technical_hits = sum(1 for kw in technical_kws if kw in msg)
        billing_hits = sum(1 for kw in billing_kws if kw in msg)
        general_hits = sum(1 for kw in general_kws if kw in msg)

        scores[AgentType.TECHNICAL] += min(0.45, technical_hits * 0.18)
        scores[AgentType.BILLING] += min(0.45, billing_hits * 0.18)
        scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

        entities = req.entities or {}
        if entities.get("error_code"):
            scores[AgentType.TECHNICAL] += 0.2
        if entities.get("amount"):
            scores[AgentType.BILLING] += 0.15
        if entities.get("order_id"):
            scores[AgentType.GENERAL] += 0.1

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """检测需要多个 Agent 的复合请求。

        意图识别只返回一个主意图，因此用领域关键词补充。例如
        “登录报错并且重复扣款”需要技术和账单两方处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        technical_kws = [
            "崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500",
            "401", "bug", "technical", "login",
        ]
        billing_kws = [
            "退款", "扣款", "发票", "账单", "支付", "订阅", "refund", "invoice",
            "return", "charged", "payment", "billing", "gift card",
        ]

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
        ) or any(kw in msg for kw in technical_kws):
            targets.append(AgentType.TECHNICAL)
        if req.intent in (
            IntentCategory.BILLING,
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.REFUND,
            IntentCategory.INVOICE,
            IntentCategory.PAYMENT_ISSUE,
        ) or any(kw in msg for kw in billing_kws):
            targets.append(AgentType.BILLING)

        # 保留顺序、去重，并且只返回存在实例的类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """对低置信度的 OTHER 意图提出澄清问题。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """性能路由：选择 routing_score() 最高的实例。"""
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    def _monitor_degraded(self, agent_type: AgentType) -> bool:
        """所有专属实例的监控惩罚都很高时触发降级。"""
        if agent_type in (AgentType.GENERAL, AgentType.ESCALATION):
            return False
        agents = self._pool.get(agent_type, [])
        if not agents:
            return False
        threshold = _env_float(
            "ECHOMIND_MONITOR_FALLBACK_PENALTY",
            self.MONITOR_FALLBACK_PENALTY,
        )
        penalties = [
            float(getattr(getattr(agent, "stats", None), "monitor_penalty", 0.0) or 0.0)
            for agent in agents
        ]
        # 只要还有健康实例，就继续使用该专属类型。
        return min(penalties) >= min(max(threshold, 0.0), 0.9)

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时回退到 GeneralAgent。"""
        response = await self.execute_role(req, agent_type)
        if self.should_fallback(req, agent_type, response):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                return self.merge_fallback(response, await fallback.handle(req))
        return response

    async def execute_role(self, req, agent_type, allow_missing_fallback=True):
        """执行一次角色；工作图自行调度降级节点。"""
        agent = self._best_agent(agent_type)
        if agent is None and allow_missing_fallback:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        return await agent.handle(req)

    def should_fallback(self, req, agent_type, response):
        """复用原降级条件，任何工具或待确认状态都禁止跨角色重放。"""
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            if (
                getattr(self, "retail_mode", False)
                and self._retail_fallback_blocked(req, response)
            ):
                logger.warning(
                    "%s 失败，但本轮已进入状态操作，禁止跨角色重放",
                    agent_type.value,
                )
                return False
            return True
        return False

    @staticmethod
    def merge_fallback(failed, response):
        """保留失败角色的调用事实，不让降级掩盖先前执行。"""
        response.fallback_used = True
        response.tools_used = list(dict.fromkeys(list(failed.tools_used) + list(response.tools_used)))
        response.tool_traces = list(failed.tool_traces) + list(response.tool_traces)
        response.knowledge_hits = int(failed.knowledge_hits or 0) + int(response.knowledge_hits or 0)
        return response

    @staticmethod
    def _retail_fallback_blocked(
        req: Request, response: AgentResponse
    ) -> bool:
        """零售模式下，出现任何工具调用或待确认状态都不再跨角色重放。"""
        guard = req.retail_guard or {}
        return bool(
            guard.get("state_op_entered")
            or guard.get("unknown_result")
            or req.retail_pending
            or response.tool_traces
        )

    # ── 统计（供监控读取） ────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """把监控反馈应用到路由惩罚。

        键使用 get_stats() 中的 Agent 键，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
