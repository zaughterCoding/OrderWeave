"""Agent 工具的定义与实现。

所有 Agent 工具都集中在此。编排器只负责管理工具白名单、执行模型返回的
tool_use 块，并把结果回传给模型。

工具保持确定性和可测试性，覆盖当前请求查看、技术指导、账单字段检查、
交接摘要和共享 RAG。

订单查询、退款执行和账单变更需要真实业务系统授权，这里不做模拟。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义与处理函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """按 JSON Schema 创建 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def validate_tool_input(
    spec: AgentToolSpec,
    args: Any,
    language: str = "zh",
) -> None:
    """按工具 Schema 严格校验参数；主程序与 MCP 工具端共用同一规则。

    只覆盖本项目的工具 Schema 子集：必填、未知字段、类型、数值范围与数组
    长度／元素类型。bool 不能冒充整数或数值。校验失败抛出 ValueError。
    """
    english = str(language).lower() == "en"

    def fail(zh: str, en: str) -> None:
        raise ValueError(en if english else zh)

    if not isinstance(args, dict):
        fail("工具参数必须是 JSON 对象", "Tool arguments must be a JSON object")
    schema = spec.input_schema
    for field_name in schema.get("required", []):
        if field_name not in args:
            fail(
                f"缺少必需参数: {field_name}",
                f"Missing required argument: {field_name}",
            )
    properties = schema.get("properties", {})
    unknown = set(args) - set(properties)
    if unknown and schema.get("additionalProperties") is False:
        fail(
            f"不允许的工具参数: {', '.join(sorted(unknown))}",
            f"Unsupported tool arguments: {', '.join(sorted(unknown))}",
        )
    type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
    for key, value in args.items():
        prop = properties.get(key, {})
        expected = prop.get("type")
        if expected in type_map:
            # bool 是 int 的子类，因此不能通过数值类型的校验。
            if isinstance(value, bool) and expected in ("integer", "number"):
                fail(
                    f"参数 {key} 类型错误，期望 {expected}",
                    f"Argument {key} has the wrong type; expected {expected}",
                )
            if not isinstance(value, type_map[expected]):
                fail(
                    f"参数 {key} 类型错误，期望 {expected}",
                    f"Argument {key} has the wrong type; expected {expected}",
                )
        if expected in ("integer", "number") and isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = prop.get("minimum")
            maximum = prop.get("maximum")
            if minimum is not None and value < minimum:
                fail(
                    f"参数 {key} 小于最小值 {minimum}",
                    f"Argument {key} is below the minimum {minimum}",
                )
            if maximum is not None and value > maximum:
                fail(
                    f"参数 {key} 大于最大值 {maximum}",
                    f"Argument {key} is above the maximum {maximum}",
                )
        if expected == "array":
            if not isinstance(value, list):
                fail(
                    f"参数 {key} 类型错误，期望 array",
                    f"Argument {key} has the wrong type; expected array",
                )
            min_items = prop.get("minItems")
            max_items = prop.get("maxItems")
            if min_items is not None and len(value) < int(min_items):
                fail(
                    f"参数 {key} 少于最少项数 {min_items}",
                    f"Argument {key} has fewer than {min_items} items",
                )
            if max_items is not None and len(value) > int(max_items):
                fail(
                    f"参数 {key} 超过最多项数 {max_items}",
                    f"Argument {key} has more than {max_items} items",
                )
            item_type = (prop.get("items") or {}).get("type")
            if item_type in type_map:
                for index, item in enumerate(value):
                    if (
                        isinstance(item, bool)
                        and item_type in ("integer", "number")
                    ) or not isinstance(item, type_map[item_type]):
                        fail(
                            f"参数 {key}[{index}] 类型错误，期望 {item_type}",
                            f"Argument {key}[{index}] has the wrong type; "
                            f"expected {item_type}",
                        )


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：返回当前请求的脱敏快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：列出下一轮只需向用户询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"order_status", "logistics"}:
        fields = ["订单号或下单时间"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码，但不声称能访问日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险的排查步骤。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def check_billing_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：检查必需的核验字段是否齐全。"""
    fields = {
        "order_id": bool(req.entities.get("order_id")),
        "amount": bool(req.entities.get("amount")),
        "date": bool(req.entities.get("date")),
        "payment_channel": bool(args.get("payment_channel")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_refund": False,
        "reason": "当前工具只做字段检查，不连接订单或支付系统",
    }


def compare_amounts(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """账单工具：只计算用户提供金额之间的差值。"""
    try:
        first = float(args["amount_a"])
        second = float(args["amount_b"])
    except (KeyError, TypeError, ValueError):
        return {"success": False, "error": "amount_a 和 amount_b 必须是数字"}
    return {
        "success": True,
        "amount_a": first,
        "amount_b": second,
        "difference": round(first - second, 2),
        "interpretation": "仅表示金额差值，不代表重复扣款或退款结论",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：为人工客服生成结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要人工客服继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(
    tool_manager: Any,
    filter_for_request: Optional[Callable[[Request, Any], bool]] = None,
) -> Dict[str, AgentToolSpec]:
    """构造所有 Agent 共用的 RAG 工具。

    filter_for_request 仅在有限支持启用时传入，用于在合并前剔除不适用证据；
    默认不改变原检索链。
    """

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        extra: Dict[str, Any] = {}
        if filter_for_request is not None:
            extra["item_filter"] = lambda item: filter_for_request(req, item)
        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
            **extra,
        )
        degraded = bool(getattr(result, "degraded", False))
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
                "degraded": degraded,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
            "degraded": degraded,
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、账单和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {
                    "type": "integer",
                    "description": "返回结果条数",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "根据运行环境和是否可复现生成排障顺序，不执行修改配置等操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def billing_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_billing_fields": make_tool(
            "check_billing_fields",
            "检查账单核验字段是否齐全；不连接订单、支付或退款系统。",
            {"payment_channel": {"type": "string", "description": "支付渠道，例如微信、支付宝、银行卡"}},
            check_billing_fields,
        ),
        "compare_amounts": make_tool(
            "compare_amounts",
            "计算用户明确提供的两笔金额差值；不判断是否重复扣款，也不执行退款。",
            {
                "amount_a": {"type": "number", "description": "第一笔金额"},
                "amount_b": {"type": "number", "description": "第二笔金额"},
            },
            compare_amounts,
            required=["amount_a", "amount_b"],
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
