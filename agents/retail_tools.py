"""零售沙箱业务工具。

工具只读写本项目 SQLite 存储；身份来自服务器会话绑定，模型传入的
客户端 user_id 不参与订单授权。退款去向与确认由服务端校验。
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    make_tool,
)
from agents.retail_support import support_evidence_applicable
from mcp_tools.retail_domain import WRITE_KINDS, DomainError, canonical_args
from mcp_tools.retail_store import RetailStore, safe_calculate

_RETAIL_KNOWLEDGE_DESCRIPTION = (
    "检索本项目冻结的零售政策与商品准备建议证据，返回 evidence_id、正文和来源。"
    "只用于解释固定政策和准备注意事项，不构成退货资格、期限、运费或到账承诺。"
)

# 零售多角色权限：只有主执行者可以绑定身份、准备／提交退货和记录交接；
# 辅助角色的工具白名单只保留只读查询。
RETAIL_READ_ONLY_TOOLS = (
    "get_user_details",
    "get_order_details",
    "get_product_details",
    "get_item_details",
    "list_all_product_types",
    "calculate",
)
# 七类写工具均为两阶段：首次调用只生成待确认详情，展示并经确认后同参数再次调用才执行。
RETAIL_WRITE_TOOLS = tuple(WRITE_KINDS)
RETAIL_STATE_TOOLS = (
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    *RETAIL_WRITE_TOOLS,
    "transfer_to_human_agents",
)
RETAIL_ALL_TOOLS = RETAIL_STATE_TOOLS + RETAIL_READ_ONLY_TOOLS
# 支持模式的商品资料角色只保留公开商品查询、必要计算和知识检索；
# 身份、订单、准备／提交与交接由主执行者和原业务链负责。
RETAIL_SUPPORT_TECHNICAL_TOOLS = (
    "get_product_details",
    "calculate",
    "search_knowledge_base",
)

_CATEGORY_MARKERS = {
    "consumer_electronics": ("电子", "耳机", "电脑", "手机", "相机", "平板", "earbuds", "headphones", "laptop", "smartphone", "camera", "tablet"),
    "small_appliances": ("小家电", "烤箱", "咖啡机", "吸尘器", "净化器", "电水壶", "水壶", "搅拌机", "微波炉", "电饭煲", "toaster", "coffee maker", "blender", "vacuum"),
    "apparel_footwear": ("衣服", "服装", "鞋", "靴", "t恤", "衬衫", "外套", "裙", "尺码", "jacket", "shoes", "sneakers", "shirt", "dress"),
}
_POLICY_COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"


def retail_evidence_applicable(item: Any, packet: Optional[Dict[str, Any]], message: str) -> bool:
    """原始询问限定类别，Host限定支持商品；改写文本不能放宽范围。"""
    if not support_evidence_applicable(item, packet) or item.get("language") != "zh":
        return False
    category = item.get("category")
    if category == "product_support":
        return (item.get("product_id") == "9924732112"
                and isinstance(packet, dict) and packet.get("issue") == "bluetooth_pairing"
                and item.get("scope") == "project_zh_earbuds"
                and item.get("support_version") == "t17.1"
                and item.get("policy_release") == "t17-sandbox-v1"
                and item.get("source_kind") == "project_authored_sandbox"
                and item.get("issue") == "bluetooth_pairing")
    if item.get("product_id"):
        return False
    if (item.get("scope") != "project_zh" or item.get("policy_release") != "v1.0.1"
            or item.get("policy_commit") != _POLICY_COMMIT
            or item.get("source_kind") not in {"upstream_paraphrase", "upstream_derived", "project_authored"}):
        return False
    categories = {name for name, markers in _CATEGORY_MARKERS.items() if any(
        (re.search(r"\b" + re.escape(marker) + r"\b", message.casefold()) if marker.isascii()
         else marker in message.casefold()) for marker in markers)}
    if (not categories and isinstance(packet, dict) and packet.get("supported") is True
            and packet.get("scope_state") == "active" and packet.get("product_id") == "9924732112"):
        categories = {"consumer_electronics"}
    return category == "all" or category in _CATEGORY_MARKERS and (not categories or category in categories)


def retail_knowledge_tool(
    tool_manager: Any,
    support_enabled: bool = False,
) -> AgentToolSpec:
    """复用既有 RAG 处理函数，只替换为零售模式可见的描述。

    T12 保留的 A 召回最终只返回 5 条；这里把模型可见的 top_k 上限也
    收到 5；普通咨询候选5条，支持咨询保留原20条，前置过滤不扩大预算。
    """
    filter_for_request = lambda req, item: retail_evidence_applicable(
        item, getattr(req, "retail_support", None) if support_enabled else None,
        req.message,
    )
    shared = build_shared_rag_tools(
        tool_manager, filter_for_request=filter_for_request,
        expand_filtered_recall=support_enabled,
    )["search_knowledge_base"]
    properties = dict(shared.input_schema["properties"])
    properties["top_k"] = {
        "type": "integer",
        "description": "返回结果条数，零售知识最多 5 条",
        "minimum": 1,
        "maximum": 5,
    }
    return make_tool(
        "search_knowledge_base",
        _RETAIL_KNOWLEDGE_DESCRIPTION,
        properties,
        shared.handler,
        required=["query"],
    )


def _desc(language: str, zh: str, en: str) -> str:
    """按业务语言选择工具说明；默认中文。"""
    return en if str(language).lower() == "en" else zh


_WRITE_PURPOSE = {
    "return_delivered_order_items": ("退回已送达订单中的部分商品；退款只能到原支付方式或本人已有礼品卡。",
                                     "Return some items of a delivered order. The refund can only go to the original "
                                     "payment method or an existing gift card of the user."),
    "exchange_delivered_order_items": ("把已送达订单中的商品换成同一商品的其他规格；差价用指定支付方式支付或退还。",
                                       "Exchange items of a delivered order for other variants of the same product; "
                                       "the price difference is paid or refunded with the given payment method."),
    "cancel_pending_order": ("取消待处理（pending）订单，原因只能是 no longer needed 或 ordered by mistake。",
                             "Cancel a pending order. The reason must be 'no longer needed' or 'ordered by mistake'."),
    "modify_pending_order_items": ("把待处理订单中的商品改成同一商品的其他规格；每笔订单只能改一次，须一次列全。",
                                   "Change items of a pending order to other variants of the same product. This can "
                                   "only be done once per order, so collect every item to change first."),
    "modify_pending_order_address": ("修改待处理订单的收货地址。", "Change the shipping address of a pending order."),
    "modify_pending_order_payment": ("把待处理订单的支付方式换成用户的另一种支付方式。",
                                     "Change the payment method of a pending order to another method of the user."),
    "modify_user_address": ("修改当前已核实用户的默认地址。", "Change the default address of the verified current user."),
}


def _write_description(name: str, lang: str) -> str:
    zh, en = _WRITE_PURPOSE[name]
    return _desc(
        lang,
        zh + "两阶段执行：首次调用只校验并返回待确认详情 display_text，不修改任何数据；必须把它原样展示给用户。"
        "用户在之后的消息中明确确认后，用完全相同的参数再次调用才会执行。服务端依据真实用户消息判断确认，"
        "参数变化会生成新的待确认详情。",
        en + " Two-phase: the first call only validates and returns pending details (display_text) without "
        "changing anything; show display_text to the user exactly. After the user explicitly confirms in a later "
        "message, call this tool again with identical arguments to carry it out. The server judges confirmation "
        "from the real user message; different arguments create new pending details.")


def _string(lang: str, zh: str, en: str) -> Dict[str, Any]:
    return {"type": "string", "description": _desc(lang, zh, en)}


def _id_list(lang: str, zh: str, en: str) -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 20,
            "description": _desc(lang, zh, en)}


def _address(lang: str) -> Dict[str, Any]:
    return {field: _string(lang, label, field) for field, label in (
        ("address1", "地址第一行"), ("address2", "地址第二行，可为空"), ("city", "城市"),
        ("state", "州／省"), ("country", "国家"), ("zip", "邮编"))}


def _order(lang: str) -> Dict[str, Any]:
    return {"order_id": _string(lang, "订单号，如 #W0000000", "Order id such as '#W0000000'")}


def _payment(lang: str, zh: str, en: str) -> Dict[str, Any]:
    return {"payment_method_id": _string(lang, zh, en)}


_DIFF_ZH, _DIFF_EN = "支付或接收差价的支付方式 ID", "Payment method id for the price difference"
_WRITE_SCHEMAS = {
    "return_delivered_order_items": lambda lang: {
        **_order(lang),
        "item_ids": _id_list(lang, "要退的 item_id；同一 item_id 重复表示多件", "Item ids to return; repeat for multiple units"),
        **_payment(lang, "原支付方式或本人礼品卡 ID", "Original payment method or the user's gift card id")},
    "exchange_delivered_order_items": lambda lang: {
        **_order(lang),
        "item_ids": _id_list(lang, "要换的 item_id", "Item ids to exchange"),
        "new_item_ids": _id_list(lang, "按位置对应的同商品新 item_id", "New item ids of the same product, by position"),
        **_payment(lang, _DIFF_ZH, _DIFF_EN)},
    "cancel_pending_order": lambda lang: {
        **_order(lang),
        "reason": _string(lang, "no longer needed 或 ordered by mistake", "'no longer needed' or 'ordered by mistake'")},
    "modify_pending_order_items": lambda lang: {
        **_order(lang),
        "item_ids": _id_list(lang, "要修改的 item_id", "Item ids to modify"),
        "new_item_ids": _id_list(lang, "按位置对应的同商品新 item_id", "New item ids of the same product, by position"),
        **_payment(lang, _DIFF_ZH, _DIFF_EN)},
    "modify_pending_order_address": lambda lang: {**_order(lang), **_address(lang)},
    "modify_pending_order_payment": lambda lang: {**_order(lang), **_payment(lang, "新的支付方式 ID", "New payment method id")},
    "modify_user_address": lambda lang: {"user_id": _string(lang, "当前已核实用户 ID", "The verified user's id"),
                                         **_address(lang)},
}
_ADDRESS_REQUIRED = ["address1", "address2", "city", "state", "country", "zip"]
_WRITE_REQUIRED = {
    "return_delivered_order_items": ["order_id", "item_ids", "payment_method_id"],
    "exchange_delivered_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    "cancel_pending_order": ["order_id", "reason"],
    "modify_pending_order_items": ["order_id", "item_ids", "new_item_ids", "payment_method_id"],
    "modify_pending_order_address": ["order_id", *_ADDRESS_REQUIRED],
    "modify_pending_order_payment": ["order_id", "payment_method_id"],
    "modify_user_address": ["user_id", *_ADDRESS_REQUIRED],
}


def retail_tools(
    store: RetailStore,
    tool_manager: Any = None,
    language: Optional[str] = None,
    support_enabled: bool = False,
) -> Dict[str, AgentToolSpec]:
    """构造零售 Agent 的完整工具白名单。"""
    lang = str(language or getattr(store, "language", "zh")).lower()

    def find_user_id_by_email(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.find_user_by_email(
            req.user_id, req.conv_id, str(args.get("email", ""))
        )

    def find_user_id_by_name_zip(
        req: Any, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        return store.find_user_by_name_zip(
            req.user_id,
            req.conv_id,
            str(args.get("first_name", "")),
            str(args.get("last_name", "")),
            str(args.get("zip_code", "")),
        )

    def get_user_details(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.get_user_details(req.user_id, req.conv_id)

    def get_order_details(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.get_order_details(
            req.user_id, req.conv_id, str(args.get("order_id", ""))
        )

    def get_product_details(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.get_product_details(
            req.user_id, req.conv_id, str(args.get("product_id", ""))
        )

    def calculate(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return safe_calculate(
            str(args.get("expression", "")), language=lang
        )

    def get_item_details(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.get_item_details(req.user_id, req.conv_id, str(args.get("item_id", "")))

    def list_all_product_types(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        return store.list_all_product_types(req.user_id, req.conv_id)

    def write_handler(name: str):
        def handler(req: Any, args: Dict[str, Any]) -> Dict[str, Any]:
            return store.write_action(req.user_id, req.conv_id, name, dict(args), req.message)
        return handler

    def transfer_to_human_agents(
        req: Any, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        return store.record_handoff(
            req.user_id,
            req.conv_id,
            str(args.get("summary", "")),
            str(args.get("reason", "")),
        )

    tools: Dict[str, AgentToolSpec] = {
        "find_user_id_by_email": make_tool(
            "find_user_id_by_email",
            _desc(
                lang,
                "按用户本人提供的邮箱查找并绑定当前会话的沙箱用户。"
                "只用于身份核对，不要询问密码或验证码；绑定后不能在同一会话切换用户。",
                "Find and bind the sandbox user for this session by the email "
                "the user provides. Use it only for identity verification; do "
                "not ask for passwords or verification codes, and do not switch "
                "users within the same session after binding.",
            ),
            {
                "email": {
                    "type": "string",
                    "description": _desc(
                        lang, "用户本人提供的邮箱", "The user's own email address"
                    ),
                    "minLength": 3,
                }
            },
            find_user_id_by_email,
            required=["email"],
        ),
        "find_user_id_by_name_zip": make_tool(
            "find_user_id_by_name_zip",
            _desc(
                lang,
                "按用户本人的名、姓和邮编查找并绑定当前会话的沙箱用户。"
                "信息必须完整；不确定时先向用户询问，不要猜测。",
                "Find and bind the sandbox user for this session by the user's "
                "first name, last name, and zip code. All fields are required; "
                "ask the user when unsure and never guess.",
            ),
            {
                "first_name": {
                    "type": "string",
                    "description": _desc(lang, "名", "First name"),
                },
                "last_name": {
                    "type": "string",
                    "description": _desc(lang, "姓", "Last name"),
                },
                "zip_code": {
                    "type": "string",
                    "description": _desc(lang, "邮编", "Zip code"),
                },
            },
            find_user_id_by_name_zip,
            required=["first_name", "last_name", "zip_code"],
        ),
        "get_user_details": make_tool(
            "get_user_details",
            _desc(
                lang,
                "返回当前会话已绑定用户的资料、地址、支付方式，以及该用户的订单列表"
                "（订单号）。用户不知道订单号时，可用订单列表定位本人订单；"
                "没有绑定身份时会被拒绝。",
                "Return the profile, addresses, payment methods, and order list "
                "(order ids) of the user bound to this session. When the user "
                "does not know an order id, use the order list to locate the "
                "user's own orders. Rejected when no identity is bound.",
            ),
            {},
            get_user_details,
        ),
        "get_order_details": make_tool(
            "get_order_details",
            _desc(
                lang,
                "查询当前绑定用户的一笔订单，返回状态、商品行（item_id、数量、订单价）、"
                "配送记录和原支付方式。只能查询本人订单；不编造不存在的物流或日期。",
                "Look up one order of the currently bound user and return its "
                "status, item lines (item_id, quantity, order price), delivery "
                "records, and original payment method. Only the user's own "
                "orders can be read; never invent tracking events or dates.",
            ),
            {
                "order_id": {
                    "type": "string",
                    "description": _desc(
                        lang,
                        "订单号，例如 #W8171054",
                        "Order id, for example #W8171054",
                    ),
                    "minLength": 2,
                }
            },
            get_order_details,
            required=["order_id"],
        ),
        "get_product_details": make_tool(
            "get_product_details",
            _desc(
                lang,
                "查询公开商品目录中的商品资料。商品目录价格不能替代订单成交价，"
                "也不能据此推断退货资格。",
                "Look up a product in the public catalog. Catalog prices do not "
                "replace the order price and do not establish return eligibility.",
            ),
            {
                "product_id": {
                    "type": "string",
                    "description": _desc(lang, "商品编号", "Product id"),
                    "minLength": 1,
                }
            },
            get_product_details,
            required=["product_id"],
        ),
        "calculate": make_tool(
            "calculate",
            _desc(
                lang,
                "计算只含数字、括号和加减乘除的金额表达式。订单金额仍以订单记录为准。",
                "Evaluate an amount expression containing only numbers, "
                "parentheses, addition, subtraction, multiplication, and "
                "division. Order amounts still come from the order record.",
            ),
            {
                "expression": {
                    "type": "string",
                    "description": _desc(
                        lang,
                        "例如 (186.06 + 258.71) * 2",
                        "For example (186.06 + 258.71) * 2",
                    ),
                    "minLength": 1,
                }
            },
            calculate,
            required=["expression"],
        ),
        "get_item_details": make_tool(
            "get_item_details",
            _desc(lang, "按 item_id 查询公开商品规格（选项、价格、是否可售）。",
                  "Look up a catalog item by item_id: options, price, and availability."),
            {"item_id": {"type": "string", "description": _desc(lang, "商品规格编号", "Item id"), "minLength": 1}},
            get_item_details,
            required=["item_id"],
        ),
        "list_all_product_types": make_tool(
            "list_all_product_types",
            _desc(lang, "列出全部商品类型名称与 product_id。", "List the name and product_id of every product type."),
            {},
            list_all_product_types,
        ),
        **{name: make_tool(name, _write_description(name, lang), _WRITE_SCHEMAS[name](lang), write_handler(name),
                           required=_WRITE_REQUIRED[name]) for name in RETAIL_WRITE_TOOLS},
        "transfer_to_human_agents": make_tool(
            "transfer_to_human_agents",
            _desc(
                lang,
                "记录一次模拟人工交接事件，用于用户明确要求人工或超出范围时。"
                "不会连接真实坐席，也不会创建真实工单。",
                "Record a simulated human handoff when the user explicitly asks "
                "for a human or the issue is out of scope. It does not connect "
                "a live agent and does not create a real support ticket.",
            ),
            {
                "summary": {
                    "type": "string",
                    "description": _desc(
                        lang,
                        "交接摘要，不包含密码或验证码",
                        "Handoff summary; do not include passwords or "
                        "verification codes",
                    ),
                    "minLength": 1,
                },
                "reason": {
                    "type": "string",
                    "description": _desc(lang, "交接原因", "Reason for the handoff"),
                },
            },
            transfer_to_human_agents,
            required=["summary"],
        ),
    }
    if tool_manager is not None:
        tools["search_knowledge_base"] = retail_knowledge_tool(
            tool_manager, support_enabled=support_enabled
        )
    return tools


def is_execution_attempt(name: str, args: Dict[str, Any], pending: Any) -> bool:
    """参数与已展示详情一致的写调用会被执行；同一轮刚生成详情时须拦截这类调用。"""
    if name not in WRITE_KINDS:
        return False
    try:
        canonical = canonical_args(WRITE_KINDS[name], dict(args or {}))
    except DomainError:
        return False
    return any(p.get("kind") == WRITE_KINDS[name] and p.get("status") == "displayed" and p.get("args") == canonical
               for p in pending or [])


def retail_tool_scope(has_knowledge: bool = True) -> tuple[str, ...]:
    """供角色契约展示的可用工具范围。"""
    names = [
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        *RETAIL_READ_ONLY_TOOLS,
        *RETAIL_WRITE_TOOLS,
        "transfer_to_human_agents",
    ]
    if has_knowledge:
        names.append("search_knowledge_base")
    return tuple(names)
