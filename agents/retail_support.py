"""无线耳机有限支持的工作记录、适用范围与交接包。

本模块只处理 T17 固定沙箱范围：公开种子中的 Wireless Earbuds
（product_id=9924732112）蓝牙无法发现／配对／连接。它不执行交易动作，
不批准身份、确认、退款去向或提交；业务事实仍由 RetailStore 工具读取。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUPPORT_FORMAT_VERSION = 1
SUPPORT_PRODUCT_ID = "9924732112"
SUPPORT_PRODUCT_NAME = "Wireless Earbuds"
SUPPORT_ISSUE = "bluetooth_pairing"
SUPPORT_ISSUE_LABEL = "蓝牙无法发现／配对／连接"
SUPPORT_STEP_IDS = ("PAIR-01", "PAIR-02", "PAIR-03")
SUPPORT_EVIDENCE_IDS = (
    "T17.E01",
    "T17.E02",
    "T17.E03",
    "T17.E04",
    "T17.E05",
    "T17.E06",
)
MAX_VISIBLE_PAIRS = 12
MAX_FEEDBACK = 8
MAX_SUGGESTIONS = 8
MAX_HANDOFF_CHARS = 6000
SUPPORT_CONTEXT_LIMIT = "SUPPORT_CONTEXT_LIMIT"
SUPPORT_STATE_ERROR = "SUPPORT_STATE_ERROR"

_SCOPE_RULE = (
    "只处理 Wireless Earbuds（9924732112）的蓝牙无法发现／配对／连接。"
    "不得扩展到其他商品、无声、音质、电池、固件、驱动、维修、换货、"
    "兼容性或通用电脑网络排障；资料未定义按键时长、恢复出厂、保修和硬件"
    "损坏结论。排查失败不是退货资格；用户明确选择退货后仍走原交易确认链。"
)


@dataclass
class SupportAnalysis:
    scope_state: str = "inactive"  # active、ambiguous、out_of_scope、inactive
    product_explicit: bool = False
    product_mentioned: bool = False
    issue: str = ""
    demand_kind: str = "none"  # none、troubleshooting、return、conditional_return、human、resolved
    demand_text: str = ""
    feedback_kind: str = "none"  # none、resolved、unresolved、choice
    feedback_text: str = ""
    tried_steps: List[Dict[str, str]] = field(default_factory=list)
    return_negative: bool = False
    human_request: bool = False
    # 当前这句话对进行中退货流程的动作；与 demand_kind 分开，避免把补充回答
    # 误当成新的技术排查诉求。取值：none、start、provide_info、
    # related_question、conditional_return、cancel、switch_to_troubleshooting、human。
    return_flow_action: str = "none"
    relevant: bool = False
    reason: str = ""


def _contains_any(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


_RETURN_WORDS = (
    "退货", "退款", "退掉", "退了", "退回", "申请退", "退了吧", "退掉吧",
    "return", "refund",
)
_HUMAN_WORDS = (
    "人工", "真人", "转接", "找人工", "human", "live agent", "manager", "经理",
)
_PAIRING_WORDS = (
    "蓝牙", "配对", "连不上", "连接不上", "无法连接", "无法发现", "发现不了",
    "找不到", "搜不到", "重新连接", "bluetooth", "pair", "connect",
)
_EXPLICIT_PRODUCT_WORDS = (
    "无线耳机", "蓝牙耳机", "wireless earbuds", "earbuds", "9924732112",
)
_BARE_PRODUCT_WORDS = ("耳机", "earbuds")
_AMBIGUOUS_MULTI_WORDS = (
    "两副", "两款", "两个耳机", "两台耳机", "哪一副", "哪一款", "哪一个耳机",
    "多副", "分别退", "退哪",
)
_PRODUCT_ID_RE = re.compile(r"(?<!\d)(\d{10})(?!\d)")
_ORDER_ID_RE = re.compile(r"(?<![a-z0-9])w\d{7}(?![0-9])", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")
_OTHER_PRODUCT_WORDS = (
    "音箱", "键盘", "鼠标", "手表", "平板", "投影仪", "打印机",
)
_UNSUPPORTED_ISSUE_WORDS = (
    "没声音", "没有声音", "音质", "电池", "续航", "固件", "驱动", "维修",
    "换货", "保修", "兼容", "损坏", "充不进电", "开不了机",
)
_NEGATIVE_RETURN_WORDS = (
    "不退货", "不退", "先不退", "不想退", "不用退", "不要退", "不申请退",
    "不是要退", "别退", "先别退",
)
_DIRECT_CHOICE_WORDS = (
    "那就退了吧", "还是退了吧", "直接退", "现在就退", "帮我退", "我要退",
    "申请退货", "办理退货", "退了吧",
)
_REFUND_DESTINATION_WORDS = (
    "原支付", "原付款", "原路退回", "原路退", "原卡", "原账户", "原帐户",
    "礼品卡", "gift card", "original payment",
)
_CANCEL_RETURN_WORDS = (
    "取消退货", "取消退款", "撤销退货", "撤回退货", "取消申请", "不退了",
    "暂时不退", "先不退", "先别退", "不申请退", "不是要退",
)
_SWITCH_TO_TROUBLESHOOTING_WORDS = (
    "继续排查", "继续排障", "回去排查", "回到排查", "先排查", "先继续排查",
    "先继续检查", "先继续", "先修", "先解决", "恢复排查",
)
_CONFIRMATION_WORDS = (
    "确认", "提交", "同意", "就按", "按这个", "按这些", "按上述", "没问题",
)
_RETURN_FLOW_STATUSES = (
    "idle", "active", "awaiting_confirmation", "submitted", "cancelled",
    "switched", "handed_off",
)
_RETURN_FLOW_AWAITING = (
    "", "refund_destination", "return_information", "confirmation",
)
_RESOLVED_WORDS = (
    "解决了", "已解决", "好了", "可以了", "连上了", "配对成功", "成功了",
    "正常了", "没问题了", "搞定了",
)
_UNRESOLVED_WORDS = (
    "还是不行", "仍然不行", "还是连不上", "还是找不到", "没解决", "没用",
    "无效", "失败", "不行", "无法连接", "无法发现", "搜不到",
)

_STEP_MARKERS = {
    "PAIR-01": (
        "有电", "充满", "充好", "开机", "蓝牙已开", "打开了蓝牙", "蓝牙开着",
    ),
    "PAIR-02": (
        "断开手机", "断开其他", "关了手机蓝牙", "关闭手机蓝牙", "其他设备",
        "另一台设备", "另一个设备", "别的设备", "平板也连着",
    ),
    "PAIR-03": (
        "重新配对过", "已重新配对", "已经重新配对", "重新配对后", "重新配对仍",
        "重新配对还是", "试过重新配对", "忽略过设备", "重新搜索过",
    ),
}


def new_support_record(turn: int = 0) -> Dict[str, Any]:
    """创建独立格式版本的空记录；不包含任何业务授权字段。"""
    return {
        "format_version": SUPPORT_FORMAT_VERSION,
        "product_id": SUPPORT_PRODUCT_ID,
        "product_name": SUPPORT_PRODUCT_NAME,
        "issue": SUPPORT_ISSUE,
        "stage": "troubleshooting",
        "source_turn": int(turn),
        "last_turn": int(turn),
        "feedback": [],
        "suggestions": [],
        "taken_steps": [],
        "retrieved_evidence": [],
        "unresolved": [],
        "demand": {},
        "return_flow": new_return_flow(),
        "visible_history": [],
        "limit_flags": [],
        "limit_detail": {},
        "handoff": None,
    }


def _copy_record(record: Optional[Dict[str, Any]], turn: int) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return new_support_record(turn)
    copied = json.loads(json.dumps(record, ensure_ascii=False))
    copied.setdefault("format_version", SUPPORT_FORMAT_VERSION)
    copied.setdefault("product_id", SUPPORT_PRODUCT_ID)
    copied.setdefault("product_name", SUPPORT_PRODUCT_NAME)
    copied.setdefault("issue", SUPPORT_ISSUE)
    copied.setdefault("stage", "troubleshooting")
    copied.setdefault("source_turn", int(turn))
    for key in (
        "feedback", "suggestions", "taken_steps", "retrieved_evidence",
        "unresolved", "visible_history", "limit_flags",
    ):
        if not isinstance(copied.get(key), list):
            copied[key] = []
    if not isinstance(copied.get("demand"), dict):
        copied["demand"] = {}
    if not isinstance(copied.get("limit_detail"), dict):
        copied["limit_detail"] = {}
    copied["return_flow"] = _normalize_return_flow(copied)
    copied["last_turn"] = int(turn)
    return copied


def _step_hits(message: str) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []
    for step_id, markers in _STEP_MARKERS.items():
        if _contains_any(message, markers):
            hits.append(
                {
                    "step_id": step_id,
                    "source": "user_report",
                    "quote": message[:160],
                }
            )
    return hits


def _has_return_choice(message: str) -> bool:
    return _contains_any(message, _RETURN_WORDS)


def _is_conditional_return(message: str) -> bool:
    """条件退货只在满足条件时成立，不能当作当前提交授权。"""
    if not _has_return_choice(message):
        return False
    # 只有真实的假设连接词或“不行就”句式才是条件。缺少“我要退”等短语
    # 本身不构成条件；例如“退回原支付方式”是补充已进行中的退货信息。
    return _contains_any(
        message,
        ("如果", "要是", "万一", "假如", "解决不了", "搞不定",
         "不行就", "不行再", "再不行"),
    )


def new_return_flow() -> Dict[str, Any]:
    """退货流程的显式状态；只记录任务阶段，不构成身份或提交授权。"""
    return {
        "status": "idle",
        "awaiting": "",
        "start_turn": 0,
        "last_turn": 0,
        "last_action": "",
        "closed_reason": "",
    }


def _normalize_return_flow(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """读取旧记录时补默认值；旧 stage=return_requested 视为进行中的退货。"""
    source = record.get("return_flow") if isinstance(record, dict) else None
    flow = new_return_flow()
    if isinstance(source, dict):
        status = str(source.get("status") or "idle")
        if status not in _RETURN_FLOW_STATUSES:
            status = "idle"
        awaiting = str(source.get("awaiting") or "")
        if awaiting not in _RETURN_FLOW_AWAITING:
            awaiting = ""
        flow.update(
            {
                "status": status,
                "awaiting": awaiting,
                "start_turn": int(source.get("start_turn") or 0),
                "last_turn": int(source.get("last_turn") or 0),
                "last_action": str(source.get("last_action") or ""),
                "closed_reason": str(source.get("closed_reason") or ""),
            }
        )
    if (
        flow["status"] == "idle"
        and isinstance(record, dict)
        and str(record.get("stage") or "") == "return_requested"
    ):
        # 旧格式没有 return_flow；沿用既有阶段，保证重启后仍能续接。
        flow["status"] = "active"
    return flow


def _return_flow_active(flow: Dict[str, Any]) -> bool:
    return str(flow.get("status") or "") in ("active", "awaiting_confirmation")


def asks_refund_destination(text: str) -> bool:
    """识别可见回复中的当前去向询问；业务记录与固定脚本共用同一口径。"""
    destination = r"(?:退款去向|退款方式|退款路径|退款渠道|原支付|原付款|原路|礼品卡|退回哪里|去向)"
    request_prefix = (
        r"(?:请|麻烦|烦请)[^，,；;。！？?\n]{0,8}"
        r"(?:选择|选|告知|告诉|指定|提供|确认|二选一)"
        r"|(?:还需(?:要)?|需要)[您你]?[^，,；;。！？?\n]{0,8}(?:指定|选择|选|提供)"
    )
    request = (
        rf"(?:{request_prefix})[^，,；;。！？?\n]{{0,18}}{destination}"
        rf"|{destination}[^。！？?\n]{{0,25}}(?:二选一|还是|哪一项|哪一种|哪种|哪个|哪一个|请选择|请告知[。！？?]?$)"
        r"|(?:希望|想)[^。！？?\n]{0,25}(?:退回|退到|退款|原支付|原付款|礼品卡|原路)"
        r"|(?:退款|退回|退到)[^。！？?\n]{0,12}(?:哪里|哪儿|何处|什么账户)"
    )
    clauses = re.findall(r"[^。！？?；;，,\n]+[。！？?；;，,]?", str(text or ""))
    current = [
        clause for clause in clauses
        if not re.search(r"无需|不用|不需要|稍后|之后|以后|已选择|已经选择|已确认", clause)
    ]
    if any(re.search(request, clause) for clause in current):
        return True
    # 列表式选项与“请告知选择哪一项”常分行，允许同一回复的明确选项指代。
    return any(re.search(destination, clause) for clause in current) and any(
        re.search(
            r"(?:请|麻烦|烦请|告知|告诉我).{0,15}(?:选择|选|决定).{0,8}(?:哪|一项|一个|一种|[12一二①②])"
            r"|[您你].{0,8}倾向.{0,8}哪", clause,
        )
        for clause in current
        if not re.search(r"商品|订单|数量|颜色|尺码|型号|邮箱|姓名|邮编", clause)
    )


def _assistant_asks_refund_destination(text: str) -> bool:
    return asks_refund_destination(text)


def _assistant_asks_return_information(text: str) -> bool:
    return asks_return_information(text)


def asks_return_information(text: str) -> bool:
    """只认当前索取身份或目标的信息，不把正文回顾与确认提醒交叉匹配。"""
    return any(
        re.search(
            r"(?:请(?:您|你)?(?:提供|告知|补充|确认|核对)|告诉我|请问(?:您的|你的)?|需要[您你](?:提供|补充|告知))"
            r"[^。！？?\n]{0,45}(?:邮箱|订单号|订单|商品|item_id|姓名|邮编)", clause,
        )
        for clause in re.split(r"[。！？?\n]", str(text or ""))
        if not re.search(r"无需|不用|不需要|稍后|之后|以后", clause)
    )


def _looks_like_related_pairing_question(message: str) -> bool:
    if not _contains_any(message, _PAIRING_WORDS):
        return False
    return _contains_any(
        message, ("?", "？", "怎么", "如何", "为什么", "能不能", "是否", "可以吗", "吗")
    )


def _looks_like_independent_question(message: str) -> bool:
    """识别与当前待补信息无关的独立问题，避免被误当作补充回答。"""
    return _contains_any(
        message,
        (
            "什么时候", "多久", "怎么还", "为什么", "在哪", "哪里",
            "如何查询", "能不能查", "可以查", "查一下", "帮我查",
        ),
    )


def analyze_message(
    message: str,
    record: Optional[Dict[str, Any]] = None,
    bound_user_id: Optional[str] = None,
    has_pending: bool = False,
) -> SupportAnalysis:
    """从真实用户消息和既有记录识别有限支持范围；不调用模型。"""
    text = str(message or "").strip()
    lower = text.lower()
    result = SupportAnalysis(demand_text=text[:300])

    explicit = _contains_any(lower, _EXPLICIT_PRODUCT_WORDS)
    bare = _contains_any(lower, _BARE_PRODUCT_WORDS)
    pairing = _contains_any(lower, _PAIRING_WORDS)
    other_product = _contains_any(lower, _OTHER_PRODUCT_WORDS)
    unsupported_issue = _contains_any(lower, _UNSUPPORTED_ISSUE_WORDS)
    ambiguous_multi = _contains_any(lower, _AMBIGUOUS_MULTI_WORDS)
    product_ids = set(_PRODUCT_ID_RE.findall(lower))
    foreign_product_id = bool(product_ids - {SUPPORT_PRODUCT_ID})
    human = _contains_any(lower, _HUMAN_WORDS)
    return_choice = _has_return_choice(lower)
    negative_return = _contains_any(lower, _NEGATIVE_RETURN_WORDS)
    conditional = _is_conditional_return(lower)
    resolved = _contains_any(lower, _RESOLVED_WORDS)
    unresolved = _contains_any(lower, _UNRESOLVED_WORDS)
    tried = _step_hits(text)
    destination = _contains_any(lower, _REFUND_DESTINATION_WORDS)
    cancel_return = _contains_any(lower, _CANCEL_RETURN_WORDS)
    switch_troubleshooting = _contains_any(
        lower, _SWITCH_TO_TROUBLESHOOTING_WORDS
    )
    has_record = bool(
        isinstance(record, dict)
        and str(record.get("product_id") or "") == SUPPORT_PRODUCT_ID
    )
    flow = _normalize_return_flow(record) if has_record else new_return_flow()
    flow_active = bool(has_record and _return_flow_active(flow))
    identity_reference = bool(
        _ORDER_ID_RE.search(lower)
        or _EMAIL_RE.search(text)
        or _LONG_NUMBER_RE.search(lower)
    )
    # 进行中的退货任务优先于本轮关键词；退出必须由取消、转排查或转人工明确触发。
    if human:
        flow_action = "human"
    elif negative_return or cancel_return:
        flow_action = (
            "switch_to_troubleshooting" if switch_troubleshooting else "cancel"
        )
    elif switch_troubleshooting and flow_active:
        flow_action = "switch_to_troubleshooting"
    elif conditional:
        flow_action = "conditional_return"
    elif flow_active:
        if has_pending and _contains_any(lower, _CONFIRMATION_WORDS):
            flow_action = "none"
        elif destination or return_choice or identity_reference:
            flow_action = "provide_info"
        elif _looks_like_related_pairing_question(lower):
            flow_action = "related_question"
        elif (
            flow.get("awaiting") in ("refund_destination", "return_information")
            and len(text) <= 120
            and not _looks_like_independent_question(lower)
        ):
            # 等待必要信息时，短回答视为当前退货任务的补充或澄清，
            # 由账单角色继续处理；是否提交只由服务端确认规则决定。
            flow_action = "provide_info"
        else:
            flow_action = "none"
    elif return_choice:
        flow_action = "start"
    else:
        flow_action = "none"

    # 显式等待信息或本轮提供了退货信息才算流程续接；仅凭旧 stage
    # 不能把无关的独立问题继续注入退货上下文。
    flow_continuation = bool(
        flow_active
        and (
            flow.get("awaiting")
            in ("refund_destination", "return_information", "confirmation")
            or flow_action in ("provide_info", "related_question")
        )
    )
    # 只有带明确续接线索的消息才复用旧支持记录，避免把无关订单问题误注入。
    continuation = bool(
        has_record
        and (
            flow_continuation
            or return_choice
            or human
            or resolved
            or unresolved
            or tried
            or has_pending
            or negative_return
            or identity_reference
            or pairing
        )
    )
    existing_active = bool(has_record and continuation)

    result.product_explicit = explicit
    result.product_mentioned = explicit or bare
    result.human_request = human
    result.return_negative = negative_return
    result.return_flow_action = flow_action
    result.tried_steps = tried

    if ambiguous_multi:
        result.scope_state = "ambiguous"
        result.reason = "multiple_earbud_or_return_target_candidates"
    elif explicit and foreign_product_id:
        result.scope_state = "out_of_scope"
        result.reason = "unsupported_product_id"
    elif explicit and other_product:
        result.scope_state = "out_of_scope"
        result.reason = "other_product"
    elif explicit and unsupported_issue:
        result.scope_state = "out_of_scope"
        result.reason = "unsupported_issue"
    elif explicit and pairing:
        result.scope_state = "active"
        result.issue = SUPPORT_ISSUE
        result.reason = "explicit_product_and_pairing_issue"
    elif explicit and (return_choice or human):
        # 直接退货或人工不强制先排查；商品明确时仍可带有限工作记录。
        result.scope_state = "active"
        result.issue = SUPPORT_ISSUE if pairing else ""
        result.reason = "explicit_product_with_return_or_handoff"
    elif bare and not explicit and not existing_active:
        result.scope_state = "ambiguous"
        result.reason = "bare_earbuds_requires_product_and_issue_clarification"
    elif other_product:
        result.scope_state = "out_of_scope"
        result.reason = "other_product"
    elif unsupported_issue:
        result.scope_state = "out_of_scope"
        result.reason = "unsupported_issue"
    elif pairing and not existing_active:
        result.scope_state = "ambiguous"
        result.reason = "pairing_issue_without_explicit_supported_product"
    elif existing_active:
        result.scope_state = "active"
        result.issue = str(record.get("issue") or SUPPORT_ISSUE)
        result.reason = "existing_support_record_continuation"
    else:
        result.scope_state = "inactive"
        result.reason = "no_support_scope"

    # 续接消息可以不带商品名，但不能把无关订单问题误注入支持记录。
    if has_pending and existing_active:
        result.scope_state = "active"
        result.issue = str(record.get("issue") or SUPPORT_ISSUE)
        result.reason = "pending_confirmation_continuation"

    if human:
        result.demand_kind = "human"
    elif flow_action in ("cancel", "switch_to_troubleshooting"):
        result.demand_kind = "none"
    elif flow_action == "conditional_return":
        result.demand_kind = "conditional_return"
    elif flow_active and flow_action in ("provide_info", "none"):
        # 本轮没有新的独立诉求；保留旧 demand.kind=return，表示任务仍在进行。
        result.demand_kind = "none"
    elif flow_action == "related_question":
        result.demand_kind = "troubleshooting"
    elif return_choice:
        result.demand_kind = "return"
    elif resolved and not unresolved:
        result.demand_kind = "resolved"
    elif pairing or unresolved or tried:
        result.demand_kind = "troubleshooting"

    if resolved and not unresolved:
        result.feedback_kind = "resolved"
    elif unresolved:
        result.feedback_kind = "unresolved"
    elif tried:
        result.feedback_kind = "choice"
    result.feedback_text = text[:500] if result.feedback_kind != "none" else ""

    result.relevant = result.scope_state in ("active", "ambiguous", "out_of_scope")
    if not bound_user_id:
        # 公开咨询不绑定身份；该字段仅用于让角色知道服务器状态。
        result.relevant = result.relevant or False
    return result


def _truncate(text: Any, limit: int) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[: max(0, limit - 1)] + "…"


def _dedupe(items: List[Dict[str, Any]], keys: Tuple[str, ...]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = tuple(str(item.get(name) or "") for name in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _displayed_text_match(text: str, item: Dict[str, Any]) -> bool:
    evidence_id = str(item.get("evidence_id") or "")
    step_id = str(item.get("step_id") or "")
    title = str(item.get("title") or "")
    content = str(item.get("content") or "")
    if evidence_id and evidence_id in text:
        return True
    if step_id and step_id in text:
        return True
    if title and title in text:
        return True
    prefix = content[:24]
    return bool(prefix and prefix in text)


def _search_items(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for trace in traces or []:
        if str(trace.get("tool_name") or "") != "search_knowledge_base":
            continue
        if trace.get("result_success") is False or trace.get("success") is False:
            continue
        result = trace.get("result")
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        if result.get("degraded") is True:
            continue
        for item in result.get("results") or []:
            if isinstance(item, dict):
                items.append(item)
    return items


def update_support_record(
    record: Optional[Dict[str, Any]],
    analysis: SupportAnalysis,
    message: str,
    turn: int,
    assistant_text: str,
    tool_traces: List[Dict[str, Any]],
    pending_after: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """在最终展示文本确定后更新工作记录；检索不等于展示。"""
    updated = _copy_record(record, turn)
    if analysis.scope_state == "active":
        if analysis.product_explicit:
            updated["product_id"] = SUPPORT_PRODUCT_ID
            updated["product_name"] = SUPPORT_PRODUCT_NAME
        if analysis.issue:
            updated["issue"] = analysis.issue
    elif analysis.scope_state in ("ambiguous", "out_of_scope"):
        # 不建立支持事项，避免把不适用请求写成耳机排查进度。
        return updated

    if updated["source_turn"] <= 0:
        updated["source_turn"] = int(turn)

    if analysis.feedback_kind != "none":
        updated["feedback"].append(
            {
                "turn": int(turn),
                "kind": analysis.feedback_kind,
                "text": analysis.feedback_text,
                "source": "user_message",
            }
        )
    for step in analysis.tried_steps:
        updated["taken_steps"].append({"turn": int(turn), **step})
    updated["taken_steps"] = _dedupe(updated["taken_steps"], ("step_id",))

    # 用户明确否定或暂缓退货时，也要更新当前诉求，避免旧的条件退货继续生效。
    if analysis.demand_kind != "none" or analysis.return_negative:
        updated["demand"] = {
            "kind": analysis.demand_kind,
            "text": analysis.demand_text,
            "turn": int(turn),
            "source": "user_message",
        }

    displayed_steps: List[str] = []
    retrieved: List[Dict[str, Any]] = []
    for item in _search_items(tool_traces):
        product_id = str(item.get("product_id") or "")
        issue = str(item.get("issue") or "")
        if product_id != SUPPORT_PRODUCT_ID or issue != SUPPORT_ISSUE:
            continue
        step_id = str(item.get("step_id") or "")
        displayed = _displayed_text_match(str(assistant_text or ""), item)
        evidence = {
            "evidence_id": str(item.get("evidence_id") or item.get("id") or ""),
            "step_id": step_id,
            "title": _truncate(item.get("title"), 80),
            "displayed": displayed,
            "source_kind": str(item.get("source_kind") or ""),
            "turn": int(turn),
        }
        retrieved.append(evidence)
        if not displayed:
            continue
        displayed_steps.append(step_id)
        updated["suggestions"].append(
            {
                "turn": int(turn),
                "step_id": step_id,
                "evidence_id": evidence["evidence_id"],
                "title": evidence["title"],
                "text": _truncate(item.get("content"), 300),
                "source": "displayed_assistant_text",
                "previously_tried": step_id in {
                    str(step.get("step_id") or "")
                    for step in updated["taken_steps"]
                },
            }
        )
    updated["retrieved_evidence"] = _dedupe(
        updated["retrieved_evidence"] + retrieved, ("turn", "evidence_id")
    )
    updated["suggestions"] = _dedupe(
        updated["suggestions"], ("turn", "evidence_id")
    )

    handoff_trace = any(
        str(trace.get("tool_name") or "") == "transfer_to_human_agents"
        and trace.get("result_success") is True
        for trace in tool_traces or []
    )
    submitted_trace = any(
        str(trace.get("tool_name") or "") == "return_delivered_order_items"
        and trace.get("result_success") is True
        for trace in tool_traces or []
    )
    pending = list(pending_after or [])
    active_pending = [
        item
        for item in pending
        if str(item.get("status") or "") in ("prepared", "displayed")
    ]
    flow = _normalize_return_flow(updated)
    flow_action = str(analysis.return_flow_action or "none")
    if submitted_trace:
        flow.update(
            status="submitted",
            awaiting="",
            last_action="submitted",
            closed_reason="return_submitted",
        )
    elif handoff_trace or flow_action == "human":
        if _return_flow_active(flow):
            flow.update(status="handed_off", awaiting="")
        if flow["status"] == "handed_off":
            flow.update(
                last_action="human", closed_reason="user_requested_human"
            )
    elif flow_action == "cancel":
        if _return_flow_active(flow) or flow["status"] == "idle":
            flow.update(
                status="cancelled",
                awaiting="",
                last_action="cancel",
                closed_reason="user_cancelled",
            )
    elif flow_action == "switch_to_troubleshooting":
        if _return_flow_active(flow) or flow["status"] == "idle":
            flow.update(
                status="switched",
                awaiting="",
                last_action="switch_to_troubleshooting",
                closed_reason="user_switched_to_troubleshooting",
            )
    elif flow_action == "start":
        if not _return_flow_active(flow):
            flow["start_turn"] = int(turn)
        flow.update(
            status="active",
            awaiting="return_information",
            last_action="start",
            closed_reason="",
        )
    elif flow_action in ("provide_info", "related_question", "none"):
        if _return_flow_active(flow):
            flow["last_action"] = flow_action
    if active_pending and _return_flow_active(flow):
        flow.update(
            status="awaiting_confirmation",
            awaiting="confirmation",
            last_action="displayed",
        )
    elif _return_flow_active(flow):
        if _assistant_asks_refund_destination(assistant_text):
            flow["awaiting"] = "refund_destination"
        elif _assistant_asks_return_information(assistant_text):
            flow["awaiting"] = "return_information"
    if flow["status"] != "idle" and flow_action != "none":
        flow["last_turn"] = int(turn)
    updated["return_flow"] = flow

    if handoff_trace:
        updated["handoff"] = {
            "status": "recorded_mock",
            "turn": int(turn),
            "note": "未创建真实工单或连接真实坐席",
        }
        updated["stage"] = "handoff_recorded"
    elif flow["status"] in ("cancelled", "switched"):
        updated["stage"] = (
            "resolved"
            if analysis.feedback_kind == "resolved"
            else "troubleshooting"
        )
    elif (
        flow["status"] in ("active", "awaiting_confirmation", "submitted")
        or analysis.demand_kind == "return"
        or submitted_trace
    ):
        updated["stage"] = "return_requested"
    elif analysis.demand_kind == "resolved" or analysis.feedback_kind == "resolved":
        updated["stage"] = "resolved"
    elif (
        analysis.feedback_kind == "unresolved"
        or displayed_steps
        or any(item.get("displayed") for item in retrieved)
    ):
        updated["stage"] = "awaiting_feedback"
    elif analysis.scope_state == "active":
        updated["stage"] = "troubleshooting"

    unresolved: List[str] = []
    if updated["stage"] == "resolved":
        unresolved = []
    elif updated["stage"] == "handoff_recorded":
        unresolved = ["已记录模拟人工交接；未创建真实工单"]
    elif updated["stage"] == "return_requested":
        if any(item.get("status") == "displayed" for item in pending):
            unresolved.append("退货详情已展示，等待用户明确确认")
        elif submitted_trace:
            unresolved.append("退货申请已提交（return requested），不代表退款到账")
        elif flow["status"] == "awaiting_confirmation":
            unresolved.append("退货详情已展示，等待用户明确确认")
        elif flow.get("awaiting") == "refund_destination":
            unresolved.append("退货流程进行中，等待用户补充退款去向")
        elif flow.get("awaiting") == "return_information":
            unresolved.append("退货流程进行中，等待用户补充身份、订单或商品信息")
        else:
            unresolved.append("用户已表达退货诉求，需按原业务工具核对身份、订单和详情")
    elif analysis.demand_kind == "conditional_return":
        unresolved.append("用户提出条件退货：若排查无效再选择退货；当前未授权提交")
    if flow["status"] == "cancelled":
        unresolved.append("用户取消当前退货流程，未执行提交")
    elif flow["status"] == "switched":
        unresolved.append("用户要求暂停退货并继续排查，未执行提交")
    elif flow["status"] == "handed_off" and not handoff_trace:
        unresolved.append("用户请求人工；当前未见实际交接事件")
    if analysis.feedback_kind == "unresolved" and updated["stage"] != "resolved":
        unresolved.append("蓝牙配对问题仍未解决")
    if displayed_steps and updated["stage"] in ("awaiting_feedback", "troubleshooting"):
        unresolved.append("已展示有限建议，等待用户反馈结果")
    if analysis.return_negative:
        unresolved.append("用户当前否定或暂缓退货；未执行提交")
    if analysis.scope_state == "ambiguous":
        unresolved.append("商品或问题不明确，需先澄清")
    if analysis.scope_state == "out_of_scope":
        unresolved.append("请求超出本次沙箱资料范围，资料不足以核实")
    updated["unresolved"] = list(dict.fromkeys(unresolved))[:8]

    updated["visible_history"].append(
        {
            "turn": int(turn),
            "user": _truncate(message, 500),
            "assistant": _truncate(assistant_text, 800),
        }
    )
    flags = set(str(item) for item in updated.get("limit_flags") or [])
    detail = dict(updated.get("limit_detail") or {})
    if len(updated["visible_history"]) > MAX_VISIBLE_PAIRS:
        dropped = len(updated["visible_history"]) - MAX_VISIBLE_PAIRS
        updated["visible_history"] = updated["visible_history"][-MAX_VISIBLE_PAIRS:]
        flags.add(SUPPORT_CONTEXT_LIMIT)
        detail["visible_pairs_dropped"] = int(detail.get("visible_pairs_dropped", 0)) + dropped
    if len(updated["feedback"]) > MAX_FEEDBACK:
        dropped = len(updated["feedback"]) - MAX_FEEDBACK
        updated["feedback"] = updated["feedback"][-MAX_FEEDBACK:]
        flags.add(SUPPORT_CONTEXT_LIMIT)
        detail["feedback_dropped"] = int(detail.get("feedback_dropped", 0)) + dropped
    if len(updated["suggestions"]) > MAX_SUGGESTIONS:
        dropped = len(updated["suggestions"]) - MAX_SUGGESTIONS
        updated["suggestions"] = updated["suggestions"][-MAX_SUGGESTIONS:]
        flags.add(SUPPORT_CONTEXT_LIMIT)
        detail["suggestions_dropped"] = int(detail.get("suggestions_dropped", 0)) + dropped
    if len(updated["retrieved_evidence"]) > MAX_SUGGESTIONS * 2:
        updated["retrieved_evidence"] = updated["retrieved_evidence"][-MAX_SUGGESTIONS * 2:]
    updated["limit_flags"] = sorted(flags)
    updated["limit_detail"] = detail
    updated["last_turn"] = int(turn)
    return updated


def _json_size(value: Any) -> int:
    """返回紧凑 JSON 的实际字符数，供上下文上限判断。"""
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def record_view(record: Optional[Dict[str, Any]], max_chars: int = 3000) -> Optional[Dict[str, Any]]:
    """给角色输入的有限视图；保留当前诉求、已做步骤和待确认事实。"""
    if not isinstance(record, dict):
        return None
    view = _copy_record(record, int(record.get("last_turn") or 0))
    view["visible_history"] = list(view.get("visible_history") or [])[-6:]
    view["feedback"] = list(view.get("feedback") or [])[-6:]
    view["suggestions"] = list(view.get("suggestions") or [])[-6:]
    view["retrieved_evidence"] = list(view.get("retrieved_evidence") or [])[-8:]
    if _json_size(view) <= max_chars:
        return view
    # 先裁可见历史，再裁较早的反馈和建议；当前诉求、已做步骤和未知项不删。
    view["visible_history"] = []
    if _json_size(view) <= max_chars:
        return view
    view["suggestions"] = view["suggestions"][-3:]
    view["feedback"] = view["feedback"][-3:]
    view["retrieved_evidence"] = view["retrieved_evidence"][-4:]
    if _json_size(view) <= max_chars:
        return view
    view["suggestions"] = view["suggestions"][-1:]
    view["feedback"] = view["feedback"][-1:]
    view["retrieved_evidence"] = []
    return view


def build_support_packet(
    record: Optional[Dict[str, Any]],
    analysis: SupportAnalysis,
    message: str,
    turn: int,
    bound_user_id: Optional[str],
    pending: Optional[List[Dict[str, Any]]],
    max_chars: int = 6000,
) -> Dict[str, Any]:
    """构造交给角色输入包的支持上下文；不含测试端预期值。"""
    packet = {
        "source": "host_support_context",
        "enabled": True,
        "scope_state": analysis.scope_state,
        "supported": analysis.scope_state == "active",
        "product_id": (
            SUPPORT_PRODUCT_ID
            if analysis.product_explicit
            or (
                analysis.scope_state == "active"
                and isinstance(record, dict)
                and str(record.get("product_id") or "") == SUPPORT_PRODUCT_ID
            )
            else ""
        ),
        "product_name": SUPPORT_PRODUCT_NAME if analysis.product_explicit else "",
        "issue": analysis.issue,
        "issue_label": SUPPORT_ISSUE_LABEL if analysis.issue == SUPPORT_ISSUE else "",
        "scope_reason": analysis.reason,
        "current_demand": {
            "kind": analysis.demand_kind,
            "text": _truncate(message, 500),
            "turn": int(turn),
            "source": "user_message",
        },
        "return_flow": {
            **_normalize_return_flow(record),
            "action": analysis.return_flow_action,
            "turn": int(turn),
            "source": "host_deterministic_support_analysis",
        },
        "user_feedback": [
            {
                "kind": analysis.feedback_kind,
                "text": _truncate(message, 500),
                "turn": int(turn),
            }
        ] if analysis.feedback_kind != "none" else [],
        "taken_steps": analysis.tried_steps,
        "work_record": record_view(record, max(800, max_chars - 1800)),
        "identity": {
            "source": "server_session",
            "bound": bool(bound_user_id),
            "user_id": bound_user_id,
        },
        "pending_confirmation": [
            {
                "op_id": item.get("op_id"),
                "order_id": item.get("order_id"),
                "item_ids": (item.get("items") or {}).get("item_ids", []),
                "payment_method_id": item.get("payment_method_id"),
                "amount": item.get("amount"),
                "status": item.get("status"),
            }
            for item in (pending or [])
        ],
        "rules": _SCOPE_RULE,
        "role_handoffs": [],
        "last_handoff": None,
        "limits": {
            "visible_pairs": MAX_VISIBLE_PAIRS,
            "feedback": MAX_FEEDBACK,
            "suggestions": MAX_SUGGESTIONS,
            "handoff_chars": min(MAX_HANDOFF_CHARS, max_chars),
        },
    }
    if _json_size(packet) > max_chars:
        packet["work_record"] = record_view(record, 1200)
    return packet


def support_evidence_applicable(item: Any, packet: Optional[Dict[str, Any]]) -> bool:
    """共享知识适配器的适用过滤；旧 29 段可继续作为政策／准备证据。"""
    if not isinstance(item, dict):
        return False
    if item.get("fallback") is True or item.get("degraded") is True:
        return False
    product_id = str(item.get("product_id") or "")
    if not product_id:
        return True
    if not isinstance(packet, dict) or packet.get("enabled") is not True:
        return False
    if packet.get("scope_state") != "active" or packet.get("supported") is not True:
        return False
    if str(packet.get("product_id") or "") != SUPPORT_PRODUCT_ID:
        return False
    if product_id != SUPPORT_PRODUCT_ID:
        return False
    item_issue = str(item.get("issue") or "")
    return not item_issue or item_issue == SUPPORT_ISSUE


def handoff_summary_from_packet(
    packet: Optional[Dict[str, Any]],
    pending: Optional[List[Dict[str, Any]]],
    language: str = "zh",
) -> str:
    """确定性交接摘要；身份最终以 RetailStore 服务端状态修正。"""
    packet = packet if isinstance(packet, dict) else {}
    record = packet.get("work_record") if isinstance(packet.get("work_record"), dict) else {}
    demand = packet.get("current_demand") if isinstance(packet.get("current_demand"), dict) else {}
    identity = packet.get("identity") if isinstance(packet.get("identity"), dict) else {}
    taken = record.get("taken_steps") or []
    suggestions = record.get("suggestions") or []
    feedback = record.get("feedback") or []
    unresolved = record.get("unresolved") or []
    stage = record.get("stage") or packet.get("scope_state") or "unknown"
    flow = _normalize_return_flow(record)
    pending_items = list(pending or packet.get("pending_confirmation") or [])
    if str(language).lower() == "en":
        return _truncate(
            "Retail support handoff. Product: Wireless Earbuds (9924732112); "
            f"issue: Bluetooth discovery/pairing/connection; stage: {stage}; "
            f"current demand: {demand.get('text', '')}; "
            f"tried: {', '.join(str(item.get('step_id')) for item in taken) or 'none'}; "
            f"suggested: {', '.join(str(item.get('step_id')) for item in suggestions) or 'none'}; "
            f"feedback: {feedback[-1].get('text', '') if feedback else 'none'}; "
            f"unresolved: {'; '.join(unresolved) or 'none'}; "
            f"identity bound by server: {bool(identity.get('bound'))}; "
            f"pending not submitted: {len(pending_items)}. "
            "No real support ticket or live agent was created.",
            MAX_HANDOFF_CHARS,
        )
    lines = [
        "零售支持交接摘要（模拟）",
        f"商品：{SUPPORT_PRODUCT_NAME}（{SUPPORT_PRODUCT_ID}），问题：{SUPPORT_ISSUE_LABEL}",
        f"当前阶段：{stage}",
        f"用户当前诉求：{demand.get('text') or '未明确'}",
        "已尝试（用户报告）：" + (
            "；".join(
                f"{item.get('step_id')}（轮次 {item.get('turn')}）"
                for item in taken
            ) or "无记录"
        ),
        "已展示建议及证据：" + (
            "；".join(
                f"{item.get('step_id') or '范围说明'}／{item.get('evidence_id')}（轮次 {item.get('turn')}）"
                for item in suggestions
            ) or "无记录"
        ),
        "用户反馈：" + (
            f"{feedback[-1].get('text', '')}（轮次 {feedback[-1].get('turn')}）"
            if feedback else "无记录"
        ),
        "未解决事项：" + ("；".join(str(item) for item in unresolved) or "无记录"),
        "退货流程：" + (
            f"{flow.get('status')}；等待：{flow.get('awaiting') or '无'}"
            if flow.get("status") != "idle" else "无"
        ),
        "身份状态：以服务端绑定为准（"
        + ("已绑定" if identity.get("bound") else "未绑定")
        + "）",
        "待确认退货：" + (
            f"{len(pending_items)} 条已展示详情尚未提交"
            if pending_items else "无"
        ),
        "说明：只记录模拟交接；未连接真实坐席，未创建真实工单，未执行退货写入。",
    ]
    return _truncate("\n".join(lines), MAX_HANDOFF_CHARS)


def merge_role_handoff(
    packet: Optional[Dict[str, Any]],
    source_role: str,
    target_role: str,
    turn: int,
    response: Any,
) -> Optional[Dict[str, Any]]:
    """把辅助角色实际结果显式注入目标角色输入，保留来源和摘要 hash。"""
    if not isinstance(packet, dict) or packet.get("enabled") is not True:
        return packet
    content = str(getattr(response, "content", "") or "")
    traces = list(getattr(response, "tool_traces", []) or [])
    evidence_ids: List[str] = []
    step_ids: List[str] = []
    for item in _search_items(traces):
        if str(item.get("product_id") or "") != SUPPORT_PRODUCT_ID:
            continue
        evidence_id = str(item.get("evidence_id") or "")
        step_id = str(item.get("step_id") or "")
        if evidence_id:
            evidence_ids.append(evidence_id)
        if step_id:
            step_ids.append(step_id)
    handoff = {
        "source_role": str(source_role),
        "target_role": str(target_role),
        "source_turn": int(turn),
        "target_turn": int(turn),
        "summary": _truncate(content, 1200),
        "evidence_ids": list(dict.fromkeys(evidence_ids))[:8],
        "step_ids": list(dict.fromkeys(step_ids))[:8],
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "source": "actual_supporting_role_response",
    }
    handoffs = list(packet.get("role_handoffs") or [])
    handoffs.append(handoff)
    packet["role_handoffs"] = handoffs[-2:]
    packet["last_handoff"] = handoff
    return packet


def support_error(code: str, detail: str = "") -> str:
    return f"{code}: {detail}".strip(": ")


def is_placeholder_or_fallback(item: Any) -> bool:
    """供测试与适配器复用的降级判断。"""
    return isinstance(item, dict) and bool(
        item.get("fallback") is True
        or item.get("degraded") is True
        or "知识库降级结果" in str(item.get("title") or "")
    )


def looks_dependent_continuation(message: str, has_pending: bool) -> bool:
    """支持记录读取失败时，判断消息是否依赖前序支持状态。"""
    if has_pending:
        return True
    text = str(message or "").strip().lower()
    if len(text) <= 8:
        return True
    return _contains_any(
        text,
        (
            "还是", "那就", "接着", "继续", "刚才", "上面", "这个", "它",
            "退了", "退货", "人工", "确认", "好了", "不行", "解决",
        ),
    )


def normalize_for_hash(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "SUPPORT_FORMAT_VERSION",
    "SUPPORT_PRODUCT_ID",
    "SUPPORT_PRODUCT_NAME",
    "SUPPORT_ISSUE",
    "SUPPORT_ISSUE_LABEL",
    "SUPPORT_STEP_IDS",
    "SUPPORT_EVIDENCE_IDS",
    "MAX_VISIBLE_PAIRS",
    "MAX_FEEDBACK",
    "MAX_SUGGESTIONS",
    "MAX_HANDOFF_CHARS",
    "SUPPORT_CONTEXT_LIMIT",
    "SUPPORT_STATE_ERROR",
    "SupportAnalysis",
    "new_support_record",
    "analyze_message",
    "update_support_record",
    "build_support_packet",
    "record_view",
    "support_evidence_applicable",
    "handoff_summary_from_packet",
    "merge_role_handoff",
    "support_error",
    "is_placeholder_or_fallback",
    "looks_dependent_continuation",
    "normalize_for_hash",
]
