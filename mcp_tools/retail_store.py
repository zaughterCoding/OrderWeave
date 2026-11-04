"""零售沙箱的最小 SQLite 业务存储。

只保存公开种子中的用户、订单和商品，以及本项目的会话、待确认详情、
退货操作和事件。业务身份由服务器按会话绑定，模型传入的 user_id 不构成
订单授权。金额使用整数分计算，退货写入与确认消费在同一事务完成。
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import sqlite3
import uuid
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from mcp_tools.retail_domain import WRITE_KINDS, DomainError, apply_write, canonical_args

SCHEMA_VERSION = "1"
SUPPORT_STATE_FORMAT_VERSION = 1
MAX_SUPPORT_STATE_BYTES = 256 * 1024
EXPECTED_SEED_SHA256 = "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"
EXPECTED_SEED_COUNTS = {"users": 500, "orders": 1000, "products": 50}

_ORDER_ID_RE = re.compile(r"#?w\d{7}", re.IGNORECASE)
_QUESTION_WORDS = (
    "?",
    "？",
    "吗",
    "是否",
    "能否",
    "能不能",
    "可不可以",
    "怎么",
    "为什么",
    "什么",
    "多少",
    "多久",
    "几天",
    "什么时候",
    "如果",
    "要是",
    "假如",
    "的话",
    "但是",
    "不过",
    "或者",
    "还是",
    "改",
    "换",
    "加退",
    "多退",
    "少退",
    "另一",
    "其他",
    "别的",
)
_NEGATIVE_WORDS = (
    "先别",
    "不要",
    "不用",
    "不确认",
    "不提交",
    "别退",
    "别提交",
    "先不",
    "暂不",
    "暂时不",
    "取消",
    "不了",
    "我再想想",
    "再想想",
    "再考虑",
    "等等",
    "等一下",
    "等会",
    "稍等",
    "not ",
    "no ",
    "cancel",
    "wait",
)
_STRONG_CONFIRMATIONS = (
    "确认",
    "提交",
    "同意",
    "就按",
    "按这个",
    "按这些",
    "按上述",
    "没问题",
    "可以提交",
)
_WEAK_CONFIRMATIONS = {
    "是的",
    "对",
    "对的",
    "好的",
    "好",
    "可以",
    "行",
    "嗯",
    "嗯嗯",
}
_ALL_SCOPE_WORDS = (
    "都",
    "两单",
    "两笔",
    "两个订单",
    "全部",
    "一起",
    "这两单",
    "这两笔",
    "均",
)

# 英文确认按整句判断；否定、疑问、条件、改口和引用优先于肯定词。
_EN_NEGATIVE_RE = re.compile(
    r"\b(?:no|nope|nah|not|don't|dont|do not|doesn't|cancel|wait|hold|stop|"
    r"later|never|refuse|decline|instead|rather)\b"
)
_EN_QUESTION_RE = re.compile(
    r"\b(?:if|whether|unless|assuming|provided|can you|could you|would you|"
    r"will you|are you|is it|does it|do you|why|how|what|when|where|which|"
    r"who|should i|shall i|maybe|perhaps|not sure|unsure|question)\b|[?]"
)
_EN_CHANGE_RE = re.compile(
    r"\b(?:change|changed|switch|swap|different|another|other|add|remove|"
    r"extra|additional|more|less|also|too|as well|first|before|after|then|"
    r"actually|but|however|though)\b"
)
_EN_QUOTE_RE = re.compile(
    r'["“”]|\byou said\b|\byour message\b|\bit says\b|\bthe instruction\b|'
    r"\bthe prompt\b|\b(?:he|she|they|someone|somebody|friend|colleague|"
    r"partner|spouse|manager)\s+said\b|\baccording to\b|"
    r"\b(?:i|we)\s+(?:was|were)\s+told\b"
)
_EN_POSITIVE_RE = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|okay|ok|confirm|confirmed|approve|approved|"
    r"proceed|submit|go ahead|go for it|sounds good|that works|please do|"
    r"do it|correct|absolutely|definitely|handle|process)\b"
)
_EN_WEAK_CONFIRMATIONS = {
    "yes",
    "yeah",
    "yep",
    "yup",
    "sure",
    "ok",
    "okay",
    "correct",
    "confirmed",
    "soundsgood",
    "thatworks",
    "goahead",
    "proceed",
    "pleasedo",
    "submit",
}
_EN_ALL_SCOPE_RE = re.compile(
    r"\b(?:both|all|both orders|all of them|everything|the two orders|"
    r"both returns)\b"
)
_EN_AMOUNT_RE = re.compile(r"(?<!\d)\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)")
_EN_TERMS_ACCEPTANCE_RE = re.compile(
    r"\b(?:works? for me|fine with me|fine by me|okay with me|ok with me|"
    r"i accept|i agree|acceptable|good with me|no problem)\b"
)
_EN_AMOUNT_CONTEXT_RE = re.compile(
    r"\b(?:refund|amount|estimate|estimated|total)\b"
)
_EN_DESTINATION_RE = re.compile(
    r"\b(?:gift card|credit card|debit card|paypal|venmo|bank account|"
    r"checking account|store credit|cash|cheque|check|card)\b"
)


class RetailStoreError(RuntimeError):
    """种子、会话或数据库状态不允许继续。"""


class SupportStateError(RetailStoreError):
    """独立支持记录读写失败；不得因此回放已完成的业务写入。"""


@dataclass(frozen=True)
class ConfirmationDecision:
    approved: bool
    scope: str
    reason: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_cents(value: Any) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as ex:
        raise RetailStoreError(f"无法解析金额: {value!r}") from ex
    if not amount.is_finite():
        raise RetailStoreError(f"金额不是有限数: {value!r}")
    return int((amount * 100).to_integral_value(rounding=ROUND_HALF_UP))


def _fmt_cents(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}{cents // 100}.{cents % 100:02d}"


def _fmt_usd(cents: int) -> str:
    """订单业务金额始终为数据库原始美元，不做汇率换算。"""
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def normalize_order_id(raw: Any) -> str:
    value = str(raw or "").strip().replace("＃", "#")
    if not value:
        return ""
    if value.startswith("#"):
        return "#" + value[1:].strip().upper()
    if value.upper().startswith("W"):
        return "#" + value.upper()
    return value.upper()


def _normalize_text(text: str) -> str:
    table = str.maketrans("，。！？；：", ",.!?;:")
    return str(text or "").translate(table).strip()


def _order_ids_in(text: str) -> List[str]:
    return list(
        dict.fromkeys(
            normalize_order_id(match.group(0)) for match in _ORDER_ID_RE.finditer(text)
        )
    )


def _has_any(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in words)


def _positive_in(clause: str) -> bool:
    if _has_any(clause, _STRONG_CONFIRMATIONS):
        return True
    compact = re.sub(r"[\s,.!;:。！？；：]", "", clause)
    return compact in _WEAK_CONFIRMATIONS


def _clean_clause(clause: str) -> bool:
    """确认子句不得含否定、疑问或改口。"""
    return not _has_any(clause, _NEGATIVE_WORDS) and not _has_any(
        clause, _QUESTION_WORDS
    )


def _clauses(text: str) -> List[str]:
    # 句点和小数点、逗号和千分位都可能是金额的一部分，不能直接按字符切分。
    parts = re.split(
        r"\n|(?<!\d)[,;!?]|[,;!?](?!\d)|(?<!\d)\.(?!\d)|[，。；！？]",
        text,
    )
    return [item.strip() for item in parts if item.strip()]


_HANDOFF_IDENTITY_RE = re.compile(
    r"(?i)(?:\bidentity\b|\bverified\b|\bverification\b|\bauthenticated\b|"
    r"\bbound\b|身份|验证|认证|绑定)"
)


def _handoff_summary_with_identity(
    summary: str,
    retail_user_id: Optional[str],
    language: str,
) -> str:
    """用服务端会话状态重建交接摘要中的身份说明。

    模型摘要中的身份陈述一律丢弃，避免与真实绑定状态并存；句内仅含
    身份内容的括注可单独移除，其余可核对内容保留。未知业务事实不由
    本函数补造。
    """
    sentences: List[str] = []
    for raw in re.split(r"(?<=[.!?。！？;；])\s*|\n+", str(summary or "")):
        item = raw.strip()
        if not item:
            continue
        if _HANDOFF_IDENTITY_RE.search(item):
            # 只删除带身份陈述的括注；同句其他事实保留。剩余文本仍有
            # 身份陈述时整句丢弃，避免矛盾信息残留。
            item = re.sub(
                r"[（(][^()（）]*(?:identity|verified|verification|"
                r"authenticated|bound|身份|验证|认证|绑定)"
                r"[^()（）]*[)）]",
                "",
                item,
                flags=re.IGNORECASE,
            )
            item = re.sub(r"\s{2,}", " ", item).strip()
            if not item or _HANDOFF_IDENTITY_RE.search(item):
                continue
        sentences.append(item)
    if str(language).lower() == "en":
        if retail_user_id:
            identity = (
                "Identity status: the server session is bound to user "
                f"{retail_user_id}."
            )
        else:
            identity = "Identity status: the server session has no bound user."
    else:
        if retail_user_id:
            identity = f"身份状态：服务端会话已绑定用户 {retail_user_id}。"
        else:
            identity = "身份状态：服务端会话尚未绑定用户。"
    return " ".join([identity, *sentences]).strip()[:1200]


def _displayed_amount_acceptance(
    clause: str, pending: Dict[str, Any]
) -> bool:
    """判断子句是否只是在接受已展示的预计金额。

    仅放行带匹配金额、退款/金额语境和接受措辞的子句；金额不一致或提到
    未展示的退款去向时仍拒绝。真正的提交指令仍须由其他肯定子句给出。
    """
    lowered = clause.lower()
    if (
        _EN_NEGATIVE_RE.search(lowered)
        or _EN_QUESTION_RE.search(lowered)
        or _EN_CHANGE_RE.search(lowered)
        or _order_ids_in(clause)
        or not _EN_TERMS_ACCEPTANCE_RE.search(lowered)
        or not _EN_AMOUNT_CONTEXT_RE.search(lowered)
    ):
        return False
    amounts = [
        _to_cents(match.group(1).replace(",", ""))
        for match in _EN_AMOUNT_RE.finditer(lowered)
    ]
    if not amounts or any(
        amount != int(pending.get("amount_cents", -1)) for amount in amounts
    ):
        return False
    display_text = str(pending.get("display_text") or "").lower()
    for match in _EN_DESTINATION_RE.finditer(lowered):
        if match.group(0) not in display_text:
            return False
    return True


def _evaluate_confirmation_en(
    message: str,
    pending: Dict[str, Any],
    displayed: List[Dict[str, Any]],
) -> ConfirmationDecision:
    """英文确认判断；否定、疑问、条件和改口优先否决。"""
    text = _normalize_text(message)
    if not text:
        return ConfirmationDecision(False, "none", "The confirmation is empty")
    # 只消除详情定语中的过去时歧义，转述肯定与provided that仍不得授权。
    text = re.sub(r"\b(details(?:\s+that)?\s+you)\s+provided\b(?!\s+that\b)",
                  r"\1 supplied", text, flags=re.I)
    if len(text) > 400:
        return ConfirmationDecision(
            False, "none", "The confirmation is too long to judge safely"
        )
    if _EN_QUOTE_RE.search(text.lower()):
        return ConfirmationDecision(
            False, "none", "Quoted or reported wording does not confirm"
        )
    if _EN_NEGATIVE_RE.search(text.lower()):
        return ConfirmationDecision(
            False, "none", "The message contains a negative or delaying expression"
        )
    if _EN_QUESTION_RE.search(text.lower()):
        return ConfirmationDecision(
            False, "none", "The message contains a question, condition, or change request"
        )
    if _EN_CHANGE_RE.search(text.lower()):
        return ConfirmationDecision(
            False, "none", "The message contains a change or additional request"
        )

    target = normalize_order_id(pending.get("order_id", ""))
    mentioned = _order_ids_in(text)
    clauses = _clauses(text)

    fillers = {
        "please", "thanks", "thank you", "great", "perfect", "fine", "good",
        "thanks for your help", "thank you for your help",
        "thanks for laying out the details", "thank you for confirming the details",
        "thank you for preparing the details and for your help",
    }

    def _positive_clauses(related: List[str]) -> bool:
        # 至少一个实质肯定子句，其余只能是“please/thanks”等礼貌填充。
        if not related:
            return False
        saw_positive = False
        for clause in related:
            lowered = clause.lower()
            if _EN_NEGATIVE_RE.search(lowered):
                return False
            if _EN_QUESTION_RE.search(lowered) or _EN_CHANGE_RE.search(lowered):
                return False
            if _displayed_amount_acceptance(clause, pending):
                # 这是对已展示金额的接受，本身不等于提交指令；继续找肯定子句。
                continue
            compact = re.sub(r"[\s,.!;:。！？；：]", "", lowered)
            scope_only = re.sub(
                r"\b(?:both|all|the|two|orders?|returns?|of|them|everything|items?)\b",
                "",
                lowered,
            )
            scope_only = re.sub(r"[\s,.!;:。！？；：]", "", scope_only)
            if _EN_POSITIVE_RE.search(lowered) or compact in _EN_WEAK_CONFIRMATIONS:
                saw_positive = True
                continue
            if re.sub(r"\s+", " ", lowered) in fillers or not scope_only:
                continue
            return False
        return saw_positive

    if len(displayed) <= 1:
        if mentioned and target not in mentioned:
            return ConfirmationDecision(
                False, "single", "The confirmation refers to another order"
            )
        if mentioned:
            related = [
                clause
                for clause in clauses
                if target.lower() in clause.lower()
                or target.lstrip("#").lower() in clause.lower()
            ]
            if _positive_clauses(related):
                return ConfirmationDecision(True, "single", "Explicit confirmation")
            return ConfirmationDecision(
                False, "single", "The clause naming the order is not a clear confirmation"
            )
        if _positive_clauses(clauses):
            return ConfirmationDecision(
                True, "single", "Clear affirmative after the details were shown"
            )
        return ConfirmationDecision(
            False,
            "single",
            "No clear confirmation was found; show the details again and ask explicitly",
        )

    if mentioned:
        if target not in mentioned:
            return ConfirmationDecision(
                False, "single", "The current message does not confirm this order"
            )
        related = [
            clause
            for clause in clauses
            if target.lower() in clause.lower()
            or target.lstrip("#").lower() in clause.lower()
        ]
        if _positive_clauses(related):
            return ConfirmationDecision(
                True, "single", "Confirmed this order among multiple pending details"
            )
        return ConfirmationDecision(
            False, "single", "The clause for this order is not a clear confirmation"
        )

    if _EN_ALL_SCOPE_RE.search(text.lower()) and _positive_clauses(clauses):
        return ConfirmationDecision(
            True, "all", "Explicitly confirmed all displayed details"
        )
    return ConfirmationDecision(
        False,
        "none",
        "Multiple pending details exist, but the message does not confirm a clear scope",
    )


def evaluate_confirmation(
    message: str,
    pending: Dict[str, Any],
    displayed: List[Dict[str, Any]],
    language: str = "zh",
) -> ConfirmationDecision:
    """判断当前用户消息是否明确批准指定待确认详情。

    单笔详情接受明确的“确认/提交/同意/就按这些”或整句简短肯定；
    多笔详情必须点名订单或明确覆盖全部，否定和改口不会被当成同意。
    """
    if str(language).lower() == "en":
        return _evaluate_confirmation_en(message, pending, displayed)
    text = _normalize_text(message)
    if not text:
        return ConfirmationDecision(False, "none", "确认语句为空")
    if len(text) > 160:
        return ConfirmationDecision(False, "none", "确认语句过长，无法可靠判断范围")

    target = normalize_order_id(pending.get("order_id", ""))
    mentioned = _order_ids_in(text)
    if len(displayed) <= 1:
        if _has_any(text, _NEGATIVE_WORDS):
            return ConfirmationDecision(False, "none", "消息包含否定或暂缓表达")
        if _has_any(text, _QUESTION_WORDS):
            return ConfirmationDecision(False, "none", "消息包含疑问、条件或改口表达")
        if mentioned and target not in mentioned:
            return ConfirmationDecision(False, "single", "确认指向其他订单")
        if _has_any(text, _STRONG_CONFIRMATIONS):
            return ConfirmationDecision(True, "single", "明确确认")
        compact = re.sub(r"[\s,.!;:。！？；：]", "", text)
        if compact in _WEAK_CONFIRMATIONS:
            return ConfirmationDecision(True, "single", "紧接展示后的简短肯定")
        return ConfirmationDecision(
            False,
            "single",
            "未检测到明确的确认语句；请重新展示详情并请用户明确确认",
        )

    # 多笔详情：点名订单时按子句判断，明确“都/两单”时才覆盖全部。
    if mentioned:
        approved_any = False
        for item in displayed:
            order_id = normalize_order_id(item.get("order_id", ""))
            if order_id not in mentioned:
                continue
            related = [
                clause
                for clause in _clauses(text)
                if order_id.lower() in clause.lower()
                or order_id.lstrip("#").lower() in clause.lower()
            ]
            if not related:
                continue
            if all(_clean_clause(clause) and _positive_in(clause) for clause in related):
                if order_id == target:
                    approved_any = True
            elif order_id == target:
                return ConfirmationDecision(
                    False, "single", "该订单的确认子句含否定、疑问或改口"
                )
        if target not in mentioned:
            return ConfirmationDecision(False, "single", "当前消息未确认该订单")
        if approved_any:
            return ConfirmationDecision(True, "single", "多笔详情中仅确认了该订单")
        return ConfirmationDecision(False, "single", "确认子句不明确")

    if (
        _has_any(text, _ALL_SCOPE_WORDS)
        and _clean_clause(text)
        and _positive_in(text)
    ):
        return ConfirmationDecision(True, "all", "明确确认全部已展示详情")
    return ConfirmationDecision(
        False,
        "none",
        "存在多笔待确认详情，但当前消息未明确确认范围",
    )


def _payment_label(
    payment: Optional[Dict[str, Any]],
    payment_id: str,
    language: str = "zh",
) -> str:
    payment = payment or {}
    source = str(payment.get("source") or "").strip()
    english = str(language).lower() == "en"
    if source == "credit_card":
        brand = str(payment.get("brand") or ("credit card" if english else "信用卡"))
        last_four = str(payment.get("last_four") or "")
        if english:
            suffix = f" ending in {last_four}" if last_four else ""
            return f"credit card {brand}{suffix} ({payment_id})"
        suffix = f" 尾号 {last_four}" if last_four else ""
        return f"信用卡 {brand}{suffix}（{payment_id}）"
    if source == "paypal":
        return f"PayPal ({payment_id})" if english else f"PayPal（{payment_id}）"
    if source == "gift_card":
        balance = payment.get("balance")
        if english:
            suffix = (
                f", balance ${float(balance):,.2f}" if balance is not None else ""
            )
            return f"gift card ({payment_id}{suffix})"
        suffix = f"，余额 ¥{balance}" if balance is not None else ""
        return f"礼品卡（{payment_id}{suffix}）"
    label = source or ("payment method" if english else "支付方式")
    return f"{label} ({payment_id})" if english else f"{label}（{payment_id}）"


def _options_text(options: Any, language: str = "zh") -> str:
    english = str(language).lower() == "en"
    if not isinstance(options, dict) or not options:
        return "no additional options" if english else "无额外选项"
    separator = ", " if english else "，"
    return separator.join(f"{key}: {value}" for key, value in options.items())


class RetailStore:
    """公开种子 + 会话状态的最小持久层。"""

    def __init__(
        self,
        db_path: str | Path,
        seed_path: str | Path,
        expected_seed_sha256: str = EXPECTED_SEED_SHA256,
        language: str = "zh",
    ):
        self.db_path = Path(db_path).expanduser().resolve()
        self.seed_path = Path(seed_path).expanduser().resolve()
        self.expected_seed_sha256 = expected_seed_sha256
        self.language = str(language or "zh").lower()
        if self.language not in {"zh", "en"}:
            raise RetailStoreError(f"不支持的语言: {language}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _t(self, zh: str, en: str) -> str:
        """按当前业务语言选择提示语；默认中文。"""
        return en if self.language == "en" else zh

    # ── 连接与初始化 ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        seed_hash = _sha256_file(self.seed_path)
        if self.expected_seed_sha256 and seed_hash != self.expected_seed_sha256:
            raise RetailStoreError(
                "公开种子哈希不一致，拒绝导入: "
                f"{seed_hash} != {self.expected_seed_sha256}"
            )

        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema(connection)
            has_meta = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
            ).fetchone()
            meta_rows = (
                connection.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
                if has_meta
                else 0
            )
            table_counts = {
                name: connection.execute(
                    f"SELECT COUNT(*) FROM {name}"
                ).fetchone()[0]
                for name in ("users", "orders", "products")
            }
            if meta_rows == 0:
                if any(table_counts.values()):
                    raise RetailStoreError(
                        "数据库已有业务数据但缺少元数据，拒绝自动覆盖；"
                        f"请使用新库或显式指定其他路径: {self.db_path}"
                    )
                self._import_seed(connection, seed_hash)
                return
            self._verify_existing(connection, seed_hash)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email_lower TEXT NOT NULL,
                first_lower TEXT NOT NULL,
                last_lower TEXT NOT NULL,
                zip TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email_lower);
            CREATE INDEX IF NOT EXISTS idx_users_name_zip
                ON users(first_lower, last_lower, zip);
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT,
                turn_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id)
            );
            CREATE TABLE IF NOT EXISTS pending_returns (
                op_id TEXT PRIMARY KEY,
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                items_json TEXT NOT NULL,
                payment_method_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                display_text TEXT NOT NULL,
                status TEXT NOT NULL,
                prepared_turn INTEGER NOT NULL,
                displayed_turn INTEGER,
                display_appended INTEGER NOT NULL DEFAULT 0,
                order_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                consumed_at TEXT,
                superseded_by TEXT,
                note TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pending_session
                ON pending_returns(client_user_id, conv_id, status);
            CREATE TABLE IF NOT EXISTS return_operations (
                op_id TEXT PRIMARY KEY,
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT NOT NULL,
                order_id TEXT NOT NULL,
                items_json TEXT NOT NULL,
                payment_method_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retail_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT,
                event_type TEXT NOT NULL,
                order_id TEXT,
                op_id TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_session
                ON retail_events(client_user_id, conv_id, event_id);
            CREATE TABLE IF NOT EXISTS retail_support_state (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                format_version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id)
            );
            CREATE TABLE IF NOT EXISTS retail_dialogue (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id)
            );
            CREATE TABLE IF NOT EXISTS pending_actions (
                op_id TEXT PRIMARY KEY,
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                target_key TEXT NOT NULL,
                order_id TEXT,
                args_json TEXT NOT NULL,
                preview_json TEXT NOT NULL,
                amount_cents INTEGER,
                display_text TEXT NOT NULL,
                status TEXT NOT NULL,
                prepared_turn INTEGER NOT NULL,
                displayed_turn INTEGER,
                display_appended INTEGER NOT NULL DEFAULT 0,
                base_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                consumed_at TEXT,
                superseded_by TEXT,
                note TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_pending_actions_session
                ON pending_actions(client_user_id, conv_id, status);
            CREATE TABLE IF NOT EXISTS action_operations (
                op_id TEXT PRIMARY KEY,
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                retail_user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                order_id TEXT,
                args_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS confirmation_decisions (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                op_id TEXT NOT NULL,
                approved INTEGER NOT NULL,
                source TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id, turn, op_id)
            );
            CREATE TABLE IF NOT EXISTS conversation_log (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                role TEXT NOT NULL,
                kind TEXT NOT NULL,
                agent TEXT,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id, seq)
            );
            CREATE TABLE IF NOT EXISTS task_facts (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                processed_turn INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id)
            );
            CREATE TABLE IF NOT EXISTS context_state (
                client_user_id TEXT NOT NULL,
                conv_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (client_user_id, conv_id)
            );
            """
        )

    def _import_seed(
        self,
        connection: sqlite3.Connection,
        seed_hash: str,
    ) -> None:
        try:
            data = json.loads(self.seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            raise RetailStoreError(f"无法读取公开种子: {ex}") from ex

        if not isinstance(data, dict):
            raise RetailStoreError("公开种子顶层必须是对象")
        for key, expected in EXPECTED_SEED_COUNTS.items():
            value = data.get(key)
            if not isinstance(value, dict) or len(value) != expected:
                raise RetailStoreError(
                    f"公开种子 {key} 数量不符: "
                    f"{len(value) if isinstance(value, dict) else 'invalid'} != {expected}"
                )

        users = data["users"]
        orders = data["orders"]
        products = data["products"]
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO users(user_id, email_lower, first_lower, last_lower, zip, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        str(user_id),
                        str(record.get("email", "")).strip().lower(),
                        str((record.get("name") or {}).get("first_name", "")).strip().lower(),
                        str((record.get("name") or {}).get("last_name", "")).strip().lower(),
                        str((record.get("address") or {}).get("zip", "")).strip(),
                        json.dumps(record, ensure_ascii=False, sort_keys=True),
                    )
                    for user_id, record in users.items()
                ],
            )
            connection.executemany(
                "INSERT INTO orders(order_id, user_id, status, version, payload)"
                " VALUES (?, ?, ?, 1, ?)",
                [
                    (
                        str(record.get("order_id") or order_id),
                        str(record.get("user_id", "")),
                        str(record.get("status", "")),
                        json.dumps(record, ensure_ascii=False, sort_keys=True),
                    )
                    for order_id, record in orders.items()
                ],
            )
            connection.executemany(
                "INSERT INTO products(product_id, payload) VALUES (?, ?)",
                [
                    (
                        str(record.get("product_id") or product_id),
                        json.dumps(record, ensure_ascii=False, sort_keys=True),
                    )
                    for product_id, record in products.items()
                ],
            )
            meta = {
                "schema_version": SCHEMA_VERSION,
                "seed_sha256": seed_hash,
                "seed_source": str(self.seed_path),
                "users_count": str(len(users)),
                "orders_count": str(len(orders)),
                "products_count": str(len(products)),
                "imported_at": _now(),
            }
            connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                list(meta.items()),
            )
            self._verify_records(connection, users, orders, products)
            connection.execute("COMMIT")
        except RetailStoreError:
            connection.execute("ROLLBACK")
            raise
        except Exception as ex:
            connection.execute("ROLLBACK")
            raise RetailStoreError(f"导入公开种子失败: {ex}") from ex

    @staticmethod
    def _verify_records(
        connection: sqlite3.Connection,
        users: Dict[str, Any],
        orders: Dict[str, Any],
        products: Dict[str, Any],
    ) -> None:
        for table, source, key in (
            ("users", users, "user_id"),
            ("orders", orders, "order_id"),
            ("products", products, "product_id"),
        ):
            rows = connection.execute(
                f"SELECT payload FROM {table}"
            ).fetchall()
            if len(rows) != len(source):
                raise RetailStoreError(f"{table} 导入后数量不一致")
            stored = {
                json.loads(row[0]).get(key): json.loads(row[0]) for row in rows
            }
            for record_id, record in source.items():
                if stored.get(record_id) != record:
                    raise RetailStoreError(
                        f"{table} 导入后记录不完整: {record_id}"
                    )

    def _verify_existing(
        self,
        connection: sqlite3.Connection,
        seed_hash: str,
    ) -> None:
        meta = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT key, value FROM meta").fetchall()
        }
        if meta.get("schema_version") != SCHEMA_VERSION:
            raise RetailStoreError(
                "数据库结构版本不一致，拒绝自动重置: "
                f"{meta.get('schema_version')} != {SCHEMA_VERSION}"
            )
        if meta.get("seed_sha256") != seed_hash:
            raise RetailStoreError(
                "数据库已用其他种子初始化，拒绝自动重置: "
                f"{meta.get('seed_sha256')} != {seed_hash}"
            )
        for table, expected in (
            ("users", 500),
            ("orders", 1000),
            ("products", 50),
        ):
            actual = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            if actual != expected:
                raise RetailStoreError(
                    f"数据库 {table} 数量与固定种子不一致: {actual} != {expected}"
                )

    # ── 会话与身份 ───────────────────────────────────────────────────────

    def _ensure_session(
        self,
        connection: sqlite3.Connection,
        client_user_id: str,
        conv_id: str,
    ) -> sqlite3.Row:
        now = _now()
        connection.execute(
            "INSERT OR IGNORE INTO sessions"
            "(client_user_id, conv_id, retail_user_id, turn_count, created_at, updated_at)"
            " VALUES (?, ?, NULL, 0, ?, ?)",
            (client_user_id, conv_id, now, now),
        )
        row = connection.execute(
            "SELECT * FROM sessions WHERE client_user_id=? AND conv_id=?",
            (client_user_id, conv_id),
        ).fetchone()
        if row is None:
            raise RetailStoreError("无法建立会话")
        return row

    def begin_turn(self, client_user_id: str, conv_id: str) -> int:
        with self._tx() as connection:
            self._ensure_session(connection, client_user_id, conv_id)
            connection.execute(
                "UPDATE sessions SET turn_count=turn_count+1, updated_at=?"
                " WHERE client_user_id=? AND conv_id=?",
                (_now(), client_user_id, conv_id),
            )
            row = connection.execute(
                "SELECT turn_count FROM sessions"
                " WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id),
            ).fetchone()
            return int(row["turn_count"])

    def visible_history(self, client_user_id: str, conv_id: str) -> List[Dict[str, str]]:
        """短对话独立于外部记忆，仅供消歧，不提供身份或交易授权。"""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload_json FROM retail_dialogue WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id)).fetchone()
        pairs = json.loads(row["payload_json"]) if row else []
        return [{"role": role, "content": str(pair[role])[:2000]} for pair in pairs[-2:]
                for role in ("user", "assistant") if isinstance(pair.get(role), str) and pair[role].strip()]

    def last_response_guard(self, client_user_id: str, conv_id: str) -> Dict[str, Any]:
        """只读同会话上轮守卫计数；该元数据不进入模型历史或业务授权。"""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload_json FROM retail_dialogue WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id)).fetchone()
        pairs = json.loads(row["payload_json"]) if row else []
        return {**(pairs[-1].get("response_guard") or {}), "turn": pairs[-1].get("turn")} if pairs else {}

    def save_visible_turn(self, client_user_id: str, conv_id: str, turn: int, message: str, reply: str,
                          *, response_guard: Optional[Dict[str, Any]] = None) -> None:
        """每会话只保留两轮，同一轮重复收口不追加历史。"""
        with self._tx() as connection:
            row = connection.execute("SELECT payload_json FROM retail_dialogue WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id)).fetchone()
            pairs = json.loads(row["payload_json"]) if row else []
            pairs = [p for p in pairs if p.get("turn") != turn]
            pairs.append({"turn": turn, "user": message[:2000], "assistant": reply[:2000],
                          "response_guard": {key: response_guard[key] for key in ("recovery_signature", "recovery_count")
                                             if response_guard and key in response_guard}})
            connection.execute("INSERT INTO retail_dialogue VALUES (?,?,?) ON CONFLICT(client_user_id,conv_id) DO UPDATE SET payload_json=excluded.payload_json",
                (client_user_id, conv_id, json.dumps(pairs[-2:], ensure_ascii=False)))

    def save_turn_messages(self, client_user_id: str, conv_id: str, turn: int,
                           rows: List[Dict[str, Any]]) -> None:
        """整轮写入完整消息（含工具调用与结果）；同一轮重复收口时整体替换，日志只追加。"""
        with self._tx() as connection:
            connection.execute("DELETE FROM conversation_log WHERE client_user_id=? AND conv_id=? AND turn=?",
                               (client_user_id, conv_id, int(turn)))
            row = connection.execute("SELECT COALESCE(MAX(seq),0) FROM conversation_log"
                                     " WHERE client_user_id=? AND conv_id=?", (client_user_id, conv_id)).fetchone()
            seq = int(row[0])
            for item in rows:
                seq += 1
                connection.execute(
                    "INSERT INTO conversation_log VALUES (?,?,?,?,?,?,?,?,?)",
                    (client_user_id, conv_id, seq, int(turn), item["role"], item["kind"], item.get("agent"),
                     json.dumps(item["content"], ensure_ascii=False, default=str), _now()))

    def conversation_log(self, client_user_id: str, conv_id: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT seq, turn, role, kind, agent, content_json FROM conversation_log"
                " WHERE client_user_id=? AND conv_id=? ORDER BY seq", (client_user_id, conv_id)).fetchall()
        return [{"seq": r["seq"], "turn": r["turn"], "role": r["role"], "kind": r["kind"], "agent": r["agent"],
                 "content": json.loads(r["content_json"])} for r in rows]

    def task_facts(self, client_user_id: str, conv_id: str) -> Tuple[List[Dict[str, Any]], int]:
        """会话事实账本与已处理到的轮次；事实来自用户陈述，不提供授权。"""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload_json, processed_turn FROM task_facts"
                                     " WHERE client_user_id=? AND conv_id=?", (client_user_id, conv_id)).fetchone()
        return (json.loads(row["payload_json"]), int(row["processed_turn"])) if row else ([], 0)

    def save_task_facts(self, client_user_id: str, conv_id: str, facts: List[Dict[str, Any]], turn: int) -> None:
        with self._tx() as connection:
            connection.execute(
                "INSERT INTO task_facts VALUES (?,?,?,?,?) ON CONFLICT(client_user_id,conv_id) DO UPDATE SET"
                " payload_json=excluded.payload_json, processed_turn=excluded.processed_turn, updated_at=excluded.updated_at",
                (client_user_id, conv_id, json.dumps(facts, ensure_ascii=False), int(turn), _now()))

    def context_state(self, client_user_id: str, conv_id: str) -> Dict[str, Any]:
        """上下文压缩锚点与 token 换算系数；只影响模型看到的历史，不影响业务授权。"""
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT payload_json FROM context_state WHERE client_user_id=? AND conv_id=?",
                                     (client_user_id, conv_id)).fetchone()
        return json.loads(row["payload_json"]) if row else {}

    def save_context_state(self, client_user_id: str, conv_id: str, payload: Dict[str, Any]) -> None:
        with self._tx() as connection:
            connection.execute(
                "INSERT INTO context_state VALUES (?,?,?,?) ON CONFLICT(client_user_id,conv_id)"
                " DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at",
                (client_user_id, conv_id, json.dumps(payload, ensure_ascii=False), _now()))

    def current_turn(self, client_user_id: str, conv_id: str) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT turn_count FROM sessions"
                " WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id),
            ).fetchone()
            return int(row["turn_count"]) if row else 0

    def bound_user_id(self, client_user_id: str, conv_id: str) -> Optional[str]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT retail_user_id FROM sessions"
                " WHERE client_user_id=? AND conv_id=?",
                (client_user_id, conv_id),
            ).fetchone()
            return str(row["retail_user_id"]) if row and row["retail_user_id"] else None

    def _require_binding(
        self,
        connection: sqlite3.Connection,
        client_user_id: str,
        conv_id: str,
    ) -> str:
        row = connection.execute(
            "SELECT retail_user_id FROM sessions"
            " WHERE client_user_id=? AND conv_id=?",
            (client_user_id, conv_id),
        ).fetchone()
        if row is None or not row["retail_user_id"]:
            raise RetailStoreError(
                self._t(
                    "尚未完成身份核对；请先按邮箱或姓名加邮编查找当前用户。",
                    "Identity has not been verified; look up the current user "
                    "by email or by full name and zip code first.",
                )
            )
        return str(row["retail_user_id"])

    def _bind(
        self,
        client_user_id: str,
        conv_id: str,
        retail_user_id: str,
        method: str,
    ) -> Dict[str, Any]:
        with self._tx() as connection:
            session = self._ensure_session(connection, client_user_id, conv_id)
            bound = session["retail_user_id"]
            if bound and bound != retail_user_id:
                return {
                    "success": False,
                    "error": self._t(
                        "本会话已绑定其他用户，不能切换；请新开会话。",
                        "This session is already bound to another user; "
                        "start a new session to switch users.",
                    ),
                }
            if not bound:
                connection.execute(
                    "UPDATE sessions SET retail_user_id=?, updated_at=?"
                    " WHERE client_user_id=? AND conv_id=?",
                    (retail_user_id, _now(), client_user_id, conv_id),
                )
                self._event(
                    connection,
                    client_user_id,
                    conv_id,
                    retail_user_id,
                    "identity_bound",
                    {"method": method},
                )
            user = connection.execute(
                "SELECT payload FROM users WHERE user_id=?",
                (retail_user_id,),
            ).fetchone()
            if user is None:
                raise RetailStoreError(
                    self._t("绑定的用户不存在", "The bound user does not exist")
                )
            record = json.loads(user["payload"])
            name = record.get("name") or {}
            return {
                "success": True,
                "user_id": retail_user_id,
                "name": f"{name.get('first_name', '')} {name.get('last_name', '')}".strip(),
                "method": method,
                "already_bound": bool(bound),
            }

    def find_user_by_email(
        self,
        client_user_id: str,
        conv_id: str,
        email: str,
    ) -> Dict[str, Any]:
        email_lower = str(email or "").strip().lower()
        if not email_lower:
            return {
                "success": False,
                "error": self._t("邮箱不能为空", "Email cannot be empty"),
            }
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT user_id FROM users WHERE email_lower=?",
                (email_lower,),
            ).fetchall()
        if not rows:
            return {
                "success": False,
                "error": self._t(
                    "未找到与该邮箱匹配的沙箱用户；请核对后重试。",
                    "No sandbox user matches that email; check it and try again.",
                ),
            }
        if len(rows) > 1:
            return {
                "success": False,
                "error": self._t(
                    "该邮箱匹配到多个沙箱用户，无法安全绑定；请改用姓名和邮编。",
                    "That email matches multiple sandbox users; use the full "
                    "name and zip code instead.",
                ),
            }
        return self._bind(
            client_user_id, conv_id, str(rows[0]["user_id"]), "email"
        )

    def find_user_by_name_zip(
        self,
        client_user_id: str,
        conv_id: str,
        first_name: str,
        last_name: str,
        zip_code: str,
    ) -> Dict[str, Any]:
        first = str(first_name or "").strip().lower()
        last = str(last_name or "").strip().lower()
        zip_value = str(zip_code or "").strip()
        if not first or not last or not zip_value:
            return {
                "success": False,
                "error": self._t(
                    "姓名或邮编不完整；请提供名、姓和邮编。",
                    "The name or zip code is incomplete; provide the first "
                    "name, last name, and zip code.",
                ),
            }
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT user_id FROM users"
                " WHERE first_lower=? AND last_lower=? AND zip=?",
                (first, last, zip_value),
            ).fetchall()
        if not rows:
            return {
                "success": False,
                "error": self._t(
                    "未找到匹配的沙箱用户；请核对姓名和邮编后重试。",
                    "No sandbox user matches that name and zip code; check "
                    "them and try again.",
                ),
            }
        if len(rows) > 1:
            return {
                "success": False,
                "error": self._t(
                    "该姓名和邮编匹配到多个用户，无法安全绑定；请改用邮箱。",
                    "That name and zip code match multiple users; use the "
                    "email instead.",
                ),
            }
        return self._bind(
            client_user_id, conv_id, str(rows[0]["user_id"]), "name_zip"
        )

    # ── 只读业务查询 ─────────────────────────────────────────────────────

    def _user_record(
        self,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> Dict[str, Any]:
        row = connection.execute(
            "SELECT payload FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise RetailStoreError(
                self._t("用户记录不存在", "The user record does not exist")
            )
        return json.loads(row["payload"])

    def get_user_details(
        self,
        client_user_id: str,
        conv_id: str,
    ) -> Dict[str, Any]:
        try:
            with closing(self._connect()) as connection:
                user_id = self._require_binding(connection, client_user_id, conv_id)
                record = self._user_record(connection, user_id)
        except RetailStoreError as ex:
            return {"success": False, "error": str(ex)}
        return {"success": True, "user": record}

    def _order_row(
        self,
        connection: sqlite3.Connection,
        order_id: str,
        user_id: str,
    ) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM orders WHERE order_id=? AND user_id=?",
            (order_id, user_id),
        ).fetchone()

    def get_order_details(
        self,
        client_user_id: str,
        conv_id: str,
        order_id: str,
    ) -> Dict[str, Any]:
        normalized = normalize_order_id(order_id)
        try:
            with closing(self._connect()) as connection:
                user_id = self._require_binding(connection, client_user_id, conv_id)
                row = self._order_row(connection, normalized, user_id)
                if row is None:
                    return {
                        "success": False,
                        "error": self._t(
                            "未找到属于当前用户的该订单；请核对订单号。",
                            "No order with that number belongs to the current "
                            "user; check the order number.",
                        ),
                    }
                order = json.loads(row["payload"])
                user = self._user_record(connection, user_id)
        except RetailStoreError as ex:
            return {"success": False, "error": str(ex)}
        history = order.get("payment_history") or []
        original_id = next(
            (
                str(item.get("payment_method_id"))
                for item in history
                if item.get("payment_method_id")
            ),
            "",
        )
        original = (user.get("payment_methods") or {}).get(original_id, {})
        return {
            "success": True,
            "order": order,
            "order_version": int(row["version"]),
            "original_payment_method_id": original_id,
            "original_payment_description": _payment_label(
                original, original_id, self.language
            )
            if original_id
            else "",
        }

    def get_product_details(
        self,
        client_user_id: str,
        conv_id: str,
        product_id: str,
    ) -> Dict[str, Any]:
        """商品目录是公开资料；用户和订单仍按绑定身份限制。"""
        product_key = str(product_id or "").strip()
        if not product_key:
            return {
                "success": False,
                "error": self._t("product_id 不能为空", "product_id cannot be empty"),
            }
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload FROM products WHERE product_id=?",
                (product_key,),
            ).fetchone()
        if row is None:
            return {
                "success": False,
                "error": self._t("商品记录不存在", "The product record does not exist"),
            }
        return {"success": True, "product": json.loads(row["payload"])}

    # ── 退货准备与确认 ───────────────────────────────────────────────────

    def _event(
        self,
        connection: sqlite3.Connection,
        client_user_id: str,
        conv_id: str,
        retail_user_id: Optional[str],
        event_type: str,
        payload: Dict[str, Any],
        order_id: Optional[str] = None,
        op_id: Optional[str] = None,
    ) -> None:
        connection.execute(
            "INSERT INTO retail_events"
            "(client_user_id, conv_id, retail_user_id, event_type, order_id, op_id,"
            " payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                client_user_id,
                conv_id,
                retail_user_id,
                event_type,
                order_id,
                op_id,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                _now(),
            ),
        )

    def list_events(
        self,
        client_user_id: str,
        conv_id: str,
    ) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM retail_events"
                " WHERE client_user_id=? AND conv_id=? ORDER BY event_id",
                (client_user_id, conv_id),
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "event_type": str(row["event_type"]),
                "order_id": row["order_id"],
                "op_id": row["op_id"],
                "payload": json.loads(row["payload_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _parse_items(value: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"item_ids": [], "lines": []}
        return parsed if isinstance(parsed, dict) else {"item_ids": [], "lines": []}

    # ── 写操作：两阶段待确认动作 ─────────────────────────────────────────

    _EVENT_PREFIX = {"return": "return", "exchange": "exchange", "cancel": "cancel",
                     "modify_items": "modify_items", "modify_address": "modify_address",
                     "modify_payment": "modify_payment", "modify_user_address": "modify_user_address"}

    @classmethod
    def executed_event(cls, kind: str) -> str:
        # 退货沿用原事件名，便于既有恢复核对和评测统计。
        return "return_requested" if kind == "return" else f"{cls._EVENT_PREFIX[kind]}_executed"

    @staticmethod
    def _fingerprint(order: Optional[Dict[str, Any]], user: Dict[str, Any]) -> str:
        # 礼品卡余额不进入指纹：同批已确认的退款会改余额，余额是否足够由执行时的业务校验保证。
        user = dict(user or {})
        methods = user.get("payment_methods")
        if isinstance(methods, dict):
            user["payment_methods"] = {k: {f: v for f, v in m.items() if f != "balance"} if isinstance(m, dict) else m
                                       for k, m in methods.items()}
        data = json.dumps([order, user], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def _rebase_displayed(self, connection: sqlite3.Connection, client_user_id: str, conv_id: str,
                          user_id: str, op_id: str, old_order: Optional[Dict[str, Any]],
                          old_user: Dict[str, Any], new_user: Dict[str, Any]) -> None:
        """刚执行的已确认写入只会让同会话其他待确认项的基准过期：
        仅当某项基准恰好等于本次写入前的状态时，才把基准前移到写入后状态；其他变化仍须重新展示。"""
        old_id = (old_order or {}).get("order_id")
        rows = connection.execute(
            "SELECT op_id, kind, args_json, base_fingerprint FROM pending_actions"
            " WHERE client_user_id=? AND conv_id=? AND retail_user_id=? AND status='displayed' AND op_id<>?",
            (client_user_id, conv_id, user_id, op_id)).fetchall()
        for row in rows:
            try:
                _, after_order, _ = self._action_target(connection, row["kind"], json.loads(row["args_json"]), user_id)
            except RetailStoreError:
                continue
            before_order = old_order if after_order and after_order.get("order_id") == old_id else after_order
            if row["base_fingerprint"] != self._fingerprint(before_order, old_user):
                continue
            connection.execute("UPDATE pending_actions SET base_fingerprint=?, updated_at=? WHERE op_id=?",
                               (self._fingerprint(after_order, new_user), _now(), row["op_id"]))

    def _action_target(self, connection: sqlite3.Connection, kind: str, args: Dict[str, Any],
                       user_id: str) -> Tuple[Optional[sqlite3.Row], Optional[Dict[str, Any]], Dict[str, Any]]:
        user = self._user_record(connection, user_id)
        if kind == "modify_user_address":
            return None, None, user
        row = self._order_row(connection, args["order_id"], user_id)
        if row is None:
            raise RetailStoreError(self._t(
                "未找到属于当前用户的该订单；请核对订单号。",
                "No order with that number belongs to the current user; check the order number."))
        return row, json.loads(row["payload"]), user

    def _products(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        rows = connection.execute("SELECT product_id, payload FROM products").fetchall()
        return {str(r["product_id"]): json.loads(r["payload"]) for r in rows}

    def _action_display(self, kind: str, args: Dict[str, Any], order: Optional[Dict[str, Any]],
                        user: Dict[str, Any], preview: Dict[str, Any], op_id: str) -> str:
        """服务端生成的完整待确认详情；模型必须原样展示，用户据此确认。"""
        english = self.language == "en"
        methods = user.get("payment_methods") or {}

        def method(payment_id: str) -> str:
            return _payment_label(methods.get(payment_id), payment_id, self.language)

        def item_line(item: Dict[str, Any]) -> str:
            return (f"- {item.get('name')} (item_id: {item.get('item_id')}, {_options_text(item.get('options'), 'en')}),"
                    f" price {_fmt_usd(_to_cents(item.get('price')))}" if english else
                    f"- {item.get('name')}（item_id: {item.get('item_id')}，{_options_text(item.get('options'))}），"
                    f"价格 ¥{_fmt_cents(_to_cents(item.get('price')))}")

        def by_id(item_id: str) -> Dict[str, Any]:
            return next((i for i in (order or {}).get("items", []) if i.get("item_id") == item_id), {"item_id": item_id})

        def address(a: Dict[str, Any]) -> str:
            return ", ".join(str(a[f]) for f in ("address1", "address2", "city", "state", "zip", "country") if a.get(f))

        lines: List[str] = []
        if kind in {"return", "exchange", "modify_items"}:
            lines.append("Items:" if english else "商品：")
            lines += [item_line(by_id(i)) for i in args["item_ids"]]
        if kind in {"exchange", "modify_items"}:
            products = preview.get("new_items") or []
            lines.append("New items:" if english else "新商品：")
            lines += [item_line(i) for i in products]
            diff = preview.get("price_difference", 0)
            lines.append((f"Price difference: {_fmt_usd(_to_cents(abs(diff)))} "
                          + ("to pay" if diff > 0 else "to refund" if diff < 0 else "")) if english else
                         f"差价：¥{_fmt_cents(_to_cents(abs(diff)))}" + ("（需补付）" if diff > 0 else "（将退还）" if diff < 0 else ""))
            lines.append(("Payment method for the difference: " if english else "差价支付／退还方式：")
                         + method(args["payment_method_id"]))
        if kind == "return":
            lines.append((f"Estimated refund amount: {_fmt_usd(_to_cents(preview['refund_amount']))} "
                          "(based on the order price; this is not a completed refund)") if english else
                         f"预计申请金额：¥{_fmt_cents(_to_cents(preview['refund_amount']))}（按订单成交价计算，不代表已到账）")
            lines.append(("Refund destination: " if english else "退款去向：") + method(args["payment_method_id"]))
        if kind == "cancel":
            lines.append(("Reason: " if english else "取消原因：") + args["reason"])
            lines.append((f"Refund amount: {_fmt_usd(_to_cents(preview['refund_amount']))} to the original payment methods"
                          if english else f"退款金额：¥{_fmt_cents(_to_cents(preview['refund_amount']))}，退回原支付方式"))
        if kind in {"modify_address", "modify_user_address"}:
            old = (order or {}).get("address") if kind == "modify_address" else user.get("address")
            lines.append(("Current address: " if english else "当前地址：") + address(old or {}))
            lines.append(("New address: " if english else "新地址：") + address(args))
        if kind == "modify_payment":
            current = (order or {}).get("payment_history", [{}])[0].get("payment_method_id", "")
            lines.append(("Current payment method: " if english else "当前支付方式：") + method(current))
            lines.append(("New payment method: " if english else "新支付方式：") + method(args["payment_method_id"]))
        titles = {
            "return": ("RETURN REQUEST DETAILS", "退货申请详情"),
            "exchange": ("EXCHANGE REQUEST DETAILS", "换货申请详情"),
            "cancel": ("ORDER CANCELLATION DETAILS", "取消订单详情"),
            "modify_items": ("ORDER ITEM CHANGE DETAILS", "修改订单商品详情"),
            "modify_address": ("ORDER ADDRESS CHANGE DETAILS", "修改订单地址详情"),
            "modify_payment": ("ORDER PAYMENT CHANGE DETAILS", "修改订单支付方式详情"),
            "modify_user_address": ("ACCOUNT ADDRESS CHANGE DETAILS", "修改账户默认地址详情"),
        }
        title = titles[kind][0 if english else 1]
        head = ([f"{title} (PENDING CONFIRMATION)", f"Operation ID: {op_id}"] if english else
                [f"【{title}（待确认）】", f"操作编号：{op_id}"])
        if order is not None:
            head.append((f"Order: {order['order_id']} (status: {order['status']})") if english else
                        f"订单号：{order['order_id']}（状态：{order['status']}）")
        tail = ("Nothing has been changed yet. Reply with an explicit confirmation, such as \"Yes, please proceed\", "
                "to carry out exactly these details." if english else
                "目前尚未做任何修改。请明确回复“确认，按这些办理”以按上述详情执行。")
        return "\n".join(head + lines + [tail])

    def active_pending(self, client_user_id: str, conv_id: str) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM pending_actions WHERE client_user_id=? AND conv_id=?"
                " AND status IN ('prepared', 'displayed') ORDER BY created_at, op_id",
                (client_user_id, conv_id)).fetchall()
        result = []
        for row in rows:
            args = json.loads(row["args_json"])
            item = {key: row[key] for key in row.keys()}
            item.update(args=args, order_id=row["order_id"] or "",
                        payment_method_id=args.get("payment_method_id"),
                        items={"item_ids": args.get("item_ids", [])},
                        amount=_fmt_cents(int(row["amount_cents"])) if row["amount_cents"] is not None else None)
            result.append(item)
        return result

    def cancel_pending(self, client_user_id: str, conv_id: str, reason: str = "user_cancelled") -> int:
        """取消本会话尚未执行的待确认动作；不修改订单或用户资料。"""
        now = _now()
        try:
            with self._tx() as connection:
                rows = connection.execute(
                    "SELECT op_id, retail_user_id, order_id, kind FROM pending_actions"
                    " WHERE client_user_id=? AND conv_id=? AND status IN ('prepared', 'displayed')",
                    (client_user_id, conv_id)).fetchall()
                if not rows:
                    return 0
                connection.execute(
                    "UPDATE pending_actions SET status='cancelled', updated_at=?, note=?"
                    " WHERE client_user_id=? AND conv_id=? AND status IN ('prepared', 'displayed')",
                    (now, f"cancelled: {str(reason or 'user_cancelled')[:120]}", client_user_id, conv_id))
                for row in rows:
                    self._event(connection, client_user_id, conv_id, row["retail_user_id"],
                                f"{self._EVENT_PREFIX[row['kind']]}_cancelled",
                                {"reason": str(reason or "user_cancelled")[:120]},
                                order_id=row["order_id"], op_id=row["op_id"])
                return len(rows)
        except sqlite3.Error as ex:
            raise SupportStateError(f"取消待确认动作失败: {ex}") from ex

    def mark_pending_displayed(self, client_user_id: str, conv_id: str, op_id: str, turn: int,
                               appended: bool) -> bool:
        with self._tx() as connection:
            row = connection.execute("SELECT * FROM pending_actions WHERE op_id=?", (op_id,)).fetchone()
            if (row is None or row["client_user_id"] != client_user_id or row["conv_id"] != conv_id
                    or row["status"] != "prepared" or int(row["prepared_turn"]) != int(turn)):
                return False
            connection.execute(
                "UPDATE pending_actions SET status='displayed', displayed_turn=?, display_appended=?,"
                " updated_at=? WHERE op_id=? AND status='prepared'",
                (turn, 1 if appended else 0, _now(), op_id))
            self._event(connection, client_user_id, conv_id, str(row["retail_user_id"]),
                        f"{self._EVENT_PREFIX[row['kind']]}_displayed",
                        {"appended": bool(appended), "turn": int(turn)},
                        order_id=row["order_id"], op_id=op_id)
            return True

    def record_confirmation_decisions(self, client_user_id: str, conv_id: str, turn: int,
                                      decisions: List[Dict[str, Any]], source: str) -> None:
        """保存本轮由真实用户消息得出的确认判定；执行时只认同一轮的判定。"""
        with self._tx() as connection:
            for decision in decisions:
                connection.execute(
                    "INSERT OR REPLACE INTO confirmation_decisions"
                    "(client_user_id, conv_id, turn, op_id, approved, source, reason, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (client_user_id, conv_id, int(turn), str(decision["op_id"]),
                     1 if decision.get("approved") else 0, source, str(decision.get("reason") or "")[:300], _now()))

    def confirmation_decision(self, client_user_id: str, conv_id: str, turn: int,
                              op_id: str) -> Optional[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM confirmation_decisions WHERE client_user_id=? AND conv_id=? AND turn=? AND op_id=?",
                (client_user_id, conv_id, int(turn), op_id)).fetchone()
        return dict(row) if row else None

    def write_action(self, client_user_id: str, conv_id: str, tool_name: str, raw_args: Dict[str, Any],
                     user_message: str, *, allow_execute: bool = True) -> Dict[str, Any]:
        """写工具统一入口：首次调用生成待确认详情；展示后经本轮确认、参数一致的再次调用才执行。"""
        kind = WRITE_KINDS[tool_name]
        turn = self.current_turn(client_user_id, conv_id)
        try:
            args = canonical_args(kind, raw_args or {})
            with closing(self._connect()) as connection:
                user_id = self._require_binding(connection, client_user_id, conv_id)
                if kind == "modify_user_address" and str(raw_args.get("user_id") or user_id) != user_id:
                    raise RetailStoreError(self._t("只能修改当前已核实用户的地址。",
                                                   "Only the verified current user's address can be changed."))
                args_json = json.dumps(args, ensure_ascii=False, sort_keys=True)
                displayed = connection.execute(
                    "SELECT * FROM pending_actions WHERE client_user_id=? AND conv_id=? AND kind=?"
                    " AND args_json=? AND status='displayed' ORDER BY created_at DESC LIMIT 1",
                    (client_user_id, conv_id, kind, args_json)).fetchone()
        except (RetailStoreError, DomainError) as ex:
            return {"success": False, "error": str(ex)}
        if displayed is not None and allow_execute and int(displayed["displayed_turn"] or turn) < turn:
            return self._execute_action(client_user_id, conv_id, kind, dict(displayed), user_message, turn)
        return self._prepare_action(client_user_id, conv_id, kind, args, turn)

    def _prepare_action(self, client_user_id: str, conv_id: str, kind: str, args: Dict[str, Any],
                        turn: int) -> Dict[str, Any]:
        try:
            with closing(self._connect()) as connection:
                user_id = self._require_binding(connection, client_user_id, conv_id)
                _, order, user = self._action_target(connection, kind, args, user_id)
                products = self._products(connection)
                new_order, _, preview = apply_write(kind, args, order, user, products)
        except (RetailStoreError, DomainError) as ex:
            return {"success": False, "error": str(ex)}
        if kind in {"exchange", "modify_items"}:
            preview["new_items"] = []
            for old_id, new_id in zip(args["item_ids"], args["new_item_ids"]):
                product_id = next(i["product_id"] for i in order["items"] if i["item_id"] == old_id)
                variant = products[product_id]["variants"][new_id]
                preview["new_items"].append({"name": products[product_id].get("name"), "item_id": new_id,
                                             "options": variant.get("options"), "price": variant.get("price")})
        op_id = f"OP-{uuid.uuid4().hex[:12]}"
        display_text = self._action_display(kind, args, order, user, preview, op_id)
        amount = preview.get("refund_amount")
        amount_cents = _to_cents(amount) if amount is not None else None
        target_key = args.get("order_id") or f"user:{user_id}"
        now = _now()
        with self._tx() as connection:
            self._ensure_session(connection, client_user_id, conv_id)
            previous = connection.execute(
                "SELECT op_id FROM pending_actions WHERE client_user_id=? AND conv_id=? AND target_key=?"
                " AND status IN ('prepared', 'displayed')", (client_user_id, conv_id, target_key)).fetchall()
            # 同一订单／账户只保留最新详情，旧详情失效后不能再被确认。
            connection.execute(
                "UPDATE pending_actions SET status='superseded', superseded_by=?, updated_at=?"
                " WHERE client_user_id=? AND conv_id=? AND target_key=? AND status IN ('prepared', 'displayed')",
                (op_id, now, client_user_id, conv_id, target_key))
            connection.execute(
                "INSERT INTO pending_actions(op_id, client_user_id, conv_id, retail_user_id, kind, target_key,"
                " order_id, args_json, preview_json, amount_cents, display_text, status, prepared_turn,"
                " displayed_turn, display_appended, base_fingerprint, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, NULL, 0, ?, ?, ?)",
                (op_id, client_user_id, conv_id, user_id, kind, target_key, args.get("order_id"),
                 json.dumps(args, ensure_ascii=False, sort_keys=True), json.dumps(preview, ensure_ascii=False),
                 amount_cents, display_text, turn, self._fingerprint(order, user), now, now))
            self._event(connection, client_user_id, conv_id, user_id, f"{self._EVENT_PREFIX[kind]}_prepared",
                        {"args": args, "superseded": [str(r["op_id"]) for r in previous]},
                        order_id=args.get("order_id"), op_id=op_id)
        return {"success": True, "action_status": "pending_confirmation", "status": "prepared", "op_id": op_id,
                "kind": kind, "order_id": args.get("order_id"), "display_text": display_text, "preview": preview,
                "note": self._t(
                    "尚未执行任何修改。必须把 display_text 原样展示给用户，等待用户在下一条消息明确确认后，"
                    "再用完全相同的参数调用本工具执行。",
                    "Nothing has been changed yet. Show display_text exactly to the user, wait for the user's "
                    "explicit confirmation in a later message, then call this tool again with identical "
                    "arguments to carry it out.")}

    def _execute_action(self, client_user_id: str, conv_id: str, kind: str, pending: Dict[str, Any],
                        user_message: str, turn: int) -> Dict[str, Any]:
        op_id = str(pending["op_id"])
        if int(pending["prepared_turn"]) >= turn:
            return {"success": False, "error": self._t("准备详情的同一轮不能执行；请等待用户确认。",
                                                       "Details cannot be carried out in the turn they were prepared.")}
        decision = self.confirmation_decision(client_user_id, conv_id, turn, op_id)
        if decision is not None:
            approved, reason, scope = bool(decision["approved"]), str(decision["reason"]), str(decision["source"])
        else:
            displayed = [p for p in self.active_pending(client_user_id, conv_id) if p["status"] == "displayed"]
            current = next((p for p in displayed if p["op_id"] == op_id), {**pending, "order_id": pending["order_id"] or ""})
            rule = evaluate_confirmation(user_message, current, displayed, language=self.language)
            approved, reason, scope = rule.approved, rule.reason, rule.scope
        args = json.loads(pending["args_json"])
        if not approved:
            self._record_rejection(client_user_id, conv_id, str(pending["retail_user_id"]),
                                   pending["order_id"], op_id, reason, {"kind": kind, "args": args})
            target = f" {pending['order_id']}" if pending["order_id"] else ""
            return {"success": False, "error_code": "confirmation_required",
                    "error": self._t(f"未获得当前用户的明确确认：{reason}",
                                     f"The current user has not given explicit confirmation: {reason}"),
                    "confirmation_hint": self._t(
                        f"尚未执行。请先解决疑问或修改详情；如确认按已展示内容办理{target}，请在下一条消息明确回复“确认，按已展示详情办理”。",
                        f"Nothing was carried out. Resolve any questions or changes first. If you want the displayed "
                        f"details{target} carried out unchanged, reply in a new message: Yes, please proceed as listed.")}
        try:
            with self._tx() as connection:
                row = connection.execute("SELECT * FROM pending_actions WHERE op_id=?", (op_id,)).fetchone()
                if row is None or row["status"] != "displayed":
                    raise RetailStoreError(self._t("该确认已使用或待确认状态已变化",
                                                   "This confirmation was already used or its pending state changed"))
                user_id = self._require_binding(connection, client_user_id, conv_id)
                if str(row["retail_user_id"]) != user_id:
                    raise RetailStoreError(self._t("会话绑定用户与待确认记录不一致",
                                                   "The session-bound user does not match the pending record"))
                order_row, order, user = self._action_target(connection, kind, args, user_id)
                if self._fingerprint(order, user) != row["base_fingerprint"]:
                    raise RetailStoreError(self._t("订单或账户在展示后已变化；请重新查询、展示并确认。",
                                                   "The order or account changed after the details were shown; "
                                                   "look it up, show the details, and confirm again."))
                new_order, new_user, _ = apply_write(kind, args, order, user, self._products(connection))
                consumed = connection.execute(
                    "UPDATE pending_actions SET status='consumed', consumed_at=?, updated_at=?"
                    " WHERE op_id=? AND status='displayed'", (_now(), _now(), op_id))
                if consumed.rowcount != 1:
                    raise RetailStoreError("该确认已被使用，不能重复执行")
                if new_order is not None:
                    written = connection.execute(
                        "UPDATE orders SET status=?, version=version+1, payload=? WHERE order_id=? AND version=?",
                        (new_order["status"], json.dumps(new_order, ensure_ascii=False, sort_keys=True),
                         new_order["order_id"], int(order_row["version"])))
                    if written.rowcount != 1:
                        raise RetailStoreError(self._t("订单状态已变化，写入未生效",
                                                       "The order changed; the write did not take effect"))
                if new_user != user:
                    connection.execute("UPDATE users SET zip=?, payload=? WHERE user_id=?",
                                       (str(new_user["address"]["zip"]),
                                        json.dumps(new_user, ensure_ascii=False, sort_keys=True), user_id))
                connection.execute(
                    "INSERT INTO action_operations(op_id, client_user_id, conv_id, retail_user_id, kind,"
                    " order_id, args_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (op_id, client_user_id, conv_id, user_id, kind, pending["order_id"],
                     pending["args_json"], _now()))
                payload = {"kind": kind, "args": args, "confirmation_scope": scope}
                if kind == "return":
                    payload.update(item_ids=args["item_ids"], payment_method_id=args["payment_method_id"],
                                   amount_cents=pending["amount_cents"])
                self._event(connection, client_user_id, conv_id, user_id, self.executed_event(kind), payload,
                            order_id=pending["order_id"], op_id=op_id)
                self._rebase_displayed(connection, client_user_id, conv_id, user_id, op_id,
                                       order, user, new_user)
        except (RetailStoreError, DomainError) as ex:
            self._record_rejection(client_user_id, conv_id, str(pending["retail_user_id"]),
                                   pending["order_id"], op_id, str(ex), {"kind": kind, "args": args})
            return {"success": False, "error": str(ex)}
        result: Dict[str, Any] = {"success": True, "action_status": "executed", "op_id": op_id, "kind": kind,
                                  "note": self._t("已按确认的详情执行。不代表退款已到账、邮件已发送或仓库已收货。",
                                                  "Carried out as confirmed. This does not mean a refund has arrived, "
                                                  "an email was sent, or the warehouse received anything.")}
        if new_order is not None:
            result.update(order_id=new_order["order_id"], status=new_order["status"], order=new_order)
            if kind == "return":
                result.update(return_items=new_order["return_items"],
                              return_payment_method_id=new_order["return_payment_method_id"])
        else:
            result.update(user_id=user_id, address=new_user["address"])
        return result

    def prepare_return(self, client_user_id: str, conv_id: str, order_id: str, item_ids: Any,
                       payment_method_id: Any) -> Dict[str, Any]:
        """兼容旧调用：只生成退货待确认详情。"""
        return self.write_action(client_user_id, conv_id, "return_delivered_order_items",
                                 {"order_id": order_id, "item_ids": item_ids, "payment_method_id": payment_method_id},
                                 "", allow_execute=False)

    def submit_return(self, client_user_id: str, conv_id: str, order_id: str, item_ids: Any,
                      payment_method_id: Any, user_message: str) -> Dict[str, Any]:
        """兼容旧调用：只在已展示的同参数详情上执行，不会新建详情。"""
        try:
            args = canonical_args("return", {"order_id": order_id, "item_ids": item_ids,
                                             "payment_method_id": payment_method_id})
        except DomainError as ex:
            return {"success": False, "error": str(ex)}
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pending_actions WHERE client_user_id=? AND conv_id=? AND kind='return'"
                " AND args_json=? AND status='displayed'",
                (client_user_id, conv_id, json.dumps(args, ensure_ascii=False, sort_keys=True))).fetchone()
        if exists is None:
            return {"success": False, "error": self._t(
                "提交参数与已展示的详情不一致或尚未展示；请重新准备、展示并确认。",
                "The submission does not match displayed details; prepare, show, and confirm them again.")}
        return self.write_action(client_user_id, conv_id, "return_delivered_order_items", args, user_message)

    def get_item_details(self, client_user_id: str, conv_id: str, item_id: str) -> Dict[str, Any]:
        """商品规格属于公开目录。"""
        key = str(item_id or "").strip()
        with closing(self._connect()) as connection:
            for product in self._products(connection).values():
                variant = (product.get("variants") or {}).get(key)
                if variant is not None:
                    return {"success": True, "item": variant, "product_id": product.get("product_id"),
                            "product_name": product.get("name")}
        return {"success": False, "error": self._t("商品规格不存在", "Item not found")}

    def list_all_product_types(self, client_user_id: str, conv_id: str) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            products = self._products(connection)
        return {"success": True, "product_types": {p.get("name"): pid for pid, p in sorted(
            products.items(), key=lambda kv: str(kv[1].get("name")))}}

    def _record_rejection(
        self,
        client_user_id: str,
        conv_id: str,
        retail_user_id: str,
        order_id: str,
        op_id: str,
        reason: str,
        payload: Dict[str, Any],
    ) -> None:
        with self._tx() as connection:
            self._event(
                connection,
                client_user_id,
                conv_id,
                retail_user_id,
                f"{self._EVENT_PREFIX.get(payload.get('kind', 'return'), 'return')}_rejected",
                {**payload, "reason": reason},
                order_id=order_id,
                op_id=op_id,
            )

    def record_handoff(
        self,
        client_user_id: str,
        conv_id: str,
        summary: str,
        reason: str,
    ) -> Dict[str, Any]:
        handoff_id = f"HO-{uuid.uuid4().hex[:12]}"
        with self._tx() as connection:
            session = self._ensure_session(connection, client_user_id, conv_id)
            retail_user_id = (
                str(session["retail_user_id"])
                if session["retail_user_id"]
                else None
            )
            corrected_summary = _handoff_summary_with_identity(
                summary, retail_user_id, self.language
            )
            self._event(
                connection,
                client_user_id,
                conv_id,
                retail_user_id,
                "handoff_requested",
                {
                    "summary": corrected_summary,
                    "reason": str(reason)[:300],
                    "identity": {
                        "source": "server_session",
                        "bound": bool(retail_user_id),
                        "user_id": retail_user_id,
                    },
                },
                op_id=handoff_id,
            )
        return {
            "success": True,
            "handoff_id": handoff_id,
            "status": "recorded_mock",
            "note": self._t(
                "已记录模拟交接事件；未连接真实坐席，也未创建真实工单。",
                "A simulated handoff event was recorded; no live agent was "
                "connected and no real support ticket was created.",
            ),
        }

    # ── 独立支持工作记录 ─────────────────────────────────────────────────

    def load_support_state(
        self,
        client_user_id: str,
        conv_id: str,
    ) -> Optional[Dict[str, Any]]:
        """读取同一会话的有限支持记录；不存在时返回 None。"""
        if not str(client_user_id or "").strip() or not str(conv_id or "").strip():
            raise SupportStateError("支持记录缺少会话隔离键")
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT format_version, payload_json "
                    "FROM retail_support_state "
                    "WHERE client_user_id = ? AND conv_id = ?",
                    (str(client_user_id), str(conv_id)),
                ).fetchone()
        except sqlite3.Error as ex:
            raise SupportStateError(f"读取支持记录失败: {ex}") from ex
        if row is None:
            return None
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError) as ex:
            raise SupportStateError(f"支持记录 JSON 损坏: {ex}") from ex
        if not isinstance(payload, dict):
            raise SupportStateError("支持记录顶层不是对象")
        if int(row["format_version"]) != SUPPORT_STATE_FORMAT_VERSION:
            raise SupportStateError(
                f"不支持的支持记录格式版本: {row['format_version']}"
            )
        return payload

    def save_support_state(
        self,
        client_user_id: str,
        conv_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """同一事务内覆盖本会话支持记录；失败保留旧记录。"""
        if not str(client_user_id or "").strip() or not str(conv_id or "").strip():
            raise SupportStateError("支持记录缺少会话隔离键")
        if not isinstance(payload, dict):
            raise SupportStateError("支持记录必须是对象")
        version = payload.get("format_version")
        if version != SUPPORT_STATE_FORMAT_VERSION:
            raise SupportStateError(f"不支持的支持记录格式版本: {version}")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as ex:
            raise SupportStateError(f"支持记录无法序列化: {ex}") from ex
        if len(encoded.encode("utf-8")) > MAX_SUPPORT_STATE_BYTES:
            raise SupportStateError(
                f"支持记录超过 {MAX_SUPPORT_STATE_BYTES} 字节，拒绝静默截断"
            )
        now = _now()
        try:
            with self._tx() as connection:
                connection.execute(
                    """
                    INSERT INTO retail_support_state (
                        client_user_id, conv_id, format_version,
                        payload_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(client_user_id, conv_id) DO UPDATE SET
                        format_version = excluded.format_version,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(client_user_id),
                        str(conv_id),
                        SUPPORT_STATE_FORMAT_VERSION,
                        encoded,
                        now,
                    ),
                )
        except sqlite3.Error as ex:
            raise SupportStateError(f"写入支持记录失败: {ex}") from ex

    # ── 供验收使用的快照与统计 ───────────────────────────────────────────

    def business_snapshot(self) -> Dict[str, Any]:
        """返回完整业务表的逻辑数据，供测试和验收比较差异。"""
        with closing(self._connect()) as connection:
            result: Dict[str, Any] = {}
            for table, key in (
                ("users", "user_id"),
                ("orders", "order_id"),
                ("products", "product_id"),
            ):
                rows = connection.execute(f"SELECT payload FROM {table}").fetchall()
                result[table] = {
                    json.loads(row["payload"])[key]: json.loads(row["payload"])
                    for row in rows
                }
            return result

    def stats(self) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            counts = {
                table: connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
                for table in (
                    "users",
                    "orders",
                    "products",
                    "sessions",
                    "pending_actions",
                    "action_operations",
                    "retail_events",
                    "retail_support_state",
                )
            }
            meta = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM meta").fetchall()
            }
        return {
            "db_path": str(self.db_path),
            "seed_sha256": meta.get("seed_sha256"),
            "schema_version": meta.get("schema_version"),
            "counts": counts,
        }


def safe_calculate(expression: str, language: str = "zh") -> Dict[str, Any]:
    """只计算金额所需的四则运算，不执行名称、属性或函数调用。"""
    english = str(language).lower() == "en"
    text = str(expression or "").strip()
    if not text:
        return {
            "success": False,
            "error": "expression cannot be empty" if english else "expression 不能为空",
        }
    if len(text) > 200:
        return {
            "success": False,
            "error": "expression is too long" if english else "expression 过长",
        }
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as ex:
        return {
            "success": False,
            "error": (
                f"Could not parse the expression: {ex.msg}"
                if english
                else f"表达式无法解析: {ex.msg}"
            ),
        }

    def evaluate(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            else:
                if right == 0:
                    raise ValueError(
                        "Cannot divide by 0" if english else "不能除以 0"
                    )
                result = left / right
            if abs(result) > Decimal("1000000000000"):
                raise ValueError(
                    "The result is outside the allowed range"
                    if english
                    else "结果超出允许范围"
                )
            return result
        raise ValueError(
            f"Unsupported expression node: {type(node).__name__}"
            if english
            else f"不允许的表达式节点: {type(node).__name__}"
        )

    try:
        value = evaluate(tree)
    except (ValueError, InvalidOperation) as ex:
        return {"success": False, "error": str(ex)}
    quantized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {
        "success": True,
        "expression": text,
        "result": f"{quantized:.2f}",
        "note": (
            "For arithmetic only; order amounts still come from "
            "get_order_details."
            if english
            else "仅用于金额计算；订单金额仍以 get_order_details 为准。"
        ),
    }
