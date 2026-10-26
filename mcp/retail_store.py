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

    def active_pending(
        self,
        client_user_id: str,
        conv_id: str,
    ) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM pending_returns"
                " WHERE client_user_id=? AND conv_id=?"
                " AND status IN ('prepared', 'displayed')"
                " ORDER BY created_at, op_id",
                (client_user_id, conv_id),
            ).fetchall()
        return [
            {
                **{key: row[key] for key in row.keys()},
                "items": self._parse_items(row["items_json"]),
                "amount": _fmt_cents(int(row["amount_cents"])),
            }
            for row in rows
        ]

    def cancel_pending(
        self,
        client_user_id: str,
        conv_id: str,
        reason: str = "user_cancelled",
    ) -> int:
        """取消本会话尚未提交的待确认详情；不修改订单或退货申请。"""
        now = _now()
        try:
            with self._tx() as connection:
                rows = connection.execute(
                    "SELECT op_id, retail_user_id, order_id FROM pending_returns"
                    " WHERE client_user_id=? AND conv_id=?"
                    " AND status IN ('prepared', 'displayed')",
                    (client_user_id, conv_id),
                ).fetchall()
                if not rows:
                    return 0
                connection.execute(
                    "UPDATE pending_returns SET status='cancelled', updated_at=?,"
                    " note=? WHERE client_user_id=? AND conv_id=?"
                    " AND status IN ('prepared', 'displayed')",
                    (
                        now,
                        f"cancelled: {str(reason or 'user_cancelled')[:120]}",
                        client_user_id,
                        conv_id,
                    ),
                )
                for row in rows:
                    self._event(
                        connection,
                        client_user_id,
                        conv_id,
                        row["retail_user_id"],
                        "return_cancelled",
                        {"reason": str(reason or "user_cancelled")[:120]},
                        order_id=row["order_id"],
                        op_id=row["op_id"],
                    )
                return len(rows)
        except sqlite3.Error as ex:
            raise SupportStateError(f"取消待确认退货失败: {ex}") from ex

    def _validate_item_request(
        self,
        order: Dict[str, Any],
        item_ids: Any,
    ) -> Tuple[List[str], List[Dict[str, Any]], int]:
        if not isinstance(item_ids, list):
            raise RetailStoreError(
                self._t("item_ids 必须是数组", "item_ids must be an array")
            )
        if not item_ids:
            raise RetailStoreError(
                self._t(
                    "item_ids 不能为空；空列表不能提交退货",
                    "item_ids cannot be empty; an empty list cannot be returned",
                )
            )
        if len(item_ids) > 20:
            raise RetailStoreError(
                self._t(
                    "一次最多提交 20 件退货商品",
                    "At most 20 return items can be submitted at once",
                )
            )
        requested: List[str] = []
        for item_id in item_ids:
            if not isinstance(item_id, str) or not item_id.strip():
                raise RetailStoreError(
                    self._t(
                        "item_ids 中的每一项都必须是非空字符串",
                        "Each entry in item_ids must be a non-empty string",
                    )
                )
            requested.append(item_id.strip())

        available = Counter(
            str(item.get("item_id")) for item in (order.get("items") or [])
        )
        wanted = Counter(requested)
        for item_id, count in wanted.items():
            if available.get(item_id, 0) == 0:
                raise RetailStoreError(
                    self._t(
                        f"商品 {item_id} 不属于该订单；请使用订单中的具体 item_id。",
                        f"Item {item_id} is not in this order; use the exact "
                        "item_id from the order.",
                    )
                )
            if count > available[item_id]:
                raise RetailStoreError(
                    self._t(
                        f"商品 {item_id} 申请 {count} 件，超过订单中的 {available[item_id]} 件。",
                        f"Requested {count} of item {item_id}, but the order "
                        f"contains {available[item_id]}.",
                    )
                )

        order_items = order.get("items") or []
        lines: List[Dict[str, Any]] = []
        total_cents = 0
        for item_id, count in wanted.items():
            source = next(
                item for item in order_items if str(item.get("item_id")) == item_id
            )
            unit_cents = _to_cents(source.get("price"))
            line_cents = unit_cents * count
            total_cents += line_cents
            lines.append(
                {
                    "item_id": item_id,
                    "product_id": str(source.get("product_id", "")),
                    "name": str(source.get("name", "")),
                    "options": source.get("options") or {},
                    "unit_price": _fmt_cents(unit_cents),
                    "quantity": count,
                    "line_amount": _fmt_cents(line_cents),
                }
            )
        if total_cents <= 0:
            raise RetailStoreError(
                self._t(
                    "退货金额必须大于 0",
                    "The return amount must be greater than 0",
                )
            )
        return requested, lines, total_cents

    @staticmethod
    def _original_payment_id(order: Dict[str, Any]) -> str:
        for item in order.get("payment_history") or []:
            payment_id = str(item.get("payment_method_id") or "").strip()
            if payment_id:
                return payment_id
        return ""

    def _resolve_payment(
        self,
        user: Dict[str, Any],
        order: Dict[str, Any],
        payment_method_id: Any,
    ) -> Tuple[str, Dict[str, Any], bool]:
        payment_id = str(payment_method_id or "").strip()
        if not payment_id:
            raise RetailStoreError(
                self._t("payment_method_id 不能为空", "payment_method_id cannot be empty")
            )
        payments = user.get("payment_methods") or {}
        payment = payments.get(payment_id)
        original_id = self._original_payment_id(order)
        is_original = bool(original_id) and payment_id == original_id
        if is_original:
            return payment_id, payment or {}, True
        if payment is None:
            raise RetailStoreError(
                self._t(
                    "退款去向不属于当前用户的支付资料；只能使用原支付方式或本人已有礼品卡。",
                    "The refund destination is not among the current user's "
                    "payment methods; use the original payment method or one "
                    "of the user's gift cards.",
                )
            )
        if str(payment.get("source")) != "gift_card":
            raise RetailStoreError(
                self._t(
                    "退款去向必须是原支付方式或当前用户已有的礼品卡；"
                    "不能改到其他信用卡、PayPal 或他人支付方式。",
                    "The refund destination must be the original payment "
                    "method or one of the user's gift cards; it cannot be "
                    "another credit card, PayPal, or someone else's method.",
                )
            )
        return payment_id, payment, False

    def prepare_return(
        self,
        client_user_id: str,
        conv_id: str,
        order_id: str,
        item_ids: Any,
        payment_method_id: Any,
    ) -> Dict[str, Any]:
        normalized = normalize_order_id(order_id)
        turn = self.current_turn(client_user_id, conv_id)
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
                status = str(order.get("status", ""))
                if status != "delivered":
                    return {
                        "success": False,
                        "error": self._t(
                            f"订单 {normalized} 当前状态为 {status}；"
                            "只有 delivered（已送达）订单可以提交退货申请。",
                            f"Order {normalized} is currently {status}; only "
                            "delivered orders can be returned.",
                        ),
                    }
                requested, lines, total_cents = self._validate_item_request(
                    order, item_ids
                )
                payment_id, payment, is_original = self._resolve_payment(
                    user, order, payment_method_id
                )
                order_version = int(row["version"])
        except RetailStoreError as ex:
            return {"success": False, "error": str(ex)}

        op_id = f"RT-{uuid.uuid4().hex[:12]}"
        payment_description = _payment_label(payment, payment_id, self.language)
        if self.language == "en":
            payment_note = "original payment method" if is_original else "existing gift card"
            display_lines = "\n".join(
                f"- {line['name']} (item_id: {line['item_id']}, "
                f"{_options_text(line['options'], 'en')}) x {line['quantity']}, "
                f"unit price {_fmt_usd(_to_cents(line['unit_price']))}, "
                f"line total {_fmt_usd(_to_cents(line['line_amount']))}"
                for line in lines
            )
            display_text = (
                "RETURN REQUEST DETAILS (PENDING CONFIRMATION)\n"
                f"Operation ID: {op_id}\n"
                f"Order: {normalized}\n"
                "Order status: delivered\n"
                f"Items to return:\n{display_lines}\n"
                f"Number of items: {len(requested)}\n"
                f"Estimated refund amount: {_fmt_usd(total_cents)} "
                "(based on the order price; this is not a completed refund)\n"
                f"Refund destination: {payment_description} ({payment_note})\n"
                "Reply with an explicit confirmation, such as \"Yes, please "
                "submit this return\", to proceed. After submission the order "
                "status becomes return requested. This does not mean the refund "
                "has been issued, an email has been sent, or the warehouse has "
                "received the items."
            )
        else:
            payment_note = "原支付方式" if is_original else "本人已有礼品卡"
            display_lines = "\n".join(
                f"- {line['name']}（item_id: {line['item_id']}，{_options_text(line['options'])}）"
                f" × {line['quantity']}，单价 ¥{line['unit_price']}，"
                f"小计 ¥{line['line_amount']}"
                for line in lines
            )
            display_text = (
                "【退货申请详情（待确认）】\n"
                f"操作编号：{op_id}\n"
                f"订单号：{normalized}\n"
                f"订单状态：delivered（已送达）\n"
                f"退货商品：\n{display_lines}\n"
                f"退货件数：{len(requested)} 件\n"
                f"预计申请金额：¥{_fmt_cents(total_cents)}（按订单成交价计算，不代表已到账）\n"
                f"退款去向：{payment_description}（{payment_note}）\n"
                "请回复“确认，就按这些申请退货”以提交。提交后订单状态将变为 "
                "return requested（已申请退货），不代表退款已到账、邮件已发送或仓库已收货。"
            )
        items_payload = {"item_ids": requested, "lines": lines}
        now = _now()
        with self._tx() as connection:
            self._ensure_session(connection, client_user_id, conv_id)
            previous = connection.execute(
                "SELECT op_id FROM pending_returns"
                " WHERE client_user_id=? AND conv_id=? AND order_id=?"
                " AND status IN ('prepared', 'displayed')",
                (client_user_id, conv_id, normalized),
            ).fetchall()
            connection.execute(
                "UPDATE pending_returns SET status='superseded', superseded_by=?,"
                " updated_at=? WHERE client_user_id=? AND conv_id=? AND order_id=?"
                " AND status IN ('prepared', 'displayed')",
                (op_id, now, client_user_id, conv_id, normalized),
            )
            connection.execute(
                "INSERT INTO pending_returns"
                "(op_id, client_user_id, conv_id, retail_user_id, order_id,"
                " items_json, payment_method_id, amount_cents, display_text, status,"
                " prepared_turn, displayed_turn, display_appended, order_version,"
                " created_at, updated_at, consumed_at, superseded_by, note)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, NULL, 0, ?, ?, ?, NULL, NULL, ?)",
                (
                    op_id,
                    client_user_id,
                    conv_id,
                    user_id,
                    normalized,
                    json.dumps(items_payload, ensure_ascii=False, sort_keys=True),
                    payment_id,
                    total_cents,
                    display_text,
                    turn,
                    order_version,
                    now,
                    now,
                    "pending details must be shown to the current user",
                ),
            )
            self._event(
                connection,
                client_user_id,
                conv_id,
                user_id,
                "return_prepared",
                {
                    "item_ids": requested,
                    "payment_method_id": payment_id,
                    "amount_cents": total_cents,
                    "superseded": [str(item["op_id"]) for item in previous],
                },
                order_id=normalized,
                op_id=op_id,
            )
        return {
            "success": True,
            "op_id": op_id,
            "order_id": normalized,
            "items": lines,
            "item_ids": requested,
            "payment_method_id": payment_id,
            "payment_description": payment_description,
            "estimated_amount": (
                _fmt_usd(total_cents)
                if self.language == "en"
                else _fmt_cents(total_cents)
            ),
            "amount_cents": total_cents,
            "display_text": display_text,
            "status": "prepared",
            "note": self._t(
                "详情尚未展示；必须把 display_text 原样展示给当前用户，并等待下一轮明确确认。",
                "The details have not been shown yet; show display_text exactly "
                "to the current user and wait for an explicit confirmation in "
                "the next turn.",
            ),
        }

    def mark_pending_displayed(
        self,
        client_user_id: str,
        conv_id: str,
        op_id: str,
        turn: int,
        appended: bool,
    ) -> bool:
        with self._tx() as connection:
            row = connection.execute(
                "SELECT * FROM pending_returns WHERE op_id=?", (op_id,)
            ).fetchone()
            if (
                row is None
                or row["client_user_id"] != client_user_id
                or row["conv_id"] != conv_id
                or row["status"] != "prepared"
                or int(row["prepared_turn"]) != int(turn)
            ):
                return False
            connection.execute(
                "UPDATE pending_returns SET status='displayed', displayed_turn=?,"
                " display_appended=?, updated_at=? WHERE op_id=? AND status='prepared'",
                (turn, 1 if appended else 0, _now(), op_id),
            )
            self._event(
                connection,
                client_user_id,
                conv_id,
                str(row["retail_user_id"]),
                "return_displayed",
                {"appended": bool(appended), "turn": int(turn)},
                order_id=str(row["order_id"]),
                op_id=op_id,
            )
            return True

    def submit_return(
        self,
        client_user_id: str,
        conv_id: str,
        order_id: str,
        item_ids: Any,
        payment_method_id: Any,
        user_message: str,
    ) -> Dict[str, Any]:
        normalized = normalize_order_id(order_id)
        turn = self.current_turn(client_user_id, conv_id)
        try:
            wanted_ids, _, _ = self._validate_item_request_shape(item_ids)
            payment_id = str(payment_method_id or "").strip()
            if not payment_id:
                raise RetailStoreError("payment_method_id 不能为空")
            with closing(self._connect()) as connection:
                user_id = self._require_binding(connection, client_user_id, conv_id)
                displayed_rows = connection.execute(
                    "SELECT * FROM pending_returns"
                    " WHERE client_user_id=? AND conv_id=? AND status='displayed'"
                    " ORDER BY created_at, op_id",
                    (client_user_id, conv_id),
                ).fetchall()
                prepared_rows = connection.execute(
                    "SELECT * FROM pending_returns"
                    " WHERE client_user_id=? AND conv_id=? AND status='prepared'",
                    (client_user_id, conv_id),
                ).fetchall()
        except RetailStoreError as ex:
            return {"success": False, "error": str(ex)}

        if not displayed_rows:
            if prepared_rows:
                return {
                    "success": False,
                    "error": self._t(
                        "退货详情尚未展示给用户，不能提交；请先展示并等待下一轮明确确认。",
                        "The return details have not been shown to the user; "
                        "show them first and wait for an explicit confirmation "
                        "in the next turn.",
                    ),
                }
            return {
                "success": False,
                "error": self._t(
                    "没有已展示的待确认详情；请先准备并展示退货详情。",
                    "There are no displayed pending details; prepare and show "
                    "the return details first.",
                ),
            }

        displayed = [
            {
                **{key: row[key] for key in row.keys()},
                "items": self._parse_items(row["items_json"]),
            }
            for row in displayed_rows
        ]
        match = None
        for candidate in displayed:
            signature = Counter(
                str(item) for item in candidate["items"].get("item_ids", [])
            )
            if (
                candidate["order_id"] == normalized
                and signature == Counter(wanted_ids)
                and candidate["payment_method_id"] == payment_id
            ):
                match = candidate
                break
        if match is None:
            return {
                "success": False,
                "error": self._t(
                    "提交参数与已展示的详情不一致；请重新准备、展示并确认新详情。",
                    "The submission does not match the displayed details; "
                    "prepare, show, and confirm the new details again.",
                ),
            }
        if int(match["prepared_turn"]) == turn:
            return {
                "success": False,
                "error": self._t(
                    "准备详情的同一轮不能提交；请先向用户展示并等待明确确认。",
                    "Details cannot be submitted in the same turn they are "
                    "prepared; show them and wait for an explicit confirmation.",
                ),
            }
        if match["displayed_turn"] is None or int(match["displayed_turn"]) >= turn:
            return {
                "success": False,
                "error": self._t(
                    "确认必须发生在详情实际展示之后；请重新展示全部详情后再确认。",
                    "Confirmation must come after the details were actually "
                    "shown; show all details again before confirming.",
                ),
            }

        decision = evaluate_confirmation(
            user_message, match, displayed, language=self.language
        )
        if not decision.approved:
            self._record_rejection(
                client_user_id,
                conv_id,
                str(match["retail_user_id"]),
                normalized,
                str(match["op_id"]),
                decision.reason,
                {"item_ids": wanted_ids, "payment_method_id": payment_id},
            )
            return {
                "success": False,
                "error": self._t(
                    f"未获得当前用户的明确确认：{decision.reason}",
                    "The current user has not given explicit confirmation: "
                    f"{decision.reason}",
                ),
                "confirmation_scope": decision.scope,
                "error_code": "confirmation_required",
                "confirmation_hint": self._t(
                    "退货尚未提交。请先解决疑问或修改详情；如确认按已展示内容办理，"
                    f"请在下一条消息明确回复“确认，按已展示详情提交订单{normalized}的退货”。",
                    "The return was not submitted. Resolve any questions or changes first. "
                    "If you want the unchanged displayed details submitted, reply in a new message: "
                    f"Yes, please submit this return for order {normalized} as listed.",
                ),
            }

        try:
            with self._tx() as connection:
                pending_row = connection.execute(
                    "SELECT * FROM pending_returns WHERE op_id=?",
                    (match["op_id"],),
                ).fetchone()
                if (
                    pending_row is None
                    or pending_row["status"] != "displayed"
                    or pending_row["client_user_id"] != client_user_id
                    or pending_row["conv_id"] != conv_id
                ):
                    raise RetailStoreError(
                        self._t(
                            "该确认已使用或待确认状态已变化",
                            "This confirmation was already used or its pending "
                            "state has changed",
                        )
                    )
                user_id = self._require_binding(connection, client_user_id, conv_id)
                if str(pending_row["retail_user_id"]) != user_id:
                    raise RetailStoreError(
                        self._t(
                            "会话绑定用户与待确认记录不一致",
                            "The session-bound user does not match the pending "
                            "confirmation record",
                        )
                    )

                order_row = self._order_row(connection, normalized, user_id)
                if order_row is None:
                    raise RetailStoreError(
                        self._t(
                            "订单不属于当前用户或已不存在",
                            "The order does not belong to the current user or "
                            "no longer exists",
                        )
                    )
                order_version = int(order_row["version"])
                if order_version != int(pending_row["order_version"]):
                    raise RetailStoreError(
                        self._t(
                            "订单在展示后已发生变化；请重新查询、展示并确认。",
                            "The order changed after the details were shown; "
                            "look it up, show the details, and confirm again.",
                        )
                    )
                order = json.loads(order_row["payload"])
                if str(order.get("status")) != "delivered":
                    raise RetailStoreError(
                        self._t(
                            f"订单当前状态为 {order.get('status')}；只有 delivered 可提交退货。",
                            f"The order is currently {order.get('status')}; only "
                            "delivered orders can be returned.",
                        )
                    )
                requested, _, total_cents = self._validate_item_request(
                    order, json.loads(pending_row["items_json"]).get("item_ids")
                )
                if total_cents != int(pending_row["amount_cents"]):
                    raise RetailStoreError(
                        self._t(
                            "订单金额已变化；请重新展示并确认。",
                            "The order amount changed; show the details and "
                            "confirm again.",
                        )
                    )
                user = self._user_record(connection, user_id)
                resolved_id, _, _ = self._resolve_payment(
                    user, order, pending_row["payment_method_id"]
                )
                if resolved_id != pending_row["payment_method_id"]:
                    raise RetailStoreError(
                        self._t(
                            "退款去向已变化；请重新展示并确认。",
                            "The refund destination changed; show the details "
                            "and confirm again.",
                        )
                    )

                consumed = connection.execute(
                    "UPDATE pending_returns SET status='consumed', consumed_at=?,"
                    " updated_at=? WHERE op_id=? AND status='displayed'",
                    (_now(), _now(), match["op_id"]),
                )
                if consumed.rowcount != 1:
                    raise RetailStoreError("该确认已被使用，不能重复提交")

                updated_order = copy.deepcopy(order)
                updated_order["status"] = "return requested"
                # 与官方工具一致，订单中的 return_items 按 item_id 排序保存。
                updated_order["return_items"] = sorted(requested)
                updated_order["return_payment_method_id"] = resolved_id
                result = connection.execute(
                    "UPDATE orders SET status=?, version=version+1, payload=?"
                    " WHERE order_id=? AND version=? AND status='delivered'",
                    (
                        "return requested",
                        json.dumps(updated_order, ensure_ascii=False, sort_keys=True),
                        normalized,
                        order_version,
                    ),
                )
                if result.rowcount != 1:
                    raise RetailStoreError(
                        self._t(
                            "订单状态已变化，退货写入未生效",
                            "The order status changed; the return was not written",
                        )
                    )
                connection.execute(
                    "INSERT INTO return_operations"
                    "(op_id, client_user_id, conv_id, retail_user_id, order_id,"
                    " items_json, payment_method_id, amount_cents, status, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'return requested', ?)",
                    (
                        match["op_id"],
                        client_user_id,
                        conv_id,
                        user_id,
                        normalized,
                        pending_row["items_json"],
                        resolved_id,
                        int(pending_row["amount_cents"]),
                        _now(),
                    ),
                )
                self._event(
                    connection,
                    client_user_id,
                    conv_id,
                    user_id,
                    "return_requested",
                    {
                        "item_ids": requested,
                        "payment_method_id": resolved_id,
                        "amount_cents": int(pending_row["amount_cents"]),
                        "confirmation_scope": decision.scope,
                    },
                    order_id=normalized,
                    op_id=match["op_id"],
                )
        except RetailStoreError as ex:
            self._record_rejection(
                client_user_id,
                conv_id,
                str(match["retail_user_id"]),
                normalized,
                str(match["op_id"]),
                str(ex),
                {"item_ids": wanted_ids, "payment_method_id": payment_id},
            )
            return {"success": False, "error": str(ex)}
        except Exception as ex:
            self._record_rejection(
                client_user_id,
                conv_id,
                str(match["retail_user_id"]),
                normalized,
                str(match["op_id"]),
                f"unexpected: {ex}",
                {"item_ids": wanted_ids, "payment_method_id": payment_id},
            )
            return {
                "success": False,
                "error": self._t(
                    f"退货写入失败: {ex}",
                    f"The return could not be written: {ex}",
                ),
            }

        return {
            "success": True,
            "op_id": match["op_id"],
            "order_id": normalized,
            "status": "return requested",
            "return_items": sorted(requested),
            "return_payment_method_id": resolved_id,
            "estimated_amount": (
                _fmt_usd(int(match["amount_cents"]))
                if self.language == "en"
                else _fmt_cents(int(match["amount_cents"]))
            ),
            "note": self._t(
                "已提交退货申请，订单状态为 return requested。"
                "这不表示退款已到账、仓库已收货、邮件已发送或已创建真实工单。",
                "The return request was submitted and the order status is "
                "return requested. This does not mean the refund has been "
                "issued, the warehouse received the items, an email was sent, "
                "or a real support ticket was created.",
            ),
        }

    @staticmethod
    def _validate_item_request_shape(item_ids: Any) -> Tuple[List[str], int, int]:
        """提交前的参数形状检查；完整归属与数量在事务内复查。"""
        if not isinstance(item_ids, list):
            raise RetailStoreError(
                self._t("item_ids 必须是数组", "item_ids must be an array")
            )
        if not item_ids:
            raise RetailStoreError(
                self._t("item_ids 不能为空", "item_ids cannot be empty")
            )
        if len(item_ids) > 20:
            raise RetailStoreError(
                self._t(
                    "一次最多提交 20 件退货商品",
                    "At most 20 return items can be submitted at once",
                )
            )
        result: List[str] = []
        for item_id in item_ids:
            if not isinstance(item_id, str) or not item_id.strip():
                raise RetailStoreError(
                    self._t(
                        "item_ids 中的每一项都必须是非空字符串",
                        "Each entry in item_ids must be a non-empty string",
                    )
                )
            result.append(item_id.strip())
        return result, len(result), len(set(result))

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
                "return_rejected",
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
                    "pending_returns",
                    "return_operations",
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
