"""端到端的意图识别。

三路信号通过加权投票融合：
  1. LLM 语义理解（70%）：处理复杂表述和上下文。
  2. Embedding 相似度（20%）：快速匹配模板话术。
  3. 关键词模式（10%）：零延迟兜底。

置信度低于阈值时降级为 OTHER。LLM 与 Embedding 分支并发执行，
关键词匹配同步运行。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

# ── 意图结果数据结构 ─────────────────────────────────────────────────────────
class IntentCategory(Enum):
    QUERY      = "query"       # 信息查询
    COMPLAINT  = "complaint"   # 不满与抱怨
    REQUEST    = "request"     # 操作请求
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 升级或转人工请求
    TECHNICAL  = "technical"   # 技术问题
    BILLING    = "billing"     # 账单或退款
    ACCOUNT    = "account"     # 账户管理
    FEEDBACK   = "feedback"    # 正面反馈
    ORDER_STATUS = "order_status"        # 订单状态
    LOGISTICS = "logistics"              # 物流配送
    REFUND = "refund"                    # 退款或退货
    INVOICE = "invoice"                  # 发票
    PAYMENT_ISSUE = "payment_issue"      # 支付或扣款异常
    ACCOUNT_SECURITY = "account_security" # 账户安全
    TECHNICAL_LOGIN = "technical_login"  # 登录与认证失败
    TECHNICAL_CRASH = "technical_crash"  # 崩溃或错误码
    HUMAN_HANDOFF = "human_handoff"      # 转人工
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

@dataclass
class IntentResult:
    intent:     IntentCategory # 识别出的意图类别
    confidence: float # 置信度，范围 0-1
    urgency:    UrgencyLevel # 紧急程度
    intent_group: str # 所属大类：query、billing、technical、account 或 escalation
    entities:   Dict[str, List[str]]   # 从消息中抽取的实体
    reasoning:  str # LLM 返回的简短说明
    latency_ms: float # 识别耗时（毫秒）
    source_scores: Dict[str, float] = field(default_factory=dict) # 各信号置信度：llm、embedding、pattern、refined_by_pattern


# LLM 示例与 Embedding 匹配共用的少样本模板。
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"],
    IntentCategory.COMPLAINT:  ["等了好几个小时！", "服务太差了！", "一直没人处理！"],
    IntentCategory.REQUEST:    ["帮我取消订单", "我需要修改地址", "请协助退款"],
    IntentCategory.GREETING:   ["你好", "嗨，有人吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "转人工客服", "找你们经理"],
    IntentCategory.TECHNICAL:  ["应用一直崩溃", "无法登录", "出现500错误"],
    IntentCategory.BILLING:    ["为什么扣了两次款？", "申请退款", "发票问题"],
    IntentCategory.ACCOUNT:    ["修改邮箱", "注销账户", "更新个人信息"],
    IntentCategory.FEEDBACK:   ["服务很棒！", "非常满意", "给个好评"],
    IntentCategory.ORDER_STATUS: ["我的订单现在是什么状态？", "订单有没有发货？", "订单处理到哪一步了？"],
    IntentCategory.LOGISTICS: ["快递什么时候到？", "物流一直不更新", "配送要多久？"],
    IntentCategory.REFUND: ["我要申请退款", "退货退款怎么处理？", "退款多久到账？"],
    IntentCategory.INVOICE: ["帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"],
    IntentCategory.PAYMENT_ISSUE: ["为什么重复扣款？", "支付失败怎么办？", "这个月多扣了钱"],
    IntentCategory.ACCOUNT_SECURITY: ["账户被盗了", "发现异常登录", "我要重置密码"],
    IntentCategory.TECHNICAL_LOGIN: ["登录一直报401", "验证码收不到", "无法登录账号"],
    IntentCategory.TECHNICAL_CRASH: ["应用一直崩溃", "页面报500错误", "系统闪退"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找人工", "请升级处理"],
}

_SPECIFIC_INTENTS = {
    IntentCategory.ORDER_STATUS,
    IntentCategory.LOGISTICS,
    IntentCategory.REFUND,
    IntentCategory.INVOICE,
    IntentCategory.PAYMENT_ISSUE,
    IntentCategory.ACCOUNT_SECURITY,
    IntentCategory.TECHNICAL_LOGIN,
    IntentCategory.TECHNICAL_CRASH,
    IntentCategory.HUMAN_HANDOFF,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.BILLING,
    IntentCategory.TECHNICAL,
    IntentCategory.ACCOUNT,
    IntentCategory.ESCALATION,
}

_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.ORDER_STATUS: IntentCategory.QUERY,
    IntentCategory.LOGISTICS: IntentCategory.QUERY,
    IntentCategory.REFUND: IntentCategory.BILLING,
    IntentCategory.INVOICE: IntentCategory.BILLING,
    IntentCategory.PAYMENT_ISSUE: IntentCategory.BILLING,
    IntentCategory.ACCOUNT_SECURITY: IntentCategory.ACCOUNT,
    IntentCategory.TECHNICAL_LOGIN: IntentCategory.TECHNICAL,
    IntentCategory.TECHNICAL_CRASH: IntentCategory.TECHNICAL,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}

# 紧急度关键词。
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，避免引入 numpy 依赖。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """基于 Anthropic 兼容 API 的意图识别器。

    不加载本地模型。模板 Embedding 在首次请求时惰性计算，之后复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ):
        if not model:
            raise ValueError("model is required")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # 本地字符 n-gram 向量始终可用。若客户端暴露 embeddings 资源，
        # _embed_text 会优先使用它，否则退回本地向量。
        self._embedding_enabled = True

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]。
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 与 Embedding 分支并发执行；禁用时跳过 Embedding。
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)
        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )

        # 缓存需要上限：达到 1000 条时丢弃最旧的一半。
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    # ── 在线学习 ──────────────────────────────────────────────────────────────

    def learn(self, message: str, correct: IntentCategory) -> None:
        """把纠正后的样本加入模板，并使相关缓存失效。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)
            self._cache.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别信号 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """信号 1：结合少样本示例和上下文的 LLM 语义理解。"""
        message = self._clean_text(message)
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每个类别只取一个示例，保持 prompt 简短
        )
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是客服意图分析专家。根据示例判断用户意图，返回 JSON。
如果用户问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如退款优先返回 refund，发票优先返回 invoice，登录故障优先返回 technical_login。

        {ctx}
        用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        last_error: Optional[Exception] = None
        # 推理偶尔耗尽预算仍未输出 JSON；有界重试一次，避免静默降级。
        for attempt in range(2):
            try:
                resp = await self.client.messages.create(
                    model=self.model,
                    # 意图判断会先输出推理，需要给完整 JSON 留出足够空间。
                    max_tokens=2048,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = extract_text_content(resp.content)
                s, e = raw.find("{"), raw.rfind("}") + 1
                if s < 0 or e <= s:
                    raise ValueError("模型未返回 JSON 文本")
                data = json.loads(raw[s:e])
                try:
                    data["intent"] = IntentCategory(data["intent"])
                except ValueError:
                    data["intent"] = IntentCategory.OTHER
                return data
            except Exception as ex:
                last_error = ex
                if attempt == 0:
                    logger.warning(f"LLM 识别失败，重试一次: {ex}")
        logger.warning(f"LLM 识别失败: {last_error}")
        return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """信号 2：与模板话术做 Embedding 相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """信号 3：同步的关键词兜底。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工"],
            IntentCategory.ORDER_STATUS: ["订单状态", "发货了吗", "处理到哪", "order status"],
            IntentCategory.LOGISTICS: ["物流", "快递", "配送", "运单", "delivery", "shipping"],
            IntentCategory.REFUND: ["退款", "退货", "refund", "return"],
            IntentCategory.INVOICE: ["发票", "抬头", "税号", "invoice"],
            IntentCategory.PAYMENT_ISSUE: ["重复扣款", "多扣", "支付失败", "扣费", "payment failed"],
            IntentCategory.ACCOUNT_SECURITY: ["被盗", "异常登录", "重置密码", "两步验证", "安全"],
            IntentCategory.TECHNICAL_LOGIN: ["无法登录", "登录失败", "401", "验证码"],
            IntentCategory.TECHNICAL_CRASH: ["崩溃", "闪退", "500", "报错", "crash"],
        }
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible", "等了很久"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票融合 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各信号得分。"""
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        group_scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            weighted_score = w * conf
            scores[cat] = scores.get(cat, 0.0) + weighted_score
            group = _INTENT_GROUPS.get(cat, cat)
            group_scores[group] = group_scores.get(group, 0.0) + weighted_score

        # 先选出胜出的大类，再在该类内选出最佳的具体意图。
        best_group = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_group]
        best = max(
            (cat for cat in scores if _INTENT_GROUPS.get(cat, cat) == best_group),
            key=scores.get,
        )
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if (best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS
                and _INTENT_GROUPS.get(pat_intent, pat_intent) == best_group
                and pat_conf >= 0.5 and best_score < 0.8):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体抽取 ──────────────────────────────────────────────────────────────

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则抽取高价值实体，避免额外的 LLM 调用。"""
        message = self._clean_text(message)
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """惰性计算全部模板的 Embedding；每个类别只执行一次。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """构造文本向量。

        客户端提供 embeddings 资源时优先使用；否则退回本地字符 n-gram
        向量，使融合逻辑在无 Embedding 服务时仍可工作。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """无远端 Embedding 服务时使用的稳定哈希字符 n-gram 向量。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        # 缓存包含实体和紧急度，因此键必须覆盖历史窗口内所有可能
        # 影响识别的输入字符。
        payload = {"message": self._clean_text(message)}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", "")),
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 一个明确的业务关键词即可给出可用置信度；命中越多分数越高。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """丢弃无法编码为 UTF-8 的字符（例如孤立代理项）。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
