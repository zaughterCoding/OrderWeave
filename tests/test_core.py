import asyncio
from types import SimpleNamespace

import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_utils import extract_text_content


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.BILLING, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.BILLING
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.BILLING, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


def test_extract_text_content_accepts_mixed_blocks():
    content = [
        "plain string",
        {"type": "text", "text": "dict block"},
        SimpleNamespace(type="text", text="object block"),
        {"text": "dict without type"},
        {"type": "tool_use", "id": "tool-1"},
        SimpleNamespace(type="thinking", thinking="ignored"),
        {"type": "text", "text": ""},
    ]

    assert extract_text_content(content) == "\n".join(
        ["plain string", "dict block", "object block", "dict without type"]
    )
    assert extract_text_content(None) == ""


class _FakeMessages:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)]
        )


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


class _PromptFakeMessages:
    """Return a different intent depending on a marker in the prompt."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        intent = "technical_login" if "TAIL-B" in prompt else "refund"
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text=(
                        f'{{"intent": "{intent}", "confidence": 0.9, '
                        '"reasoning": "fake intent"}'
                    ),
                )
            ]
        )


class _PromptFakeClient:
    def __init__(self):
        self.messages = _PromptFakeMessages()


def make_cache_recognizer(client):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.client = client
    recognizer.model = "fake-model"
    recognizer.threshold = 0.5
    recognizer._embedding_enabled = False
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0
    return recognizer


def test_intent_cache_does_not_share_entities_between_long_messages():
    client = _PromptFakeClient()
    recognizer = make_cache_recognizer(client)
    prefix = "同一段较长的客服工单描述" * 30

    first = asyncio.run(
        recognizer.recognize(f"{prefix} order_id: AAAA1111")
    )
    second = asyncio.run(
        recognizer.recognize(f"{prefix} order_id: BBBB2222")
    )

    assert first is not second
    assert first.entities["order_id"] == ["AAAA1111"]
    assert second.entities["order_id"] == ["BBBB2222"]
    assert len(client.messages.calls) == 2


def test_intent_cache_uses_the_full_recent_history_window():
    client = _PromptFakeClient()
    recognizer = make_cache_recognizer(client)
    prefix = "最近对话的公共前缀" * 30

    first = asyncio.run(
        recognizer.recognize(
            "请查看这个问题",
            history=[{"role": "user", "content": prefix + "TAIL-A"}],
        )
    )
    second = asyncio.run(
        recognizer.recognize(
            "请查看这个问题",
            history=[{"role": "user", "content": prefix + "TAIL-B"}],
        )
    )

    assert first.intent is IntentCategory.REFUND
    assert second.intent is IntentCategory.TECHNICAL_LOGIN
    assert len(client.messages.calls) == 2


def test_identical_message_and_history_still_hit_the_cache():
    client = _PromptFakeClient()
    recognizer = make_cache_recognizer(client)
    history = [{"role": "user", "content": "我要申请退款"}]

    first = asyncio.run(recognizer.recognize("请查看退款进度", history=history))
    second = asyncio.run(recognizer.recognize("请查看退款进度", history=history))

    assert second is first
    assert len(client.messages.calls) == 1


def test_recognize_with_fake_client():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.client = _FakeClient(
        '{"intent": "refund", "confidence": 0.9, "reasoning": "user asks for a refund"}'
    )
    recognizer.model = "fake-model"
    recognizer.threshold = 0.5
    recognizer._embedding_enabled = True
    recognizer._tpl_embeddings = {}
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0

    history = [{"role": "user", "content": "订单有问题"}]
    result = asyncio.run(recognizer.recognize("我要申请退款", history=history))

    assert result.intent is IntentCategory.REFUND
    assert result.confidence >= 0.5
    assert result.intent_group == "billing"
    assert result.urgency is UrgencyLevel.LOW
    assert result.source_scores["llm"] == pytest.approx(0.9)
    assert recognizer.client.messages.calls == 1

    cached = asyncio.run(recognizer.recognize("我要申请退款", history=history))
    assert cached is result
    assert recognizer.client.messages.calls == 1
    assert recognizer.cache_stats == {
        "size": 1,
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
    }
