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
