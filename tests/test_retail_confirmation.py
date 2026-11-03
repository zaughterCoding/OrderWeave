"""确认分类器：只采纳已知操作的明确判定，漏判默认不批准，失败时交由规则回退。"""
import asyncio

from agents.retail_confirmation import classify_confirmations


class _Client:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error
        self.messages = self

    async def create(self, **kwargs):
        if self.error:
            raise self.error
        block = type("B", (), {"type": "text", "text": self.text})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn"})()


PENDING = [{"op_id": "OP-1", "display_text": "cancel"}, {"op_id": "OP-2", "display_text": "address"}]


def test_unknown_and_missing_operations_are_not_approved():
    client = _Client('{"decisions": [{"op_id": "OP-1", "approved": true}, {"op_id": "OP-9", "approved": true}]}')
    result = asyncio.run(classify_confirmations(client, "m", "yes", "", PENDING))
    assert {r["op_id"]: r["approved"] for r in result} == {"OP-1": True, "OP-2": False}


def test_failure_returns_none_for_rule_fallback():
    assert asyncio.run(classify_confirmations(_Client(error=RuntimeError("x")), "m", "yes", "", PENDING)) is None
    assert asyncio.run(classify_confirmations(_Client("not json"), "m", "yes", "", PENDING)) is None
