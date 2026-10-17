import asyncio
import os
import uuid
from types import SimpleNamespace

from memory.conversation_memory import MemoryManager, MsgRole

REDIS_URL = os.getenv(
    "REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))


class PromptFakeMessages:
    """Deterministic fake model for summaries and profile extraction."""

    def __init__(self, marker):
        self.marker = marker

    async def create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        if "合并" in prompt:
            text = f"合并摘要，标记 {self.marker}"
        elif "用户偏好" in prompt:
            text = (
                '{"preferences": ["偏好退款"], '
                '"entities": {"问题类型": ["退款"]}}'
            )
        else:
            text = f"用户咨询退款，标记 {self.marker}"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)]
        )


class PromptFakeAnthropic:
    def __init__(self, marker):
        self.messages = PromptFakeMessages(marker)


def test_real_redis_and_chroma_memory_round_trip(monkeypatch):
    marker = f"memory-marker-{uuid.uuid4().hex}"
    user_a = f"memory-test-a-{uuid.uuid4().hex}"
    user_b = f"memory-test-b-{uuid.uuid4().hex}"
    conv_a1 = f"conv-a1-{uuid.uuid4().hex}"
    conv_a2 = f"conv-a2-{uuid.uuid4().hex}"
    conv_b = f"conv-b-{uuid.uuid4().hex}"

    fake_model = PromptFakeAnthropic(marker)
    monkeypatch.setattr(
        "memory.conversation_memory.AsyncAnthropic",
        lambda **kwargs: fake_model,
    )

    manager = MemoryManager(
        redis_url=REDIS_URL,
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        api_key="test-key",
        model="fake-model",
    )

    async def scenario():
        assert manager._use_server is True, (
            "independent Chroma server is required at "
            f"{CHROMA_HOST}:{CHROMA_PORT}; run compose.yml first"
        )

        for index in range(15):
            await manager.add_message(
                user_a,
                conv_a1,
                MsgRole.USER,
                f"m{index} {marker}",
            )

        context = await manager.get_context(user_a, conv_a1, query="退款")
        assert [message.content for message in context.recent_messages] == [
            f"m{index} {marker}" for index in range(10, 15)
        ]
        assert marker in context.summary
        assert any(marker in item for item in context.relevant_history)

        cross_session = await manager.get_context(user_a, conv_a2, query="退款")
        assert any(marker in item for item in cross_session.relevant_history)

        await manager.update_profile(user_a, conv_a1)
        profiled = await manager.get_context(user_a, conv_a1, query="退款")
        assert profiled.user_profile.get("preferences") == ["偏好退款"]

        await manager.add_message(
            user_a, conv_a1, MsgRole.USER, f"m15 {marker}"
        )
        latest = await manager.get_context(user_a, conv_a1, query="退款")
        assert [
            message.content for message in latest.recent_messages[-5:]
        ] == [f"m{index} {marker}" for index in range(11, 16)]

        other_user = await manager.get_context(user_b, conv_b, query="退款")
        assert other_user.relevant_history == []
        assert other_user.user_profile == {}

    async def cleanup():
        await asyncio.to_thread(
            manager._episodic.delete, where={"user_id": user_a}
        )
        await asyncio.to_thread(
            manager._episodic.delete, where={"user_id": user_b}
        )
        await asyncio.to_thread(
            manager._profile.delete,
            ids=[
                manager._profile_doc_id(user_a),
                manager._profile_doc_id(user_b),
            ],
        )
        await manager._redis.delete(
            manager._wm_key(user_a, conv_a1),
            manager._summary_key(user_a, conv_a1),
            manager._wm_key(user_a, conv_a2),
            manager._summary_key(user_a, conv_a2),
            manager._wm_key(user_b, conv_b),
            manager._summary_key(user_b, conv_b),
        )
        await manager.close()

    async def run():
        try:
            await scenario()
        finally:
            await cleanup()

    asyncio.run(run())
