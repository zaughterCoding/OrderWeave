import asyncio
from types import SimpleNamespace

from memory.conversation_memory import MemoryManager, MsgRole


class FakeRedis:
    """Minimal async Redis list/key store with LPUSH newest-at-head semantics."""

    def __init__(self):
        self.lists = {}
        self.values = {}
        self.expires = {}

    async def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    async def lrange(self, key, start, end):
        items = self.lists.get(key, [])
        if end == -1:
            return items[start:]
        return items[start : end + 1]

    async def llen(self, key):
        return len(self.lists.get(key, []))

    async def expire(self, key, ttl):
        self.expires[key] = ttl
        return True

    async def delete(self, *keys):
        deleted = 0
        for key in keys:
            if key in self.lists:
                del self.lists[key]
                deleted += 1
            if key in self.values:
                del self.values[key]
                deleted += 1
        return deleted

    async def get(self, key):
        return self.values.get(key)

    async def setex(self, key, ttl, value):
        self.values[key] = value
        self.expires[key] = ttl
        return True

    async def aclose(self):
        return None


class FakeCollection:
    def __init__(self):
        self.added = []

    def add(self, **kwargs):
        self.added.append(kwargs)

    def query(self, **kwargs):
        return {"documents": []}


class FilterCollection:
    def __init__(self):
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        return {"documents": [["matched"] * 5]}


class FailingCollection:
    def add(self, **kwargs):
        raise RuntimeError("chroma unavailable")


class FakeMessages:
    def __init__(self, response="摘要", error=None, started=None, release=None):
        self.response = response
        self.error = error
        self.started = started
        self.release = release
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.response)]
        )


class FakeAnthropic:
    def __init__(self, response="摘要", error=None, started=None, release=None):
        self.messages = FakeMessages(
            response=response,
            error=error,
            started=started,
            release=release,
        )


def make_manager(redis_client=None, episodic=None, client=None):
    manager = MemoryManager.__new__(MemoryManager)
    manager._redis = redis_client or FakeRedis()
    manager._episodic = episodic or FakeCollection()
    manager._profile = FakeCollection()
    manager._client = client or FakeAnthropic()
    manager._model = "test-model"
    manager._locks = {}
    return manager


def test_episodic_search_combines_user_and_conversation_filters():
    collection = FilterCollection()
    manager = MemoryManager.__new__(MemoryManager)
    manager._episodic = collection

    docs = asyncio.run(
        manager._search_episodic("user-1", "conv-1", "query")
    )

    assert docs == ["matched"]
    assert collection.calls[0]["where"] == {
        "$and": [{"user_id": "user-1"}, {"conv_id": "conv-1"}]
    }


def test_compression_keeps_latest_five_in_ascending_order():
    manager = make_manager()

    async def scenario():
        for index in range(15):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )
        return await manager._get_working_memory("user-1", "conv-1")

    messages = asyncio.run(scenario())

    assert [message.content for message in messages] == [
        "m10",
        "m11",
        "m12",
        "m13",
        "m14",
    ]
    assert manager._redis.values["summary:user-1:conv-1"] == "摘要"
    assert len(manager._episodic.added) == 1


def test_concurrent_append_is_not_lost_during_compression():
    async def scenario():
        started = asyncio.Event()
        release = asyncio.Event()
        manager = make_manager(
            client=FakeAnthropic(started=started, release=release)
        )

        for index in range(14):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )

        compressing = asyncio.create_task(
            manager.add_message("user-1", "conv-1", MsgRole.USER, "m14")
        )
        await started.wait()

        appended = asyncio.create_task(
            manager.add_message("user-1", "conv-1", MsgRole.USER, "m15")
        )
        await asyncio.sleep(0.01)

        release.set()
        await asyncio.gather(compressing, appended)

        return await manager._get_working_memory("user-1", "conv-1")

    messages = asyncio.run(scenario())

    assert [message.content for message in messages] == [
        "m10",
        "m11",
        "m12",
        "m13",
        "m14",
        "m15",
    ]


def test_summary_failure_preserves_working_memory():
    manager = make_manager(client=FakeAnthropic(error=RuntimeError("llm down")))

    async def scenario():
        for index in range(15):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )
        return await manager._get_working_memory("user-1", "conv-1")

    messages = asyncio.run(scenario())

    assert [message.content for message in messages] == [
        f"m{index}" for index in range(15)
    ]
    assert "summary:user-1:conv-1" not in manager._redis.values
    assert manager._episodic.added == []


def test_empty_summary_preserves_working_memory():
    manager = make_manager(client=FakeAnthropic(response=""))

    async def scenario():
        for index in range(15):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )
        return await manager._get_working_memory("user-1", "conv-1")

    messages = asyncio.run(scenario())

    assert len(messages) == 15
    assert "summary:user-1:conv-1" not in manager._redis.values
    assert manager._episodic.added == []


def test_episodic_storage_failure_preserves_working_memory():
    manager = make_manager(episodic=FailingCollection())

    async def scenario():
        for index in range(15):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )
        return await manager._get_working_memory("user-1", "conv-1")

    messages = asyncio.run(scenario())

    assert len(messages) == 15
    assert "summary:user-1:conv-1" not in manager._redis.values


def test_recovery_compresses_the_whole_backlog_after_summary_failure():
    manager = make_manager(client=FakeAnthropic(error=RuntimeError("llm down")))
    key = manager._wm_key("user-1", "conv-1")

    async def scenario():
        for index in range(21):
            await manager.add_message(
                "user-1", "conv-1", MsgRole.USER, f"m{index}"
            )

        raw_backlog = list(manager._redis.lists[key])
        window = await manager._get_working_memory("user-1", "conv-1")

        manager._client.messages.error = None
        manager._client.messages.response = "恢复摘要"
        await manager.add_message("user-1", "conv-1", MsgRole.USER, "m21")

        summary_prompts = [
            call["messages"][0]["content"]
            for call in manager._client.messages.calls
            if "总结以下对话" in call["messages"][0]["content"]
        ]
        return (
            raw_backlog,
            window,
            await manager._get_working_memory("user-1", "conv-1"),
            summary_prompts,
            manager._redis.values.get("summary:user-1:conv-1"),
            list(manager._episodic.added),
        )

    raw_backlog, window, messages, prompts, summary, episodic = asyncio.run(
        scenario()
    )

    assert len(raw_backlog) == 21
    assert [message.content for message in window] == [
        f"m{index}" for index in range(1, 21)
    ]
    assert [message.content for message in messages] == [
        "m17", "m18", "m19", "m20", "m21"
    ]
    assert len(prompts) > 1
    recovery_prompt = prompts[-1]
    expected_input = "\n".join(f"user: m{index}" for index in range(17))
    assert expected_input in recovery_prompt
    assert summary == "恢复摘要"
    assert len(episodic) == 1
