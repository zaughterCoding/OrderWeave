"""Integration coverage for the HTTP API and interactive CLI.

The test uses a deterministic fake model with the real independent Redis and
ChromaDB services. It never calls a paid model endpoint.
"""
import asyncio
import builtins
import os
import uuid
from types import SimpleNamespace

import chromadb
import redis
from fastapi.testclient import TestClient

import api.main as api_main
from mcp.knowledge_base import KnowledgeBase

REDIS_URL = os.getenv(
    "REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))


def _text_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)]
    )


def _last_message_text(kwargs):
    messages = kwargs.get("messages") or []
    if not messages:
        return ""
    content = messages[-1].get("content")
    return content if isinstance(content, str) else ""


def _has_tool_result(kwargs):
    for message in kwargs.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


class FakeMessages:
    """Reply by prompt shape so every component can share one fake client."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        prompt = "\n".join(
            message.get("content", "")
            for message in messages
            if isinstance(message.get("content"), str)
        )
        system = kwargs.get("system") or ""
        if system:
            prompt = f"{system}\n{prompt}"

        if _has_tool_result(kwargs):
            return _text_response(
                "FAKE_AGENT_REPLY: 根据知识库，退款申请需在七天内提交。"
            )
        if "意图分析专家" in prompt:
            return _text_response(
                '{"intent": "refund", "confidence": 0.9, '
                '"reasoning": "fake intent"}'
            )
        if "搜索子查询" in prompt:
            return _text_response('["退款政策", "订单退款说明"]')
        if "相关性打分" in prompt:
            return _text_response("[0, 1, 2, 3, 4]")
        if "客服质量评估专家" in prompt:
            return _text_response(
                '{"relevance": 0.9, "accuracy": 0.9, '
                '"completeness": 0.9, "helpfulness": 0.9}'
            )
        if "提炼或更新用户偏好" in prompt:
            return _text_response(
                '{"preferences": ["关注退款"], '
                '"entities": {"问题类型": ["退款"]}}'
            )
        if "总结以下对话" in prompt or "合并为一段" in prompt:
            return _text_response("假摘要：用户咨询退款政策。")

        last_message = _last_message_text(kwargs)
        if kwargs.get("tools") and "检索知识库" in last_message:
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        id="toolu_fake_1",
                        name="search_knowledge_base",
                        input={"query": "退款政策", "top_k": 2},
                    )
                ]
            )
        return _text_response("FAKE_AGENT_REPLY: 已收到您的问题。")


class FakeAnthropic:
    def __init__(self):
        self.messages = FakeMessages()


def _cleanup(
    user_id: str,
    conv_id: str,
    collection_name: str,
) -> None:
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        client.delete(
            f"wm:{user_id}:{conv_id}",
            f"summary:{user_id}:{conv_id}",
        )
        client.close()
    except Exception:
        pass

    try:
        chroma = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        try:
            episodic = chroma.get_collection("episodic")
            episodic.delete(where={"user_id": user_id})
        except Exception:
            pass
        try:
            profiles = chroma.get_collection("user_profile")
            profiles.delete(ids=[f"user_profile:{user_id}"])
        except Exception:
            pass
        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass
    except Exception:
        pass


def test_http_api_routes_with_fake_model_and_real_storage(monkeypatch, tmp_path):
    collection_name = f"api_test_{uuid.uuid4().hex}"
    user_id = f"api-test-user-{uuid.uuid4().hex}"
    conv_id = f"api-test-conv-{uuid.uuid4().hex}"
    marker = f"API-MARKER-{uuid.uuid4().hex}"

    monkeypatch.setattr(KnowledgeBase, "COLLECTION_NAME", collection_name)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("CHROMA_HOST", CHROMA_HOST)
    monkeypatch.setenv("CHROMA_PORT", str(CHROMA_PORT))
    monkeypatch.setenv(
        "CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma")
    )
    monkeypatch.setenv("MONITOR_INTERVAL_S", "3600")
    monkeypatch.setenv("MONITOR_PROMETHEUS_PORT", "0")
    monkeypatch.setenv("MONITOR_WEBHOOK_URL", "")
    monkeypatch.setenv(
        "EVAL_BASELINE_PATH", str(tmp_path / "baseline.json")
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(tmp_path / "reports"))

    fake = FakeAnthropic()
    for target in (
        "agents.agent_orchestrator.AsyncAnthropic",
        "core.intent_recognizer.AsyncAnthropic",
        "evaluation.evaluator.AsyncAnthropic",
        "mcp.tool_manager.AsyncAnthropic",
        "memory.conversation_memory.AsyncAnthropic",
    ):
        monkeypatch.setattr(target, lambda **kwargs: fake)

    first_message = f"请检索知识库：{marker} 的退款政策"

    try:
        with TestClient(api_main.app) as client:
            assert api_main._memory._use_server is True, (
                "independent Redis/Chroma services are required at "
                f"{CHROMA_HOST}:{CHROMA_PORT}"
            )
            assert api_main._knowledge_base._use_server is True

            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            skills = client.get("/skills")
            assert skills.status_code == 200
            assert skills.json()["count"] == 3

            reloaded = client.post("/skills/reload")
            assert reloaded.status_code == 200
            assert reloaded.json()["count"] == 3

            added = client.post(
                "/knowledge/add",
                json={
                    "documents": [
                        {
                            "title": f"合成退款文档 {marker}",
                            "content": (
                                f"火星退款条款 {marker}：订单号 MARS-ALPHA "
                                "在七天内可以申请退款。"
                            ),
                        }
                    ]
                },
            )
            assert added.status_code == 200
            assert added.json()["added_chunks"] >= 1

            uploaded = client.post(
                "/knowledge/upload",
                files={
                    "file": (
                        "synthetic_policy.txt",
                        f"月球配送说明 {marker}：订单号 LUNA-BETA 使用星际加急配送。".encode(
                            "utf-8"
                        ),
                        "text/plain",
                    )
                },
            )
            assert uploaded.status_code == 200
            assert uploaded.json()["added_chunks"] >= 1

            stats = client.get("/knowledge/stats")
            assert stats.status_code == 200
            assert stats.json()["total_chunks"] > 0

            search = client.post(
                "/search",
                params={"query": f"{marker} 退款", "top_k": 3},
            )
            assert search.status_code == 200
            payload = search.json()
            assert payload["query"] == f"{marker} 退款"
            assert payload["results"]

            upload_search = client.post(
                "/search",
                params={"query": "LUNA-BETA 星际加急配送", "top_k": 3},
            )
            assert upload_search.status_code == 200
            assert any(
                "LUNA-BETA" in item.get("content", "")
                for item in upload_search.json()["results"]
            )

            first_chat = client.post(
                "/chat",
                json={
                    "message": first_message,
                    "user_id": user_id,
                    "conv_id": conv_id,
                },
            )
            assert first_chat.status_code == 200
            first_payload = first_chat.json()
            assert first_payload["conv_id"] == conv_id
            assert "search_knowledge_base" in first_payload["tools_used"]
            assert first_payload["knowledge_used"] is True
            assert first_payload["response"].startswith("FAKE_AGENT_REPLY")

            second_chat = client.post(
                "/chat",
                json={
                    "message": "我刚才问了什么？",
                    "user_id": user_id,
                    "conv_id": conv_id,
                },
            )
            assert second_chat.status_code == 200
            assert second_chat.json()["conv_id"] == conv_id

            agent_calls = [
                kwargs
                for kwargs in fake.messages.calls
                if kwargs.get("tools")
            ]
            assert agent_calls
            second_agent_messages = "\n".join(
                message.get("content", "")
                for message in agent_calls[-1].get("messages", [])
                if isinstance(message.get("content"), str)
            )
            assert first_message in second_agent_messages

            trace = client.get(
                f"/trace/tool/{first_payload['request_id']}"
            )
            assert trace.status_code == 200
            trace_payload = trace.json()
            assert trace_payload["found"] is True
            assert trace_payload["trace"]["tool_calls"]

            recent = client.get("/trace/tools", params={"limit": 5})
            assert recent.status_code == 200
            assert recent.json()["items"]

            monitor = client.get("/monitor")
            assert monitor.status_code == 200
            monitor_payload = monitor.json()
            assert "agent_stats" in monitor_payload
            assert "tool_stats" in monitor_payload

            metrics = client.get("/metrics")
            assert metrics.status_code == 200
            assert metrics.headers["content-type"].startswith("text/plain")

            evaluation = client.post(
                "/eval/run",
                json={
                    "intent_cases": [],
                    "dialog_cases": [{"question": "退款怎么处理？"}],
                },
            )
            assert evaluation.status_code == 200
            eval_payload = evaluation.json()
            assert eval_payload["total"] == 1
            assert eval_payload["passed"] == 1
            assert eval_payload["grader_errors"] == 0
            assert eval_payload["avg_scores"]["relevance"] == 0.9
    finally:
        _cleanup(user_id, conv_id, collection_name)


def test_cli_quit_exits_and_closes_memory(monkeypatch):
    closed = {"value": False}

    class FakeMemory:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            closed["value"] = True

    fake = FakeAnthropic()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setattr(
        "agents.agent_orchestrator.AsyncAnthropic",
        lambda **kwargs: fake,
    )
    monkeypatch.setattr(
        "core.intent_recognizer.AsyncAnthropic",
        lambda **kwargs: fake,
    )
    monkeypatch.setattr(
        "memory.conversation_memory.MemoryManager",
        FakeMemory,
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": "quit")

    asyncio.run(api_main._cli())

    assert closed["value"] is True
