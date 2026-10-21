"""T13 零售模式的 API/CLI 集成回归。

使用确定性 Fake 模型，但使用真实 Redis、ChromaDB 和独立 SQLite；
覆盖多轮身份核对、准备展示、确认写入、重启持久化、轨迹与冻结资源。
绝不调用付费模型端点。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import uuid
from types import SimpleNamespace

import chromadb
import pytest
import redis
from fastapi.testclient import TestClient

import api.main as api_main
from agents.retail_tools import retail_tool_scope
from evaluation.retail_ledger import BudgetedClient
from mcp.retail_store import RetailStore

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = os.path.join(
    ROOT, "evaluation", "data", "retail_sandbox", "db.json"
)
POLICY = os.path.join(
    ROOT, "evaluation", "data", "retail_sandbox", "policy.md"
)
KNOWLEDGE = os.path.join(
    ROOT, "evaluation", "data", "retail_sandbox", "knowledge.jsonl"
)
REDIS_URL = os.getenv(
    "REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))

EMAIL = "chen.silva2698@example.com"
ORDER = "#W3069600"
ITEM = "5012998807"


def _text(content: str):
    return SimpleNamespace(content=[{"type": "text", "text": content}])


def _tool_use(name: str, payload: dict, call_id: str = "toolu_retail"):
    return SimpleNamespace(
        content=[
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": payload,
            }
        ]
    )


def _last_tool_result(messages):
    for message in reversed(messages or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                try:
                    return json.loads(block.get("content") or "{}")
                except json.JSONDecodeError:
                    return {}
    return None


def _current_user_message(messages) -> str:
    for message in reversed(messages or []):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        if content.startswith(("[结构化实体]", "[角色输入契约]")):
            continue
        return content
    return ""


class RetailFakeMessages:
    """按确定性脚本驱动零售工具链，不模拟自然语言理解。"""

    def __init__(self, email: str = EMAIL, order_id: str = ORDER, item_id: str = ITEM):
        self.email = email
        self.order_id = order_id
        self.item_id = item_id
        self.calls = []
        self.pending_args = None
        self.displayed_text = ""
        self._tool_index = 0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        system = kwargs.get("system") or ""
        prompt = system + "\n" + "\n".join(
            message.get("content", "")
            for message in messages
            if isinstance(message.get("content"), str)
        )

        if "搜索子查询" in prompt:
            return _text('["退货期限与条件"]')
        if "相关性打分" in prompt:
            return _text("[0, 1, 2, 3, 4]")
        if "总结以下对话" in prompt or "合并为一段" in prompt:
            return _text("假摘要：用户咨询退货申请。")
        if "提炼或更新用户偏好" in prompt:
            return _text(
                '{"preferences": ["关注退货"], '
                '"entities": {"问题类型": ["退货"]}}'
            )
        if "零售客服 Agent" not in system:
            return _text("FAKE_UNEXPECTED_PROMPT")

        return self._agent_response(messages)

    def _next_id(self) -> str:
        self._tool_index += 1
        return f"toolu_retail_{self._tool_index}"

    def _agent_response(self, messages):
        result = _last_tool_result(messages)
        if result is not None:
            if result.get("success") and "order" in result:
                payment = result.get("original_payment_method_id") or ""
                return _tool_use(
                    "prepare_return",
                    {
                        "order_id": self.order_id,
                        "item_ids": [self.item_id],
                        "payment_method_id": payment,
                    },
                    self._next_id(),
                )
            if result.get("success") and "display_text" in result:
                self.pending_args = {
                    "order_id": self.order_id,
                    "item_ids": [self.item_id],
                    "payment_method_id": result.get("payment_method_id", ""),
                }
                self.displayed_text = result["display_text"]
                return _text(
                    "请核对以下退货详情，确认后我再提交。\n\n"
                    + result["display_text"]
                )
            if result.get("success") and result.get("status") == "return requested":
                return _text(
                    "已提交退货申请，订单状态为 return requested。"
                    "这不代表退款已到账、邮件已发送或仓库已收货。"
                )
            if result.get("success") and "user_id" in result:
                return _tool_use(
                    "get_order_details",
                    {"order_id": self.order_id},
                    self._next_id(),
                )
            if result.get("results") is not None:
                items = result.get("results") or []
                if items:
                    first = items[0]
                    return _text(
                        f"依据冻结知识 {first.get('evidence_id', '')}："
                        f"{first.get('content', '')[:120]}"
                    )
                return _text("知识库未返回可用证据，无法核实该政策。")
            return _text("工具未成功：" + str(result.get("error", "未知原因")))

        message = _current_user_message(messages)
        if "确认" in message and self.pending_args:
            return _tool_use(
                "return_delivered_order_items",
                dict(self.pending_args),
                self._next_id(),
            )
        if any(word in message for word in ("期限", "政策", "准备建议", "知识")):
            return _tool_use(
                "search_knowledge_base",
                {"query": message[:80], "top_k": 5},
                self._next_id(),
            )
        return _tool_use(
            "find_user_id_by_email",
            {"email": self.email},
            self._next_id(),
        )


class RetailFakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = RetailFakeMessages(**kwargs)


def _configure_retail(monkeypatch, tmp_path, name: str):
    collection = f"t13k{name}"
    suffix = f"_t13{name}"
    prefix = f"t13test{name}:"
    monkeypatch.setenv("RETAIL_MODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", REDIS_URL)
    monkeypatch.setenv("CHROMA_HOST", CHROMA_HOST)
    monkeypatch.setenv("CHROMA_PORT", str(CHROMA_PORT))
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("RETAIL_DB_PATH", str(tmp_path / "retail.db"))
    monkeypatch.setenv("RETAIL_SEED_PATH", SEED)
    monkeypatch.setenv("RETAIL_POLICY_PATH", POLICY)
    monkeypatch.setenv("RETAIL_KNOWLEDGE_PATH", KNOWLEDGE)
    monkeypatch.setenv("RETAIL_COLLECTION", collection)
    monkeypatch.setenv("RETAIL_EMBEDDING_MODEL", "minilm")
    monkeypatch.setenv("RETAIL_BUDGET_LEDGER", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("RETAIL_BUDGET_CNY", "5")
    monkeypatch.setenv("RETAIL_INPUT_CNY_PER_MTOK", "1")
    monkeypatch.setenv("RETAIL_OUTPUT_CNY_PER_MTOK", "1")
    monkeypatch.setenv("RETAIL_PROFILE_ENABLED", "false")
    monkeypatch.setenv("RETAIL_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("RETAIL_MCP_STDERR_LOG", str(tmp_path / "mcp_stderr.log"))
    monkeypatch.setenv("RETAIL_RUN_ID", f"t13-{name}")
    monkeypatch.setenv("RETAIL_CASE_ID", "integration")
    monkeypatch.setenv("RETAIL_CLIENT_USER", f"client-{name}")
    monkeypatch.setenv("RETAIL_CONV_ID", f"conv-{name}")
    monkeypatch.setenv("RETAIL_MEMORY_KEY_PREFIX", prefix)
    monkeypatch.setenv("RETAIL_MEMORY_COLLECTION_SUFFIX", suffix)
    monkeypatch.setenv("RETAIL_AGENT_MAX_TOKENS", "512")
    return collection, suffix, prefix


def _patch_fake_client(monkeypatch, fake):
    def build(api_key, base_url, ledger, stage):
        return BudgetedClient(fake, ledger, stage)

    monkeypatch.setattr("agents.retail_runtime.build_budgeted_client", build)


def _cleanup(collection: str, suffix: str, prefix: str) -> None:
    try:
        client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        keys = list(client.scan_iter(match=f"{prefix}*", count=100))
        if keys:
            client.delete(*keys)
        client.close()
    except Exception:
        pass
    try:
        chroma = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        for name in (collection, f"episodic{suffix}", f"user_profile{suffix}"):
            try:
                chroma.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass


def _mcp_log(tmp_path) -> str:
    return (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")


def _mcp_pids(text: str):
    started = re.findall(r"started pid=(\d+)", text)
    pids = {int(value) for value in re.findall(r"pid=(\d+)", text)}
    return started, pids


def _assert_process_stopped(pid: int) -> None:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert f'"{pid}"' not in result.stdout


def test_retail_api_multiturn_display_confirm_and_knowledge(
    monkeypatch, tmp_path
):
    name = uuid.uuid4().hex[:8]
    collection, suffix, prefix = _configure_retail(monkeypatch, tmp_path, name)
    fake = RetailFakeAnthropic()
    _patch_fake_client(monkeypatch, fake)
    conv = f"conv-api-{name}"
    client_user = f"client-api-{name}"

    try:
        with TestClient(api_main.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            body = health.json()
            assert body["mode"] == "retail"
            assert body["store"]["counts"]["users"] == 500
            assert body["budget"]["budget_cny"] == 5

            skills = client.get("/skills")
            assert skills.json()["count"] == 4

            stats = client.get("/knowledge/stats")
            assert stats.status_code == 200
            assert stats.json() == {
                "total_chunks": 29,
                "collection": collection,
                "embedding_model": "minilm",
                "frozen": True,
                "mode": "retail",
            }
            assert client.post(
                "/knowledge/add", json={"documents": []}
            ).status_code == 409
            assert client.post("/eval/run", json={}).status_code == 409

            first = client.post(
                "/chat",
                json={
                    "message": (
                        f"我的邮箱是 {EMAIL}，想退 {ORDER} 里的 Makeup Kit，"
                        "退款到原支付方式。"
                    ),
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert first.status_code == 200, first.text
            first_body = first.json()
            assert first_body["agent_type"] == "retail"
            assert first_body["agent_types"] == ["retail"]
            assert set(first_body["tools_used"]) == {
                "find_user_id_by_email",
                "get_order_details",
                "prepare_return",
            }
            assert "【退货申请详情（待确认）】" in first_body["response"]
            assert ORDER in first_body["response"]
            assert "Makeup Kit" in first_body["response"]
            assert "¥258.71" in first_body["response"]
            assert "请回复“确认，就按这些申请退货”" in first_body["response"]

            store = RetailStore(tmp_path / "retail.db", SEED)
            assert store.bound_user_id(client_user, conv) == "chen_silva_7485"
            pending = store.active_pending(client_user, conv)
            assert len(pending) == 1
            assert pending[0]["status"] == "displayed"
            assert pending[0]["displayed_turn"] == store.current_turn(
                client_user, conv
            )
            assert (
                store.get_order_details(client_user, conv, ORDER)["order"][
                    "status"
                ]
                == "delivered"
            )

            trace = client.get(f"/trace/tool/{first_body['request_id']}")
            assert trace.status_code == 200
            trace_body = trace.json()
            assert trace_body["found"] is True
            assert [
                item["tool_name"]
                for item in trace_body["trace"]["tool_traces"]
            ] == [
                "find_user_id_by_email",
                "get_order_details",
                "prepare_return",
            ]
            recent = client.get("/trace/tools?limit=5")
            assert recent.status_code == 200
            assert recent.json()["items"]

            second = client.post(
                "/chat",
                json={
                    "message": "确认，就按这些申请退货",
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert second.status_code == 200, second.text
            second_body = second.json()
            assert "return requested" in second_body["response"]
            assert "不代表退款已到账" in second_body["response"]
            assert "return_delivered_order_items" in second_body["tools_used"]

            order = store.get_order_details(client_user, conv, ORDER)["order"]
            assert order["status"] == "return requested"
            assert order["return_items"] == [ITEM]
            assert order["return_payment_method_id"] == "credit_card_1565124"
            assert store.stats()["counts"]["return_operations"] == 1
            assert store.active_pending(client_user, conv) == []

            third = client.post(
                "/chat",
                json={
                    "message": "退货期限是多久？有哪些准备建议？",
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert third.status_code == 200, third.text
            third_body = third.json()
            assert "search_knowledge_base" in third_body["tools_used"]
            assert third_body["knowledge_used"] is True
            assert "K" in third_body["response"]

            runtime = api_main._retail_runtime
            assert runtime is not None
            assert set(runtime.mcp_client.tools()) == set(
                retail_tool_scope(has_knowledge=False)
            )
            assert "search_knowledge_base" in runtime.agent.get_tools()
            assert runtime.mcp_client.call_count == 4

            monitor = client.get("/monitor")
            assert monitor.status_code == 200
            monitor_body = monitor.json()
            assert monitor_body["mode"] == "retail"
            assert len(monitor_body["recent_traces"]) >= 3
            assert monitor_body["budget"]["requests"] > 0
            assert monitor_body["budget"]["reservation_bound_violations"] == 0

        log_text = _mcp_log(tmp_path)
        started, pids = _mcp_pids(log_text)
        assert len(started) == 1
        assert len(pids) == 1
        for tool_name in (
            "find_user_id_by_email",
            "get_order_details",
            "prepare_return",
            "return_delivered_order_items",
        ):
            assert log_text.count(f"call tool={tool_name} ") == 1
        assert "call tool=search_knowledge_base" not in log_text
        _assert_process_stopped(pids.pop())

        event_rows = [
            json.loads(line)
            for line in (tmp_path / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [row["turn"] for row in event_rows] == [1, 2, 3]
        assert event_rows[0]["display_appended"] == []
        assert event_rows[1]["store_events"][-1]["event_type"] == (
            "return_requested"
        )
        assert event_rows[2]["knowledge_hits"] > 0
    finally:
        _cleanup(collection, suffix, prefix)


def test_retail_pending_survives_restart_and_confirm_after_reopen(
    monkeypatch, tmp_path
):
    name = uuid.uuid4().hex[:8]
    collection, suffix, prefix = _configure_retail(monkeypatch, tmp_path, name)
    fake = RetailFakeAnthropic()
    _patch_fake_client(monkeypatch, fake)
    conv = f"conv-restart-{name}"
    client_user = f"client-restart-{name}"

    try:
        with TestClient(api_main.app) as client:
            prepared = client.post(
                "/chat",
                json={
                    "message": (
                        f"我的邮箱是 {EMAIL}，请帮我退 {ORDER} 里的 Makeup Kit。"
                    ),
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert prepared.status_code == 200
            assert prepared.json()["tools_used"][-1] == "prepare_return"

        log_text = _mcp_log(tmp_path)
        started, pids = _mcp_pids(log_text)
        assert len(started) == 1
        assert len(pids) == 1
        first_pid = int(started[0])
        _assert_process_stopped(first_pid)

        # 服务完全关闭后重新启动，待确认详情与轮次必须保留。
        with TestClient(api_main.app) as client:
            store = RetailStore(tmp_path / "retail.db", SEED)
            assert store.active_pending(client_user, conv)[0]["status"] == (
                "displayed"
            )

            confirmed = client.post(
                "/chat",
                json={
                    "message": "确认，就按这些申请退货",
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert confirmed.status_code == 200, confirmed.text
            order = store.get_order_details(client_user, conv, ORDER)["order"]
            assert order["status"] == "return requested"
            assert order["return_items"] == [ITEM]

        log_text = _mcp_log(tmp_path)
        started, pids = _mcp_pids(log_text)
        assert len(started) == 2
        assert len(pids) == 2
        assert first_pid in pids
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        assert "call tool=search_knowledge_base" not in log_text
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(collection, suffix, prefix)


def test_retail_cli_uses_same_service_and_writes_return(
    monkeypatch, tmp_path, capsys
):
    name = uuid.uuid4().hex[:8]
    collection, suffix, prefix = _configure_retail(monkeypatch, tmp_path, name)
    fake = RetailFakeAnthropic()
    _patch_fake_client(monkeypatch, fake)
    client_user = f"client-cli-{name}"
    conv = f"conv-cli-{name}"
    monkeypatch.setenv("RETAIL_CLIENT_USER", client_user)
    monkeypatch.setenv("RETAIL_CONV_ID", conv)

    replies = iter(
        [
            f"我的邮箱是 {EMAIL}，请帮我退 {ORDER} 里的 Makeup Kit。",
            "确认，就按这些申请退货",
            "退出",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    try:
        asyncio.run(api_main._retail_cli())
        output = capsys.readouterr().out
        assert "零售模式：会话" in output
        assert "【退货申请详情（待确认）】" in output
        assert "return requested" in output
        assert "再见" in output

        store = RetailStore(tmp_path / "retail.db", SEED)
        order = store.get_order_details(client_user, conv, ORDER)["order"]
        assert order["status"] == "return requested"
        assert order["return_items"] == [ITEM]
        assert store.stats()["counts"]["return_operations"] == 1

        events = store.list_events(client_user, conv)
        assert [event["event_type"] for event in events] == [
            "identity_bound",
            "return_prepared",
            "return_displayed",
            "return_requested",
        ]

        log_text = _mcp_log(tmp_path)
        started, pids = _mcp_pids(log_text)
        assert len(started) == 1
        assert len(pids) == 1
        for tool_name in (
            "find_user_id_by_email",
            "get_order_details",
            "prepare_return",
            "return_delivered_order_items",
        ):
            assert log_text.count(f"call tool={tool_name} ") == 1
        assert "call tool=search_knowledge_base" not in log_text
        _assert_process_stopped(pids.pop())
    finally:
        _cleanup(collection, suffix, prefix)
