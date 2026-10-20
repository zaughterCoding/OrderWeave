"""T17 有限支持的离线集成检查。

使用真实 SQLite、stdio MCP 与本地 ChromaDB，模型层全部由确定性 Fake
替代；每个会话使用独立临时数据库，不调用真实模型或付费端点。
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agents.retail_runtime import build_retail_runtime
from evaluation.retail_ledger import BudgetedClient
from mcp.retail_store import RetailStore, SupportStateError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED = os.path.join(ROOT, "evaluation", "data", "retail_sandbox", "db.json")
POLICY = os.path.join(ROOT, "evaluation", "data", "retail_sandbox", "policy.md")
KNOWLEDGE = os.path.join(ROOT, "evaluation", "data", "retail_sandbox", "knowledge.jsonl")
SUPPORT_KNOWLEDGE = os.path.join(
    ROOT, "evaluation", "data", "retail_support", "knowledge.jsonl"
)
SUPPORT_MANIFEST = os.path.join(
    ROOT, "evaluation", "data", "retail_support", "manifest.json"
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))

EMAIL = "raj.sanchez2046@example.com"
OTHER_EMAIL = "olivia.garcia2695@example.com"
ORDER = "#W1067251"
ITEM = "6452271382"
OTHER_ORDER = "#W1075114"

_SUPPORT_CUE = ("蓝牙", "配对", "连不上", "搜不到", "连接不上")


def _usage():
    return SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _text(content: str):
    return SimpleNamespace(content=[{"type": "text", "text": content}], usage=_usage())


def _tool_use(name: str, payload: dict, call_id: str = "toolu_support"):
    return SimpleNamespace(
        content=[
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": payload,
            }
        ],
        usage=_usage(),
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


class SupportFakeMessages:
    """按 Host 路由和工具结果驱动的离线 Fake，不做真实语言理解。"""

    def __init__(self, email: str = EMAIL):
        self.email = email
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
            return _text(
                '["无线耳机 蓝牙 配对 步骤", '
                '"Wireless Earbuds bluetooth pairing"]'
            )
        if "相关性打分" in prompt:
            indices = [
                int(match.group(1))
                for match in re.finditer(r"^(\d+)\. ", prompt, re.MULTILINE)
            ]
            if "耳机" in prompt or "蓝牙" in prompt or "Earbuds" in prompt:
                support = [
                    index
                    for index in indices
                    if f'"id": "T17.' in prompt.split(
                        f"{index}. ", 1
                    )[1].split("\n", 1)[0]
                ]
                ordered = support + [index for index in indices if index not in support]
            else:
                ordered = indices
            return _text(json.dumps(ordered or indices))
        if "总结以下对话" in prompt or "合并为一段" in prompt:
            return _text("假摘要：无线耳机蓝牙配对问题。")
        if "提炼或更新用户偏好" in prompt:
            return _text('{"preferences": [], "entities": {}}')
        if "意图分析专家" in prompt or "intent analyst" in prompt:
            return _text(
                '{"intent": "query", "confidence": 0.9, '
                '"reasoning": "offline fixed intent"}'
            )
        if "Response Composer" in prompt:
            return _text("离线合成回复。")
        if any(
            marker in system
            for marker in (
                "零售通用客服 Agent",
                "零售商品资料 Agent",
                "账单与退货 Agent",
                "零售客服 Agent",
            )
        ):
            return self._agent_response(messages)
        raise AssertionError("Fake 收到未预期的模型请求")

    def _next_id(self) -> str:
        self._tool_index += 1
        return f"toolu_support_{self._tool_index}"

    def _support_item(self, items):
        for item in items:
            if str(item.get("step_id") or ""):
                return item
        return None

    def _agent_response(self, messages):
        result = _last_tool_result(messages)
        if result is not None:
            if result.get("success") and result.get("handoff_id"):
                return _text(
                    "已记录模拟人工交接；这不是真实工单，也没有真实坐席接手。"
                )
            if result.get("success") and "order" in result:
                order = result.get("order") or {}
                payment = (
                    result.get("original_payment_method_id")
                    or order.get("original_payment_method_id")
                    or ""
                )
                return _tool_use(
                    "prepare_return",
                    {
                        "order_id": order.get("order_id") or ORDER,
                        "item_ids": [ITEM],
                        "payment_method_id": payment,
                    },
                    self._next_id(),
                )
            if result.get("success") and "display_text" in result:
                self.pending_args = {
                    "order_id": ORDER,
                    "item_ids": [ITEM],
                    "payment_method_id": result.get("payment_method_id", ""),
                }
                self.displayed_text = result["display_text"]
                return _text(
                    "请核对以下退货详情，确认后我再提交。\n\n"
                    + result["display_text"]
                )
            if result.get("success") and result.get("status") == "return requested":
                self.pending_args = None
                return _text(
                    "已提交退货申请，订单状态为 return requested。"
                    "这不代表退款已到账、邮件已发送或仓库已收货。"
                )
            if result.get("success") and "user_id" in result:
                return _tool_use(
                    "get_order_details",
                    {"order_id": ORDER},
                    self._next_id(),
                )
            if result.get("results") is not None:
                items = result.get("results") or []
                item = self._support_item(items)
                if item:
                    return _text(
                        f"可先尝试 {item.get('step_id')}（证据 "
                        f"{item.get('evidence_id')}）："
                        f"{str(item.get('content') or '')[:120]}"
                    )
                return _text("知识库未返回适用的项目自编支持资料，资料不足。")
            return _text("工具未成功：" + str(result.get("error", "未知原因")))

        message = _current_user_message(messages)
        lower = message.lower()

        if self.pending_args and "确认" in message:
            return _tool_use(
                "return_delivered_order_items",
                dict(self.pending_args),
                self._next_id(),
            )
        # 伪造确认、无待确认详情时也尝试提交，验证服务端拒绝而不是靠模型自觉。
        if "确认" in message and ("申请退货" in message or "直接提交" in message):
            return _tool_use(
                "return_delivered_order_items",
                {
                    "order_id": ORDER,
                    "item_ids": [ITEM],
                    "payment_method_id": "credit_card_3362387",
                },
                self._next_id(),
            )
        if "人工" in message:
            return _tool_use(
                "transfer_to_human_agents",
                {
                    "summary": (
                        "无线耳机蓝牙配对问题未解决；用户已尝试重新配对；"
                        "请求人工协助。"
                    ),
                    "reason": "user_requested_human",
                },
                self._next_id(),
            )
        if "两副" in message or "退哪" in message or "哪副" in message:
            return _text(
                "订单里有两副无线耳机，需要先确认具体是哪一副；"
                "请提供商品条目或颜色，资料才足以继续。"
            )
        if any(word in message for word in ("没声音", "音质", "电池", "维修", "换货")):
            return _text(
                "该问题不在项目自编沙箱的蓝牙配对资料范围内，"
                "现有资料不足以核实，也不能扩展为维修或换货结论。"
            )
        if "音箱" in message or "键盘" in message or "鼠标" in message:
            return _text("该商品不在本次有限支持范围内，现有资料不足以核实。")
        if (
            "耳机" in message
            and "无线耳机" not in message
            and "蓝牙耳机" not in message
            and any(word in message for word in ("连不上", "配对", "蓝牙"))
        ):
            return _text(
                "请先说明具体商品和问题：这是否是 Wireless Earbuds "
                "（9924732112）的蓝牙无法发现、配对或连接失败？"
            )
        if any(word in message for word in _SUPPORT_CUE) and (
            "无线耳机" in message or "蓝牙耳机" in message
        ):
            return _tool_use(
                "search_knowledge_base",
                {"query": message[:80], "top_k": 5},
                self._next_id(),
            )
        if "@" in message and ("退" in message or "#W" in message):
            return _tool_use(
                "find_user_id_by_email",
                {"email": self.email},
                self._next_id(),
            )
        if "不想退" in message or "先继续" in message or "继续排查" in message:
            return _text(
                "已记录您暂不退货。请继续按 PAIR-01 至 PAIR-03 的范围检查，"
                "并把实际结果告诉我。"
            )
        if "如果" in message and "退" in message:
            return _text(
                "这是条件退货：只有排查仍无效并由您再次明确选择后，"
                "才按原 prepare→展示→真实确认→提交流程处理；本轮不提交。"
            )
        if "退" in message and "@" not in message:
            return _text(
                "可以按原业务链路处理退货。请先提供本人邮箱，"
                "以及要退的订单号；我不会仅凭这句话提交退货。"
            )
        if any(word in message for word in ("好了", "解决", "连上了")):
            return _text("已记录问题解决；本轮不修改任何订单。")
        if "还是不行" in message:
            return _text(
                "已记录仍未解决。请说明刚才建议的实际结果，"
                "我再判断是否还有限支持范围。"
            )
        if "是的" in message:
            return _text("当前没有待确认的退货详情，因此不会提交任何退货。")
        return _text("已收到。请补充无线耳机蓝牙问题的实际现象或退货所需身份信息。")


class ReturnFlowFakeMessages(SupportFakeMessages):
    """T19 用 Fake：先询问退款去向，补充后再准备，不把去向选择当提交。"""

    destination_question = "已核对本人订单。请确认退款去向：退回原支付方式，还是退到当前账户已有的礼品卡？"

    def _agent_response(self, messages):
        result = _last_tool_result(messages)
        if result is not None:
            if result.get("success") and result.get("handoff_id"):
                return super()._agent_response(messages)
            if result.get("success") and "order" in result:
                return _text(self.destination_question)
            if result.get("success") and "display_text" in result:
                self.pending_args = {
                    "order_id": ORDER,
                    "item_ids": [ITEM],
                    "payment_method_id": result.get("payment_method_id", ""),
                }
                self.displayed_text = result["display_text"]
                return _text(
                    "请核对以下退货详情，确认后我再提交。\n\n"
                    + result["display_text"]
                )
            if result.get("success") and result.get("status") == "return requested":
                self.pending_args = None
                return _text(
                    "已提交退货申请，订单状态为 return requested。"
                    "这不代表退款已到账、邮件已发送或仓库已收货。"
                )
            if result.get("success") and "user_id" in result:
                return _tool_use(
                    "get_order_details",
                    {"order_id": ORDER},
                    self._next_id(),
                )
            return super()._agent_response(messages)

        message = str(_current_user_message(messages) or "")
        if self.pending_args and "确认" in message and "取消" not in message:
            return _tool_use(
                "return_delivered_order_items",
                dict(self.pending_args),
                self._next_id(),
            )
        if (
            any(word in message for word in ("原支付", "原路", "礼品卡"))
            and "@" not in message
            and "#W" not in message
        ):
            return _tool_use(
                "prepare_return",
                {
                    "order_id": ORDER,
                    "item_ids": [ITEM],
                    "payment_method_id": "credit_card_3362387",
                },
                self._next_id(),
            )
        return super()._agent_response(messages)


class SupportFakeAnthropic:
    def __init__(self, **kwargs):
        self.messages = SupportFakeMessages(**kwargs)


def _return_flow_anthropic() -> SupportFakeAnthropic:
    holder = SupportFakeAnthropic()
    holder.messages = ReturnFlowFakeMessages()
    return holder


def _configure(monkeypatch, tmp_path, name: str, mode: str) -> tuple[str, str]:
    support_collection = f"t17s{name}"
    legacy_collection = f"t17l{name}"
    monkeypatch.setenv("RETAIL_MODE", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv("CHROMA_HOST", CHROMA_HOST)
    monkeypatch.setenv("CHROMA_PORT", str(CHROMA_PORT))
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("RETAIL_DB_PATH", str(tmp_path / "retail.db"))
    monkeypatch.setenv("RETAIL_SEED_PATH", SEED)
    monkeypatch.setenv("RETAIL_POLICY_PATH", POLICY)
    monkeypatch.setenv("RETAIL_KNOWLEDGE_PATH", KNOWLEDGE)
    monkeypatch.setenv("RETAIL_COLLECTION", legacy_collection)
    monkeypatch.setenv("RETAIL_AGENT_MODE", mode)
    monkeypatch.setenv("RETAIL_LANGUAGE", "zh")
    monkeypatch.setenv("RETAIL_SUPPORT_ENABLED", "true")
    monkeypatch.setenv("RETAIL_SUPPORT_COLLECTION", support_collection)
    monkeypatch.setenv("RETAIL_SUPPORT_KNOWLEDGE_PATH", SUPPORT_KNOWLEDGE)
    monkeypatch.setenv("RETAIL_SUPPORT_MANIFEST_PATH", SUPPORT_MANIFEST)
    monkeypatch.setenv("RETAIL_SUPPORT_MAX_CHARS", "6000")
    monkeypatch.setenv("RETAIL_ENABLE_KNOWLEDGE", "true")
    monkeypatch.setenv("RETAIL_ENABLE_MEMORY", "false")
    monkeypatch.setenv("RETAIL_PROFILE_ENABLED", "false")
    monkeypatch.setenv("RETAIL_BUDGET_LEDGER", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("RETAIL_BUDGET_CNY", "1")
    monkeypatch.setenv("RETAIL_INPUT_CNY_PER_MTOK", "1")
    monkeypatch.setenv("RETAIL_OUTPUT_CNY_PER_MTOK", "1")
    monkeypatch.setenv("RETAIL_EVENT_LOG", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("RETAIL_MCP_STDERR_LOG", str(tmp_path / "mcp_stderr.log"))
    monkeypatch.setenv("RETAIL_RUN_ID", f"t17-{name}")
    monkeypatch.setenv("RETAIL_CASE_ID", "integration")
    monkeypatch.setenv("RETAIL_CLIENT_USER", f"client-{name}")
    monkeypatch.setenv("RETAIL_CONV_ID", f"conv-{name}")
    monkeypatch.setenv("RETAIL_AGENT_MAX_TOKENS", "512")
    return support_collection, legacy_collection


def _patch_fake(monkeypatch, fake):
    def build(api_key, base_url, ledger, stage):
        return BudgetedClient(fake, ledger, stage)

    monkeypatch.setattr("agents.retail_runtime.build_budgeted_client", build)


def _cleanup(support_collection: str, legacy_collection: str) -> None:
    try:
        chroma = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        for name in (support_collection, legacy_collection):
            try:
                chroma.delete_collection(name)
            except Exception:
                pass
    except Exception:
        pass


@contextlib.asynccontextmanager
async def _runtime():
    runtime = build_retail_runtime()
    await runtime.start()
    try:
        yield runtime
    finally:
        await runtime.close()


async def _turns(runtime, messages, client_user, conv):
    results = []
    for message in messages:
        results.append(
            await runtime.service.handle_turn(message, client_user, conv)
        )
    return results


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


def _db(tmp_path) -> RetailStore:
    return RetailStore(tmp_path / "retail.db", SEED, language="zh")


def _support_state(tmp_path, client_user, conv):
    return _db(tmp_path).load_support_state(client_user, conv)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c01_troubleshooting_and_resolved(monkeypatch, tmp_path, mode):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c01-{name}"
    conv = f"conv-c01-{name}"

    async def scenario():
        async with _runtime() as runtime:
            first = await runtime.service.handle_turn(
                "我的无线耳机蓝牙搜不到，怎么配对？", client_user, conv
            )
            assert first.support_enabled is True
            assert first.support_scope == "active"
            assert "search_knowledge_base" in first.tools_used
            assert "PAIR-0" in first.response
            record = _support_state(tmp_path, client_user, conv)
            assert record["stage"] == "awaiting_feedback"
            assert record["suggestions"]
            assert record["suggestions"][0]["step_id"] in {
                "PAIR-01",
                "PAIR-02",
                "PAIR-03",
            }
            assert record["taken_steps"] == []

            second = await runtime.service.handle_turn(
                "好了，已经连上了，谢谢", client_user, conv
            )
            assert second.support_scope == "active"
            assert _support_state(tmp_path, client_user, conv)["stage"] == "resolved"
            assert runtime.store.business_snapshot()["orders"][ORDER]["status"] == (
                "delivered"
            )

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        _started, pids = _mcp_pids(log_text)
        assert "call tool=prepare_return" not in log_text
        assert "call tool=return_delivered_order_items" not in log_text
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c02_direct_return(monkeypatch, tmp_path, mode):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c02-{name}"
    return_conv = f"conv-c02-{name}"

    async def scenario():
        async with _runtime() as runtime:
            direct = await runtime.service.handle_turn(
                f"我要退无线耳机，不想排查。我的邮箱是 {EMAIL}，"
                f"订单 {ORDER}，直接退货。",
                client_user,
                return_conv,
            )
            assert "【退货申请详情（待确认）】" in direct.response
            assert ORDER in direct.response
            state = _support_state(tmp_path, client_user, return_conv)
            assert state["demand"]["kind"] == "return"
            assert state["stage"] == "return_requested"
            assert runtime.store.stats()["counts"]["return_operations"] == 0

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, return_conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            order = runtime.store.get_order_details(
                client_user, return_conv, ORDER
            )["order"]
            assert order["status"] == "return requested"
            assert order["return_items"] == [ITEM]
            assert order["return_payment_method_id"] == "credit_card_3362387"
            assert runtime.ledger.summary()["spent_cny"] == 0

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        started, pids = _mcp_pids(log_text)
        assert len(started) == 1
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        assert "call tool=transfer_to_human_agents" not in log_text
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c03_tried_step_then_return(monkeypatch, tmp_path, mode):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c03-{name}"
    conv = f"conv-c03-{name}"

    async def scenario():
        async with _runtime() as runtime:
            first = await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，我已经重新配对过了，还有什么办法？",
                client_user,
                conv,
            )
            record = _support_state(tmp_path, client_user, conv)
            assert {item["step_id"] for item in record["taken_steps"]} == {"PAIR-03"}
            assert record["suggestions"]
            assert record["suggestions"][0]["step_id"] != "PAIR-03"
            assert "search_knowledge_base" in first.tools_used

            chosen = await runtime.service.handle_turn(
                "还是不行，那就退了吧", client_user, conv
            )
            state = _support_state(tmp_path, client_user, conv)
            assert state["demand"]["kind"] == "return"
            assert state["stage"] == "return_requested"
            assert state["suggestions"]
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert "订单" in chosen.response or "邮箱" in chosen.response

            prepared = await runtime.service.handle_turn(
                f"我的邮箱是 {EMAIL}，订单 {ORDER}，申请退货，退无线耳机。",
                client_user,
                conv,
            )
            assert "【退货申请详情（待确认）】" in prepared.response
            assert runtime.store.stats()["counts"]["return_operations"] == 0

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            final = _support_state(tmp_path, client_user, conv)
            assert final["taken_steps"]
            assert final["suggestions"]
            assert any(
                "退货" in item for item in final["unresolved"]
            )

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c04_actual_handoff_uses_mcp_event(monkeypatch, tmp_path, mode):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c04-{name}"
    conv = f"conv-c04-{name}"

    async def scenario():
        async with _runtime() as runtime:
            await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，怎么办？", client_user, conv
            )
            result = await runtime.service.handle_turn(
                "还是不行，帮我转人工", client_user, conv
            )
            assert result.escalated is True
            assert "transfer_to_human_agents" in result.tools_used
            assert "真实工单" in result.response or "未创建" in result.response
            state = _support_state(tmp_path, client_user, conv)
            assert state["stage"] == "handoff_recorded"
            assert state["handoff"]["status"] == "recorded_mock"
            events = runtime.store.list_events(client_user, conv)
            assert any(item["event_type"] == "handoff_requested" for item in events)
            assert runtime.store.stats()["counts"]["return_operations"] == 0

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log_text.count("call tool=transfer_to_human_agents ") == 1
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c05_scope_clarification_and_no_support_record(monkeypatch, tmp_path, mode):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c05-{name}"

    async def scenario():
        async with _runtime() as runtime:
            before = runtime.store.business_snapshot()
            bare = await runtime.service.handle_turn(
                "耳机连不上", client_user, f"conv-c05-bare-{name}"
            )
            assert bare.support_scope == "ambiguous"
            assert _support_state(
                tmp_path, client_user, f"conv-c05-bare-{name}"
            ) is None

            unsupported = await runtime.service.handle_turn(
                "无线耳机没声音，能修吗？", client_user, f"conv-c05-out-{name}"
            )
            assert unsupported.support_scope == "out_of_scope"
            assert _support_state(
                tmp_path, client_user, f"conv-c05-out-{name}"
            ) is None
            assert runtime.store.business_snapshot() == before

    try:
        asyncio.run(scenario())
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c06_user_and_conversation_isolation_with_two_earbuds(
    monkeypatch, tmp_path, mode
):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_a = f"client-a-{name}"
    client_b = f"client-b-{name}"
    conv_a = f"conv-a-{name}"
    conv_b = f"conv-b-{name}"

    async def scenario():
        async with _runtime() as runtime:
            ambiguous = await runtime.service.handle_turn(
                f"我的订单 {OTHER_ORDER} 里有两副耳机，我想退一副，退哪副？",
                client_a,
                conv_a,
            )
            assert ambiguous.support_scope == "ambiguous"
            assert _support_state(tmp_path, client_a, conv_a) is None
            assert runtime.store.stats()["counts"]["return_operations"] == 0

            active = await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，帮我看看", client_b, conv_b
            )
            assert active.support_scope == "active"
            assert _support_state(tmp_path, client_b, conv_b) is not None
            assert _support_state(tmp_path, client_a, conv_a) is None
            assert _support_state(tmp_path, client_a, conv_b) is None
            assert _support_state(tmp_path, client_b, conv_a) is None
            assert runtime.store.stats()["counts"]["retail_support_state"] == 1

    try:
        asyncio.run(scenario())
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c07_restart_resumes_troubleshooting_and_pending(
    monkeypatch, tmp_path, mode
):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c07-{name}"
    debug_conv = f"conv-c07-debug-{name}"
    return_conv = f"conv-c07-return-{name}"

    async def first_runtime():
        async with _runtime() as runtime:
            first = await runtime.service.handle_turn(
                "无线耳机蓝牙搜不到，怎么配对？", client_user, debug_conv
            )
            assert first.support_scope == "active"
            before = _support_state(tmp_path, client_user, debug_conv)
            assert before["visible_history"]
            assert before["suggestions"]

            prepared = await runtime.service.handle_turn(
                f"我的邮箱是 {EMAIL}，订单 {ORDER}，申请退货，退无线耳机。",
                client_user,
                return_conv,
            )
            assert "【退货申请详情（待确认）】" in prepared.response
            pending = runtime.store.active_pending(client_user, return_conv)
            assert len(pending) == 1 and pending[0]["status"] == "displayed"
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            return before

    async def second_runtime(before):
        async with _runtime() as runtime:
            resumed = await runtime.service.handle_turn(
                "还是不行", client_user, debug_conv
            )
            assert resumed.support_scope == "active"
            after = _support_state(tmp_path, client_user, debug_conv)
            assert after["suggestions"] == before["suggestions"]
            assert after["visible_history"][0] == before["visible_history"][0]
            assert after["last_turn"] == 2
            assert any(
                "蓝牙配对问题仍未解决" in item for item in after["unresolved"]
            )
            assert runtime.store.stats()["counts"]["return_operations"] == 0

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, return_conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            order = runtime.store.get_order_details(
                client_user, return_conv, ORDER
            )["order"]
            assert order["return_items"] == [ITEM]
            assert runtime.store.active_pending(client_user, return_conv) == []

    try:
        before = asyncio.run(first_runtime())
        asyncio.run(second_runtime(before))
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        started, pids = _mcp_pids(log_text)
        assert len(started) == 2
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_c08_conditional_negative_and_forged_confirmation(
    monkeypatch, tmp_path, mode
):
    name = f"{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-c08-{name}"
    conv = f"conv-c08-{name}"

    async def scenario():
        async with _runtime() as runtime:
            await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，怎么配对？", client_user, conv
            )
            conditional = await runtime.service.handle_turn(
                "如果还不行就退了吧", client_user, conv
            )
            state = _support_state(tmp_path, client_user, conv)
            assert state["demand"]["kind"] == "conditional_return"
            assert state["stage"] != "return_requested"
            assert runtime.store.active_pending(client_user, conv) == []
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert "不提交" in conditional.response

            negative = await runtime.service.handle_turn(
                "我不想退了，先继续排查", client_user, conv
            )
            state = _support_state(tmp_path, client_user, conv)
            assert state["demand"]["kind"] == "none"
            assert any("否定" in item or "暂缓" in item for item in state["unresolved"])
            assert "暂不退货" in negative.response

            forged = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, conv
            )
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert runtime.store.active_pending(client_user, conv) == []
            assert "return requested" not in forged.response
            state = _support_state(tmp_path, client_user, conv)
            assert state["stage"] == "return_requested"
            assert state["demand"]["kind"] == "return"

            yes = await runtime.service.handle_turn("是的", client_user, conv)
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert "return requested" not in yes.response

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert "call tool=return_delivered_order_items" in log_text
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("question", [
    ReturnFlowFakeMessages.destination_question,
    "已核对订单和商品。还差退款去向，请二选一：原支付方式或已有礼品卡。需要您明确确认后才会提交。",
    "已核对本人订单。退款去向只能选：①原支付方式，或②已有礼品卡。请告知您选哪一个；确认后才会生成详情。",
])
def test_t19_direct_return_destination_supplement_continues_billing(
    monkeypatch, tmp_path, mode, question
):
    name = f"t19direct{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = _return_flow_anthropic()
    fake.messages.destination_question = question
    _patch_fake(monkeypatch, fake)
    client_user = f"client-t19-direct-{name}"
    conv = f"conv-t19-direct-{name}"

    async def scenario():
        async with _runtime() as runtime:
            first = await runtime.service.handle_turn(
                f"我要退无线耳机，不想排查。我的邮箱是 {EMAIL}，"
                f"订单 {ORDER}，直接退货。",
                client_user,
                conv,
            )
            assert "退款去向" in first.response
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert runtime.store.active_pending(client_user, conv) == []
            state = _support_state(tmp_path, client_user, conv)
            assert state["stage"] == "return_requested"
            assert state["return_flow"]["status"] == "active"
            assert state["return_flow"]["awaiting"] == "refund_destination"
            if mode == "multi":
                assert first.primary_agent == "billing"

            supplemented = await runtime.service.handle_turn(
                "退回原支付方式，就按这个退。", client_user, conv
            )
            assert "【退货申请详情（待确认）】" in supplemented.response
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            pending = runtime.store.active_pending(client_user, conv)
            assert len(pending) == 1 and pending[0]["status"] == "displayed"
            state = _support_state(tmp_path, client_user, conv)
            assert state["return_flow"]["status"] == "awaiting_confirmation"
            assert state["return_flow"]["awaiting"] == "confirmation"
            if mode == "multi":
                assert supplemented.primary_agent == "billing"
                assert (
                    supplemented.routing_reason
                    == "support_return_flow_continuation"
                )

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            events = runtime.store.list_events(client_user, conv)
            assert sum(
                item["event_type"] == "return_requested" for item in events
            ) == 1
            assert runtime.store.active_pending(client_user, conv) == []
            assert (
                _support_state(tmp_path, client_user, conv)["return_flow"]["status"]
                == "submitted"
            )

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_t19_troubleshooting_then_return_keeps_billing_across_turns(
    monkeypatch, tmp_path, mode
):
    name = f"t19debug{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    _patch_fake(monkeypatch, _return_flow_anthropic())
    client_user = f"client-t19-debug-{name}"
    conv = f"conv-t19-debug-{name}"

    async def scenario():
        async with _runtime() as runtime:
            first = await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，怎么配对？", client_user, conv
            )
            assert first.support_scope == "active"
            assert _support_state(tmp_path, client_user, conv)["suggestions"]
            if mode == "multi":
                assert first.primary_agent == "technical"

            chosen = await runtime.service.handle_turn(
                f"还是不行，我要退货。我的邮箱是 {EMAIL}，"
                f"订单 {ORDER}，退无线耳机。",
                client_user,
                conv,
            )
            assert "退款去向" in chosen.response
            state = _support_state(tmp_path, client_user, conv)
            assert state["return_flow"]["awaiting"] == "refund_destination"
            assert state["suggestions"]
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if mode == "multi":
                assert chosen.primary_agent == "billing"

            prepared = await runtime.service.handle_turn(
                "退回原支付方式", client_user, conv
            )
            assert "【退货申请详情（待确认）】" in prepared.response
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if mode == "multi":
                assert prepared.primary_agent == "billing"
                assert prepared.routing_reason == "support_return_flow_continuation"

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            state = _support_state(tmp_path, client_user, conv)
            assert state["stage"] == "return_requested"
            assert state["return_flow"]["status"] == "submitted"

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert log_text.count("call tool=prepare_return ") == 1
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["multi"])
def test_t19_restart_preserves_return_phase_and_pending(
    monkeypatch, tmp_path, mode
):
    name = f"t19restart{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    _patch_fake(monkeypatch, _return_flow_anthropic())
    client_user = f"client-t19-restart-{name}"
    conv = f"conv-t19-restart-{name}"

    async def first_runtime():
        async with _runtime() as runtime:
            await runtime.service.handle_turn(
                f"我要退无线耳机，不想排查。我的邮箱是 {EMAIL}，"
                f"订单 {ORDER}，直接退货。",
                client_user,
                conv,
            )
            prepared = await runtime.service.handle_turn(
                "退回原支付方式", client_user, conv
            )
            assert "【退货申请详情（待确认）】" in prepared.response
            assert runtime.store.active_pending(client_user, conv)[0][
                "status"
            ] == "displayed"
            return _support_state(tmp_path, client_user, conv)

    async def second_runtime(before):
        # 新 Fake 没有上一进程的待确认参数，只能依赖 SQLite 中的 pending。
        _patch_fake(monkeypatch, _return_flow_anthropic())
        async with _runtime() as runtime:
            resumed = _support_state(tmp_path, client_user, conv)
            assert resumed["return_flow"] == before["return_flow"]
            assert resumed["stage"] == "return_requested"
            pending = runtime.store.active_pending(client_user, conv)
            assert len(pending) == 1 and pending[0]["status"] == "displayed"

            confirmed = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, conv
            )
            assert "return requested" in confirmed.response
            assert runtime.store.stats()["counts"]["return_operations"] == 1
            assert runtime.store.active_pending(client_user, conv) == []

    try:
        before = asyncio.run(first_runtime())
        asyncio.run(second_runtime(before))
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        started, pids = _mcp_pids(log_text)
        assert len(started) == 2
        assert log_text.count("call tool=return_delivered_order_items ") == 1
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_t19_cancel_switch_and_human_close_return_without_writes(
    monkeypatch, tmp_path, mode
):
    name = f"t19exit{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    _patch_fake(monkeypatch, _return_flow_anthropic())
    client_user = f"client-t19-exit-{name}"

    async def start_return(runtime, conv):
        await runtime.service.handle_turn(
            f"我要退无线耳机，不想排查。我的邮箱是 {EMAIL}，"
            f"订单 {ORDER}，直接退货。",
            client_user,
            conv,
        )
        assert (
            _support_state(tmp_path, client_user, conv)["return_flow"]["status"]
            == "active"
        )

    async def scenario():
        async with _runtime() as runtime:
            cancel_conv = f"conv-t19-cancel-{name}"
            await start_return(runtime, cancel_conv)
            cancelled = await runtime.service.handle_turn(
                "取消退货", client_user, cancel_conv
            )
            state = _support_state(tmp_path, client_user, cancel_conv)
            assert state["return_flow"]["status"] == "cancelled"
            assert state["return_flow"]["awaiting"] == ""
            assert state["stage"] != "return_requested"
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if mode == "multi":
                assert cancelled.primary_agent == "general"
                assert (
                    cancelled.routing_reason == "support_return_flow_cancelled"
                )

            switch_conv = f"conv-t19-switch-{name}"
            await start_return(runtime, switch_conv)
            switched = await runtime.service.handle_turn(
                "先继续排查吧", client_user, switch_conv
            )
            state = _support_state(tmp_path, client_user, switch_conv)
            assert state["return_flow"]["status"] == "switched"
            assert state["stage"] == "troubleshooting"
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if mode == "multi":
                assert switched.primary_agent == "technical"
                assert (
                    switched.routing_reason
                    == "support_return_flow_switched_to_troubleshooting"
                )

            human_conv = f"conv-t19-human-{name}"
            await start_return(runtime, human_conv)
            human = await runtime.service.handle_turn(
                "请转人工", client_user, human_conv
            )
            state = _support_state(tmp_path, client_user, human_conv)
            assert human.escalated is True
            assert state["return_flow"]["status"] == "handed_off"
            assert state["stage"] == "handoff_recorded"
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            if mode == "multi":
                assert human.primary_agent == "escalation"

            stale_conv = f"conv-t19-stale-{name}"
            await start_return(runtime, stale_conv)
            await runtime.service.handle_turn(
                "退回原支付方式", client_user, stale_conv
            )
            assert runtime.store.active_pending(client_user, stale_conv)
            await runtime.service.handle_turn(
                "取消退货", client_user, stale_conv
            )
            assert runtime.store.active_pending(client_user, stale_conv) == []
            replay = await runtime.service.handle_turn(
                "确认，就按这些申请退货", client_user, stale_conv
            )
            assert "return requested" not in replay.response
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            events = runtime.store.list_events(client_user, stale_conv)
            assert not any(
                item["event_type"] == "return_requested" for item in events
            )

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        assert "call tool=prepare_return " in log_text
        assert "call tool=return_delivered_order_items" in log_text
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


def test_support_state_errors_are_explicit_and_do_not_replay(monkeypatch, tmp_path):
    name = f"stateerr{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, "single"
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-stateerr-{name}"
    conv = f"conv-stateerr-{name}"

    async def scenario():
        async with _runtime() as runtime:
            before = runtime.store.business_snapshot()

            def fail_load(user: str, conversation: str):
                raise SupportStateError("模拟读取失败")

            original_load = runtime.store.load_support_state
            runtime.store.load_support_state = fail_load
            blocked = await runtime.service.handle_turn(
                f"我的邮箱是 {EMAIL}，订单 {ORDER}，申请退货，退无线耳机。",
                client_user,
                conv,
            )
            assert "SUPPORT_STATE_ERROR" in blocked.response
            assert blocked.support_state_error
            assert blocked.tools_used == []
            assert runtime.store.stats()["counts"]["return_operations"] == 0
            assert runtime.store.active_pending(client_user, conv) == []
            assert runtime.store.business_snapshot() == before
            runtime.store.load_support_state = original_load

            def fail_save(user: str, conversation: str, payload: dict):
                raise SupportStateError("模拟写入失败")

            runtime.store.save_support_state = fail_save
            saved = await runtime.service.handle_turn(
                "无线耳机蓝牙连不上，怎么办？", client_user, conv
            )
            assert "[SUPPORT_STATE_ERROR]" in saved.response
            assert saved.support_state_error
            assert runtime.store.business_snapshot() == before

    try:
        asyncio.run(scenario())
        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_api_lifespan_assembles_support_runtime(monkeypatch, tmp_path, mode):
    name = f"api{mode}{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, mode
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-api-{name}"
    conv = f"conv-api-{name}"

    try:
        with TestClient(api_main.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            body = health.json()
            assert body["mode"] == "retail"
            assert body["support"]["enabled"] is True
            assert body["support"]["collection"] == support_collection

            stats = client.get("/knowledge/stats")
            assert stats.status_code == 200
            stats_body = stats.json()
            assert stats_body["total_chunks"] == 35
            assert stats_body["collection"] == support_collection
            assert stats_body["support_material"] == "project_authored_sandbox"

            response = client.post(
                "/chat",
                json={
                    "message": "无线耳机蓝牙连不上，怎么配对？",
                    "user_id": client_user,
                    "conv_id": conv,
                },
            )
            assert response.status_code == 200, response.text
            payload = response.json()
            assert payload["support_enabled"] is True
            assert payload["support_scope"] == "active"
            assert payload["support_stage"] == "awaiting_feedback"
            assert "search_knowledge_base" in payload["tools_used"]
            assert "PAIR-0" in payload["response"]

            runtime = api_main._retail_runtime
            assert runtime.config.support_enabled is True
            assert runtime.config.agent_mode == mode
            assert runtime.ledger.summary()["spent_cny"] == 0

        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


def test_cli_assembles_support_runtime(monkeypatch, tmp_path, capsys):
    name = f"cli{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, "single"
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)
    client_user = f"client-cli-{name}"
    conv = f"conv-cli-{name}"
    monkeypatch.setenv("RETAIL_CLIENT_USER", client_user)
    monkeypatch.setenv("RETAIL_CONV_ID", conv)
    replies = iter(["无线耳机蓝牙连不上，怎么配对？", "退出"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    try:
        asyncio.run(api_main._retail_cli())
        output = capsys.readouterr().out
        assert "有限支持 开启" in output
        assert "PAIR-0" in output
        state = _support_state(tmp_path, client_user, conv)
        assert state["stage"] == "awaiting_feedback"

        log_text = (tmp_path / "mcp_stderr.log").read_text(encoding="utf-8")
        _started, pids = _mcp_pids(log_text)
        for pid in pids:
            _assert_process_stopped(pid)
    finally:
        _cleanup(support_collection, legacy_collection)


def test_real_chroma_support_collection_has_35_applicable_segments(
    monkeypatch, tmp_path
):
    name = f"chroma{uuid.uuid4().hex[:6]}"
    support_collection, legacy_collection = _configure(
        monkeypatch, tmp_path, name, "single"
    )
    fake = SupportFakeAnthropic()
    _patch_fake(monkeypatch, fake)

    async def scenario():
        async with _runtime() as runtime:
            knowledge = runtime.knowledge_base
            assert knowledge is not None
            assert knowledge.doc_count == 35
            ids = knowledge.list_ids()
            assert len(ids) == 35
            positive = knowledge.search("无线耳机 蓝牙 配对 步骤", top_k=5)
            steps = {item.get("step_id") for item in positive if item.get("step_id")}
            assert steps & {"PAIR-01", "PAIR-02", "PAIR-03"}
            for item in positive:
                if item.get("step_id"):
                    assert item["product_id"] == "9924732112"
                    assert item["issue"] == "bluetooth_pairing"
                    assert item["source_kind"] == "project_authored_sandbox"
            negative = knowledge.search("烤箱退货期限是多久", top_k=5)
            assert negative

            # 二次导入幂等，不自动清理或重建集合。
            from agents.retail_runtime import (
                _load_knowledge,
                _load_support_knowledge,
            )

            rows = list(_load_knowledge(Path(KNOWLEDGE))) + list(
                _load_support_knowledge(Path(SUPPORT_KNOWLEDGE))
            )
            added = knowledge.upsert_evidence(rows)
            assert added == 0
            assert knowledge.doc_count == 35

    try:
        asyncio.run(scenario())
    finally:
        _cleanup(support_collection, legacy_collection)
