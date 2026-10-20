"""零售业务工具经本地 MCP 的确定性对照与协议边界回归。

工具服务是真实子进程，业务状态来自两个同源但独立的 SQLite。直调与
MCP 使用相同种子和相同轮次序列，比较每个工具结果与最终业务表；模型
不参与本文件。
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Tuple

import mcp.types as types
import pytest
from mcp.shared.exceptions import McpError

from agents.agent_orchestrator import Request
from agents.retail_tools import retail_tools
from mcp.retail_client import (
    RetailMCPClient,
    RetailMCPProtocolError,
    RetailMCPTransportError,
)
from mcp.retail_protocol import build_context_meta
from mcp.retail_store import RetailStore

ROOT = Path(__file__).resolve().parent.parent
SEED = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"

EMAIL = "chen.silva2698@example.com"
ORDER = "#W3069600"
ITEM = "5012998807"
PRODUCT = "5149340237"
PAYMENT = "credit_card_1565124"

OTHER_FIRST = "Ivan"
OTHER_LAST = "Santos"
OTHER_ZIP = "75277"

VOLATILE_ID = re.compile(r"\b(?:RT|HO)-[0-9a-f]{12}\b")
ISO_TIME = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?(?:Z|[+-][0-9:]+)?"
)


def _normalize(value: Any) -> Any:
    """把随机操作号与时间戳替换为稳定占位，其余值原样比较。"""
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str):
        return ISO_TIME.sub("<TS>", VOLATILE_ID.sub("<ID>", value))
    return value


def _db_state(path: Path) -> Dict[str, Any]:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    state: Dict[str, Any] = {}
    try:
        for table, order in (
            ("meta", "key"),
            ("users", "user_id"),
            ("orders", "order_id"),
            ("products", "product_id"),
            ("sessions", "client_user_id, conv_id"),
            ("pending_returns", "client_user_id, conv_id, order_id, status"),
            ("return_operations", "client_user_id, conv_id, order_id"),
            ("retail_events", "event_id"),
        ):
            rows = connection.execute(
                f"SELECT * FROM {table} ORDER BY {order}"
            ).fetchall()
            state[table] = [_normalize(dict(row)) for row in rows]
    finally:
        connection.close()
    return state


def _request(
    user_id: str,
    conv_id: str,
    turn: int,
    message: str,
    request_id: str,
) -> Request:
    return Request(
        message=message,
        user_id=user_id,
        conv_id=conv_id,
        turn=turn,
        request_id=request_id,
    )


def _direct_call(
    specs: Dict[str, Any],
    user_id: str,
    conv_id: str,
    turn: int,
    name: str,
    args: Dict[str, Any],
    message: str,
    request_id: str,
) -> Dict[str, Any]:
    req = _request(user_id, conv_id, turn, message, request_id)
    result = specs[name].handler(req, args)
    assert isinstance(result, dict)
    return result


def _client_pids(log_path: Path) -> Tuple[int, set[int]]:
    text = log_path.read_text(encoding="utf-8")
    started = re.findall(r"started pid=(\d+)", text)
    pids = {int(value) for value in re.findall(r"pid=(\d+)", text)}
    return len(started), pids


async def _mcp_call(
    client: RetailMCPClient,
    user_id: str,
    conv_id: str,
    turn: int,
    name: str,
    args: Dict[str, Any],
    message: str,
    request_id: str,
) -> Dict[str, Any]:
    return await client.call_tool(
        name,
        _request(user_id, conv_id, turn, message, request_id),
        args,
    )


@asynccontextmanager
async def _started_client(
    tmp_path: Path, name: str = "mcp"
) -> AsyncIterator[Tuple[Path, RetailStore, RetailMCPClient, Path]]:
    db = tmp_path / f"{name}.db"
    stderr_path = tmp_path / f"{name}-stderr.log"
    store = RetailStore(db, SEED)
    specs = retail_tools(store, None, language="zh")
    client = RetailMCPClient(
        db_path=db,
        seed_path=SEED,
        language="zh",
        expected_specs=specs,
        project_root=ROOT,
        startup_timeout=20,
        call_timeout=20,
        stderr_log_path=stderr_path,
    )
    await client.start()
    try:
        yield db, store, client, stderr_path
    finally:
        await client.close()


async def _raw_call(
    client: RetailMCPClient,
    name: str,
    arguments: Dict[str, Any],
    meta: Any,
) -> types.CallToolResult:
    params = types.CallToolRequestParams(
        name=name, arguments=arguments
    )
    if meta is not None:
        params = types.CallToolRequestParams(
            name=name, arguments=arguments, _meta=meta
        )
    request = types.ClientRequest(
        types.CallToolRequest(method="tools/call", params=params)
    )
    return await client._session.send_request(
        request,
        types.CallToolResult,
        request_read_timeout_seconds=timedelta(seconds=10),
    )


def test_direct_and_mcp_match_on_seeded_databases(tmp_path):
    direct_db = tmp_path / "direct.db"
    mcp_db = tmp_path / "mcp.db"
    stderr_path = tmp_path / "mcp-stderr.log"

    direct_store = RetailStore(direct_db, SEED)
    mcp_store = RetailStore(mcp_db, SEED)
    direct_specs = retail_tools(direct_store, None, language="zh")
    expected_specs = retail_tools(mcp_store, None, language="zh")
    assert set(expected_specs) == set(direct_specs)
    assert len(expected_specs) == 9

    user_a = "client-compare-a"
    conv_a = "conv-compare-a"
    user_b = "client-compare-b"
    conv_b = "conv-compare-b"

    client = RetailMCPClient(
        db_path=mcp_db,
        seed_path=SEED,
        language="zh",
        expected_specs=expected_specs,
        project_root=ROOT,
        startup_timeout=20,
        call_timeout=20,
        stderr_log_path=stderr_path,
    )

    async def scenario() -> None:
        await client.start()
        assert client.protocol_version
        assert set(client.tools()) == set(expected_specs)
        try:
            turn_a1 = direct_store.begin_turn(user_a, conv_a)
            assert mcp_store.begin_turn(user_a, conv_a) == turn_a1
            steps = [
                (
                    "find_user_id_by_email",
                    {"email": EMAIL},
                    "请核对我的邮箱",
                    "cmp-a-01",
                ),
                (
                    "get_user_details",
                    {},
                    "查看我的资料",
                    "cmp-a-02",
                ),
                (
                    "get_order_details",
                    {"order_id": ORDER},
                    "查询订单",
                    "cmp-a-03",
                ),
                (
                    "get_product_details",
                    {"product_id": PRODUCT},
                    "查询商品",
                    "cmp-a-04",
                ),
                (
                    "calculate",
                    {"expression": "(186.06 + 258.71) * 2"},
                    "计算金额",
                    "cmp-a-05",
                ),
            ]
            for name, args, message, request_id in steps:
                direct = _direct_call(
                    direct_specs,
                    user_a,
                    conv_a,
                    turn_a1,
                    name,
                    args,
                    message,
                    request_id,
                )
                mcp = await _mcp_call(
                    client,
                    user_a,
                    conv_a,
                    turn_a1,
                    name,
                    args,
                    message,
                    request_id,
                )
                assert _normalize(mcp) == _normalize(direct), name

            prepare_args = {
                "order_id": ORDER,
                "item_ids": [ITEM],
                "payment_method_id": PAYMENT,
            }
            direct_prepare = _direct_call(
                direct_specs,
                user_a,
                conv_a,
                turn_a1,
                "prepare_return",
                prepare_args,
                "准备退货",
                "cmp-a-06",
            )
            mcp_prepare = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn_a1,
                "prepare_return",
                prepare_args,
                "准备退货",
                "cmp-a-06",
            )
            assert _normalize(mcp_prepare) == _normalize(direct_prepare)
            assert direct_prepare["success"] is True
            assert mcp_prepare["success"] is True
            assert direct_store.mark_pending_displayed(
                user_a, conv_a, direct_prepare["op_id"], turn_a1, True
            )
            assert mcp_store.mark_pending_displayed(
                user_a, conv_a, mcp_prepare["op_id"], turn_a1, True
            )

            turn_a2 = direct_store.begin_turn(user_a, conv_a)
            assert mcp_store.begin_turn(user_a, conv_a) == turn_a2
            confirm_args = dict(prepare_args)
            direct_submit = _direct_call(
                direct_specs,
                user_a,
                conv_a,
                turn_a2,
                "return_delivered_order_items",
                confirm_args,
                "确认，就按这些申请退货",
                "cmp-a-07",
            )
            mcp_submit = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn_a2,
                "return_delivered_order_items",
                confirm_args,
                "确认，就按这些申请退货",
                "cmp-a-07",
            )
            assert _normalize(mcp_submit) == _normalize(direct_submit)
            assert direct_submit["success"] is True
            assert mcp_submit["success"] is True
            assert direct_submit["status"] == "return requested"

            turn_a3 = direct_store.begin_turn(user_a, conv_a)
            assert mcp_store.begin_turn(user_a, conv_a) == turn_a3
            handoff_args = {
                "summary": "用户要求人工继续处理退货",
                "reason": "需要人工确认",
            }
            direct_handoff = _direct_call(
                direct_specs,
                user_a,
                conv_a,
                turn_a3,
                "transfer_to_human_agents",
                handoff_args,
                "请转人工",
                "cmp-a-08",
            )
            mcp_handoff = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn_a3,
                "transfer_to_human_agents",
                handoff_args,
                "请转人工",
                "cmp-a-08",
            )
            assert _normalize(mcp_handoff) == _normalize(direct_handoff)

            turn_b1 = direct_store.begin_turn(user_b, conv_b)
            assert mcp_store.begin_turn(user_b, conv_b) == turn_b1
            name_args = {
                "first_name": OTHER_FIRST,
                "last_name": OTHER_LAST,
                "zip_code": OTHER_ZIP,
            }
            direct_bind = _direct_call(
                direct_specs,
                user_b,
                conv_b,
                turn_b1,
                "find_user_id_by_name_zip",
                name_args,
                "按姓名和邮编核对",
                "cmp-b-01",
            )
            mcp_bind = await _mcp_call(
                client,
                user_b,
                conv_b,
                turn_b1,
                "find_user_id_by_name_zip",
                name_args,
                "按姓名和邮编核对",
                "cmp-b-01",
            )
            assert _normalize(mcp_bind) == _normalize(direct_bind)
            assert direct_bind["success"] is True
            assert mcp_bind["success"] is True

            assert client.call_count == 9
        finally:
            await client.close()

    asyncio.run(scenario())

    assert _db_state(mcp_db) == _db_state(direct_db)
    started, pids = _client_pids(stderr_path)
    assert started == 1
    assert len(pids) == 1
    state = _db_state(mcp_db)
    assert len(state["return_operations"]) == 1
    assert state["pending_returns"][0]["status"] == "consumed"
    assert [
        row["event_type"] for row in state["retail_events"]
    ] == [
        "identity_bound",
        "return_prepared",
        "return_displayed",
        "return_requested",
        "handoff_requested",
        "identity_bound",
    ]


def test_context_unknown_tool_and_argument_validation(tmp_path):
    async def scenario() -> None:
        async with _started_client(tmp_path, "boundaries") as (
            _db,
            store,
            client,
            _stderr,
        ):
            user_a = "client-context-a"
            conv_a = "conv-context-a"
            turn = store.begin_turn(user_a, conv_a)
            valid = build_context_meta(
                user_a, conv_a, "ctx-valid", turn, "核对身份", "zh"
            )

            control = await _raw_call(
                client,
                "find_user_id_by_email",
                {"email": EMAIL},
                valid,
            )
            assert control.structuredContent["success"] is True

            cases = [
                (
                    "get_user_details",
                    {},
                    None,
                    "缺少受信任的调用上下文",
                ),
                (
                    "get_user_details",
                    {},
                    {**valid, "turn": 999},
                    "轮次不一致",
                ),
                (
                    "get_user_details",
                    {},
                    {k: v for k, v in valid.items() if k != "message"},
                    "message",
                ),
                (
                    "get_user_details",
                    {},
                    {**valid, "language": "en"},
                    "语言",
                ),
                (
                    "get_user_details",
                    {},
                    {**valid, "client_user_id": "intruder"},
                    "轮次不一致",
                ),
            ]
            for name, args, meta, keyword in cases:
                with pytest.raises(McpError) as raised:
                    await _raw_call(client, name, args, meta)
                assert raised.value.error.code == types.INVALID_PARAMS
                assert keyword in str(raised.value)

            with pytest.raises(McpError) as unknown:
                await _raw_call(client, "no_such_tool", {}, valid)
            assert unknown.value.error.code == types.METHOD_NOT_FOUND

            bad_arguments = [
                ("find_user_id_by_email", {}, "缺少必需参数"),
                ("find_user_id_by_email", {"email": 123}, "类型错误"),
                (
                    "find_user_id_by_email",
                    {"email": EMAIL, "extra": "x"},
                    "不允许的工具参数",
                ),
                (
                    "prepare_return",
                    {
                        "order_id": ORDER,
                        "item_ids": [],
                        "payment_method_id": PAYMENT,
                    },
                    "少于最少项数",
                ),
                (
                    "prepare_return",
                    {
                        "order_id": ORDER,
                        "item_ids": "not-a-list",
                        "payment_method_id": PAYMENT,
                    },
                    "期望 array",
                ),
                ("calculate", {"expression": 12}, "类型错误"),
            ]
            for name, args, keyword in bad_arguments:
                with pytest.raises(RetailMCPProtocolError) as raised:
                    await client.call_tool(
                        name,
                        _request(user_a, conv_a, turn, "参数校验", "ctx-bad"),
                        args,
                    )
                assert raised.value.code == types.INVALID_PARAMS
                assert keyword in str(raised.value)

            assert store.stats()["counts"]["return_operations"] == 0
            assert store.stats()["counts"]["retail_events"] == 1

    asyncio.run(scenario())


def test_unbound_unauthorized_and_confirmation_boundaries(tmp_path):
    async def scenario() -> None:
        async with _started_client(tmp_path, "business") as (
            _db,
            store,
            client,
            _stderr,
        ):
            user_a = "client-business-a"
            conv_a = "conv-business-a"
            user_b = "client-business-b"
            conv_b = "conv-business-b"

            turn1 = store.begin_turn(user_a, conv_a)
            unbound = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn1,
                "get_order_details",
                {"order_id": ORDER},
                "查询订单",
                "biz-01",
            )
            assert unbound["success"] is False
            assert "身份" in unbound["error"] or "绑定" in unbound["error"]

            await _mcp_call(
                client,
                user_a,
                conv_a,
                turn1,
                "find_user_id_by_email",
                {"email": EMAIL},
                "我的邮箱",
                "biz-02",
            )
            unauthorized = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn1,
                "get_order_details",
                {"order_id": "#W6893533"},
                "查询他人订单",
                "biz-03",
            )
            assert unauthorized["success"] is False
            assert "当前用户" in unauthorized["error"]

            prepare_args = {
                "order_id": ORDER,
                "item_ids": [ITEM],
                "payment_method_id": PAYMENT,
            }
            prepared = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn1,
                "prepare_return",
                prepare_args,
                "准备退货",
                "biz-04",
            )
            assert prepared["success"] is True
            assert store.mark_pending_displayed(
                user_a, conv_a, prepared["op_id"], turn1, True
            )
            same_turn = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn1,
                "return_delivered_order_items",
                prepare_args,
                "确认，就按这些申请退货",
                "biz-05",
            )
            assert same_turn["success"] is False
            assert "同一轮" in same_turn["error"]

            turn2 = store.begin_turn(user_a, conv_a)
            negated = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn2,
                "return_delivered_order_items",
                prepare_args,
                "先别提交，我再看看",
                "biz-06",
            )
            assert negated["success"] is False
            assert "未获得" in negated["error"]

            turn3 = store.begin_turn(user_a, conv_a)
            changed = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn3,
                "return_delivered_order_items",
                prepare_args,
                "确认，但是把数量改成两件",
                "biz-07",
            )
            assert changed["success"] is False
            assert store.stats()["counts"]["return_operations"] == 0

            turn4 = store.begin_turn(user_a, conv_a)
            amount_statement = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn4,
                "return_delivered_order_items",
                prepare_args,
                "预计金额不对，先别提交",
                "biz-08",
            )
            assert amount_statement["success"] is False
            assert store.stats()["counts"]["return_operations"] == 0

            turn5 = store.begin_turn(user_a, conv_a)
            submitted = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn5,
                "return_delivered_order_items",
                prepare_args,
                "确认，就按这些申请退货",
                "biz-09",
            )
            assert submitted["success"] is True
            assert submitted["status"] == "return requested"
            assert store.stats()["counts"]["return_operations"] == 1

            turn6 = store.begin_turn(user_a, conv_a)
            duplicate = await _mcp_call(
                client,
                user_a,
                conv_a,
                turn6,
                "return_delivered_order_items",
                prepare_args,
                "确认，就按这些申请退货",
                "biz-10",
            )
            assert duplicate["success"] is False
            assert store.stats()["counts"]["return_operations"] == 1

            turn_b1 = store.begin_turn(user_b, conv_b)
            await _mcp_call(
                client,
                user_b,
                conv_b,
                turn_b1,
                "find_user_id_by_name_zip",
                {
                    "first_name": OTHER_FIRST,
                    "last_name": OTHER_LAST,
                    "zip_code": OTHER_ZIP,
                },
                "按姓名邮编核对",
                "biz-11",
            )
            cross_session = await _mcp_call(
                client,
                user_b,
                conv_b,
                turn_b1,
                "return_delivered_order_items",
                prepare_args,
                "确认，就按这些申请退货",
                "biz-12",
            )
            assert cross_session["success"] is False
            assert store.stats()["counts"]["return_operations"] == 1

            order = store.get_order_details(
                user_a, conv_a, ORDER
            )["order"]
            assert order["status"] == "return requested"
            assert order["return_items"] == [ITEM]

    asyncio.run(scenario())


def test_two_sessions_confirm_one_order_at_a_time(tmp_path):
    async def scenario() -> None:
        async with _started_client(tmp_path, "interleave") as (
            _db,
            store,
            client,
            _stderr,
        ):
            conv1 = "conv-interleave-1"
            conv2 = "conv-interleave-2"
            user1 = "client-interleave-1"
            user2 = "client-interleave-2"

            turn1 = store.begin_turn(user1, conv1)
            assert store.begin_turn(user2, conv2) == turn1
            for user, conv, request_id in (
                (user1, conv1, "int-01"),
                (user2, conv2, "int-02"),
            ):
                bound = await _mcp_call(
                    client,
                    user,
                    conv,
                    turn1,
                    "find_user_id_by_email",
                    {"email": EMAIL},
                    "我的邮箱",
                    request_id,
                )
                assert bound["success"] is True

            details1 = await _mcp_call(
                client,
                user1,
                conv1,
                turn1,
                "get_order_details",
                {"order_id": ORDER},
                "查询第一笔订单",
                "int-03",
            )
            order2_id = "#W2598834"
            details2 = await _mcp_call(
                client,
                user2,
                conv2,
                turn1,
                "get_order_details",
                {"order_id": order2_id},
                "查询第二笔订单",
                "int-04",
            )
            assert details1["success"] is True
            assert details2["success"] is True

            args1 = {
                "order_id": ORDER,
                "item_ids": [details1["order"]["items"][0]["item_id"]],
                "payment_method_id": details1["original_payment_method_id"],
            }
            args2 = {
                "order_id": order2_id,
                "item_ids": [details2["order"]["items"][0]["item_id"]],
                "payment_method_id": details2["original_payment_method_id"],
            }
            prepared1 = await _mcp_call(
                client,
                user1,
                conv1,
                turn1,
                "prepare_return",
                args1,
                "准备第一笔",
                "int-05",
            )
            prepared2 = await _mcp_call(
                client,
                user2,
                conv2,
                turn1,
                "prepare_return",
                args2,
                "准备第二笔",
                "int-06",
            )
            assert prepared1["success"] is True
            assert prepared2["success"] is True
            assert store.mark_pending_displayed(
                user1, conv1, prepared1["op_id"], turn1, True
            )
            assert store.mark_pending_displayed(
                user2, conv2, prepared2["op_id"], turn1, True
            )

            turn2 = store.begin_turn(user1, conv1)
            assert store.begin_turn(user2, conv2) == turn2
            wrong_session = await _mcp_call(
                client,
                user2,
                conv2,
                turn2,
                "return_delivered_order_items",
                args1,
                "确认，就按这些申请退货",
                "int-07",
            )
            assert wrong_session["success"] is False
            assert store.stats()["counts"]["return_operations"] == 0
            assert store.active_pending(user2, conv2)[0]["status"] == "displayed"

            first = await _mcp_call(
                client,
                user1,
                conv1,
                turn2,
                "return_delivered_order_items",
                args1,
                "确认，就按这些申请退货",
                "int-08",
            )
            assert first["success"] is True
            assert store.stats()["counts"]["return_operations"] == 1
            assert store.active_pending(user2, conv2)[0]["status"] == "displayed"

            second = await _mcp_call(
                client,
                user2,
                conv2,
                turn2,
                "return_delivered_order_items",
                args2,
                "确认，就按这些申请退货",
                "int-09",
            )
            assert second["success"] is True
            assert store.stats()["counts"]["return_operations"] == 2
            assert (
                store.get_order_details(user1, conv1, ORDER)["order"]["status"]
                == "return requested"
            )
            assert (
                store.get_order_details(user2, conv2, order2_id)["order"]["status"]
                == "return requested"
            )

    asyncio.run(scenario())


def test_lost_submit_response_does_not_replay(tmp_path, monkeypatch):
    async def scenario() -> None:
        async with _started_client(tmp_path, "loss") as (
            _db,
            store,
            client,
            stderr_path,
        ):
            user = "client-loss"
            conv = "conv-loss"
            turn1 = store.begin_turn(user, conv)
            await _mcp_call(
                client,
                user,
                conv,
                turn1,
                "find_user_id_by_email",
                {"email": EMAIL},
                "我的邮箱",
                "loss-01",
            )
            args = {
                "order_id": ORDER,
                "item_ids": [ITEM],
                "payment_method_id": PAYMENT,
            }
            prepared = await _mcp_call(
                client,
                user,
                conv,
                turn1,
                "prepare_return",
                args,
                "准备退货",
                "loss-02",
            )
            assert store.mark_pending_displayed(
                user, conv, prepared["op_id"], turn1, True
            )

            turn2 = store.begin_turn(user, conv)
            original = client._session.send_request
            call_count_before = client.call_count

            async def lossy(request, result_type, **kwargs):
                result = await original(request, result_type, **kwargs)
                if request.root.params.name == "return_delivered_order_items":
                    raise McpError(
                        types.ErrorData(
                            code=408, message="Request timed out"
                        )
                    )
                return result

            monkeypatch.setattr(client._session, "send_request", lossy)
            with pytest.raises(RetailMCPTransportError):
                await _mcp_call(
                    client,
                    user,
                    conv,
                    turn2,
                    "return_delivered_order_items",
                    args,
                    "确认，就按这些申请退货",
                    "loss-03",
                )
            monkeypatch.setattr(client._session, "send_request", original)
            assert client.call_count == call_count_before + 1

            stats = store.stats()["counts"]
            assert stats["return_operations"] == 1
            order = store.get_order_details(user, conv, ORDER)["order"]
            assert order["status"] == "return requested"

            reread = await _mcp_call(
                client,
                user,
                conv,
                turn2,
                "get_order_details",
                {"order_id": ORDER},
                "确认是否写入",
                "loss-04",
            )
            assert reread["success"] is True
            assert reread["order"]["status"] == "return requested"
            assert store.stats()["counts"]["return_operations"] == 1

            text = stderr_path.read_text(encoding="utf-8")
            assert text.count("call tool=return_delivered_order_items") == 1
            started, pids = _client_pids(stderr_path)
            assert started == 1 and len(pids) == 1

    asyncio.run(scenario())


def test_result_contract_rejects_inconsistent_payloads() -> None:
    payload = {"success": True, "user_id": "u1"}

    def build(
        structured: Any = payload,
        text: str = json.dumps(payload, ensure_ascii=False),
        is_error: bool = False,
    ) -> types.CallToolResult:
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            structuredContent=structured,
            isError=is_error,
        )

    valid = build()
    assert RetailMCPClient._validate_result("get_user_details", valid) == payload

    with pytest.raises(RetailMCPProtocolError, match="isError"):
        RetailMCPClient._validate_result(
            "get_user_details", build(is_error=True)
        )
    with pytest.raises(RetailMCPProtocolError, match="不一致"):
        RetailMCPClient._validate_result(
            "get_user_details",
            build(text=json.dumps({"success": True})),
        )
    with pytest.raises(RetailMCPProtocolError, match="结构化结果"):
        RetailMCPClient._validate_result(
            "get_user_details", build(structured=None)
        )


def test_startup_failure_cleans_open_resources(tmp_path: Path) -> None:
    async def scenario() -> None:
        db = tmp_path / "startup.db"
        stderr_path = tmp_path / "startup-stderr.log"
        store = RetailStore(db, SEED)
        client = RetailMCPClient(
            db_path=db,
            seed_path=SEED,
            language="zh",
            expected_specs=retail_tools(store, None, language="zh"),
            project_root=ROOT,
            python_executable=str(tmp_path / "missing-python.exe"),
            startup_timeout=5,
            call_timeout=5,
            stderr_log_path=stderr_path,
        )

        with pytest.raises(RetailMCPTransportError):
            await client.start()
        assert client._session is None
        assert client._session_cm is None
        assert client._stdio_cm is None
        assert client._stderr_handle is None
        assert client.tools() == {}

        with pytest.raises(RetailMCPTransportError):
            await client.call_tool(
                "find_user_id_by_email",
                _request("client-startup", "conv-startup", 1, "查邮箱", "start-01"),
                {"email": EMAIL},
            )

        await client.close()
        await client.close()
        assert client._closed is True

    asyncio.run(scenario())
