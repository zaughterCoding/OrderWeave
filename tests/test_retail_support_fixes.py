"""交接当前视图与编号类型的局部回归，不发送模型请求。"""
import copy
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentType, Request
from agents.retail_handoff import build_role_handoff
from agents.retail_support import analyze_message, build_support_packet, handoff_summary_from_packet
from tests.test_retail_support import _active_return_record, _support_decision


@pytest.mark.parametrize("item", ["item_id: 6452271382", '"item_id": "6452271382"',
    "商品编号 6452271382", "商品条目编号：6452271382", "（6452271382）一件"])
def test_order_item_is_not_a_foreign_product(item):
    text = f"我要退货，订单 #W1067251，蓝色无线耳机 {item}，退款到原支付方式。"
    analysis = analyze_message(text)
    packet = build_support_packet(None, analysis, text, 1, None, [])
    assert analysis.scope_state == "active"
    assert _support_decision(packet).primary_agent == AgentType.BILLING


@pytest.mark.parametrize("text", [
    "无线耳机 product_id=1234567890，item_id=6452271382，蓝牙连不上",
    "无线耳机 product_id=1234567890，item_id=1234567890，蓝牙连不上",
    "无线耳机 1234567890 蓝牙连不上",
    "无线耳机 product_id=1234567890（6452271382）一件，申请退货",
    "申请退货，无线耳机 product_id（1234567890）一件",
])
def test_foreign_product_is_still_rejected(text):
    assert analyze_message(text).scope_state == "out_of_scope"


def test_both_supported_product_and_item_are_allowed():
    assert analyze_message("无线耳机 product_id=9924732112，item_id=6452271382，蓝牙连不上").scope_state == "active"


def test_item_id_alone_cannot_establish_supported_product():
    assert analyze_message("item_id=6452271382，蓝牙连不上").scope_state == "ambiguous"


def waiting_record():
    record = _active_return_record("confirmation")
    record["return_flow"]["status"] = "awaiting_confirmation"
    record["unresolved"] = ["退货详情已展示，等待用户明确确认", "蓝牙配对问题仍未解决"]
    record["taken_steps"] = [{"step_id": "PAIR-03", "turn": 1, "source": "user_report"}]
    return record


@pytest.mark.parametrize("language", ["zh", "en"])
def test_human_summary_uses_current_exit_without_fabricating_success(language):
    record = waiting_record()
    before = copy.deepcopy(record)
    text = "这笔退货先不要提交，请转人工"
    analysis = analyze_message(text, record, None, True)
    packet = build_support_packet(record, analysis, text, 2, None, [])
    summary = handoff_summary_from_packet(packet, [], language)
    assert packet["return_flow"]["status"] == "handed_off"
    assert packet["return_flow"]["awaiting"] == ""
    assert "handoff_requested" in summary
    assert "awaiting_confirmation" not in summary and "等待用户明确确认" not in summary
    assert "PAIR-03" in summary
    assert "handoff_recorded" not in summary
    assert record == before
    req = Request(text, "user", "conv", turn=2, retail_support=packet)
    store = SimpleNamespace(active_pending=lambda *args: [], list_events=lambda *args: [], bound_user_id=lambda *args: None)
    handoff = build_role_handoff(store, req, {}, "escalation")
    assert handoff["task"]["phase"] == "handoff_requested"
    assert handoff["pending_refs"] == []
    assert "退货详情已展示，等待用户明确确认" not in handoff["support_refs"]["unresolved"]
    assert "蓝牙配对问题仍未解决" in handoff["support_refs"]["unresolved"]


def test_explicit_empty_pending_does_not_reuse_packet_pending():
    record = waiting_record()
    text = "转人工"
    packet = build_support_packet(record, analyze_message(text, record), text, 2, None, [])
    packet["pending_confirmation"] = [{"op_id": "old", "status": "displayed"}]
    summary = handoff_summary_from_packet(packet, [], "en")
    assert "pending not submitted: 0" in summary
    assert "handoff_recorded" not in summary


def test_real_waiting_still_remains_confirmation():
    record = waiting_record()
    pending = [{"op_id": "current", "order_id": "#W1067251", "status": "displayed", "prepared_turn": 1}]
    text = "我只问金额，还没确认"
    packet = build_support_packet(record, analyze_message(text, record, None, True), text, 2, None, pending)
    req = Request(text, "user", "conv", turn=2, retail_support=packet)
    store = SimpleNamespace(active_pending=lambda *args: pending, list_events=lambda *args: [], bound_user_id=lambda *args: None)
    assert build_role_handoff(store, req, {}, "billing")["task"]["phase"] == "confirmation"
    assert packet["return_flow"]["status"] == "awaiting_confirmation"


@pytest.mark.parametrize("message,status", [("取消退货", "cancelled"), ("我不退了，继续排查", "switched")])
def test_exit_view_keeps_history_but_clears_current_wait(message, status):
    record = waiting_record()
    packet = build_support_packet(record, analyze_message(message, record, None, True), message, 2, None, [])
    assert packet["return_flow"]["status"] == status
    assert packet["current_state"]["phase"] == status
    assert packet["current_state"]["turn"] == 2
    assert packet["work_record"]["return_flow"]["status"] == "awaiting_confirmation"
    assert packet["work_record"]["last_turn"] == 1
    assert "等待用户明确确认" not in str(packet["current_state"]["unresolved"])


def test_pending_not_cleared_cannot_claim_exit_is_complete():
    record = waiting_record()
    pending = [{"status": "displayed", "op_id": "current"}]
    packet = build_support_packet(record, analyze_message("转人工", record, None, True), "转人工", 2, None, pending)
    assert packet["return_flow"]["status"] == "awaiting_confirmation"
    assert packet["current_state"]["phase"] == "confirmation"


def test_previous_handoff_does_not_become_a_new_human_request():
    record = waiting_record()
    record["return_flow"].update(status="handed_off", awaiting="")
    record["stage"] = "handoff_recorded"
    record["handoff"] = {"status": "recorded_mock", "turn": 1}
    packet = build_support_packet(record, analyze_message("无线耳机还是连不上", record), "无线耳机还是连不上", 2, None, [])
    assert packet["current_state"]["phase"] == "executing"
    assert packet["return_flow"]["action"] != "human"


@pytest.mark.parametrize("engine", ["legacy", "langgraph"])
def test_failed_handoff_never_claims_recorded_and_keeps_return_cancelled(tmp_path, engine):
    from evaluation.demo_retail_roles import build
    async def scenario():
        store, _, service = build(tmp_path, "human", engine)
        store.find_user_by_email("demo-client", "demo-conversation", "raj.sanchez2046@example.com")
        turn = store.begin_turn("demo-client", "demo-conversation")
        prepared = store.prepare_return("demo-client", "demo-conversation", "#W1067251", ["6452271382"], "credit_card_3362387")
        store.mark_pending_displayed("demo-client", "demo-conversation", prepared["op_id"], turn, False)
        store.save_support_state("demo-client", "demo-conversation", waiting_record())
        service.support_enabled = True
        orchestrator = service.agent.orchestrator
        tools = dict(orchestrator.shared_tools)
        calls = []
        def failed(req, args):
            calls.append(args)
            return {"success": False, "error": "固定交接失败"}
        tools["transfer_to_human_agents"] = replace(tools["transfer_to_human_agents"], handler=failed)
        orchestrator.set_shared_tools(tools)
        try:
            result = await service.handle_turn("转人工", "demo-client", "demo-conversation")
            assert not result.escalated and not result.success
            assert len(calls) == 1
            assert "handoff_requested" in calls[0]["summary"]
            assert "handoff_recorded" not in calls[0]["summary"]
            assert "awaiting_confirmation" not in calls[0]["summary"]
            assert not store.active_pending("demo-client", "demo-conversation")
            assert not any(e["event_type"] in {"return_requested", "handoff_requested"} for e in store.list_events("demo-client", "demo-conversation"))
            record = store.load_support_state("demo-client", "demo-conversation")
            assert record["stage"] == "handoff_requested"
            if service.workflow:
                assert result.workflow["role_handoff"]["task"]["phase"] == "handoff_requested"
                assert result.workflow["next"] == []
        finally:
            if service.workflow:
                await service.workflow.close()
    asyncio.run(scenario())
