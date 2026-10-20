"""T17 有限支持资料、工作记录与适用边界的单元回归。"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentOrchestrator, AgentType, Request
from agents.retail_agent import RetailAgent
from agents.retail_multi_agent import RetailBillingAgent, RetailTechnicalAgent
from agents.retail_runtime import (
    COMBINED_KNOWLEDGE_SEGMENTS,
    EXPECTED_KNOWLEDGE_SEGMENTS,
    EXPECTED_SUPPORT_KNOWLEDGE_SHA256,
    RetailRuntimeConfig,
    _load_knowledge,
    _load_support_knowledge,
    _load_support_manifest,
    _sha256_file,
)
from agents.retail_support import (
    MAX_HANDOFF_CHARS,
    SUPPORT_CONTEXT_LIMIT,
    SUPPORT_EVIDENCE_IDS,
    SUPPORT_PRODUCT_ID,
    SUPPORT_STEP_IDS,
    analyze_message,
    build_support_packet,
    handoff_summary_from_packet,
    looks_dependent_continuation,
    merge_role_handoff,
    new_support_record,
    support_evidence_applicable,
    update_support_record,
)
from agents.retail_tools import (
    RETAIL_READ_ONLY_TOOLS,
    RETAIL_SUPPORT_TECHNICAL_TOOLS,
    retail_tools,
)
from mcp.retail_store import RetailStore, SupportStateError
from mcp.tool_manager import ToolResult

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json"
OLD_KNOWLEDGE = ROOT / "evaluation" / "data" / "retail_sandbox" / "knowledge.jsonl"
SUPPORT_DIR = ROOT / "evaluation" / "data" / "retail_support"
SUPPORT_KNOWLEDGE = SUPPORT_DIR / "knowledge.jsonl"
SUPPORT_MANIFEST = SUPPORT_DIR / "manifest.json"
SCENARIOS = SUPPORT_DIR / "scenarios.json"


def _store(tmp_path: Path) -> RetailStore:
    return RetailStore(tmp_path / "retail.db", SEED)


def _packet(record=None, message="无线耳机蓝牙连不上", turn=1):
    analysis = analyze_message(message, record, None, False)
    return analysis, build_support_packet(
        record, analysis, message, turn, None, []
    )


def test_support_assets_and_combined_manifest_are_frozen():
    assert _sha256_file(SUPPORT_KNOWLEDGE) == EXPECTED_SUPPORT_KNOWLEDGE_SHA256
    manifest = _load_support_manifest(SUPPORT_MANIFEST, SUPPORT_KNOWLEDGE)
    rows = _load_support_knowledge(SUPPORT_KNOWLEDGE)
    old_rows = _load_knowledge(OLD_KNOWLEDGE)

    assert len(old_rows) == EXPECTED_KNOWLEDGE_SEGMENTS == 29
    assert len(rows) == 6
    assert len(rows) + len(old_rows) == COMBINED_KNOWLEDGE_SEGMENTS == 35
    assert tuple(row["evidence_id"] for row in rows) == SUPPORT_EVIDENCE_IDS
    assert tuple(manifest["support_knowledge"]["step_ids"]) == SUPPORT_STEP_IDS
    assert manifest["combined"]["count"] == 35
    assert manifest["legacy_knowledge"]["count"] == 29
    assert len(manifest["scenarios"]["case_ids"]) == 8

    step_rows = [row for row in rows if row.get("step_id")]
    assert {row["step_id"] for row in step_rows} == set(SUPPORT_STEP_IDS)
    for row in rows:
        assert row["product_id"] == SUPPORT_PRODUCT_ID
        assert row["issue"] == "bluetooth_pairing"
        assert row["source_kind"] == "project_authored_sandbox"
        assert row["language"] == "zh"
        assert row["evidence_id"].startswith("T17.E")

    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    assert [case["id"] for case in scenarios["cases"]] == [
        f"C{index:02d}" for index in range(1, 9)
    ]


def test_scope_demand_and_conditional_return_are_not_submit_authorization():
    analysis = analyze_message(
        "无线耳机蓝牙搜不到，怎么配对？", None, None, False
    )
    assert analysis.scope_state == "active"
    assert analysis.demand_kind == "troubleshooting"

    assert analyze_message("耳机连不上", None, None, False).scope_state == "ambiguous"
    assert (
        analyze_message("蓝牙音箱连不上", None, None, False).scope_state
        == "out_of_scope"
    )
    assert (
        analyze_message("无线耳机没声音", None, None, False).scope_state
        == "out_of_scope"
    )
    assert (
        analyze_message(
            "无线耳机连不上，两副耳机我不知道退哪副", None, None, False
        ).scope_state
        == "ambiguous"
    )

    record = new_support_record(1)
    record["taken_steps"] = [{"step_id": "PAIR-03", "turn": 1}]
    conditional = analyze_message(
        "如果还不行就退了吧", record, None, False
    )
    assert conditional.scope_state == "active"
    assert conditional.demand_kind == "conditional_return"

    direct = analyze_message("还是不行，那就退了吧", record, None, False)
    assert direct.demand_kind == "return"

    negative = analyze_message("我不想退，先继续排查", record, None, False)
    assert negative.return_negative is True
    assert negative.demand_kind == "none"

    human = analyze_message("还是不行，转人工", record, None, False)
    assert human.demand_kind == "human"


def test_restart_continuation_and_dependent_message_detection():
    record = new_support_record(1)
    record["stage"] = "awaiting_feedback"
    analysis = analyze_message("还是不行", record, "raj_sanchez_2970", False)
    assert analysis.scope_state == "active"
    assert analysis.feedback_kind == "unresolved"
    assert "蓝牙配对问题仍未解决" in update_support_record(
        record, analysis, "还是不行", 2, "请继续尝试。", [], []
    )["unresolved"]

    assert looks_dependent_continuation("还是不行", False)
    assert looks_dependent_continuation("确认", True)
    assert not looks_dependent_continuation(
        "你好，我想查询配送政策", False
    )


def test_existing_record_only_continues_with_dependent_messages():
    record = new_support_record(1)
    record["stage"] = "return_requested"

    identity = analyze_message("我的邮箱是 raj@example.com", record, None, False)
    assert identity.scope_state == "active"
    assert identity.demand_kind == "none"

    order = analyze_message("我的订单 #W1067251 什么时候发货", record, None, False)
    assert order.scope_state == "active"

    unrelated = analyze_message("我的订单什么时候发货", record, None, False)
    assert unrelated.scope_state == "inactive"
    assert analyze_message("你好", record, None, False).scope_state == "inactive"

    resolved = new_support_record(1)
    resolved["stage"] = "resolved"
    assert (
        analyze_message("还是不行", resolved, None, False).scope_state == "active"
    )


def test_record_separates_retrieved_from_displayed_and_user_reports():
    record = new_support_record(1)
    record["taken_steps"] = [
        {"step_id": "PAIR-03", "source": "user_report", "quote": "已重新配对", "turn": 1}
    ]
    analysis = analyze_message(
        "无线耳机蓝牙连不上，我已经重新配对过了", record, None, False
    )
    traces = [
        {
            "tool_name": "search_knowledge_base",
            "result_success": True,
            "result": {
                "success": True,
                "degraded": False,
                "results": [
                    {
                        "evidence_id": "T17.E02",
                        "product_id": SUPPORT_PRODUCT_ID,
                        "issue": "bluetooth_pairing",
                        "step_id": "PAIR-01",
                        "title": "PAIR-01 供电与蓝牙开关",
                        "content": "确认耳机有电并处于开机状态。",
                    },
                    {
                        "evidence_id": "T17.E03",
                        "product_id": SUPPORT_PRODUCT_ID,
                        "issue": "bluetooth_pairing",
                        "step_id": "PAIR-02",
                        "title": "PAIR-02 其他设备占用",
                        "content": "先在另一台设备断开连接。",
                    },
                ],
            },
        }
    ]
    updated = update_support_record(
        record,
        analysis,
        "无线耳机蓝牙连不上，我已经重新配对过了",
        2,
        "请先确认 PAIR-01，并检查其他设备；PAIR-03 已做过。",
        traces,
        pending_after=[],
    )
    assert len(updated["suggestions"]) == 1
    assert updated["suggestions"][0]["step_id"] == "PAIR-01"
    assert updated["suggestions"][0]["previously_tried"] is False
    assert {item["step_id"] for item in updated["retrieved_evidence"]} == {
        "PAIR-01",
        "PAIR-02",
    }
    assert {item["step_id"] for item in updated["taken_steps"]} == {"PAIR-03"}
    assert updated["stage"] == "awaiting_feedback"

    # 仅检索、未在可见回复中出现，不进入建议。
    hidden = update_support_record(
        record, analysis, "无线耳机蓝牙连不上", 2, "请稍等。", traces, []
    )
    assert hidden["suggestions"] == []


def test_record_limits_emit_flag_without_deleting_current_facts():
    record = new_support_record(1)
    record["visible_history"] = [
        {"turn": index, "user": f"u{index}", "assistant": f"a{index}"}
        for index in range(12)
    ]
    record["feedback"] = [
        {"turn": index, "kind": "unresolved", "text": f"f{index}"}
        for index in range(8)
    ]
    record["suggestions"] = [
        {
            "turn": index,
            "step_id": "PAIR-01",
            "evidence_id": f"T17.E0{index % 6 + 1}",
        }
        for index in range(8)
    ]
    analysis = analyze_message("还是不行", record, None, False)
    updated = update_support_record(
        record, analysis, "还是不行", 13, "请继续检查。", [], []
    )
    assert len(updated["visible_history"]) == 12
    assert len(updated["feedback"]) == 8
    assert SUPPORT_CONTEXT_LIMIT in updated["limit_flags"]
    assert updated["limit_detail"]["visible_pairs_dropped"] == 1
    assert updated["demand"]["text"] == "还是不行"
    assert "蓝牙配对问题仍未解决" in updated["unresolved"]


def test_support_packet_is_bounded_and_preserves_work_facts():
    record = new_support_record(1)
    record["taken_steps"] = [
        {"step_id": "PAIR-03", "source": "user_report", "quote": "已重新配对", "turn": 1}
    ]
    record["feedback"] = [
        {"turn": 1, "kind": "unresolved", "text": "还是连不上"}
    ]
    record["unresolved"] = ["蓝牙配对问题仍未解决"]
    record["visible_history"] = [
        {"turn": index, "user": "长" * 500, "assistant": "长" * 800}
        for index in range(12)
    ]
    analysis = analyze_message("我不想退，继续排查", record, None, False)
    packet = build_support_packet(
        record, analysis, "我不想退，继续排查", 2, "raj_sanchez_2970", []
    )
    encoded = json.dumps(packet, ensure_ascii=False)
    assert len(encoded) <= MAX_HANDOFF_CHARS
    assert packet["scope_state"] == "active"
    assert packet["product_id"] == SUPPORT_PRODUCT_ID
    assert packet["work_record"]["taken_steps"][0]["step_id"] == "PAIR-03"
    assert packet["work_record"]["unresolved"]
    assert packet["identity"]["bound"] is True


def test_handoff_summary_and_role_handoff_keep_actual_source():
    record = new_support_record(1)
    record["stage"] = "awaiting_feedback"
    record["taken_steps"] = [
        {"step_id": "PAIR-03", "source": "user_report", "quote": "已重新配对", "turn": 1}
    ]
    record["feedback"] = [
        {"turn": 2, "kind": "unresolved", "text": "还是不行"}
    ]
    record["suggestions"] = [
        {
            "turn": 2,
            "step_id": "PAIR-01",
            "evidence_id": "T17.E02",
            "text": "检查供电",
        }
    ]
    analysis = analyze_message("还是不行，转人工", record, None, False)
    packet = build_support_packet(
        record, analysis, "还是不行，转人工", 3, "raj_sanchez_2970", []
    )
    summary = handoff_summary_from_packet(packet, [], "zh")
    assert "PAIR-01" in summary
    assert "还是不行" in summary
    assert "未创建真实工单" in summary
    assert len(summary) <= MAX_HANDOFF_CHARS

    response = type(
        "Response",
        (),
        {
            "content": "建议先检查 PAIR-01。",
            "tool_traces": [
                {
                    "tool_name": "search_knowledge_base",
                    "result_success": True,
                    "result": {
                        "success": True,
                        "degraded": False,
                        "results": [
                            {
                                "evidence_id": "T17.E02",
                                "product_id": SUPPORT_PRODUCT_ID,
                                "issue": "bluetooth_pairing",
                                "step_id": "PAIR-01",
                            }
                        ],
                    },
                }
            ],
        },
    )()
    merged = merge_role_handoff(packet, "technical", "billing", 3, response)
    assert merged["last_handoff"]["source_role"] == "technical"
    assert merged["last_handoff"]["evidence_ids"] == ["T17.E02"]
    assert merged["last_handoff"]["content_hash"]


def test_support_evidence_applicability_filters_wrong_product_and_degraded():
    _, packet = _packet()
    old_row = {"evidence_id": "T12.E01", "content": "固定政策"}
    good = {
        "evidence_id": "T17.E02",
        "product_id": SUPPORT_PRODUCT_ID,
        "issue": "bluetooth_pairing",
    }
    wrong = dict(good, product_id="5565631513")
    degraded = dict(good, degraded=True)
    assert support_evidence_applicable(old_row, packet) is True
    assert support_evidence_applicable(good, packet) is True
    assert support_evidence_applicable(wrong, packet) is False
    assert support_evidence_applicable(degraded, packet) is False
    assert support_evidence_applicable(good, None) is False


def test_support_state_table_is_incremental_isolated_and_restart_safe(tmp_path):
    store = _store(tmp_path)
    original = new_support_record(1)
    original["demand"] = {"kind": "troubleshooting", "text": "连不上", "turn": 1}
    store.save_support_state("u1", "c1", original)
    store.save_support_state("u2", "c1", new_support_record(1))

    assert store.load_support_state("u1", "c1")["demand"]["text"] == "连不上"
    assert store.load_support_state("u2", "c1")["demand"] == {}
    assert store.load_support_state("u1", "c2") is None
    with pytest.raises(SupportStateError):
        store.save_support_state("u1", "c1", {"format_version": 99})
    assert store.load_support_state("u1", "c1")["demand"]["text"] == "连不上"

    # 旧库没有新表时，初始化只增量建表，不改 schema_version 或种子。
    with sqlite3.connect(store.db_path) as connection:
        connection.execute("DROP TABLE retail_support_state")
    reopened = RetailStore(tmp_path / "retail.db", SEED)
    assert reopened.load_support_state("u1", "c1") is None
    assert reopened.stats()["schema_version"] == "1"
    assert reopened.stats()["counts"]["users"] == 500
    reopened.save_support_state("u1", "c1", original)
    assert reopened.load_support_state("u1", "c1")["demand"]["text"] == "连不上"


def test_support_state_save_failure_preserves_existing_record(tmp_path, monkeypatch):
    store = _store(tmp_path)
    original = new_support_record(1)
    store.save_support_state("u1", "c1", original)

    @contextlib.contextmanager
    def broken_tx():
        class Broken:
            def execute(self, *args, **kwargs):
                raise sqlite3.OperationalError("disk full")

        yield Broken()

    monkeypatch.setattr(store, "_tx", broken_tx)
    with pytest.raises(SupportStateError) as excinfo:
        store.save_support_state("u1", "c1", new_support_record(2))
    assert "写入支持记录失败" in str(excinfo.value)
    assert store.load_support_state("u1", "c1")["format_version"] == 1


def test_support_technical_role_trims_state_tools_but_keeps_retrieval(tmp_path):
    store = _store(tmp_path)
    tools = retail_tools(store, object(), language="zh", support_enabled=True)
    enabled = RetailTechnicalAgent(
        client=None,
        model="fake-model",
        language="zh",
        policy_text="",
        support_enabled=True,
    )
    disabled = RetailTechnicalAgent(
        client=None,
        model="fake-model",
        language="zh",
        policy_text="",
    )
    for agent in (enabled, disabled):
        agent.set_shared_tools(tools)

    req = Request(
        message="无线耳机蓝牙连不上",
        user_id="u",
        conv_id="c",
        turn=1,
        retail_primary="technical",
    )
    assert set(enabled._tools_for_request(req)) == set(
        RETAIL_SUPPORT_TECHNICAL_TOOLS
    )
    assert set(disabled._tools_for_request(req)) == set(RETAIL_READ_ONLY_TOOLS)
    assert "search_knowledge_base" in enabled._tools_for_request(req)

    supporting_req = Request(
        message="无线耳机蓝牙连不上",
        user_id="u",
        conv_id="c",
        turn=1,
        retail_primary="billing",
    )
    assert set(enabled._tools_for_request(supporting_req)) == {
        "get_product_details",
        "calculate",
        "search_knowledge_base",
    }
    assert "PAIR-01" in enabled.system_prompt
    assert "PAIR-01" not in disabled.system_prompt


def test_role_packet_and_single_agent_actually_carry_support_record(tmp_path):
    store = _store(tmp_path)
    record = new_support_record(1)
    record["taken_steps"] = [{"step_id": "PAIR-03", "turn": 1}]
    analysis = analyze_message(
        "还是不行，转人工", record, "raj_sanchez_2970", False
    )
    packet = build_support_packet(
        record, analysis, "还是不行，转人工", 2, "raj_sanchez_2970", []
    )
    req = Request(
        message="还是不行，转人工",
        user_id="u",
        conv_id="c",
        turn=2,
        retail_support=packet,
        retail_support_enabled=True,
    )
    agent = RetailTechnicalAgent(
        client=None,
        model="fake-model",
        language="zh",
        support_enabled=True,
    )
    role_packet = json.loads(agent._build_role_packet(req))
    assert role_packet["support"]["scope_state"] == "active"
    assert role_packet["support"]["work_record"]["taken_steps"][0]["step_id"] == "PAIR-03"

    class _CaptureMessages:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    class _CaptureClient:
        def __init__(self):
            self.messages = _CaptureMessages()

    # 用真实目标角色构造 messages，确认前序支持记录被实际消费而非只存在于状态里。
    capture = _CaptureClient()
    billing = RetailBillingAgent(
        client=capture,
        model="fake-model",
        language="zh",
        support_enabled=True,
    )
    billed_req = Request(
        message="还是不行，转人工",
        user_id="u",
        conv_id="c",
        turn=2,
        retail_support=packet,
        retail_support_enabled=True,
        retail_primary="billing",
    )
    response = asyncio.run(billing.handle(billed_req))
    assert response.success is True
    messages = capture.messages.calls[0]["messages"]
    role_input = next(
        item["content"]
        for item in messages
        if "[角色输入契约]" in str(item.get("content"))
    )
    assert '"step_id": "PAIR-03"' in role_input
    assert "还是不行，转人工" in role_input
    assert "当前诉求" in role_input or "current_demand" in role_input

    single = RetailAgent(
        client=None,
        model="fake-model",
        store=store,
        tool_manager=object(),
        language="zh",
        policy_text="固定政策",
    )
    assert single.support_enabled is False
    assert "PAIR-01" not in single.system_prompt
    single_packet = json.loads(single._build_role_packet(req))
    assert single_packet["support"]["current_demand"]["kind"] == "human"


def _active_return_record(awaiting: str = "refund_destination") -> dict:
    record = new_support_record(1)
    record["stage"] = "return_requested"
    record["demand"] = {
        "kind": "return",
        "text": "我要退订单 #W1067251 里的无线耳机",
        "turn": 1,
        "source": "user_message",
    }
    record["return_flow"] = {
        "status": "active",
        "awaiting": awaiting,
        "start_turn": 1,
        "last_turn": 1,
        "last_action": "start",
        "closed_reason": "",
    }
    return record


def _support_decision(packet: dict):
    orchestrator = SimpleNamespace(
        retail_support_enabled=True,
        _pool={
            AgentType.GENERAL: object(),
            AgentType.TECHNICAL: object(),
            AgentType.BILLING: object(),
            AgentType.ESCALATION: object(),
        },
    )
    req = Request(
        message=str((packet.get("current_demand") or {}).get("text") or ""),
        user_id="u",
        conv_id="c",
        turn=2,
        retail_support=packet,
        retail_support_enabled=True,
    )
    return AgentOrchestrator._support_routing(orchestrator, req)


def test_return_flow_supplement_stays_billing_and_keeps_return_stage():
    record = _active_return_record()
    message = "退回原支付方式。请先展示完整退货申请详情。"
    analysis = analyze_message(message, record, "raj_sanchez_2970", False)
    assert analysis.scope_state == "active"
    assert analysis.return_flow_action == "provide_info"
    assert analysis.demand_kind == "none"

    packet = build_support_packet(
        record, analysis, message, 2, "raj_sanchez_2970", []
    )
    assert packet["return_flow"]["status"] == "active"
    assert packet["return_flow"]["awaiting"] == "refund_destination"
    decision = _support_decision(packet)
    assert decision.primary_agent == AgentType.BILLING
    assert decision.reason == "support_return_flow_continuation"

    pending = [
        {
            "op_id": "OP-1",
            "order_id": "#W1067251",
            "items": {"item_ids": ["6452271382"]},
            "payment_method_id": "credit_card_3362387",
            "amount": "¥258.84",
            "status": "displayed",
        }
    ]
    updated = update_support_record(
        record,
        analysis,
        message,
        2,
        "【退货申请详情（待确认）】\n订单号：#W1067251",
        [],
        pending_after=pending,
    )
    assert updated["stage"] == "return_requested"
    assert updated["demand"]["kind"] == "return"
    assert updated["return_flow"]["status"] == "awaiting_confirmation"
    assert updated["return_flow"]["awaiting"] == "confirmation"


@pytest.mark.parametrize("reply", [
    "还差退款去向，请二选一：原支付方式或已有礼品卡。",
    "退款去向可选（只能二选一）：\n1. 原支付方式\n2. 已有礼品卡\n请告知选择哪一项。",
])
def test_destination_question_keeps_stage_despite_order_summary(reply):
    record = _active_return_record()
    message = "我的邮箱是 raj.sanchez2046@example.com，订单 #W1067251。"
    analysis = analyze_message(message, record, None, False)
    updated = update_support_record(
        record, analysis, message, 2,
        "已核对订单和商品。" + reply + "需要您明确确认后才会提交。", [], [],
    )
    assert updated["return_flow"]["awaiting"] == "refund_destination"


def test_confirmation_reminder_is_not_a_new_information_request():
    record = _active_return_record()
    message = "原支付方式。"
    analysis = analyze_message(message, record, None, False)
    updated = update_support_record(
        record, analysis, message, 2,
        "已核对订单和商品，随后会展示详情；需要您明确确认后才会提交。", [], [],
    )
    assert updated["return_flow"]["awaiting"] == "refund_destination"


def test_return_flow_cancel_switch_and_human_are_not_permanent_lock():
    record = _active_return_record()

    cancel = analyze_message("取消退货，先不退了", record, None, False)
    assert cancel.return_flow_action == "cancel"
    cancelled = update_support_record(
        record, cancel, "取消退货，先不退了", 2, "已取消待确认退货。", [], []
    )
    assert cancelled["return_flow"]["status"] == "cancelled"
    assert cancelled["return_flow"]["awaiting"] == ""
    assert cancelled["return_flow"]["closed_reason"] == "user_cancelled"
    assert cancelled["stage"] == "troubleshooting"

    switch = analyze_message("我不想退，先继续排查", record, None, False)
    assert switch.return_flow_action == "switch_to_troubleshooting"
    assert switch.demand_kind == "none"
    switched = update_support_record(
        record, switch, "我不想退，先继续排查", 2, "继续排查。", [], []
    )
    assert switched["return_flow"]["status"] == "switched"
    assert switched["return_flow"]["closed_reason"] == "user_switched_to_troubleshooting"

    human = analyze_message("还是不行，转人工", record, None, False)
    assert human.return_flow_action == "human"
    handed = update_support_record(
        record, human, "还是不行，转人工", 2, "已记录人工交接。", [], []
    )
    assert handed["return_flow"]["status"] == "handed_off"
    assert handed["return_flow"]["closed_reason"] == "user_requested_human"

    # 取消不是永久锁死；用户之后仍可重新发起退货。
    restarted = analyze_message("那还是退货吧", cancelled, None, False)
    assert restarted.return_flow_action == "start"
    reopened = update_support_record(
        cancelled, restarted, "那还是退货吧", 3, "请提供退货所需信息。", [], []
    )
    assert reopened["return_flow"]["status"] == "active"
    assert reopened["return_flow"]["awaiting"] in (
        "return_information",
        "refund_destination",
    )


def test_conditional_return_requires_real_conditional_connective():
    conditional = analyze_message("如果还不行就退了吧", new_support_record(1), None, False)
    assert conditional.demand_kind == "conditional_return"
    assert conditional.return_flow_action == "conditional_return"

    direct = analyze_message(
        "退回原支付方式。请先展示完整退货申请详情。", new_support_record(1), None, False
    )
    assert direct.demand_kind == "return"
    assert direct.return_flow_action == "start"

    record = _active_return_record()
    supplement = analyze_message(
        "退回原支付方式。请先展示完整退货申请详情。", record, None, False
    )
    assert supplement.demand_kind == "none"
    assert supplement.return_flow_action == "provide_info"

    human = analyze_message("转人工", record, None, False)
    packet = build_support_packet(record, human, "转人工", 2, None, [])
    assert _support_decision(packet).primary_agent == AgentType.ESCALATION


def test_cancel_and_switch_routing_leave_billing_without_business_write():
    record = _active_return_record()
    for message, expected_agent, expected_reason in (
        (
            "取消退货，先不退了",
            AgentType.GENERAL,
            "support_return_flow_cancelled",
        ),
        (
            "我不想退，先继续排查",
            AgentType.TECHNICAL,
            "support_return_flow_switched_to_troubleshooting",
        ),
    ):
        analysis = analyze_message(message, record, None, False)
        packet = build_support_packet(record, analysis, message, 2, None, [])
        decision = _support_decision(packet)
        assert decision.primary_agent == expected_agent
        assert decision.reason == expected_reason


def test_return_flow_restart_recovery_reads_sqlite_state(tmp_path):
    store = _store(tmp_path)
    record = _active_return_record()
    store.save_support_state("u1", "c1", record)

    reloaded = store.load_support_state("u1", "c1")
    assert reloaded["return_flow"]["status"] == "active"
    analysis = analyze_message(
        "退回原支付方式。请先展示完整退货申请详情。",
        reloaded,
        None,
        False,
    )
    assert analysis.return_flow_action == "provide_info"
    pending = [
        {
            "op_id": "OP-1",
            "order_id": "#W1067251",
            "items": {"item_ids": ["6452271382"]},
            "status": "displayed",
        }
    ]
    updated = update_support_record(
        reloaded, analysis, "退回原支付方式。", 2, "已生成待确认详情。", [], pending
    )
    store.save_support_state("u1", "c1", updated)

    again = store.load_support_state("u1", "c1")
    assert again["return_flow"]["status"] == "awaiting_confirmation"
    assert again["return_flow"]["awaiting"] == "confirmation"
    assert again["stage"] == "return_requested"


def test_cancel_pending_invalidates_displayed_details_without_writing_order(tmp_path):
    store = _store(tmp_path)
    assert store.find_user_by_email("u", "c", "raj.sanchez2046@example.com")["success"]
    turn = store.begin_turn("u", "c")
    prepared = store.prepare_return(
        "u", "c", "#W1067251", ["6452271382"], "credit_card_3362387"
    )
    assert prepared["success"]
    pending = store.active_pending("u", "c")[0]
    assert store.mark_pending_displayed(
        "u", "c", pending["op_id"], turn, False
    )
    cancelled = store.cancel_pending("u", "c", "user_cancelled_return")
    assert cancelled == 1
    assert store.active_pending("u", "c") == []
    store.begin_turn("u", "c")
    result = store.submit_return(
        "u",
        "c",
        "#W1067251",
        ["6452271382"],
        "credit_card_3362387",
        "确认，就按这些申请退货",
    )
    assert result["success"] is False
    assert store.stats()["counts"]["return_operations"] == 0
    events = store.list_events("u", "c")
    assert any(item["event_type"] == "return_cancelled" for item in events)
    assert not any(item["event_type"] == "return_requested" for item in events)


def test_runtime_config_fails_explicitly_for_conflicting_support_settings(monkeypatch):
    for name, value in {
        "RETAIL_MODE": "1",
        "ANTHROPIC_API_KEY": "test-key",
        "LLM_MODEL": "fake-model",
        "RETAIL_INPUT_CNY_PER_MTOK": "1",
        "RETAIL_OUTPUT_CNY_PER_MTOK": "1",
        "RETAIL_COLLECTION": "t17legacy",
        "RETAIL_LANGUAGE": "zh",
        "RETAIL_SUPPORT_ENABLED": "true",
        "RETAIL_SUPPORT_COLLECTION": "t17support",
        "RETAIL_ENABLE_KNOWLEDGE": "true",
    }.items():
        monkeypatch.setenv(name, value)

    config = RetailRuntimeConfig.from_env()
    assert config.support_enabled is True
    assert config.support_collection == "t17support"

    monkeypatch.setenv("RETAIL_LANGUAGE", "en")
    with pytest.raises(RuntimeError, match="中文"):
        RetailRuntimeConfig.from_env()

    monkeypatch.setenv("RETAIL_LANGUAGE", "zh")
    monkeypatch.setenv("RETAIL_ENABLE_KNOWLEDGE", "false")
    with pytest.raises(RuntimeError, match="RETAIL_ENABLE_KNOWLEDGE"):
        RetailRuntimeConfig.from_env()

    monkeypatch.setenv("RETAIL_ENABLE_KNOWLEDGE", "true")
    monkeypatch.setenv("RETAIL_SUPPORT_COLLECTION", "t17legacy")
    with pytest.raises(RuntimeError, match="不能与"):
        RetailRuntimeConfig.from_env()
