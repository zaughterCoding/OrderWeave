"""U09 有限事实约束与误拒答回归，全部离线。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentResponse, AgentType
from agents.retail_agent import RetailService
from agents.retail_response_guard import constrain_response
from mcp.retail_store import RetailStore

SEED = Path(__file__).resolve().parents[1] / "evaluation/data/retail_sandbox/db.json"
EMAIL = "chen.silva2698@example.com"
ORDER = "#W3069600"
ITEM = "5012998807"
PAYMENT = "credit_card_1565124"


def guard(text, **kwargs):
    args = dict(traces=[], events=[], pending_before=[], pending_after=[],
                turn=1, bound_user_id=None)
    args.update(kwargs)
    return constrain_response(text, **args)


def submitted(order="#W100", op="op-a", amount=10001, item="item-a", payment="card-a"):
    return {"event_id": 1, "event_type": "return_requested", "order_id": order,
            "op_id": op, "payload": {"amount_cents": amount, "item_ids": [item],
                                       "payment_method_id": payment}}


def trace(name, result, **kwargs):
    return {"tool_name": name, "result": result, "success": True,
            "result_success": result.get("success"), **kwargs}


@pytest.mark.parametrize("case", json.loads(
    (Path(__file__).parent / "fixtures/u09_guard_saved.json").read_text(encoding="utf-8")
)["cases"], ids=lambda case: case["query_id"])
def test_saved_policy_candidates_keep_their_complete_meaning(case):
    result = guard(case["candidate"], traces=case["traces"], events=case["events"])
    assert result.text == case["candidate"]
    assert not result.record["findings"]


@pytest.mark.parametrize("text", [
    "这并不代表：\n\n- 退款已到账。\n- 仓库已收到退货。",
    "如果提交成功：\n- 退货申请已提交；\n- 订单状态：return requested。",
    "请先核对订单状态为已送达，再整理商品行。",
    "您反馈退款已到账；这只是您的描述，实际状态尚需工具核实。",
    "订单状态是已送达吗？",
    "订单状态为 delivered 时，才可准备退货。",
])
def test_scoped_explanations_are_not_order_facts(text):
    assert guard(text).text == text


@pytest.mark.parametrize("text", [
    "仓库已收到退货。",
    "请核对订单号。订单状态：delivered。",
    "如果之后提交成功，会显示结果。但退货申请已提交。",
    "您反馈退款已到账，但仓库已收到退货。",
    "退款已到账，请问还需要帮助吗？",
    "不代表退款已到账。仓库已收到退货。",
    "我已核对订单状态为delivered。",
])
def test_explanation_cannot_authorize_a_separate_assertion(text):
    result = guard(text)
    assert result.text != text
    assert result.record["findings"]


@pytest.mark.parametrize("text,issue", [
    ("礼品卡余额不足，因此不能退款。", "gift_card_balance_refusal"),
    ("建议您申请换货。", "unsupported_exchange_advice"),
    ("退款已到账。", "unverified_external_status"),
    ("退款尚未到账，仓库还没有收到退货。", "unverified_external_status"),
    ("退货申请已提交。", "unsupported_submission_claim"),
    ("已连接真人坐席。", "unsupported_handoff_claim"),
    ("已核实您的身份。", "unsupported_identity_claim"),
    ("预计申请金额：¥123.45。", "unverified_labeled_order_fact"),
    ("订单状态：delivered。", "unverified_labeled_order_fact"),
    ("The refund has been issued.", "unverified_external_status"),
])
def test_known_unsupported_assertions_are_corrected(text, issue):
    result = guard(text)
    assert result.text != text
    assert issue in result.record["findings"]
    assert result.record["model_calls"] == 0


@pytest.mark.parametrize("text", [
    "可以换货吗？", "本沙箱不支持换货。", "退款是否已到账？",
    "退款到账状态尚未核实。", "不代表退款已到账、仓库已收货。",
    "本人已有礼品卡可用于收款。请问选择哪种退款去向？",
    "请提供本人邮箱，以便核对身份。", "请问订单 #W123 的状态是什么？",
    "请把耳机靠近手机，按 PAIR-02 重新扫描。",
    "This does not mean the refund has been issued.",
])
def test_questions_negations_and_valid_advice_are_preserved(text):
    result = guard(text)
    assert result.text == text
    assert result.record["findings"] == []


@pytest.mark.parametrize("text,issue", [
    ("后续如需退货、换货或查询订单，请走原有交易流程；届时需要先核对身份（请提供本人邮箱，或本人姓名加邮编）。", "unsupported_exchange_advice"),
    ("2. 你账户中已有的礼品卡 gift_card_2259499（余额 30.00 美元，低于这笔预计申请金额，无法覆盖）。", "gift_card_balance_refusal"),
    ("退款尚未到账，仓库也尚未收货，退货邮件是否发送我无法确认；退款到账时间请以实际处理为准。", "unverified_external_status"),
])
def test_t19_original_warning_excerpts(text, issue):
    """来自 t19-final-20261020-v2 对应 S01B／S02A／S03B 的实际公开沙箱回复。"""
    result = guard(text)
    assert issue in result.record["findings"]
    assert result.text != text


def test_submitted_text_uses_each_event_without_cross_order_facts():
    result = guard(
        "订单#W999退款999元已到账，之后可以修改退款去向。",
        events=[submitted(), submitted("#W200", "op-b", 22222, "item-b", "card-b")],
    )
    first, second = result.text.split("\n\n")
    assert "#W100" in first and "¥100.01" in first and "card-a" in first
    assert "item-a × 1" in first and "item-b" not in first
    assert "#W200" in second and "¥222.22" in second and "card-b" in second
    assert "card-a" not in second and "999" not in result.text
    assert "尚未核实" in result.text and "不支持提交后更改" in result.text
    assert len(result.record["facts"]) == 2


def test_handoff_is_only_simulated_even_if_model_claims_real_agent():
    result = guard("已连接真人。", events=[{
        "event_id": 9, "event_type": "handoff_requested", "op_id": "HO-123", "payload": {},
    }])
    assert "模拟人工交接（HO-123）" in result.text
    assert "未连接真实坐席" in result.text


@pytest.mark.parametrize("extra,kind", [
    ({"skipped": True}, "failed"),
    ({"blocked_by_guard": True, "success": False}, "failed"),
    ({"success": True}, "failed"),
    ({"success": False}, "unknown"),
])
def test_failures_and_unknown_results_never_become_success(extra, kind):
    result = guard("退货申请已提交。", traces=[trace(
        "return_delivered_order_items", {"success": False, "error": "timeout"}, **extra,
    )])
    assert result.record["facts"][0]["kind"] == kind
    assert "已提交" not in result.text
    if kind == "unknown":
        assert "不能判定成功或未写入" in result.text


def test_success_flag_without_event_is_unknown_and_event_wins_lost_response():
    claimed = trace("return_delivered_order_items", {
        "success": True, "op_id": "op-a", "order_id": "#W100",
    })
    result = guard("已完成。", traces=[claimed])
    assert result.record["facts"][0]["kind"] == "unknown"
    lost = {**claimed, "success": False, "result_success": None,
            "input": {"order_id": "#W100"},
            "result": {"success": False, "error": "response lost"}}
    result = guard("操作失败。", traces=[lost], events=[submitted()])
    assert "退货申请已提交" in result.text
    assert "event_overrides_tool_error" in result.record["findings"]


def test_order_status_requires_successful_same_user_read():
    order = {"order_id": "#W100", "user_id": "u1", "status": "delivered"}
    good = trace("get_order_details", {"success": True, "order": order})
    result = guard("订单号：#W100\n订单状态：pending", traces=[good], bound_user_id="u1")
    assert "订单状态：delivered" in result.text
    assert "pending" not in result.text
    assert "inconsistent_order_status" in result.record["findings"]
    for bad in [{**good, "skipped": True}, {**good, "success": False}, good]:
        result = guard("订单状态：delivered", traces=[bad], bound_user_id="other")
        assert "订单状态：delivered" not in result.text


def test_tool_event_conflict_and_other_operation_are_not_merged():
    bad = trace("return_delivered_order_items", {
        "success": True, "order_id": "#W100", "op_id": "op-a", "status": "refunded",
        "return_items": ["wrong"], "return_payment_method_id": "wrong",
    }, input={"order_id": "W100"})
    result = guard("成功。", traces=[bad], events=[submitted()])
    assert "tool_event_conflict" in result.record["findings"]
    assert "wrong" not in result.text and "¥100.01" in result.text
    bad["result"]["op_id"] = "other-op"
    result = guard("成功。", traces=[bad], events=[submitted()])
    assert {f["kind"] for f in result.record["facts"]} == {"submitted", "unknown"}


def test_unidentified_failure_cannot_borrow_another_operations_event():
    result = guard("已完成。", traces=[trace(
        "return_delivered_order_items", {"success": False}, success=False,
    )], events=[submitted()])
    assert {f["kind"] for f in result.record["facts"]} == {"submitted", "unknown"}


def test_skipped_call_is_not_supported_by_another_calls_success_event():
    result = guard("成功。", traces=[trace("return_delivered_order_items", {
        "success": False,
    }, skipped=True, input={"order_id": "#W100"})], events=[submitted()])
    assert {f["kind"] for f in result.record["facts"]} == {"submitted", "failed"}


def test_two_order_statuses_are_rendered_by_object_not_by_value_membership():
    traces = [trace("get_order_details", {"success": True, "order": {
        "order_id": key, "user_id": "u1", "status": status,
    }}) for key, status in [("#W100", "delivered"), ("#W200", "pending")]]
    result = guard("订单号：#W100，订单状态：pending。订单号：#W200，订单状态：delivered。",
                   traces=traces, bound_user_id="u1")
    assert "订单号：#W100，订单状态：delivered" in result.text
    assert "订单号：#W200，订单状态：pending" in result.text


def test_english_results_keep_usd_and_do_not_invent_external_status():
    result = guard("The refund has been issued.", language="en", events=[submitted(amount=101)])
    assert "Estimated refund amount: $1.01" in result.text
    assert "have not been verified" in result.text
    assert "¥" not in result.text


def test_displayed_details_and_matching_amount_are_not_mistaken_for_hallucination():
    pending = {"op_id": "op", "order_id": "#W100", "amount_cents": 12345,
               "status": "displayed", "display_text": "服务端原始待确认详情。"}
    text = "预计申请金额：¥123.45。"
    assert guard(text, pending_after=[pending]).text == text
    result = guard(pending["display_text"], pending_after=[pending])
    assert result.text.strip() == pending["display_text"]
    assert "123.456" not in guard("预计申请金额：¥123.456。", pending_after=[pending]).text


def test_valid_localized_order_status_keeps_other_answer_content():
    text = "订单状态：delivered（已送达）。请问您需要查询哪件商品？"
    result = guard(text, bound_user_id="u1", traces=[trace("get_order_details", {
        "success": True, "order": {"order_id": "#W100", "user_id": "u1", "status": "delivered"},
    })])
    assert result.text == text and not result.record["findings"]


@pytest.mark.parametrize("status", ["**已送达（delivered）**", "pending"])
def test_status_check_preserves_refund_destination_question(status):
    text = f"已核对订单 #W100，订单状态为{status}。请回复退款去向：原支付方式还是本人已有礼品卡？"
    result = guard(text, bound_user_id="u1", traces=[trace("get_order_details", {
        "success": True, "order": {"order_id": "#W100", "user_id": "u1", "status": "delivered"},
    })])
    assert "请回复退款去向：原支付方式还是本人已有礼品卡？" in result.text
    if status.startswith("**"):
        assert result.text == text
    else:
        assert "pending" not in result.text and "delivered" in result.text


def test_original_post_submission_warning_cannot_offer_to_change_destination():
    text = "如需将退款改到本人已有的礼品卡，或需要人工介入，请告诉我。"
    result = guard(text, events=[submitted()])
    assert "如需将退款改到" not in result.text
    assert "不支持提交后更改退款去向" in result.text


class ScriptAgent:
    agent_type = AgentType.RETAIL

    def __init__(self, handler):
        self.handler = handler

    async def handle(self, request):
        return self.handler(request)


def test_service_prepares_then_submits_and_logs_final_text_across_restart(tmp_path):
    store = RetailStore(tmp_path / "retail.db", SEED)
    store.find_user_by_email("client", "conv", EMAIL)
    messages = []

    class Memory:
        async def get_context(self, *args, **kwargs):
            return SimpleNamespace(to_prompt_text=lambda: "")

        async def add_message(self, *args):
            messages.append(args[-1])

    def prepare(req):
        result = store.prepare_return("client", "conv", ORDER, [ITEM], PAYMENT)
        assert result["success"]
        return AgentResponse(agent_type=AgentType.RETAIL, content="退款999元已到账。",
                             success=True, tool_traces=[trace("prepare_return", result)])

    log = tmp_path / "turns.jsonl"
    service = RetailService(store, ScriptAgent(prepare), memory=Memory(), event_log_path=log)
    first = asyncio.run(service.handle_turn("我要退货", "client", "conv"))
    pending = store.active_pending("client", "conv")[0]
    assert pending["status"] == "displayed" and pending["display_text"] in first.response
    assert "999" not in first.response
    assert messages[-1] == first.response
    assert store.stats()["counts"]["return_operations"] == 0

    def submit(req):
        result = store.submit_return("client", "conv", ORDER, [ITEM], PAYMENT, req.message)
        assert result["success"], result
        return AgentResponse(agent_type=AgentType.RETAIL, content="退款已到账，稍后可以修改退款去向。",
                             success=True, tool_traces=[trace("return_delivered_order_items", result)])

    service.agent = ScriptAgent(submit)
    second = asyncio.run(service.handle_turn("确认，就按这些申请退货", "client", "conv"))
    assert "退货申请已提交" in second.response and ORDER in second.response
    assert "尚未核实" in second.response and messages[-1] == second.response
    assert store.stats()["counts"]["return_operations"] == 1
    restarted = RetailService(RetailStore(tmp_path / "retail.db", SEED), ScriptAgent(
        lambda req: AgentResponse(agent_type=AgentType.RETAIL, content="退货申请已提交。", success=True)
    ), event_log_path=log)
    third = asyncio.run(restarted.handle_turn("查一下现在状态", "client", "conv"))
    assert "本轮没有可核验" in third.response
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert not rows[-1]["store_events"]
    assert rows[-1]["response_guard"]["facts"] == []
    assert rows[1]["assistant_message"] == second.response
    assert service.get_tool_trace(second.request_id)["response_guard"] == second.response_guard
    assert store.stats()["counts"]["return_operations"] == 1
