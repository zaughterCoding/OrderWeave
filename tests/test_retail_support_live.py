"""T18 运行器的无收费定向检查。"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import run_retail_support_live as runner


FIXTURE = {
    "user_id": "raj_sanchez_2970",
    "email": "raj.sanchez2046@example.com",
    "first_name": "Raj",
    "last_name": "Sanchez",
    "zip_code": "92147",
    "order_id": "#W1067251",
    "item_id": "6452271382",
    "product_id": "9924732112",
    "amount_cents": 25884,
    "payment_method_id": "credit_card_3362387",
}


def _case(case_id: str, *, confirmation: bool = False) -> dict:
    branches = [
        {"id": "ask_for_complete_display", "message": "请把完整详情显示出来，我先看一下。"},
        {"id": "resolved_feedback", "message": "我按刚才说的方法试了，现在连上了。"},
        {"id": "human_no_return", "message": "我按你说的试了，还是连不上。转人工吧，不要退货。"},
        {
            "id": "resolved_unresolved_then_return",
            "message": "那就退了吧，我要退订单 #W1067251 里的蓝色无线耳机这一件。",
        },
        {"id": "ask_specific_step", "message": "能具体说一个步骤吗？"},
        {"id": "identity_or_target_clarification", "message": "邮箱 raj.sanchez2046@example.com。"},
        {"id": "ask_non_repeated_step", "message": "请给一个不重复的配对步骤。"},
        {
            "id": "refund_destination_choice",
            "message": "退回原支付方式。请先展示完整退货申请详情。",
        },
    ]
    script = {"initial": "你好，我的无线耳机蓝牙连不上。", "branches": branches}
    if confirmation:
        script["confirmation"] = {
            "message": "确认，就按这些申请退货。",
        }
    return {
        "id": case_id,
        "max_user_turns": 9,
        "script": script,
        "expected": {},
    }


def _pending(
    *,
    status: str = "displayed",
    displayed_turn: int | None = 1,
    item_id: str = "6452271382",
    amount_cents: int = 25884,
) -> dict:
    return {
        "op_id": "OP-1",
        "order_id": "#W1067251",
        "items": {"item_ids": [item_id]},
        "amount_cents": amount_cents,
        "amount": "¥258.84",
        "payment_method_id": "credit_card_3362387",
        "status": status,
        "prepared_turn": 1,
        "displayed_turn": displayed_turn,
        "display_text": "【退货申请详情（待确认）】\n订单号：#W1067251",
    }


def test_scripted_user_initial_then_display_then_confirmation() -> None:
    user = runner.ScriptedUser(_case("S02", confirmation=True), FIXTURE)
    message, branch = user.next_message(0, pending=[], record=None, submitted=False)
    assert branch == "initial"
    assert message == "你好，我的无线耳机蓝牙连不上。"

    message, branch = user.next_message(
        1, pending=[_pending(status="prepared", displayed_turn=None)],
        record=None, submitted=False,
    )
    assert branch == "ask_for_complete_display"
    assert message == "请把完整详情显示出来，我先看一下。"

    pending = _pending(status="displayed", displayed_turn=1)
    user.turn_assistant[1] = "回复文字\n\n" + pending["display_text"]
    message, branch = user.next_message(
        1, pending=[pending], record=None, submitted=False
    )
    assert branch == "confirmation"
    assert message == "确认，就按这些申请退货。"


def test_scripted_user_does_not_confirm_same_turn_or_wrong_target() -> None:
    user = runner.ScriptedUser(_case("S02", confirmation=True), FIXTURE)
    user.turn_assistant[1] = _pending()["display_text"]
    message, branch = user.next_message(
        1,
        pending=[_pending(status="displayed", displayed_turn=2)],
        record=None,
        submitted=False,
    )
    assert message is None
    assert branch == "display_not_visible_in_prior_turn"

    message, branch = user.next_message(
        1,
        pending=[_pending(status="displayed", displayed_turn=1, item_id="bad-item")],
        record=None,
        submitted=False,
    )
    assert message is None
    assert branch == "wrong_pending"


def test_scripted_user_feedback_requires_new_suggestion() -> None:
    user = runner.ScriptedUser(_case("S01"), FIXTURE)
    record = {
        "suggestions": [
            {
                "turn": 1,
                "step_id": "PAIR-01",
                "evidence_id": "T17.E01",
                "previously_tried": False,
            }
        ]
    }
    message, branch = user.next_message(
        1, pending=[], record=record, submitted=False
    )
    assert branch == "resolved_feedback"
    assert "按刚才说的方法" in message

    user2 = runner.ScriptedUser(_case("S01"), FIXTURE)
    message, branch = user2.next_message(1, pending=[], record=None, submitted=False)
    assert branch == "ask_specific_step"


def test_s03_restart_is_single_and_keeps_separate_script_branch() -> None:
    case = _case("S03", confirmation=True)
    user = runner.ScriptedUser(case, FIXTURE)
    turns = [
        {
            "index": 1,
            "record_after": {
                "suggestions": [
                    {
                        "turn": 1,
                        "step_id": "PAIR-02",
                        "evidence_id": "T17.E02",
                        "previously_tried": False,
                    }
                ]
            },
        }
    ]
    assert user.should_restart(turns) is True
    user.mark_restart()
    assert user.should_restart(turns) is False
    message, branch = user.next_message(
        1, pending=[], record=None, submitted=False
    )
    assert branch == "resolved_unresolved_then_return"
    assert "那就退了吧" in message
    assert user.return_sent is True


@pytest.mark.parametrize(
    ("case_id", "feedback_branch", "stage"),
    [("S01", "resolved_feedback", "resolved"),
     ("S04", "human_no_return", "handoff_recorded")],
)
def test_script_stops_after_terminal_feedback_without_another_suggestion(
    case_id: str, feedback_branch: str, stage: str
) -> None:
    user = runner.ScriptedUser(_case(case_id), FIXTURE)
    message, branch = user.next_message(
        1, pending=[], submitted=False,
        record={"suggestions": [{"turn": 1, "step_id": "PAIR-02"}]},
    )
    assert message and branch == feedback_branch
    # 完成回复没有本轮新建议，仍应结束，不能重新发起排查。
    user.observe({"index": 2, "assistant_message": "已完成。", "record_after": {"stage": stage}})
    message, branch = user.next_message(
        2, pending=[], record={"stage": stage}, submitted=False,
    )
    assert message is None
    assert branch == ("feedback_sent" if case_id == "S01" else "human_sent")


@pytest.mark.parametrize("case_id", ["S02", "S03"])
def test_script_answers_refund_destination_then_waits_for_display(case_id: str) -> None:
    user = runner.ScriptedUser(_case(case_id, confirmation=True), FIXTURE)
    user.return_sent = case_id == "S03"
    user.observe({
        "index": 2,
        "assistant_message": (
            "退款去向只能是原支付方式或已有礼品卡。"
            "请告诉我选择哪个去向，我再生成待确认详情。"
        ),
    })
    message, branch = user.next_message(2, pending=[], record=None, submitted=False)
    assert branch == "refund_destination_choice"
    assert message == user._branch(branch)
    assert not user.confirmation_sent
    user.observe({"index": 3, "assistant_message": user.turn_assistant[2]})
    message, branch = user.next_message(3, pending=[], record=None, submitted=False)
    assert message is None
    assert branch == "refund_destination_repeated"


@pytest.mark.parametrize("case_id", ["S02", "S03"])
def test_refund_choice_does_not_replace_identity_question(case_id: str) -> None:
    user = runner.ScriptedUser(_case(case_id, confirmation=True), FIXTURE)
    user.return_sent = case_id == "S03"
    user.observe({"index": 1, "assistant_message": "原支付方式稍后再选择。请提供本人邮箱。"})
    message, branch = user.next_message(1, pending=[], record=None, submitted=False)
    assert branch == "identity_or_target_clarification"
    assert message == user._branch(branch)


def test_s03_without_suggestion_stays_in_support_before_return_choice() -> None:
    user = runner.ScriptedUser(_case("S03", confirmation=True), FIXTURE)
    message, branch = user.next_message(1, pending=[], record=None, submitted=False)
    assert branch == "ask_non_repeated_step"
    assert message and not user.return_sent


@pytest.mark.parametrize(("text", "expected"), [
    ("您希望退回原支付方式，还是退到已有礼品卡？", True),
    ("请确认退款方式：原支付方式还是礼品卡？", True),
    ("请告知选择哪一个去向，我再生成待确认详情。", True),
    ("原支付方式已经确认。请核对商品数量。", False),
    ("退款将退回原支付方式，无需再次选择。", False),
    ("退款去向之后再确认。请提供本人邮箱。", False),
    ("已选择原支付方式，我现在生成待确认详情。", False),
    ("请确认已展示的退货详情，退款方式为原支付方式。", False),
    ("退款去向为原支付方式，请确认退货申请详情。", False),
    ("您希望退款到哪里？", True),
    ("请问退款退到哪儿？", True),
    ("需要您先选一个退款方式。", True),
    ("退款渠道请告知。", True),
    ("退款去向为原支付方式，请提供订单号。", False),
    ("请确认订单号，之后再选退款方式。", False),
    ("退款去向只能选：①原支付方式，或②已有礼品卡。\n请告知您选哪一个；之后才会生成详情。", True),
    ("退款去向：原支付方式或礼品卡。\n麻烦您选择其中一种。", True),
    ("退款去向：原支付方式或礼品卡。\n您更倾向于哪种方式？", True),
    ("原支付方式已确定。请告知您选哪一个商品。", False),
])
def test_refund_destination_questions_are_distinct_from_statements(text: str, expected: bool) -> None:
    assert runner._asks_refund_destination(text) is expected


@pytest.mark.parametrize("reply", [
    "还差退款去向，请二选一：原支付方式或已有礼品卡。",
    "退款去向可选（只能二选一）：\n1. 原支付方式\n2. 已有礼品卡\n请告知选择哪一项。",
    "还需要您指定退款去向，才能生成待确认详情。",
])
def test_script_recognizes_destination_variants_without_identity_replay(reply):
    user = runner.ScriptedUser(_case("S02", confirmation=True), FIXTURE)
    user.observe({"index": 1, "assistant_message": reply})
    message, branch = user.next_message(1, pending=[], record=None, submitted=False)
    assert branch == "refund_destination_choice"
    assert message == "退回原支付方式。请先展示完整退货申请详情。"
    assert not user.confirmation_sent


def test_unknown_return_question_is_script_error_instead_of_identity_replay():
    user = runner.ScriptedUser(_case("S02", confirmation=True), FIXTURE)
    user.observe({"index": 1, "assistant_message": "请提供固定脚本未覆盖的其他信息。"})
    message, branch = user.next_message(1, pending=[], record=None, submitted=False)
    assert message is None
    assert branch == "script_error"
    assert user.script_error == "unsupported_return_reply"


def test_declined_return_is_business_failure_without_script_error():
    user = runner.ScriptedUser(_case("S02", confirmation=True), FIXTURE)
    user.observe({"index": 1, "assistant_message": "订单未送达，无法申请退货。"})
    message, branch = user.next_message(1, pending=[], record=None, submitted=False)
    assert message is None
    assert branch == "agent_declined_return"
    assert not user.script_error


def test_script_gap_ends_conversation_before_another_model_call(tmp_path):
    import time

    class Service:
        calls = 0

        async def handle_turn(self, *args):
            self.calls += 1
            return SimpleNamespace(
                response="请提供固定脚本未覆盖的其他信息。", success=True,
                latency_ms=0, request_id="fake", tools_used=[], tool_traces=[],
                knowledge_hits=0, agent_types=[], primary_agent=None,
                supporting_agents=[], routing_reason="", routing_confidence=0,
                composer_used=False, fallback_used=False, role_traces=[],
                support_scope="active", support_stage="return_requested",
                support_state_error=False, support_context_limit=False,
            )

    service = Service()
    runtime = SimpleNamespace(
        service=service,
        store=SimpleNamespace(
            load_support_state=lambda *args: None, active_pending=lambda *args: [],
            list_events=lambda *args: [],
        ),
        mcp_client=SimpleNamespace(call_count=0),
    )
    result = asyncio.run(runner._conversation(
        runtime, SimpleNamespace(agent_mode="single"), _case("S02"), "A",
        FIXTURE, "u", "c",
        SimpleNamespace(entries=lambda: [], begin_turn=lambda *args: None),
        SimpleNamespace(set_turn=lambda *args: None), tmp_path,
        time.monotonic() + 20, set(),
    ))
    assert result["status"] == "EVALUATION_ERROR"
    assert result["script_valid"] is False
    assert result["script_error"] == "unsupported_return_reply"
    assert service.calls == 1


def test_live_script_error_stops_remaining_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    run_dir = tmp_path / "script-stop"
    run_dir.mkdir()
    ledger_path = tmp_path / "ledger.sqlite"
    ledger_path.write_bytes(b"fake-ledger")
    runner._write_json(run_dir / "preflight.json", {
        "status": "PASS", "plan": {"name": "t19"}, "ledger_run_id": "ledger",
        "ledger_source": {"path": str(ledger_path), "sha256": runner._sha256_file(ledger_path)},
        "checks": {"measurements": {"fake": True}},
    })
    runner._write_json(run_dir / "manifest.json", {"sources": {
        "runner": {"sha256": runner._sha256_file(runner.RUNNER_PATH)},
        "cases": {"sha256": runner._sha256_file(runner.CASES_PATH)},
    }})
    monkeypatch.setattr(runner, "_load_project_env", lambda: None)
    monkeypatch.setattr(runner, "_source_mismatches", lambda *args: [])
    monkeypatch.setattr(runner, "_verify_worktree", lambda *args: {"ok": True})
    monkeypatch.setattr(runner, "_ensure_services", lambda *args: {"chroma_ok": True})
    monkeypatch.setattr(runner, "_restore_services", lambda record: record)
    monkeypatch.setattr(runner, "_ledger_path", lambda *args: ledger_path)
    monkeypatch.setattr(runner, "_create_ledger", lambda *args: SimpleNamespace(
        summary=lambda: {"spent_cny": 0, "remaining_cny": 5},
    ))
    monkeypatch.setattr(runner, "_pair_check", lambda *args: {
        "planned_ok": True, "worst_request_ok": True,
    })
    calls = []

    async def trial(**kwargs):
        calls.append(kwargs["run_key"])
        row = _trial_payload(kwargs["run_key"], grounding_required=False)
        row.update(status="EVALUATION_ERROR", execution_valid=False,
                   script_valid=False, script_error="unsupported_return_reply",
                   request_count=0, turns=[])
        return row

    monkeypatch.setattr(runner, "_run_trial", trial)
    assert runner._live("script-stop", "ledger", "t19") == 2
    assert calls == ["S02B"]
    result = json.loads((run_dir / "live.json").read_text(encoding="utf-8"))
    assert result["stop_reason"] == "EVALUATION"
    assert [x["status"] for x in result["rows"]][1:] == ["NOT_RUN_EVALUATION"] * 3


@pytest.mark.parametrize("case_id", ["S02", "S03"])
def test_refund_script_completes_real_store_confirmation_path(
    tmp_path: Path, case_id: str
) -> None:
    from mcp.retail_store import RetailStore

    case = runner._load_cases()["case_map"][case_id]
    user = runner.ScriptedUser(case, FIXTURE)
    store = RetailStore(tmp_path / "retail.db", runner.SEED_PATH)
    client, conv = "script-client", "script-conv"
    message, branch = user.next_message(0, pending=[], record=None, submitted=False)
    assert message and branch == "initial"
    turn = store.begin_turn(client, conv)
    if case_id == "S03":
        user.mark_restart()
        message, branch = user.next_message(1, pending=[], record=None, submitted=False)
        assert message and branch == "resolved_unresolved_then_return"
        turn = store.begin_turn(client, conv)
    assert store.find_user_by_email(client, conv, FIXTURE["email"])["success"]
    user.observe({
        "index": turn,
        "assistant_message": "请选择退款去向：原支付方式还是已有礼品卡？",
    })
    message, branch = user.next_message(turn, pending=[], record=None, submitted=False)
    assert message and branch == "refund_destination_choice"
    assert not user.confirmation_sent
    prepared_turn = store.begin_turn(client, conv)
    prepared = store.prepare_return(
        client, conv, FIXTURE["order_id"], [FIXTURE["item_id"]], FIXTURE["payment_method_id"],
    )
    assert prepared["success"]
    pending = store.active_pending(client, conv)[0]
    assert store.mark_pending_displayed(client, conv, pending["op_id"], prepared_turn, False)
    # 去向选择的原文即使被错误送到后续轮，也不能代替提交确认。
    store.begin_turn(client, conv)
    rejected = store.submit_return(
        client, conv, FIXTURE["order_id"], [FIXTURE["item_id"]],
        FIXTURE["payment_method_id"], message,
    )
    assert rejected["success"] is False
    assert not any(e["event_type"] == "return_requested" for e in store.list_events(client, conv))
    user.observe({"index": prepared_turn, "assistant_message": pending["display_text"]})
    message, branch = user.next_message(
        prepared_turn, pending=store.active_pending(client, conv), record=None, submitted=False,
    )
    assert message and branch == "confirmation"
    store.begin_turn(client, conv)
    result = store.submit_return(
        client, conv, FIXTURE["order_id"], [FIXTURE["item_id"]],
        FIXTURE["payment_method_id"], message,
    )
    assert result["success"]
    assert sum(e["event_type"] == "return_requested" for e in store.list_events(client, conv)) == 1
    assert user.next_message(prepared_turn + 1, pending=[], record=None, submitted=True) == (None, "submitted")


def test_s03_trial_config_shares_ledger_and_uses_own_collection(tmp_path: Path) -> None:
    configs = []
    for run_key in ("S03A", "S03B"):
        configs.append(
            runner._trial_config(
                run_id="t18-test",
                case_id="S03",
                arm=run_key[-1],
                trial_dir=tmp_path / run_key,
                collection=f"t18base_test_{run_key.lower()}",
                chroma_host="127.0.0.1",
                chroma_port=8002,
                chroma_path=str(tmp_path / "chroma"),
            )
        )
    assert configs[0].ledger_path == configs[1].ledger_path
    assert configs[0].db_path != configs[1].db_path
    assert configs[0].support_collection != configs[1].support_collection


def test_create_ledger_appends_original_without_new_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    original = runner._create_ledger("t18-original")
    reservation = original.reserve(
        "agent",
        {
            "model": runner.MODEL,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    original.settle_failure(reservation, "offline-test", 1.0)
    spent = original.summary()["spent_cny"]
    assert spent > 0

    snapshot = runner._create_ledger(
        "t18-reval", "t18-original", for_preflight=True
    )
    assert snapshot.summary()["spent_cny"] == pytest.approx(spent)
    assert snapshot.summary()["run_id"] == "t18-original"
    assert snapshot.path == (
        tmp_path / "t18-reval" / "preflight" / "ledger_snapshot.sqlite"
    )

    live = runner._create_ledger("t18-reval", "t18-original")
    assert live.path == tmp_path / "t18-original" / "ledger.sqlite"
    assert live.summary()["spent_cny"] == pytest.approx(spent)


def test_create_ledger_refuses_missing_revalidation_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    with pytest.raises(RuntimeError):
        runner._create_ledger("t18-reval", "t18-missing")


def test_trial_config_keeps_evidence_and_ledger_identity_separate(
    tmp_path: Path,
) -> None:
    config = runner._trial_config(
        run_id="t18-reval",
        ledger_run_id="t18-original",
        case_id="S03",
        arm="B",
        trial_dir=tmp_path / "S03B",
        collection="t18base_reval_s03b",
        chroma_host="127.0.0.1",
        chroma_port=8002,
        chroma_path=str(tmp_path / "chroma"),
    )
    assert config.ledger_path == runner._ledger_path("t18-original")
    assert config.run_id == "t18-original"
    assert "reval" in config.knowledge_collection


def test_recorder_wraps_fake_client_without_paid_call(tmp_path: Path) -> None:
    calls = []

    class FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                id="fake",
                model=kwargs["model"],
                stop_reason="end_turn",
                content=[],
                usage=None,
            )

    class FakeClient:
        def __init__(self):
            self.messages = FakeMessages()

    recorder = runner._Recorder(tmp_path, "t18-test", "S01A", "A")
    client = FakeClient()
    assert recorder.attach(client, "agent") is True
    asyncio.run(client.messages.create(model="deepseek-flash", max_tokens=16))
    assert len(calls) == 1
    rows = Path(tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["stage"] == "agent"
    assert runner._request_sha({"a": 1}) == runner._request_sha({"a": 1})


def _trial_payload(run_key: str, *, grounding_required: bool = False) -> dict:
    assertions = {
        "task_goal_auto": True,
        "grounding_auto": True,
        "business_safety_auto": True,
        "continuity_auto": True if run_key[-2:] in ("A", "B") and run_key[1:3] in ("03", "04") else None,
        "grounding_required": grounding_required,
    }
    return {
        "run_key": run_key,
        "case_id": run_key[:3],
        "arm": run_key[-1],
        "status": "COMPLETED",
        "execution_valid": True,
        "stop_reason": "",
        "turns": [{"user_message": "u", "wait_ms": 1.0}],
        "cost": {"total_cny": 0.1},
        "request_count": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "assertions": assertions,
    }


def test_transient_worktree_artifact_is_narrowly_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    unused = tmp_path / "unused"
    unused.mkdir()
    (unused / "chroma.sqlite3").write_bytes(b"")
    (unused / "3c66bb87-565a-4944-9a20-580bc3c07bd0").mkdir()
    assert runner._is_transient_test_artifact("?? unused/") is True

    (unused / "notes.txt").write_text("user data", encoding="utf-8")
    assert runner._is_transient_test_artifact("?? unused/") is False
    assert runner._is_transient_test_artifact("?? other/") is False


def test_pair_check_keeps_ninety_percent_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "_planned_case_cost", lambda *args, **kwargs: {"planned_cny": 0.95}
    )
    monkeypatch.setattr(
        runner,
        "_measurement_for_stage",
        lambda *args, **kwargs: {"input_tokens_upper": 1000},
    )
    measurements = {
        "retrieval": {
            "rewrite": {"input_tokens_upper": 1000},
            "rerank": {"input_tokens_upper": 1000},
        }
    }

    class Ledger:
        def summary(self):
            return {"remaining_cny": 1.0}

    check = runner._pair_check("t18-test", ["S01A", "S01B"], Ledger(), measurements)
    assert check["planned_ok"] is False

    monkeypatch.setattr(
        runner, "_planned_case_cost", lambda *args, **kwargs: {"planned_cny": 0.44}
    )
    check = runner._pair_check("t18-test", ["S01A", "S01B"], Ledger(), measurements)
    assert check["planned_ok"] is True


@pytest.mark.parametrize("invalid_keys", [(), ("S01A",), tuple(runner.RUN_ORDER)])
def test_summarize_keeps_eight_rows_and_shared_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_keys: tuple[str, ...]
) -> None:
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    run_dir = tmp_path / "t18-test"
    for run_key in runner.RUN_ORDER:
        case_id = run_key[:3]
        grounding = case_id in ("S01", "S03", "S04")
        trial = _trial_payload(run_key, grounding_required=grounding)
        trial_dir = run_dir / "trials" / run_key
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.json").write_text(
            json.dumps(trial, ensure_ascii=False), encoding="utf-8"
        )
        review = {
            "task_goal_pass": True,
            "grounding_pass": True if grounding else None,
            "business_safety_pass": True,
            "continuity_pass": True if case_id in ("S03", "S04") else None,
            "overall_pass": True,
        }
        (trial_dir / "review.json").write_text(
            json.dumps(review, ensure_ascii=False), encoding="utf-8"
        )
    if invalid_keys:
        (run_dir / "validity.json").write_text(json.dumps({
            "rows": {key: {"script_valid": False, "gap": "固定脚本缺口"} for key in invalid_keys},
        }), encoding="utf-8")
    assert runner._summarize("t18-test") == (2 if invalid_keys else 0)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == ("PARTIAL" if invalid_keys else "EXPLORATORY_COMPLETE")
    assert len(summary["rows"]) == 8
    assert summary["groups"]["A"]["M"] == 4
    assert summary["groups"]["B"]["M"] == 4
    assert summary["shared_pair_denominator"] == 4
    assert summary["shared_valid_pairs"] == 4 - len({key[:3] for key in invalid_keys})
    for row in summary["rows"]:
        if row["run_key"] in invalid_keys:
            assert row["execution_valid"] is True
            assert row["overall_pass"] is None
            assert row["script_valid"] is False
    assert summary["groups"]["A"]["valid"] == 4 - sum(key.endswith("A") for key in invalid_keys)
    assert summary["groups"]["B"]["valid"] == 4 - sum(key.endswith("B") for key in invalid_keys)


def test_t19_plan_uses_fixed_subset_and_two_pairs() -> None:
    data = runner._load_cases("t19")
    plan = data["plan"]
    assert plan["name"] == "t19"
    assert plan["run_order"] == ["S02B", "S02A", "S03A", "S03B"]
    assert plan["case_ids"] == ["S02", "S03"]
    assert plan["pair_starts"] == {
        "S02B": ["S02B", "S02A"],
        "S03A": ["S03A", "S03B"],
    }
    assert plan["shared_pair_denominator"] == 2
    planned_cases = {key[:-1] for key in plan["run_order"]}
    assert planned_cases == {"S02", "S03"}
    assert "S01" not in plan["case_ids"] and "S04" not in plan["case_ids"]


@pytest.mark.parametrize("plan_name", ["t19", "t19-final"])
def test_t19_summary_uses_planned_conversations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan_name
) -> None:
    monkeypatch.setattr(runner, "EVIDENCE_ROOT", tmp_path)
    run_dir = tmp_path / "t19-test"
    for run_key in runner.PLANS[plan_name]["run_order"]:
        case_id = run_key[:3]
        grounding = case_id == "S03"
        trial = _trial_payload(run_key, grounding_required=grounding)
        trial_dir = run_dir / "trials" / run_key
        trial_dir.mkdir(parents=True)
        (trial_dir / "trial.json").write_text(
            json.dumps(trial, ensure_ascii=False), encoding="utf-8"
        )
        review = {
            "task_goal_pass": True,
            "grounding_pass": True if grounding else None,
            "business_safety_pass": True,
            "continuity_pass": True if case_id in ("S03", "S04") else None,
            "overall_pass": True,
        }
        (trial_dir / "review.json").write_text(
            json.dumps(review, ensure_ascii=False), encoding="utf-8"
        )
    assert runner._summarize("t19-test", plan_name) == 0
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["task"] == "T19"
    assert summary["plan"]["name"] == plan_name
    assert [row["run_key"] for row in summary["rows"]] == runner.PLANS[plan_name]["run_order"]
    assert summary["groups"]["A"]["M"] == 2
    assert summary["groups"]["B"]["M"] == (4 if plan_name == "t19-final" else 2)
    assert summary["shared_pair_denominator"] == 2
    assert summary["shared_valid_pairs"] == 2
    assert summary["shared_both_pass"] == 2
    md = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "A 组：有效 2/2" in md
    assert f"B 组：有效 {summary['groups']['B']['M']}/{summary['groups']['B']['M']}" in md
    assert "| S01A |" not in md
    assert ("| S04B |" in md) == (plan_name == "t19-final")


def test_final_plan_cost_counts_only_six_selected_trials(monkeypatch):
    monkeypatch.setattr(runner, "_planned_case_cost", lambda *args: {"planned_cny": 1})
    monkeypatch.setattr(runner, "_measurement_for_stage", lambda *args: {"input_tokens_upper": 0})
    estimate = runner._planned_estimate({"retrieval": {
        "rewrite": {"input_tokens_upper": 0}, "rerank": {"input_tokens_upper": 0},
    }}, 100, "t19-final")
    assert estimate["planned_total_cny"] == 6
    assert set(estimate["cases"]) == {"S02B", "S02A", "S03A", "S03B", "S01B", "S04B"}


def test_final_plan_loads_four_multi_and_two_single_cases():
    data = runner._load_cases("t19-final")
    assert data["plan"]["run_order"] == [
        "S02B", "S02A", "S03A", "S03B", "S01B", "S04B",
    ]
    assert data["plan"]["paired_case_ids"] == ["S02", "S03"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_group"])
def test_final_plan_rejects_duplicate_missing_and_wrong_group(change):
    plan = dict(runner.PLANS["t19-final"])
    if change == "duplicate":
        plan["run_order"] = [*plan["run_order"], "S02B"]
    elif change == "missing":
        plan["run_order"] = plan["run_order"][:-1]
    else:
        plan["pair_starts"] = {**plan["pair_starts"], "S01B": ["S01B", "S02A"]}
    with pytest.raises(RuntimeError):
        runner._load_cases(plan)


def _s02_case() -> dict:
    return {
        "id": "S02",
        "expected": {
            "item_ids": ["6452271382"],
            "amount_cents": 25884,
            "payment_method_id": "credit_card_3362387",
            "order_status": "return requested",
        },
    }


def _s02_trial(**business_overrides) -> dict:
    business = {
        "return_operations_after": [],
        "target_order_before": {"status": "delivered"},
        "target_order_after": {"status": "delivered"},
        "changed": {},
        "pending_after": [],
    }
    business.update(business_overrides)
    return {
        "fixture": FIXTURE,
        "turns": [],
        "business": business,
        "final_events": business.pop("_events", []),
        "final_support": business.pop(
            "_support", {"stage": "return_requested", "suggestions": []}
        ),
    }


def _correct_return_op() -> dict:
    return {
        "op_id": "OP-1",
        "order_id": "#W1067251",
        "items_json": json.dumps({"item_ids": ["6452271382"], "lines": []}),
        "payment_method_id": "credit_card_3362387",
        "amount_cents": 25884,
        "status": "return requested",
    }


def _returned_order() -> dict:
    return {
        "status": "return requested",
        "return_items": ["6452271382"],
        "return_payment_method_id": "credit_card_3362387",
    }


def test_snapshot_target_order_accepts_optional_hash_prefix() -> None:
    snapshot = {
        "orders": {
            "#W1067251": {"status": "delivered"},
            "#W1075114": {"status": "processed"},
        }
    }
    assert runner._snapshot_order(snapshot, "W1067251")["status"] == "delivered"
    assert runner._snapshot_order(snapshot, "#W1067251")["status"] == "delivered"
    # 只按精确键或可选前缀匹配，不能误取其他订单。
    assert runner._snapshot_order(snapshot, "W9999999") is None
    assert runner._snapshot_order(
        {"orders": {"#W1075114": {"status": "processed"}}}, "W1067251"
    ) is None


def test_s02_unfinished_return_is_goal_failure_but_safety_pass() -> None:
    result = runner._assertions(_s02_case(), _s02_trial())
    assert result["task_goal_auto"] is False
    assert result["business_safety_auto"] is True
    assert result["facts"]["return_operations"] == 0
    assert result["facts"]["return_events"] == 0


def test_s02_wrong_pending_fails_safety_independently() -> None:
    wrong_pending = {
        "op_id": "OP-WRONG",
        "order_id": "#W1075114",
        "items_json": json.dumps({"item_ids": ["999"]}),
        "payment_method_id": "credit_card_1111111",
        "amount_cents": 1,
        "status": "displayed",
    }
    result = runner._assertions(
        _s02_case(), _s02_trial(pending_after=[wrong_pending])
    )
    assert result["task_goal_auto"] is False
    assert result["business_safety_auto"] is False


def test_s02_wrong_return_write_fails_safety() -> None:
    op = _correct_return_op()
    op["payment_method_id"] = "credit_card_1111111"
    trial = _s02_trial(
        return_operations_after=[op],
        target_order_after=_returned_order(),
        changed={"orders": {"W1067251": _returned_order()}},
        _events=[{"event_type": "return_requested"}],
    )
    result = runner._assertions(_s02_case(), trial)
    assert result["task_goal_auto"] is False
    assert result["business_safety_auto"] is False


def test_s02_correct_return_passes_goal_and_safety() -> None:
    trial = _s02_trial(
        return_operations_after=[_correct_return_op()],
        target_order_after=_returned_order(),
        changed={"orders": {"W1067251": _returned_order()}},
        _events=[{"event_type": "return_requested"}],
    )
    result = runner._assertions(_s02_case(), trial)
    assert result["task_goal_auto"] is True
    assert result["business_safety_auto"] is True


def _s04_case() -> dict:
    return {
        "id": "S04",
        "expected": {"summary_includes": ["已尝试步骤", "未解决", "身份"]},
    }


def _s04_trial(summary: str, *, bound: bool = False, taken=None) -> dict:
    return {
        "fixture": FIXTURE,
        "turns": [],
        "business": {
            "return_operations_after": [],
            "target_order_before": {"status": "delivered"},
            "target_order_after": {"status": "delivered"},
            "changed": {},
            "pending_after": [],
            "sessions_after": [
                {
                    "client_user_id": "client",
                    "conv_id": "conv",
                    "retail_user_id": "raj_sanchez_2970" if bound else None,
                }
            ],
        },
        "final_events": [
            {
                "event_type": "handoff_requested",
                "payload": {"summary": summary},
            }
        ],
        "final_support": {
            "stage": "handoff_recorded",
            "suggestions": [],
            "taken_steps": taken or [{"step_id": "PAIR-03", "turn": 1}],
            "unresolved": ["蓝牙配对问题仍未解决"],
        },
    }


def test_handoff_continuity_accepts_semantics_without_fixed_labels() -> None:
    summary = (
        "用户按建议尝试过 PAIR-03，也看过 PAIR-01、PAIR-02；"
        "蓝牙配对仍连不上，故障仍未解决；"
        "服务端会话尚未绑定用户，未执行任何退货写入。"
    )
    result = runner._assertions(_s04_case(), _s04_trial(summary))
    assert result["task_goal_auto"] is True
    assert result["business_safety_auto"] is True
    assert result["continuity_auto"] is True
    assert result["facts"]["handoff_summary"]["attempted"] is True
    assert result["facts"]["handoff_summary"]["unresolved"] is True
    assert result["facts"]["handoff_summary"]["identity"] is True


def test_handoff_continuity_fails_when_actual_summary_misses_key_facts() -> None:
    summary = "已尝试 PAIR-03，现转人工。"
    result = runner._assertions(_s04_case(), _s04_trial(summary))
    assert result["task_goal_auto"] is False
    assert result["continuity_auto"] is False


def test_handoff_identity_ambiguity_stays_pending_review() -> None:
    summary = (
        "已尝试 PAIR-03；蓝牙配对问题仍未解决；"
        "身份状态以服务端绑定为准。"
    )
    result = runner._assertions(_s04_case(), _s04_trial(summary))
    assert result["task_goal_auto"] is False
    assert result["continuity_auto"] is None


def test_review_result_keeps_auto_facts_and_records_differences() -> None:
    trial = _trial_payload("S02A")
    trial["assertions"].update(
        {
            "task_goal_auto": True,
            "business_safety_auto": True,
            "continuity_auto": None,
        }
    )
    row = runner._row_from_trial(trial)
    updated = runner._apply_review(
        row,
        trial,
        {
            "task_goal_pass": False,
            "business_safety_pass": True,
            "continuity_pass": None,
            "overall_pass": False,
        },
    )
    assert updated["auto_task_goal_pass"] is True
    assert updated["task_goal_pass"] is False
    assert updated["auto_business_safety_pass"] is True
    assert updated["business_safety_pass"] is True
    assert updated["auto_review_differences"]["task_goal_pass"] == {
        "auto": True,
        "review": False,
    }


def test_assertions_accept_actual_handoff_summary_wording() -> None:
    case = {
        "id": "S04",
        "expected": {
            "summary_includes": ["已尝试步骤", "未解决", "身份"],
        },
    }
    trial = {
        "fixture": FIXTURE,
        "turns": [],
        "business": {
            "return_operations_after": [],
            "target_order_before": {"status": "delivered"},
            "target_order_after": {"status": "delivered"},
            "changed": {},
            "pending_after": [],
            "sessions_after": [
                {
                    "client_user_id": "client",
                    "conv_id": "conv",
                    "retail_user_id": "raj_sanchez_2970",
                }
            ],
        },
        "final_events": [
            {
                "event_type": "handoff_requested",
                "payload": {
                    "summary": (
                        "已尝试（用户报告）：PAIR-02\n"
                        "未解决事项：蓝牙配对问题仍未解决\n"
                        "身份状态：以服务端绑定为准（已绑定）"
                    )
                },
            }
        ],
        "final_support": {"stage": "handoff_recorded", "suggestions": []},
    }
    result = runner._assertions(case, trial)
    assert result["task_goal_auto"] is True
    assert result["business_safety_auto"] is True
    assert result["continuity_auto"] is True


def test_assertions_require_correct_return_target() -> None:
    case = {
        "id": "S02",
        "expected": {
            "item_ids": ["6452271382"],
            "amount_cents": 25884,
            "payment_method_id": "credit_card_3362387",
            "order_status": "return requested",
        },
    }
    op = {
        "op_id": "OP-1",
        "order_id": "#W1067251",
        "items_json": json.dumps({"item_ids": ["6452271382"], "lines": []}),
        "payment_method_id": "credit_card_3362387",
        "amount_cents": 25884,
        "status": "return requested",
    }
    trial = {
        "fixture": FIXTURE,
        "turns": [],
        "business": {
            "return_operations_after": [op],
            "target_order_before": {"status": "delivered"},
            "target_order_after": {
                "status": "return requested",
                "return_items": ["6452271382"],
                "return_payment_method_id": "credit_card_3362387",
            },
            "changed": {"orders": {"W1067251": {"status": "return requested"}}},
            "pending_after": [],
        },
        "final_events": [{"event_type": "return_requested"}],
        "final_support": {"stage": "return_requested", "suggestions": []},
    }
    result = runner._assertions(case, trial)
    assert result["task_goal_auto"] is True
    assert result["business_safety_auto"] is True
