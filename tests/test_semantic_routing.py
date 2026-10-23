"""选择性路由的Fake工程验证；不以合成分数证明语义质量。"""
import asyncio
import json

import numpy as np
import pytest

from agents.agent_orchestrator import AgentType, Request
from core.retail_semantic_router import (
    FORMAT_VERSION, MESSAGE_FORMAT_VERSION, LABEL_ROLES, RetailSemanticRouter,
    format_input, task_view, visible_history,
)
from evaluation.demo_retail_roles import build
from mcp.qwen_embedding import IDENTITY


class FakeEncoder:
    def __init__(self):
        self.inputs = []
        self.closed = False

    def __call__(self, texts):
        self.inputs.extend(texts)
        return np.eye(1, 1024)

    def close(self):
        self.closed = True


def fake_model(tmp_path, label="greeting", score=0.95):
    labels = sorted(LABEL_ROLES)
    weights = np.zeros((1024, len(labels)))
    weights[0, labels.index(label)] = score
    value = {"version": "fake-engineering-only", "format_version": FORMAT_VERSION,
             "encoder_identity": IDENTITY, "labels": labels, "weights": weights.tolist(),
             "bias": [0.0] * len(labels), "accepted_labels": ["greeting", "refund"],
             "score_threshold": 0.75, "margin_threshold": 0.3}
    path = tmp_path / "fake-head.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_linear_head_state_conflict_and_no_text_only_cache(tmp_path):
    encoder = FakeEncoder()
    router = RetailSemanticRouter(fake_model(tmp_path), encoder)
    first = router.classify("同一句短答", state=task_view())
    second = router.classify("同一句短答", state={**task_view(), "flow_status": "active", "awaiting": "refund_destination"})
    assert first["path"] == "local" and first["role"] == "general"
    assert second["path"] == "llm" and second["reason"] == "state_conflict"
    assert len(encoder.inputs) == 2 and encoder.inputs[0] != encoder.inputs[1]
    router.close()
    assert encoder.closed


@pytest.mark.parametrize("label,score,reason", [("greeting", 0.6, "uncertain"), ("other", 1.0, "unsupported_label")])
def test_low_score_and_unsupported_class_reject(tmp_path, label, score, reason):
    result = RetailSemanticRouter(fake_model(tmp_path, label, score), FakeEncoder()).classify("测试")
    assert result["path"] == "llm" and result["reason"] == reason


def test_invalid_artifact_fails_closed_without_encoding(tmp_path):
    path = fake_model(tmp_path)
    value = json.loads(path.read_text())
    value["encoder_identity"] = "different"
    path.write_text(json.dumps(value))
    encoder = FakeEncoder()
    router = RetailSemanticRouter(path, encoder)
    for _ in range(2):
        assert router.classify("你好")["reason"] == "unavailable"
    assert encoder.inputs == [] and encoder.closed


def test_classifier_failure_does_not_close_shared_rag_encoder(tmp_path):
    encoder = FakeEncoder()
    router = RetailSemanticRouter(tmp_path / "missing.json", encoder, owns_encoder=False)
    assert router.classify("你好")["reason"] == "unavailable"
    router.close()
    assert not encoder.closed


@pytest.mark.parametrize("similarity,expected", [(0.9, "local"), (0.4, "llm")])
def test_v2_rejects_high_linear_score_outside_semantic_support(tmp_path, similarity, expected):
    path = fake_model(tmp_path)
    model = json.loads(path.read_text())
    references = np.eye(len(model["labels"]), 1024)
    references[model["labels"].index("greeting")] = 0
    references[model["labels"].index("greeting"), :2] = [similarity, (1 - similarity ** 2) ** 0.5]
    model.update(format_version=MESSAGE_FORMAT_VERSION, reference_vectors=references.tolist(),
                 reference_labels=model["labels"], similarity_threshold=0.75)
    path.write_text(json.dumps(model))
    encoder = FakeEncoder()
    router = RetailSemanticRouter(path, encoder)
    result = router.classify("高线性分数仍需语义支持")
    assert result["path"] == expected and result["score"] == 0.95
    assert result["similarity"] == pytest.approx(similarity)
    if expected == "llm":
        assert result["reason"] == "outside_training_support"
    active = router.classify("高线性分数仍需语义支持", state={**task_view(), "flow_status": "active"})
    assert active["path"] == "llm" and len(encoder.inputs) == 2
    assert encoder.inputs[0] == encoder.inputs[1]


def test_v2_input_is_not_silently_truncated_and_invalid_references_fail_closed(tmp_path):
    path = fake_model(tmp_path)
    encoder = FakeEncoder()
    router = RetailSemanticRouter(path, encoder)
    assert router.classify("问" * 2001)["reason"] == "unsupported_input" and not encoder.inputs
    model = json.loads(path.read_text())
    model.update(format_version=MESSAGE_FORMAT_VERSION, reference_vectors=[[0] * 1024],
                 reference_labels=["greeting"], similarity_threshold=0.75)
    path.write_text(json.dumps(model))
    assert RetailSemanticRouter(path, encoder).classify("你好")["reason"] == "unavailable"


def test_default_model_is_self_contained_and_groups_do_not_overlap():
    from core.retail_semantic_router import DEFAULT_MODEL

    router = RetailSemanticRouter(DEFAULT_MODEL, FakeEncoder())
    router._load()
    assert DEFAULT_MODEL.parent.name == "core"
    assert router.model["format_version"] == MESSAGE_FORMAT_VERSION
    assert router.model["method"] == "ridge_linear_one_hot_with_semantic_support"
    assert router.model["accepted_labels"] == ["greeting"]
    assert len(router.model["reference_vectors"]) == 48
    groups = [set(values) for values in router.model["groups"].values()]
    assert sum(map(len, groups)) == len(set().union(*groups))


def test_history_is_visible_bounded_and_task_has_no_authority_fields():
    record = {"visible_history": [{"user": str(i), "assistant": "a" * 500, "turn": i} for i in range(5)]}
    history = visible_history(record)
    assert len(history) == 4 and history[0]["content"] == "3"
    assert len(history[1]["content"]) == 300
    text = format_input("是的", history, {**task_view(), "bound_user_id": "forged", "approved": True})
    assert "forged" not in text and "approved" not in text


@pytest.mark.parametrize("score,active,path", [(0.95, False, "local"), (0.6, False, "llm"), (0.95, True, "llm")])
def test_shared_boundary_precedes_llm_and_falls_back_once(tmp_path, score, active, path):
    async def scenario():
        _, calls, service = build(tmp_path, "prepare")
        orchestrator = service.agent.orchestrator
        orchestrator._retail_semantic_router = RetailSemanticRouter(fake_model(tmp_path, score=score), FakeEncoder())
        req = Request("知道了", "u", "c", retail_task={**task_view(), "flow_status": "active" if active else "idle"})
        try:
            result = await orchestrator.prepare_route(req)
            assert req.retail_routing["path"] == path
            assert len(calls.calls) == (0 if path == "local" else 1)
            if path == "local":
                assert result.primary_agent == AgentType.GENERAL and not result.supporting_agents
        finally:
            orchestrator._retail_semantic_router.close()
            if service.workflow:
                await service.workflow.close()
    asyncio.run(scenario())


def test_unavailable_router_original_path_and_fallback_cache_are_state_scoped(tmp_path):
    async def scenario():
        _, calls, service = build(tmp_path, "prepare")
        orchestrator = service.agent.orchestrator
        orchestrator._retail_semantic_router = RetailSemanticRouter(tmp_path / "missing.json")
        try:
            for status in ("idle", "active", "active"):
                req = Request("原支付方式", "u", "c", retail_task={**task_view(), "flow_status": status})
                await orchestrator.prepare_route(req)
                assert req.retail_routing["reason"] == "unavailable"
            assert len(calls.calls) == 2
            assert orchestrator._intent_recognizer.cache_hits == 1
            content = [c["messages"][0]["content"] for c in calls.calls]
            assert content[0] != content[1] and "服务端调度语境" in content[0]
        finally:
            if service.workflow:
                await service.workflow.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("message,pending,role", [("确认", [{}], AgentType.BILLING), ("转人工", [], AgentType.ESCALATION)])
def test_rules_never_encode_or_classify(tmp_path, message, pending, role):
    class ForbiddenRouter:
        def classify(self, *args):
            raise AssertionError("确定性分支不应进入编码器")

    async def scenario():
        _, calls, service = build(tmp_path, "prepare")
        orchestrator = service.agent.orchestrator
        orchestrator._retail_semantic_router = ForbiddenRouter()
        try:
            req = Request(message, "u", "c", retail_pending=pending)
            result = await orchestrator.prepare_route(req)
            assert result.primary_agent == role
            assert req.retail_routing["path"] == "rule" and not calls.calls
        finally:
            if service.workflow:
                await service.workflow.close()
    asyncio.run(scenario())


def test_disable_switch_and_nonretail_do_not_use_router(tmp_path):
    async def scenario():
        _, calls, service = build(tmp_path, "prepare")
        orchestrator = service.agent.orchestrator
        try:
            await orchestrator.prepare_route(Request("原支付方式", "u", "c"))
            assert len(calls.calls) == 1
            orchestrator.retail_mode = False
            orchestrator._retail_semantic_router = object()
            req = Request("另一个普通问题", "u", "c")
            await orchestrator.prepare_route(req)
            assert len(calls.calls) == 2 and req.retail_routing == {}
        finally:
            if service.workflow:
                await service.workflow.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("message,returning,role", [("还没好", False, AgentType.TECHNICAL), ("先别退了，再试连接", True, AgentType.TECHNICAL)])
def test_requested_short_state_phrases_keep_existing_task_boundary(tmp_path, message, returning, role):
    from agents.retail_support import analyze_message, build_support_packet, new_support_record

    _, _, service = build(tmp_path, "prepare")
    service.agent.orchestrator.retail_support_enabled = True
    record = new_support_record(1)
    if returning:
        record["stage"] = "return_collecting"
        record["return_flow"].update(status="active", awaiting="refund_destination", start_turn=1, last_turn=1)
    packet = build_support_packet(record, analyze_message(message, record), message, 2, None, [])
    req = Request(message, "u", "c", retail_support=packet, retail_task=task_view(packet))
    decision = service.agent.orchestrator._support_routing(req)
    assert decision is not None and decision.primary_agent == role
