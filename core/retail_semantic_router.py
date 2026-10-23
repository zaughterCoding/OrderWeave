"""零售中文选择性分类；输出仅供路由，不能提供业务授权。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from mcp.qwen_embedding import IDENTITY, QwenEmbeddingFunction


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "core/retail_semantic_model.json"
FORMAT_VERSION = "retail-intent-document-v1"
MESSAGE_FORMAT_VERSION = "retail-intent-message-v2"
LABEL_ROLES = {"order_status": "general", "query": "general", "greeting": "general",
               "refund": "billing", "human_handoff": "escalation", "other": "general"}
QUICK_LABELS = {"order_status", "query", "refund", "greeting"}


def task_view(packet=None, pending=None):
    """只传调度语境，不把身份或金额变成模型输入事实。"""
    packet = packet or {}
    flow = packet.get("return_flow") or {}
    current = packet.get("current_state") or {}
    return {"stage": str(current.get("stage") or "none"),
            "flow_status": str(flow.get("status") or "idle"),
            "awaiting": str(flow.get("awaiting") or ""),
            "action": str(flow.get("action") or "none"),
            "pending": bool(pending)}


def visible_history(record):
    history = []
    for pair in (record or {}).get("visible_history", [])[-2:]:
        for role in ("user", "assistant"):
            text = pair.get(role)
            if isinstance(text, str) and text.strip():
                history.append({"role": role, "content": text[:300]})
    return history


def format_input(message, history=None, state=None, format_version=FORMAT_VERSION):
    """分类单独使用文档向量，不调用RAG查询指令。"""
    if format_version == MESSAGE_FORMAT_VERSION:
        # 任务由权威状态单独约束，避免固定上下文淹没本轮话语的语义差异。
        return "用户消息：" + str(message).strip()[:2000]
    state = state or task_view()
    safe = {k: state.get(k, task_view()[k]) for k in task_view()}
    recent = [{"role": h["role"], "content": str(h.get("content") or "")[:300]}
              for h in (history or [])[-4:] if h.get("role") in ("user", "assistant")]
    return ("当前用户消息：" + str(message)[:2000] + "\n当前任务：" +
            json.dumps(safe, ensure_ascii=False, sort_keys=True) + "\n最近可见对话：" +
            json.dumps(recent, ensure_ascii=False, sort_keys=True))


def state_consistent(label, state):
    """活跃任务中的疑似切换交回原路径；不据分类结果修改任务。"""
    state = state or task_view()
    active = state.get("pending") or state.get("flow_status") in {"active", "awaiting_confirmation"}
    if active and label != "refund":
        return False
    if state.get("stage") == "troubleshooting" and label in {"greeting", "order_status", "refund"}:
        return False
    return True


class RetailSemanticRouter:
    def __init__(self, path=DEFAULT_MODEL, encoder=None, owns_encoder=True):
        self.path = Path(path)
        self.encoder = encoder
        self.owns_encoder = owns_encoder
        self.model = None
        self.disabled_reason = ""

    def _load(self):
        if self.disabled_reason:
            raise RuntimeError(self.disabled_reason)
        if self.model is not None:
            return
        model = json.loads(self.path.read_text(encoding="utf-8"))
        labels = model["labels"]
        weights = np.asarray(model["weights"], dtype=np.float64)
        bias = np.asarray(model["bias"], dtype=np.float64)
        if (model["format_version"] not in {FORMAT_VERSION, MESSAGE_FORMAT_VERSION} or model["encoder_identity"] != IDENTITY
                or set(labels) != set(LABEL_ROLES) or len(labels) != len(LABEL_ROLES)
                or weights.shape != (1024, len(labels)) or bias.shape != (len(labels),)
                or not np.isfinite(weights).all() or not np.isfinite(bias).all()
                or not set(model["accepted_labels"]) <= QUICK_LABELS
                or not 0 <= model["score_threshold"] <= 2
                or not 0 <= model["margin_threshold"] <= 2):
            raise ValueError("语义分类器格式或向量身份不符")
        if model["format_version"] == MESSAGE_FORMAT_VERSION:
            references = np.asarray(model["reference_vectors"], dtype=np.float64)
            reference_labels = model["reference_labels"]
            if (references.shape != (len(reference_labels), 1024) or not len(reference_labels)
                    or set(reference_labels) != set(labels) or not np.isfinite(references).all()
                    or not np.allclose(np.linalg.norm(references, axis=1), 1.0, atol=1e-4)
                    or not 0 <= model["similarity_threshold"] <= 1):
                raise ValueError("语义拒接参考无效")
            self.references = references
            self.reference_labels = np.asarray(reference_labels)
        self.model, self.weights, self.bias = model, weights, bias
        if self.encoder is None:
            self.encoder = QwenEmbeddingFunction(timeout=90)

    def classify(self, message, history=None, state=None):
        started = time.perf_counter()
        info = {"path": "llm", "reason": "unavailable", "encoder_calls": 0,
                "encoding_ms": 0.0, "head_ms": 0.0}
        try:
            self._load()
            if not str(message).strip() or len(str(message)) > 2000:
                info.update(reason="unsupported_input", total_ms=(time.perf_counter() - started) * 1000)
                return info
            text = format_input(message, history, state, self.model["format_version"])
            encoded = time.perf_counter()
            info["encoder_calls"] = 1
            x = np.asarray(self.encoder([text]), dtype=np.float64)
            info["encoding_ms"] = (time.perf_counter() - encoded) * 1000
            if (x.shape != (1, 1024) or not np.isfinite(x).all()
                    or not np.allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-4)):
                raise ValueError("分类编码无效")
            scored = time.perf_counter()
            scores = (x @ self.weights + self.bias)[0]
            order = np.argsort(scores)
            top = int(order[-1])
            label = self.model["labels"][top]
            score, margin = float(scores[top]), float(scores[top] - scores[order[-2]])
            similarity = None
            if self.model["format_version"] == MESSAGE_FORMAT_VERSION:
                similarity = float(np.max(x @ self.references[self.reference_labels == label].T))
            info.update(predicted_intent=label, score=score, margin=margin,
                        similarity=similarity, version=self.model["version"],
                        head_ms=(time.perf_counter() - scored) * 1000)
            if label not in self.model["accepted_labels"]:
                info["reason"] = "unsupported_label"
            elif score < self.model["score_threshold"] or margin < self.model["margin_threshold"]:
                info["reason"] = "uncertain"
            elif similarity is not None and similarity < self.model["similarity_threshold"]:
                info["reason"] = "outside_training_support"
            elif not state_consistent(label, state):
                info["reason"] = "state_conflict"
            else:
                info.update(path="local", reason="semantic_support_and_state_agree", intent=label,
                            role=LABEL_ROLES[label])
        except Exception as exc:
            self.disabled_reason = type(exc).__name__
            info.update(path="llm", reason="unavailable", error=type(exc).__name__)
            if self.encoder is not None and self.owns_encoder:
                try:
                    self.encoder.close()
                except Exception:
                    pass
        info["total_ms"] = (time.perf_counter() - started) * 1000
        return info

    def close(self):
        if self.encoder is not None and self.owns_encoder:
            self.encoder.close()
