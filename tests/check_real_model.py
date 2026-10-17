"""T8 真实模型可用性验收的人工运行入口。

该文件不匹配 pytest 的默认收集规则，不会被全量测试自动执行。
运行示例：

    uv run --locked --offline python tests/check_real_model.py --budget-cny 5

脚本会启动真实 uvicorn 服务，使用项目根 `.env` 中的真实模型配置，
并把每次模型调用的阶段、usage、耗时与保守费用写入工作目录。
它只包装 SDK 客户端以记录真实返回值，不替换模型输出、不修改业务 Prompt。
"""
from __future__ import annotations

import argparse
import asyncio
import builtins
import contextlib
import io
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Iterable, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8 真实模型验收脚本")
    parser.add_argument(
        "--budget-cny",
        type=float,
        required=True,
        help="本次真实模型调用总费用上限（人民币）",
    )
    parser.add_argument(
        "--run-id",
        default=time.strftime("t8_%Y%m%d_%H%M%S"),
        help="本次运行的唯一标识，用于合成用户、会话与知识文档",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="证据与报告输出目录；默认 docs/reports/t8-runs/<run-id>",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


# 运行时由 configure() 初始化；导入模块本身不解析命令行、不写文件。
ARGS: Optional[argparse.Namespace] = None
RUN_ID = ""
WORK_DIR = ROOT

# 保守计费口径：DeepSeek 官方高峰价，汇率取 2026-10-16 Frankfurter 的 1 USD = 6.6862 CNY。
PRICE_CNY_PER_TOKEN = {
    "input_miss": 0.30 * 6.6862 / 1_000_000,
    "input_hit": 0.006 * 6.6862 / 1_000_000,
    "output": 1.20 * 6.6862 / 1_000_000,
}

# 只匹配肯定式的退款完成表述；否定句“无法承诺退款成功”不属于越权承诺。
REFUND_SUCCESS_CLAIMS = (
    "已退款",
    "已为您退款",
    "已完成退款",
    "退款已成功",
    "已成功退款",
)

_CLAUSE_BOUNDARIES = "。！？；\n，,"
_NEGATION_MARKERS = (
    "不会",
    "无法",
    "不能",
    "没有",
    "未",
    "不",
    "并非",
    "拒绝",
    "避免",
)


def has_affirmative_refund_success_claim(text: str) -> bool:
    """只把肯定式表述视为越权承诺，忽略“不会声称……已退款成功”。"""
    for claim in REFUND_SUCCESS_CLAIMS:
        start = 0
        while True:
            index = text.find(claim, start)
            if index < 0:
                break
            left = max(
                (text.rfind(ch, 0, index) for ch in _CLAUSE_BOUNDARIES),
                default=-1,
            ) + 1
            prefix = text[left:index]
            if not any(marker in prefix for marker in _NEGATION_MARKERS):
                return True
            start = index + len(claim)
    return False


class BudgetExceeded(BaseException):
    """剩余预算不足以覆盖下一次可能的最坏费用。

    继承 ``BaseException``，避免被业务层的 ``except Exception`` 吞掉后
    静默降级为本地规则，导致超出费用上限仍继续调用。
    """


class Reservation:
    """一次在途模型调用的最坏费用预留。"""

    __slots__ = ("stage", "worst_cny", "released")

    def __init__(self, stage: str, worst_cny: float):
        self.stage = stage
        self.worst_cny = worst_cny
        self.released = False


class Recorder:
    """记录真实 SDK 调用，并在发起前预留最坏情况费用。"""

    def __init__(self, budget_cny: float, work_dir: pathlib.Path):
        self.budget_cny = float(budget_cny)
        self.work_dir = work_dir
        self.spent_cny = 0.0
        # 在途请求按最坏费用占用额度，避免并发请求同时通过同一余额。
        self.reserved_cny = 0.0
        self.calls: List[Dict[str, Any]] = []
        self.estimate_misses: List[Dict[str, Any]] = []
        self._reservations: Dict[int, Reservation] = {}
        self._lock = threading.Lock()
        self._usage_path = work_dir / "usage.jsonl"
        self._usage_path.write_text("", encoding="utf-8")

    @staticmethod
    def _prompt_text(kwargs: Dict[str, Any]) -> str:
        parts: List[str] = []
        system = kwargs.get("system")
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
        for message in kwargs.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text":
                        parts.append(str(block.get("text", "")))
                    elif block_type == "tool_use":
                        parts.append(
                            json.dumps(block.get("input"), ensure_ascii=False, default=str)
                        )
                    elif block_type == "tool_result":
                        result_content = block.get("content")
                        if isinstance(result_content, str):
                            parts.append(result_content)
                        elif isinstance(result_content, list):
                            for item in result_content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    parts.append(str(item.get("text", "")))
        tools = kwargs.get("tools")
        if tools:
            parts.append(json.dumps(tools, ensure_ascii=False))
        return "\n".join(parts)

    @staticmethod
    def infer_stage(kwargs: Dict[str, Any]) -> str:
        prompt = Recorder._prompt_text(kwargs)
        rules = (
            ("客服意图分析专家", "intent"),
            ("将以下用户查询改写", "query_rewrite"),
            ("根据用户查询，对以下检索结果", "rerank"),
            ("客服 Response Composer", "composer"),
            ("用 2-3 句话总结以下对话", "summary"),
            ("你是对话摘要器", "summary_merge"),
            ("提炼或更新用户偏好", "profile"),
            ("客服质量评估专家", "judge"),
        )
        for needle, stage in rules:
            if needle in prompt:
                return stage
        if kwargs.get("tools"):
            return "agent"
        return "unclassified"

    @staticmethod
    def estimate_worst_cny(kwargs: Dict[str, Any]) -> float:
        """按完整请求载荷估算最坏费用，输入字符数覆盖 tool_result 等所有内容。"""
        payload = {
            "system": kwargs.get("system"),
            "messages": kwargs.get("messages") or [],
            "tools": kwargs.get("tools"),
            "tool_choice": kwargs.get("tool_choice"),
        }
        try:
            serialized = json.dumps(payload, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(payload)
        # 中文按 1 字符约 1 token 保守估算；英文会被高估，只会更安全。
        input_est = max(1, len(serialized))
        output_est = int(kwargs.get("max_tokens") or 1024)
        return (
            input_est * PRICE_CNY_PER_TOKEN["input_miss"]
            + output_est * PRICE_CNY_PER_TOKEN["output"]
        )

    def reserve(self, stage: str, kwargs: Dict[str, Any]) -> Reservation:
        """原子预留本次请求的最坏费用；余额不足时拒绝发起请求。"""
        worst = self.estimate_worst_cny(kwargs)
        with self._lock:
            if self.spent_cny + self.reserved_cny + worst > self.budget_cny:
                raise BudgetExceeded(
                    f"阶段 {stage} 最坏费用 {worst:.4f} CNY 将超出剩余预算"
                    f"（已花费 {self.spent_cny:.4f}，在途预留 {self.reserved_cny:.4f}）"
                )
            reservation = Reservation(stage, worst)
            self._reservations[id(reservation)] = reservation
            self.reserved_cny += worst
            return reservation

    def _release_locked(self, reservation: Optional[Reservation]) -> Optional[float]:
        if reservation is None or reservation.released:
            return None
        reservation.released = True
        self._reservations.pop(id(reservation), None)
        self.reserved_cny = max(0.0, self.reserved_cny - reservation.worst_cny)
        return reservation.worst_cny

    def _append_locked(self, entry: Dict[str, Any]) -> None:
        self.calls.append(entry)
        with self._usage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def record(
        self,
        stage: str,
        kwargs: Dict[str, Any],
        response: Any,
        latency_ms: float,
        reservation: Optional[Reservation] = None,
    ) -> None:
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        cost = (
            input_tokens * PRICE_CNY_PER_TOKEN["input_miss"]
            + output_tokens * PRICE_CNY_PER_TOKEN["output"]
            + cache_read * PRICE_CNY_PER_TOKEN["input_hit"]
        )
        blocks = getattr(response, "content", []) or []
        tool_uses = [
            getattr(block, "name", "")
            for block in blocks
            if getattr(block, "type", "") == "tool_use"
        ]
        text_parts = [
            str(getattr(block, "text", ""))
            for block in blocks
            if getattr(block, "type", "") == "text"
        ]
        entry = {
            "stage": stage,
            # 只保留开头，用于定位截断和格式问题，不写入完整业务内容。
            "text_preview": "".join(text_parts)[:160],
            "content_block_types": [
                getattr(block, "type", "") for block in blocks
            ],
            "requested_model": kwargs.get("model"),
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            "tool_uses": tool_uses,
            "estimated_worst_cny": (
                round(reservation.worst_cny, 6) if reservation else None
            ),
            "cost_cny_conservative": round(cost, 6),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            worst = self._release_locked(reservation)
            self.spent_cny += cost
            if worst is not None and cost > worst + 1e-12:
                miss = {
                    "stage": stage,
                    "estimated_worst_cny": round(worst, 6),
                    "actual_cny": round(cost, 6),
                }
                self.estimate_misses.append(miss)
                entry["estimate_exceeded"] = miss
            self._append_locked(entry)

    def record_failure(
        self,
        reservation: Reservation,
        kwargs: Dict[str, Any],
        latency_ms: float,
        error: BaseException,
    ) -> None:
        """失败请求可能已被供应商处理，按预留上限保守计费。"""
        worst = reservation.worst_cny
        error_text = f"{type(error).__name__}: {error}"
        entry = {
            "stage": reservation.stage,
            "text_preview": "",
            "content_block_types": [],
            "requested_model": kwargs.get("model"),
            "response_model": None,
            "response_id": None,
            "stop_reason": None,
            "failed": True,
            "error": error_text[:500],
            "latency_ms": round(latency_ms, 1),
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "tool_uses": [],
            "estimated_worst_cny": round(worst, 6),
            "cost_cny_conservative": round(worst, 6),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self._lock:
            self._release_locked(reservation)
            self.spent_cny += worst
            self._append_locked(entry)

    def summary(self) -> Dict[str, Any]:
        stages: Dict[str, Dict[str, Any]] = {}
        for call in self.calls:
            bucket = stages.setdefault(
                call["stage"],
                {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_cny": 0.0},
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += call["input_tokens"]
            bucket["output_tokens"] += call["output_tokens"]
            bucket["cost_cny"] = round(
                bucket["cost_cny"] + call["cost_cny_conservative"], 6
            )
        return {
            "budget_cny": self.budget_cny,
            "spent_cny_conservative": round(self.spent_cny, 6),
            "reserved_cny": round(self.reserved_cny, 6),
            "remaining_cny": round(
                self.budget_cny - self.spent_cny - self.reserved_cny, 6
            ),
            "model_requests": len(self.calls),
            "failed_requests": sum(1 for call in self.calls if call.get("failed")),
            "estimate_misses": list(self.estimate_misses),
            "stages": stages,
            "price_basis": {
                "provider": "DeepSeek Anthropic-compatible Messages API",
                "model": os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL"),
                "official_pricing_url": "https://api-docs.deepseek.com/quick_start/pricing",
                "peak_prices_usd_per_million": {
                    "input_cache_hit": 0.006,
                    "input_cache_miss": 0.30,
                    "output": 1.20,
                },
                "usd_cny": 6.6862,
                "usd_cny_date": "2026-10-16",
                "note": "按高峰价估算，未扣除低峰折扣；这是保守估算，不是供应商实际账单。",
            },
        }


RECORDER: Optional[Recorder] = None
_REAL_CLIENTS: List[Any] = []
_REAL_ASYNC_ANTHROPIC = None


def _constrained_client_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """关闭 SDK 隐式重试，使每次实际尝试都经过预算预留与记录。"""
    constrained = dict(kwargs)
    constrained["max_retries"] = 0
    return constrained


class RecordingMessages:
    def __init__(self, real_messages: Any):
        self._real = real_messages

    async def create(self, **kwargs: Any) -> Any:
        if RECORDER is None:
            raise RuntimeError("调用前必须先执行 configure()")
        stage = RECORDER.infer_stage(kwargs)
        reservation = RECORDER.reserve(stage, kwargs)
        started = time.monotonic()
        try:
            response = await self._real.create(**kwargs)
        except BaseException as exc:
            RECORDER.record_failure(
                reservation,
                kwargs,
                (time.monotonic() - started) * 1000,
                exc,
            )
            raise
        RECORDER.record(
            stage,
            kwargs,
            response,
            (time.monotonic() - started) * 1000,
            reservation,
        )
        return response


class RecordingClient:
    """透传真实 AsyncAnthropic，只记录真实响应与 usage。"""

    def __init__(self, real: Any):
        self._real = real
        self.messages = RecordingMessages(real.messages)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _patch_clients() -> None:
    global _REAL_ASYNC_ANTHROPIC
    from anthropic import AsyncAnthropic as real_cls

    _REAL_ASYNC_ANTHROPIC = real_cls

    def factory(**kwargs: Any) -> RecordingClient:
        real = real_cls(**_constrained_client_kwargs(kwargs))
        _REAL_CLIENTS.append(real)
        return RecordingClient(real)

    import agents.agent_orchestrator as orchestrator_module
    import core.intent_recognizer as intent_module
    import evaluation.evaluator as evaluator_module
    import mcp.tool_manager as tool_module
    import memory.conversation_memory as memory_module

    for module in (
        orchestrator_module,
        intent_module,
        evaluator_module,
        tool_module,
        memory_module,
    ):
        module.AsyncAnthropic = factory


def _restore_clients() -> None:
    if _REAL_ASYNC_ANTHROPIC is None:
        return
    import agents.agent_orchestrator as orchestrator_module
    import core.intent_recognizer as intent_module
    import evaluation.evaluator as evaluator_module
    import mcp.tool_manager as tool_module
    import memory.conversation_memory as memory_module

    for module in (
        orchestrator_module,
        intent_module,
        evaluator_module,
        tool_module,
        memory_module,
    ):
        module.AsyncAnthropic = _REAL_ASYNC_ANTHROPIC


class CaseLog:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.path.write_text("", encoding="utf-8")
        self.items: List[Dict[str, Any]] = []

    def add(
        self,
        *,
        case: str,
        assertions: Dict[str, Any],
        required: Iterable[str],
        **payload: Any,
    ) -> Dict[str, Any]:
        """由唯一的必需断言集合生成状态，其余断言仅作诊断信息。"""
        required_keys = list(required)
        missing = [key for key in required_keys if key not in assertions]
        failed = [key for key in required_keys if not assertions.get(key)]
        informational_failed = [
            key
            for key, value in assertions.items()
            if key not in required_keys and not value
        ]
        payload.update(
            {
                "case": case,
                "assertions": assertions,
                "required": required_keys,
                "required_failed": failed,
                "required_missing": missing,
                "informational_failed": informational_failed,
                "status": "pass" if not missing and not failed else "fail",
            }
        )
        payload.setdefault("run_id", RUN_ID)
        self.items.append(payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload


CASES: Optional[CaseLog] = None


# 业务层已观察到会静默降级的告警；作为补充证据，不单独承担判定。
DEGRADATION_MARKERS = (
    ("Response Composer 失败", "composer_fallback"),
    ("查询改写未解析到 JSON 数组", "query_rewrite_parse_fallback"),
    ("查询改写失败，使用原始查询", "query_rewrite_error_fallback"),
    ("重排失败，返回原始顺序", "rerank_fallback"),
    ("更新用户画像失败", "profile_failure"),
    ("生成摘要失败", "summary_failure"),
    ("生成摘要为空", "summary_empty"),
    ("合并摘要失败", "summary_merge_fallback"),
    ("LLM 识别失败", "intent_failure"),
    ("LLM Judge 失败", "judge_failure"),
    ("降级到 GeneralAgent", "agent_fallback"),
)


class DegradationCapture(logging.Handler):
    """把已声明会降级的 WARNING 转为机器可读记录。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.items: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        for marker, code in DEGRADATION_MARKERS:
            if marker in message:
                self.items.append(
                    {
                        "code": code,
                        "logger": record.name,
                        "level": record.levelname,
                        "message": message[:500],
                    }
                )
                break


DEGRADATIONS = DegradationCapture()


def _install_degradation_capture() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, DegradationCapture):
            root.removeHandler(handler)
    root.addHandler(DEGRADATIONS)


def _apply_environment() -> None:
    os.chdir(ROOT)
    os.environ["MONITOR_INTERVAL_S"] = "2"
    os.environ["MONITOR_WEBHOOK_URL"] = ""
    os.environ.pop("ALERT_WEBHOOK_URL", None)
    os.environ["MONITOR_PROMETHEUS_PORT"] = "9109"
    os.environ["EVAL_REPORT_DIR"] = str(WORK_DIR / "eval_reports")
    os.environ["EVAL_BASELINE_PATH"] = str(WORK_DIR / "baseline.json")


def configure(args: argparse.Namespace, *, apply_env: bool = True) -> None:
    """初始化全局运行状态；导入模块时不做任何解析或写盘。"""
    global ARGS, RUN_ID, WORK_DIR, RECORDER, CASES
    ARGS = args
    RUN_ID = args.run_id
    if apply_env:
        # 相对 --work-dir 与正式运行时一样以项目根为基准。
        os.chdir(ROOT)
    WORK_DIR = pathlib.Path(
        args.work_dir or ROOT / "docs" / "reports" / "t8-runs" / RUN_ID
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if apply_env:
        _apply_environment()
    RECORDER = Recorder(args.budget_cny, WORK_DIR)
    CASES = CaseLog(WORK_DIR / "cases.jsonl")
    DEGRADATIONS.items.clear()
    if apply_env:
        _install_degradation_capture()


def _short(value: Any, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _call_ids(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_id": payload.get("user_id"),
        "conv_id": payload.get("conv_id"),
        "request_id": payload.get("request_id"),
    }


def _find_tool_trace(trace: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    for item in trace.get("tool_calls") or []:
        if item.get("tool_name") == name:
            return item
    return None


def _cleanup_user(
    user_id: str,
    conv_id: Optional[str] = None,
    *,
    delete_profile: bool = True,
    redis_client: Any = None,
    chroma_client: Any = None,
) -> None:
    """清理本次运行拥有的数据。

    ``conv_id`` 存在时只清理该会话；固定 CLI 用户必须传
    ``delete_profile=False``，避免删除其他会话或 HTTP 同名用户留下的画像。
    """
    try:
        owns_redis = redis_client is None
        if owns_redis:
            import redis

            client = redis.Redis.from_url(
                os.getenv(
                    "REDIS_URL",
                    "redis://:agentecommerce123@127.0.0.1:6380/0",
                ),
                decode_responses=True,
            )
        else:
            client = redis_client
        if conv_id:
            keys = [f"wm:{user_id}:{conv_id}", f"summary:{user_id}:{conv_id}"]
        else:
            keys = list(client.scan_iter(match=f"wm:{user_id}:*"))
            keys += list(client.scan_iter(match=f"summary:{user_id}:*"))
        if keys:
            client.delete(*keys)
        if owns_redis:
            client.close()
    except Exception:
        pass

    try:
        owns_chroma = chroma_client is None
        embedding_function = None
        if owns_chroma:
            import chromadb
            from memory.conversation_memory import _MemoryEmbeddingFunction

            chroma = chromadb.HttpClient(
                host=os.getenv("CHROMA_HOST", "127.0.0.1"),
                port=int(os.getenv("CHROMA_PORT", "8002")),
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            embedding_function = _MemoryEmbeddingFunction()
        else:
            chroma = chroma_client
        try:
            episodic = chroma.get_collection(
                "episodic", embedding_function=embedding_function
            )
            # 指定会话时只删除本次会话，避免误删同一用户的历史情景记忆。
            where = (
                {"$and": [{"user_id": user_id}, {"conv_id": conv_id}]}
                if conv_id
                else {"user_id": user_id}
            )
            episodic.delete(where=where)
        except Exception:
            pass
        if delete_profile:
            try:
                profiles = chroma.get_collection(
                    "user_profile", embedding_function=embedding_function
                )
                profiles.delete(ids=[f"user_profile:{user_id}"])
            except Exception:
                pass
    except Exception:
        pass


def _delete_docs(titles: Iterable[str]) -> None:
    import chromadb
    from mcp.knowledge_base import _ProjectEmbeddingFunction

    try:
        chroma = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            port=int(os.getenv("CHROMA_PORT", "8002")),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        collection = chroma.get_collection(
            "knowledge_base",
            embedding_function=_ProjectEmbeddingFunction(),
        )
        for title in titles:
            try:
                collection.delete(where={"title": title})
            except Exception:
                pass
    except Exception:
        pass


async def _wait_for_profile(user_id: str, timeout_s: float = 60.0) -> Optional[str]:
    import chromadb

    chroma = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "127.0.0.1"),
        port=int(os.getenv("CHROMA_PORT", "8002")),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            collection = chroma.get_collection("user_profile")
            result = await asyncio.to_thread(
                collection.get,
                ids=[f"user_profile:{user_id}"],
            )
            documents = result.get("documents") or []
            if documents:
                return documents[0]
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return None


async def _read_working_memory(user_id: str, conv_id: str) -> List[Dict[str, Any]]:
    import redis

    client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"),
        decode_responses=True,
    )
    raws = await asyncio.to_thread(
        client.lrange, f"wm:{user_id}:{conv_id}", 0, -1
    )
    summary = await asyncio.to_thread(
        client.get, f"summary:{user_id}:{conv_id}"
    )
    client.close()
    messages = [json.loads(raw) for raw in reversed(raws)]
    return messages, summary


async def _http_json(client: Any, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    response = await client.request(method, path, **kwargs)
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = response.json()
    else:
        # Prometheus 文本必须完整保留，否则无法解析业务指标。
        body = {"raw": response.text}
    return {"status": response.status_code, "body": body, "headers": dict(response.headers)}


async def run_acceptance() -> Dict[str, Any]:
    import httpx
    import uvicorn

    from core.skill_loader import SkillManager

    routes: Dict[str, Dict[str, Any]] = {}

    def record_route(name: str, result: Dict[str, Any]) -> None:
        routes[name] = {
            "status": result["status"],
            "body_summary": _short(result.get("body", {}), 500),
        }

    _patch_clients()
    import api.main as api_main

    config = uvicorn.Config(
        api_main.app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://{ARGS.host}:{ARGS.port}"
    deadline = time.time() + 45
    ready = False
    while time.time() < deadline:
        try:
            probe = httpx.get(f"{base}/health", timeout=2)
            if probe.status_code == 200:
                ready = True
                break
        except Exception:
            await asyncio.sleep(0.5)
    if not ready:
        server.should_exit = True
        thread.join(timeout=30)
        raise RuntimeError("uvicorn 未在 45 秒内就绪")

    marker = RUN_ID.upper().replace("_", "-")
    users = {
        "general": f"t8_{RUN_ID}_general",
        "technical": f"t8_{RUN_ID}_technical",
        "billing": f"t8_{RUN_ID}_billing",
        "rag": f"t8_{RUN_ID}_rag",
        "memory": f"t8_{RUN_ID}_memory",
        "memory_other": f"t8_{RUN_ID}_memory_other",
        "multi": f"t8_{RUN_ID}_multi",
        "escalation": f"t8_{RUN_ID}_escalation",
        "refund": f"t8_{RUN_ID}_refund",
        "eval": f"t8_{RUN_ID}_eval",
    }
    convs = {name: f"t8_{RUN_ID}_conv_{name}" for name in users}
    # V5 使用独立会话，避免 V4 已有 4 条消息影响“压缩后保留 5 条”的断言。
    convs["memory_v5"] = f"t8_{RUN_ID}_conv_memory_v5"
    added_titles: List[str] = []
    cli_conv: Optional[str] = None

    try:
        async with httpx.AsyncClient(base_url=base, timeout=180.0) as client:
            health = await _http_json(client, "GET", "/health")
            record_route("GET /health", health)

            skills = await _http_json(client, "GET", "/skills")
            record_route("GET /skills", skills)

            # V3 前置：写入本次唯一事实、相关资料与干扰资料。
            knowledge_docs = [
                {
                    "title": f"星桥计划总则 {marker}",
                    "content": (
                        f"测试服务星桥计划，规则编号 {marker}-RULE，"
                        "特殊办理期限 17 个工作日。适用对象仅限本次合成测试用户。"
                    ),
                },
                {
                    "title": f"星桥计划材料 {marker}",
                    "content": (
                        f"星桥计划申请材料包括身份核验单和测试服务确认函。"
                        f"规则编号 {marker}-RULE，材料齐全后进入审核。"
                    ),
                },
                {
                    "title": f"星桥计划时限 {marker}",
                    "content": (
                        "星桥计划的特殊办理期限为 17 个工作日，"
                        "不含提交当天与法定节假日。"
                    ),
                },
                {
                    "title": f"星桥计划常见问题 {marker}",
                    "content": (
                        f"问：星桥计划多久办完？答：特殊办理期限 17 个工作日。"
                        f"规则编号 {marker}-RULE。"
                    ),
                },
                {
                    "title": f"极光计划说明 {marker}",
                    "content": "极光计划的办理期限是 3 个工作日，规则编号 AURORA-01。",
                },
                {
                    "title": f"潮汐计划说明 {marker}",
                    "content": "潮汐计划的办理期限是 9 个工作日，规则编号 TIDE-02。",
                },
                {
                    "title": f"远山计划说明 {marker}",
                    "content": "远山计划的办理期限是 5 个工作日，规则编号 HILL-03。",
                },
                {
                    "title": f"清风计划说明 {marker}",
                    "content": "清风计划的办理期限是 12 个工作日，规则编号 BREEZE-04。",
                },
            ]
            added_titles = [doc["title"] for doc in knowledge_docs]
            added = await _http_json(
                client,
                "POST",
                "/knowledge/add",
                json={"documents": knowledge_docs},
            )
            record_route("POST /knowledge/add", added)

            upload_title = f"t8_upload_{RUN_ID}"
            upload_text = (
                f"测试上传资料 {marker}：潮汐上传计划的参考编号是 {marker}-UPLOAD，"
                "仅用于本次上传接口验收。"
            )
            uploaded = await _http_json(
                client,
                "POST",
                "/knowledge/upload",
                files={
                    "file": (
                        f"{upload_title}.txt",
                        upload_text.encode("utf-8"),
                        "text/plain",
                    )
                },
            )
            added_titles.append(upload_title)
            record_route("POST /knowledge/upload", uploaded)

            stats = await _http_json(client, "GET", "/knowledge/stats")
            record_route("GET /knowledge/stats", stats)

            search = await _http_json(
                client,
                "POST",
                "/search",
                params={
                    "query": f"星桥计划 {marker} 规则编号 特殊办理期限",
                    "top_k": 3,
                },
            )
            record_route("POST /search", search)
            search_results = search["body"].get("results") or []
            CASES.add(
                case="V3-search",
                input={"query": f"星桥计划 {marker} 规则编号 特殊办理期限", "top_k": 3},
                assertions={
                    "http_200": search["status"] == 200,
                    "reranked_true": search["body"].get("reranked") is True,
                    "has_unique_fact": any(
                        "17 个工作日" in str(item.get("content", ""))
                        or f"{marker}-RULE" in str(item.get("content", ""))
                        for item in search_results
                    ),
                    "results_non_empty": bool(search_results),
                },
                output_summary=_short(search_results),
                required=[
                    "http_200",
                    "reranked_true",
                    "has_unique_fact",
                    "results_non_empty",
                ],
            )

            # V2 通用对话。
            general = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": "你们的标准配送一般多久送达？",
                    "user_id": users["general"],
                    "conv_id": convs["general"],
                },
            )
            record_route("POST /chat", general)
            general_body = general["body"]
            CASES.add(
                case="V2-general",
                input={"message": "你们的标准配送一般多久送达？"},
                assertions={
                    "http_200": general["status"] == 200,
                    "agent_general": general_body.get("agent_type") == "general",
                    "response_non_empty": bool(general_body.get("response", "").strip()),
                },
                output_summary=_short(general_body.get("response", "")),
                ids=_call_ids(general_body),
                required=["http_200", "agent_general", "response_non_empty"],
            )

            # V2 技术对话：显式要求使用真实工具。
            technical = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": (
                        "Windows 上调用你们的接口一直返回 401。"
                        "请先调用 lookup_error_code 工具核对 401 的含义，"
                        "再给出三步排查建议。"
                    ),
                    "user_id": users["technical"],
                    "conv_id": convs["technical"],
                },
            )
            record_route("POST /chat", technical)
            technical_body = technical["body"]
            technical_trace = await _http_json(
                client,
                "GET",
                f"/trace/tool/{technical_body.get('request_id', '')}",
            )
            trace_calls = (technical_trace["body"].get("trace") or {}).get("tool_calls") or []
            lookup_trace = next(
                (item for item in trace_calls if item.get("tool_name") == "lookup_error_code"),
                None,
            )
            CASES.add(
                case="V2-technical",
                input={
                    "message": (
                        "Windows 上调用你们的接口一直返回 401。"
                        "请先调用 lookup_error_code 工具核对 401 的含义，再给出三步排查建议。"
                    ),
                    "intent": technical_body.get("intent"),
                    "primary_agent": technical_body.get("primary_agent"),
                    "supporting_agents": technical_body.get("supporting_agents"),
                },
                assertions={
                    "http_200": technical["status"] == 200,
                    # 路由可能把 technical 作为协作 Agent 而不是最终主 Agent；
                    # 只要它真实参与了本次请求，技术排查就成立。
                    "agent_technical": technical_body.get("agent_type") == "technical",
                    "technical_participated": "technical"
                    in (technical_body.get("agent_types") or []),
                    "tool_used": "lookup_error_code" in (technical_body.get("tools_used") or []),
                    "tool_trace_success": bool(
                        lookup_trace and lookup_trace.get("success")
                    ),
                    "tool_input_401": bool(
                        lookup_trace
                        and "401" in str((lookup_trace.get("input") or {}).get("error_code"))
                    ),
                    "response_mentions_auth": any(
                        word in technical_body.get("response", "")
                        for word in ("认证", "Token", "token", "401")
                    ),
                },
                output_summary={
                    "response": _short(technical_body.get("response", "")),
                    "agent_type": technical_body.get("agent_type"),
                    "agent_types": technical_body.get("agent_types"),
                    "routing_reason": technical_body.get("routing_reason"),
                },
                tool_evidence=_short(trace_calls),
                ids=_call_ids(technical_body),
                required=[
                    "http_200",
                    "technical_participated",
                    "tool_used",
                    "tool_trace_success",
                    "tool_input_401",
                    "response_mentions_auth",
                ],
            )

            # V2 账单对话：合成订单号、日期、金额。
            billing_message = (
                f"订单号 {marker}-ORD1，支付日期 2026-10-10，金额 128 元。"
                "请先调用 check_billing_fields 核验字段，再说明能否直接退款。"
            )
            billing = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": billing_message,
                    "user_id": users["billing"],
                    "conv_id": convs["billing"],
                },
            )
            record_route("POST /chat", billing)
            billing_body = billing["body"]
            billing_trace = await _http_json(
                client,
                "GET",
                f"/trace/tool/{billing_body.get('request_id', '')}",
            )
            billing_calls = (billing_trace["body"].get("trace") or {}).get("tool_calls") or []
            billing_tool_calls = [
                item for item in billing_calls
                if item.get("tool_name") == "check_billing_fields"
            ]
            # 模型可能先带多余参数被拒，再按 schema 重试；以任意一次成功调用为准。
            fields_trace = next(
                (item for item in billing_tool_calls if item.get("success")),
                None,
            )
            rejected_trace = next(
                (item for item in billing_tool_calls if not item.get("success")),
                None,
            )
            CASES.add(
                case="V2-billing",
                input={"message": billing_message},
                assertions={
                    "http_200": billing["status"] == 200,
                    "agent_billing": billing_body.get("agent_type") == "billing",
                    "tool_used": "check_billing_fields" in (billing_body.get("tools_used") or []),
                    "tool_trace_success": bool(fields_trace),
                    "invalid_arg_retry": bool(rejected_trace and fields_trace),
                    "entities_order_id": bool((billing_body.get("entities") or {}).get("order_id")),
                    "entities_amount": bool((billing_body.get("entities") or {}).get("amount")),
                    "entities_date": bool((billing_body.get("entities") or {}).get("date")),
                    "no_refund_claim": not has_affirmative_refund_success_claim(
                        billing_body.get("response", "")
                    ),
                },
                output_summary=_short(billing_body.get("response", "")),
                tool_evidence=_short(billing_calls),
                ids=_call_ids(billing_body),
                required=[
                    "http_200",
                    "agent_billing",
                    "tool_used",
                    "tool_trace_success",
                    "entities_order_id",
                    "entities_amount",
                    "entities_date",
                    "no_refund_claim",
                ],
            )

            # V3 HTTP 知识问答。
            rag_message = (
                f"请调用 search_knowledge_base（top_k 设为 3）检索知识库，"
                "测试服务星桥计划的规则编号和特殊办理期限"
                "分别是什么？不要凭常识猜测。"
            )
            rag = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": rag_message,
                    "user_id": users["rag"],
                    "conv_id": convs["rag"],
                },
            )
            record_route("POST /chat", rag)
            rag_body = rag["body"]
            rag_trace = await _http_json(
                client,
                "GET",
                f"/trace/tool/{rag_body.get('request_id', '')}",
            )
            rag_calls = (rag_trace["body"].get("trace") or {}).get("tool_calls") or []
            knowledge_traces = [
                item for item in rag_calls
                if item.get("tool_name") == "search_knowledge_base"
            ]
            knowledge_trace = knowledge_traces[0] if knowledge_traces else None
            # 同一轮可能检索多次；只要有一次真实重排即可满足 V3。
            rag_reranked = any(item.get("reranked") for item in knowledge_traces)
            rag_response = rag_body.get("response", "")
            CASES.add(
                case="V3-chat",
                input={"message": rag_message},
                assertions={
                    "http_200": rag["status"] == 200,
                    "knowledge_tool_used": "search_knowledge_base"
                    in (rag_body.get("tools_used") or []),
                    "knowledge_used_true": rag_body.get("knowledge_used") is True,
                    "trace_has_fragment": bool(knowledge_trace),
                    "reranked_true": rag_reranked,
                    "mentions_rule": f"{marker}-RULE" in rag_response,
                    "mentions_17_workdays": "17 个工作日" in rag_response
                    or "17个工作日" in rag_response,
                },
                output_summary=_short(rag_response),
                tool_evidence=_short(rag_calls),
                ids=_call_ids(rag_body),
                required=[
                    "http_200",
                    "knowledge_tool_used",
                    "knowledge_used_true",
                    "trace_has_fragment",
                    "reranked_true",
                    "mentions_rule",
                    "mentions_17_workdays",
                ],
            )

            # V4 同会话两轮上下文与后台画像。
            device = f"{marker}-DEVICE"
            first = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": (
                        f"我的测试设备是 {device}，偏好中文短步骤。请记住。"
                    ),
                    "user_id": users["memory"],
                    "conv_id": convs["memory"],
                },
            )
            second = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": "按刚才的设备继续说明如何排查连接超时。",
                    "user_id": users["memory"],
                    "conv_id": convs["memory"],
                },
            )
            record_route("POST /chat", first)
            record_route("POST /chat", second)
            second_body = second["body"]
            mem_messages, mem_summary = await _read_working_memory(
                users["memory"], convs["memory"]
            )
            profile_doc = await _wait_for_profile(users["memory"], timeout_s=60.0)
            profile_data: Dict[str, Any] = {}
            if profile_doc:
                try:
                    profile_data = json.loads(profile_doc)
                except Exception:
                    profile_data = {"raw": profile_doc}
            profile_text = json.dumps(profile_data, ensure_ascii=False) if profile_data else ""
            CASES.add(
                case="V4-context-and-profile",
                input={
                    "turn1": f"我的测试设备是 {device}，偏好中文短步骤。请记住。",
                    "turn2": "按刚才的设备继续说明如何排查连接超时。",
                },
                assertions={
                    "http_200": first["status"] == 200 and second["status"] == 200,
                    "second_uses_device": device in second_body.get("response", "")
                    or "该设备" in second_body.get("response", "")
                    or "刚才" in second_body.get("response", ""),
                    "redis_roles_ordered": [m.get("role") for m in mem_messages]
                    == ["user", "assistant", "user", "assistant"],
                    "profile_stored": profile_doc is not None,
                    "profile_has_preference": any(
                        word in profile_text
                        for word in ("中文", "短步骤", device, "设备")
                    ),
                },
                output_summary=_short(second_body.get("response", "")),
                profile_evidence=_short(profile_text),
                ids=_call_ids(second_body),
                required=[
                    "http_200",
                    "second_uses_device",
                    "redis_roles_ordered",
                    "profile_stored",
                    "profile_has_preference",
                ],
            )

            # V6 技术 + 账单复合问题，真实 Composer。
            multi_message = (
                "接口 401，同时本次扣款 128 元、预计 99 元。"
                "请分别给排查和核对建议，并用 compare_amounts 计算差额。"
            )
            multi = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": multi_message,
                    "user_id": users["multi"],
                    "conv_id": convs["multi"],
                },
            )
            record_route("POST /chat", multi)
            multi_body = multi["body"]
            multi_trace = await _http_json(
                client,
                "GET",
                f"/trace/tool/{multi_body.get('request_id', '')}",
            )
            multi_calls = (multi_trace["body"].get("trace") or {}).get("tool_calls") or []
            agent_types = multi_body.get("agent_types") or []
            multi_response = multi_body.get("response", "")
            composer_calls = [
                call for call in RECORDER.calls if call["stage"] == "composer"
            ]
            CASES.add(
                case="V6-multi-agent-composer",
                input={"message": multi_message},
                assertions={
                    "http_200": multi["status"] == 200,
                    "two_agents": len(agent_types) >= 2,
                    "has_technical_and_billing": "technical" in agent_types
                    and "billing" in agent_types,
                    "composer_called": bool(composer_calls),
                    "mentions_401": "401" in multi_response or "认证" in multi_response,
                    "mentions_difference_29": "29" in multi_response,
                    "compare_tool_used": "compare_amounts"
                    in (multi_body.get("tools_used") or []),
                },
                output_summary=_short(multi_response),
                tool_evidence=_short(multi_calls),
                ids=_call_ids(multi_body),
                required=[
                    "http_200",
                    "two_agents",
                    "has_technical_and_billing",
                    "composer_called",
                    "mentions_401",
                    "mentions_difference_29",
                    "compare_tool_used",
                ],
            )

            # V7 转人工与直接退款请求。
            handoff = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": "我要转人工客服。",
                    "user_id": users["escalation"],
                    "conv_id": convs["escalation"],
                },
            )
            record_route("POST /chat", handoff)
            handoff_body = handoff["body"]
            refund = await _http_json(
                client,
                "POST",
                "/chat",
                json={
                    "message": (
                        f"请直接查询我的订单 {marker}-ORD2 并立即退款。"
                    ),
                    "user_id": users["refund"],
                    "conv_id": convs["refund"],
                },
            )
            record_route("POST /chat", refund)
            refund_body = refund["body"]
            refund_response = refund_body.get("response", "")
            CASES.add(
                case="V7-handoff-and-refund-boundary",
                input={
                    "handoff": "我要转人工客服。",
                    "refund": f"请直接查询我的订单 {marker}-ORD2 并立即退款。",
                },
                assertions={
                    "handoff_http_200": handoff["status"] == 200,
                    "handoff_escalated": handoff_body.get("escalated") is True,
                    "handoff_agent": handoff_body.get("agent_type") == "escalation",
                    "handoff_no_ticket_claim": "工单已创建" not in handoff_body.get("response", "")
                    and "已创建工单" not in handoff_body.get("response", ""),
                    "refund_http_200": refund["status"] == 200,
                    "refund_no_success_claim": not has_affirmative_refund_success_claim(
                        refund_response
                    ),
                    "refund_states_boundary": any(
                        word in refund_response
                        for word in ("无法", "不能", "需要人工", "核验", "审核")
                    ),
                },
                output_summary={
                    "handoff": _short(handoff_body.get("response", "")),
                    "refund": _short(refund_response),
                },
                ids={
                    "handoff": _call_ids(handoff_body),
                    "refund": _call_ids(refund_body),
                },
                required=[
                    "handoff_http_200",
                    "handoff_escalated",
                    "handoff_agent",
                    "handoff_no_ticket_claim",
                    "refund_http_200",
                    "refund_no_success_claim",
                    "refund_states_boundary",
                ],
            )

            # V8 受控知识工具失败：真实模型 + 失败工具，不写入业务存储。
            from agents.agent_orchestrator import AgentOrchestrator, Request
            from agents.tools import make_tool
            from mcp.tool_manager import MCPToolManager, Tool

            class _FailingKnowledge:
                async def search_handler(self, params, context):
                    raise RuntimeError("T8 controlled knowledge failure")

            failing_manager = MCPToolManager(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
                model=os.getenv("LLM_MODEL") or os.environ["ANTHROPIC_MODEL"],
            )

            def failing_fallback(params, context, error):
                return [
                    {
                        "title": "知识库降级结果",
                        "content": "知识库暂时不可用，未能完成语义检索。",
                        "score": 0.0,
                        "fallback": True,
                        "error": error,
                    }
                ]

            failing_manager.register(
                Tool(
                    name="knowledge_search",
                    description="受控失败的知识检索工具",
                    handler=_FailingKnowledge().search_handler,
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                    cache_ttl=0.0,
                    supports_rerank=True,
                    fallback=failing_fallback,
                )
            )
            skill_manager = SkillManager(
                root_dir=str(ROOT / "skills"),
                max_prompt_chars=int(
                    os.getenv("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")
                ),
            )
            skill_manager.load()
            failing_orchestrator = AgentOrchestrator(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
                model=os.getenv("LLM_MODEL") or os.environ["ANTHROPIC_MODEL"],
                skill_manager=skill_manager,
                rag_tool_manager=failing_manager,
            )
            failing_request = Request(
                message=(
                    "请检索知识库，查一下星桥计划的特殊办理期限是什么。"
                    "如果无法检索，就明确说无法核实，不要猜测。"
                ),
                user_id=users["rag"],
                conv_id=f"t8_{RUN_ID}_conv_failing_tool",
            )
            failing_result = await failing_orchestrator.run(failing_request)
            failing_response = failing_result.response
            failing_traces = [
                trace
                for trace in failing_result.tool_traces
                if trace.get("tool_name") == "search_knowledge_base"
            ]
            CASES.add(
                case="V8-controlled-knowledge-failure",
                input={"message": failing_request.message},
                assertions={
                    "tool_attempted": bool(failing_traces),
                    "tool_result_success_false": bool(
                        failing_traces
                        and failing_traces[-1].get("result_success") is False
                    ),
                    "knowledge_hits_zero": failing_result.knowledge_hits == 0,
                    "response_non_empty": bool(failing_response.strip()),
                    "response_admits_unverified": any(
                        word in failing_response
                        for word in ("无法核实", "无法确认", "暂时不可用", "不能核实")
                    ),
                },
                output_summary=_short(failing_response),
                tool_evidence=_short(failing_traces),
                required=[
                    "tool_attempted",
                    "tool_result_success_false",
                    "knowledge_hits_zero",
                    "response_non_empty",
                    "response_admits_unverified",
                ],
            )

            # V9 单条对话评测 + 真实 Judge。
            eval_case = {
                "question": "退款申请提交后多久能到账？",
                "user_id": users["eval"],
                "conv_id": convs["eval"],
            }
            evaluation = await _http_json(
                client,
                "POST",
                "/eval/run",
                json={
                    "intent_cases": [],
                    "dialog_cases": [eval_case],
                },
            )
            record_route("POST /eval/run", evaluation)
            eval_body = evaluation["body"]
            eval_results = eval_body.get("results") or []
            eval_scores = (eval_results[0].get("scores") if eval_results else {}) or {}
            gate_fields = ("relevance", "accuracy", "completeness", "helpfulness")
            report_files = sorted(
                (WORK_DIR / "eval_reports").glob("report_*.json")
            )
            report_payload: Dict[str, Any] = {}
            if report_files:
                try:
                    report_payload = json.loads(
                        report_files[-1].read_text(encoding="utf-8")
                    )
                except Exception:
                    report_payload = {}
            CASES.add(
                case="V9-eval-single-dialog",
                input={
                    "intent_cases": [],
                    "dialog_cases": [eval_case],
                },
                assertions={
                    "http_200": evaluation["status"] == 200,
                    "total_one": eval_body.get("total") == 1,
                    "grader_errors_zero": eval_body.get("grader_errors") == 0,
                    "scores_numeric_0_1": all(
                        isinstance(eval_scores.get(name), (int, float))
                        and 0.0 <= float(eval_scores[name]) <= 1.0
                        for name in gate_fields
                    ),
                    "report_written": bool(report_payload),
                    "judge_called": any(
                        call["stage"] == "judge" for call in RECORDER.calls
                    ),
                },
                output_summary={
                    "scores": eval_scores,
                    "response": _short(
                        (eval_results[0].get("metadata") or {}).get("response", "")
                        if eval_results
                        else ""
                    ),
                    "report_file": str(report_files[-1]) if report_files else "",
                },
                ids={"user_id": users["eval"], "conv_id": convs["eval"]},
                required=[
                    "http_200",
                    "total_one",
                    "grader_errors_zero",
                    "scores_numeric_0_1",
                    "report_written",
                    "judge_called",
                ],
            )

            # V5 真实摘要、合并、情景召回与用户隔离。
            from memory.conversation_memory import MemoryManager, MsgRole

            memory = MemoryManager(
                redis_url=os.getenv(
                    "REDIS_URL",
                    "redis://:agentecommerce123@127.0.0.1:6380/0",
                ),
                chroma_host=os.getenv("CHROMA_HOST", "127.0.0.1"),
                chroma_port=int(os.getenv("CHROMA_PORT", "8002")),
                chroma_path=str(ROOT / "data" / "chroma"),
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url=os.getenv("ANTHROPIC_BASE_URL"),
                model=os.getenv("LLM_MODEL") or os.environ["ANTHROPIC_MODEL"],
            )
            try:
                early_fact = f"我的会员等级是星桥金卡，测试设备是 {marker}-MEM"
                for index in range(15):
                    role = MsgRole.USER if index % 2 == 0 else MsgRole.ASSISTANT
                    content = (
                        f"第 {index + 1} 条固定测试消息。{early_fact}。"
                        if index == 0
                        else f"第 {index + 1} 条固定测试消息，继续讨论退款流程。"
                    )
                    await memory.add_message(
                        users["memory"], convs["memory_v5"], role, content
                    )
                for index in range(15, 25):
                    role = MsgRole.USER if index % 2 == 0 else MsgRole.ASSISTANT
                    await memory.add_message(
                        users["memory"],
                        convs["memory_v5"],
                        role,
                        f"第 {index + 1} 条固定测试消息，补充说明退款审核时效。",
                    )
                working, summary = await _read_working_memory(
                    users["memory"], convs["memory_v5"]
                )
                memory_ctx = await memory.get_context(
                    users["memory"],
                    convs["memory_v5"],
                    query="我的会员等级和测试设备是什么？",
                )
                new_ctx = await memory.get_context(
                    users["memory"],
                    f"t8_{RUN_ID}_conv_memory_new",
                    query="我的会员等级和测试设备是什么？",
                )
                other_ctx = await memory.get_context(
                    users["memory_other"],
                    f"t8_{RUN_ID}_conv_memory_other",
                    query="我的会员等级和测试设备是什么？",
                )
                import chromadb

                chroma = chromadb.HttpClient(
                    host=os.getenv("CHROMA_HOST", "127.0.0.1"),
                    port=int(os.getenv("CHROMA_PORT", "8002")),
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                episodic = chroma.get_collection("episodic")
                episodic_result = await asyncio.to_thread(
                    episodic.get,
                    where={"user_id": users["memory"]},
                )
                episodic_docs = episodic_result.get("documents") or []
                episodic_metas = episodic_result.get("metadatas") or []
                episodic_full = "\n".join(
                    str(meta.get("full_text", ""))
                    for meta in episodic_metas
                    if isinstance(meta, dict)
                )
                evidence_text = (
                    (summary or "")
                    + "\n"
                    + "\n".join(new_ctx.relevant_history)
                    + "\n"
                    + episodic_full
                )
                summary_calls = [
                    call for call in RECORDER.calls if call["stage"] == "summary"
                ]
                merge_calls = [
                    call for call in RECORDER.calls if call["stage"] == "summary_merge"
                ]
                CASES.add(
                    case="V5-summary-merge-recall-isolation",
                    input={
                        "fixture": "15 条固定消息触发首次压缩，再追加 10 条触发合并",
                        "early_fact": early_fact,
                        "recall_query": "我的会员等级和测试设备是什么？",
                    },
                    assertions={
                        "summary_generated": bool(summary),
                        "summary_calls": len(summary_calls),
                        "merge_calls": len(merge_calls),
                        "merge_model_called": bool(merge_calls),
                        "early_fact_retained": any(
                            word in evidence_text
                            for word in ("星桥金卡", f"{marker}-MEM", "MEM")
                        ),
                        "working_window_five": len(working) == 5,
                        "episodic_records": len(episodic_docs),
                        "same_user_new_conv_recall": bool(new_ctx.relevant_history),
                        "other_user_isolated": not other_ctx.relevant_history,
                    },
                    output_summary={
                        "summary": _short(summary or ""),
                        "episodic_count": len(episodic_docs),
                        "same_user_history": _short(new_ctx.relevant_history),
                        "other_user_history": _short(other_ctx.relevant_history),
                    },
                    ids={
                        "user_id": users["memory"],
                        "conv_id": convs["memory_v5"],
                    },
                    required=[
                        "summary_generated",
                        "summary_calls",
                        "merge_calls",
                        "merge_model_called",
                        "early_fact_retained",
                        "working_window_five",
                        "episodic_records",
                        "same_user_new_conv_recall",
                        "other_user_isolated",
                    ],
                )
            finally:
                await memory.close()

            # V10 Skills 重载、Trace、监控与指标。
            reloaded = await _http_json(client, "POST", "/skills/reload")
            record_route("POST /skills/reload", reloaded)

            await asyncio.sleep(2.5)
            monitor = await _http_json(client, "GET", "/monitor")
            record_route("GET /monitor", monitor)
            metrics = await _http_json(client, "GET", "/metrics")
            record_route("GET /metrics", metrics)
            metrics_text = (
                metrics["body"].get("raw", "")
                if isinstance(metrics["body"], dict)
                else ""
            )
            executions = {
                line.split('{agent="', 1)[1].split('"', 1)[0]: float(line.rsplit(" ", 1)[1])
                for line in metrics_text.splitlines()
                if line.startswith("agent_executions_total{agent=\"")
            }
            latencies = {
                line.split('{agent="', 1)[1].split('"', 1)[0]: float(line.rsplit(" ", 1)[1])
                for line in metrics_text.splitlines()
                if line.startswith("agent_avg_latency_ms{agent=\"")
            }
            recent = await _http_json(client, "GET", "/trace/tools", params={"limit": 20})
            record_route("GET /trace/tools", recent)
            rag_trace_route: Dict[str, Any] = {"status": 0, "body": {}}
            if rag_body.get("request_id"):
                rag_trace_route = await _http_json(
                    client,
                    "GET",
                    f"/trace/tool/{rag_body['request_id']}",
                )
                record_route(
                    f"GET /trace/tool/{{request_id}}", rag_trace_route
                )
            CASES.add(
                case="V10-routes-monitor-metrics",
                input={"routes": sorted(routes)},
                assertions={
                    "skills_reload_count_3": reloaded["body"].get("count") == 3,
                    "monitor_has_stats": "agent_stats" in monitor["body"]
                    and "tool_stats" in monitor["body"],
                    "metrics_http_200": metrics["status"] == 200,
                    "agent_executions_positive": any(
                        value > 0 for value in executions.values()
                    ),
                    "agent_latency_present": bool(latencies),
                    "trace_recent_non_empty": bool(recent["body"].get("items")),
                    "trace_by_request_found": (
                        rag_trace_route["body"].get("found") is True
                        if rag_body.get("request_id")
                        else False
                    ),
                },
                output_summary={
                    "executions": executions,
                    "latencies": latencies,
                    "monitor_agents": list(
                        (monitor["body"].get("agent_stats") or {}).keys()
                    ),
                },
                required=[
                    "skills_reload_count_3",
                    "monitor_has_stats",
                    "metrics_http_200",
                    "agent_executions_positive",
                    "agent_latency_present",
                    "trace_recent_non_empty",
                    "trace_by_request_found",
                ],
            )

            # V10 真实 CLI。每轮使用唯一 conv_id，清理只触及本次会话。
            cli_conv = str(uuid.uuid4())
            cli_uuid = uuid.UUID(cli_conv)
            original_uuid4 = api_main.uuid.uuid4
            api_main.uuid.uuid4 = lambda: cli_uuid
            cli_inputs = iter(
                [
                    f"请检索知识库，测试服务星桥计划的规则编号是什么？",
                    "quit",
                ]
            )
            original_input = builtins.input
            builtins.input = lambda prompt="": next(cli_inputs)
            cli_output = io.StringIO()
            cli_error: Optional[str] = None
            try:
                with contextlib.redirect_stdout(cli_output):
                    await api_main._cli()
            except Exception as exc:
                cli_error = f"{type(exc).__name__}: {exc}"
            finally:
                builtins.input = original_input
                api_main.uuid.uuid4 = original_uuid4
            cli_text = cli_output.getvalue()
            CASES.add(
                case="V10-cli-knowledge-and-quit",
                input={
                    "message": "请检索知识库，测试服务星桥计划的规则编号是什么？",
                    "command": "quit",
                },
                assertions={
                    "no_exception": cli_error is None,
                    "quit_exited": "再见" in cli_text,
                    "mentions_rule": f"{marker}-RULE" in cli_text,
                },
                output_summary=_short(cli_text),
                ids={"user_id": "cli_user", "conv_id": cli_conv},
                required=["no_exception", "quit_exited", "mentions_rule"],
                error=cli_error,
            )

            final_health = await _http_json(client, "GET", "/health")
            record_route("GET /health", final_health)

            # V11 停机：请求退出并等待 lifespan 收尾。
            server.should_exit = True
            thread.join(timeout=30)
            shutdown = {
                "thread_alive": thread.is_alive(),
                "orchestrator_cleared": api_main._orchestrator is None,
                "memory_cleared": api_main._memory is None,
                "monitor_cleared": api_main._monitor is None,
            }
            CASES.add(
                case="V11-shutdown",
                input={"action": "uvicorn server.should_exit"},
                assertions={
                    **shutdown,
                    # thread_alive=False 才是正确结果，因此单独定义必需断言。
                    "thread_stopped": not shutdown["thread_alive"],
                },
                required=[
                    "thread_stopped",
                    "orchestrator_cleared",
                    "memory_cleared",
                    "monitor_cleared",
                ],
            )

            return {
                "ready": ready,
                "marker": marker,
                "users": users,
                "convs": convs,
                "cli_conv": cli_conv,
                "added_titles": added_titles,
                "routes": routes,
                "executions": executions,
                "latencies": latencies,
                "shutdown": shutdown,
                "cases": CASES.items,
            }
    finally:
        # 数据清理：只删除本次随机用户/会话、本次知识标题与本次 CLI 会话。
        if not server.should_exit:
            server.should_exit = True
            thread.join(timeout=30)
        for name, user_id in users.items():
            _cleanup_user(user_id, convs.get(name), delete_profile=True)
        _cleanup_user(users["memory"], convs["memory_v5"], delete_profile=True)
        if cli_conv:
            # cli_user 是共享身份，画像可能来自其他会话或 HTTP 调用，不能删除。
            _cleanup_user("cli_user", cli_conv, delete_profile=False)
        _delete_docs(added_titles)
        for real_client in _REAL_CLIENTS:
            try:
                await real_client.close()
            except Exception:
                pass
        _restore_clients()


# 验收必须真实覆盖的模型阶段；缺失任一阶段都不能签收。
REQUIRED_MODEL_STAGES = (
    "intent",
    "query_rewrite",
    "rerank",
    "agent",
    "composer",
    "summary",
    "summary_merge",
    "profile",
    "judge",
)
# 结构化阶段必须返回可用文本，不能只有 thinking 或空内容。
STRUCTURED_MODEL_STAGES = (
    "intent",
    "query_rewrite",
    "rerank",
    "composer",
    "summary",
    "summary_merge",
    "profile",
    "judge",
)


def _usable_stage_call(call: Dict[str, Any], stage: str) -> bool:
    if call.get("failed") or call.get("stop_reason") == "max_tokens":
        return False
    block_types = call.get("content_block_types") or []
    if stage in STRUCTURED_MODEL_STAGES:
        return "text" in block_types and bool(
            str(call.get("text_preview", "")).strip()
        )
    return bool(block_types)


def validate_calls(
    calls: Iterable[Dict[str, Any]],
    degradations: Iterable[Any] = (),
) -> List[Dict[str, Any]]:
    """把调用记录转为机器可读的失败原因；无原因即通过。"""
    calls = list(calls)
    reasons: List[Dict[str, Any]] = []
    for call in calls:
        if call.get("failed"):
            reasons.append(
                {
                    "code": "model_request_failed",
                    "stage": call.get("stage"),
                    "error": call.get("error"),
                }
            )
        if call.get("stop_reason") == "max_tokens":
            reasons.append(
                {
                    "code": "truncated_response",
                    "stage": call.get("stage"),
                }
            )
        if call.get("stage") == "unclassified":
            reasons.append({"code": "stage_unclassified"})
        if call.get("estimate_exceeded"):
            reasons.append(
                {
                    "code": "cost_estimate_exceeded",
                    "stage": call.get("stage"),
                    "detail": call["estimate_exceeded"],
                }
            )
    for stage in REQUIRED_MODEL_STAGES:
        stage_calls = [call for call in calls if call.get("stage") == stage]
        if not any(_usable_stage_call(call, stage) for call in stage_calls):
            reasons.append(
                {
                    "code": "required_stage_missing",
                    "stage": stage,
                }
            )
    for item in degradations:
        reasons.append({"code": "degradation", "detail": item})
    return reasons


def validate_run(
    cases: Iterable[Dict[str, Any]],
    calls: Iterable[Dict[str, Any]],
    degradations: Iterable[Any] = (),
    *,
    budget_cny: Optional[float] = None,
    spent_cny: Optional[float] = None,
) -> Dict[str, Any]:
    """统一判定用例必需断言、模型调用与费用，返回可序列化结果。"""
    cases = list(cases)
    reasons: List[Dict[str, Any]] = []
    failed_cases = [
        {
            "case": item.get("case"),
            "required_failed": item.get("required_failed") or [],
            "required_missing": item.get("required_missing") or [],
        }
        for item in cases
        if item.get("status") != "pass"
    ]
    for item in failed_cases:
        reasons.append({"code": "case_required_failed", **item})
    reasons.extend(validate_calls(calls, degradations))
    if (
        budget_cny is not None
        and spent_cny is not None
        and spent_cny > budget_cny + 1e-9
    ):
        reasons.append(
            {
                "code": "budget_exceeded",
                "budget_cny": budget_cny,
                "spent_cny": round(spent_cny, 6),
            }
        )
    return {
        "ok": not reasons,
        "reasons": reasons,
        "failed_cases": failed_cases,
        "case_count": len(cases),
    }


def main(argv: Optional[List[str]] = None) -> int:
    started = time.time()
    if RECORDER is None or CASES is None:
        configure(_parse_args(argv))
    assert RECORDER is not None and CASES is not None
    try:
        result = asyncio.run(run_acceptance())
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        failure = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "recorder": RECORDER.summary(),
        }
        (WORK_DIR / "summary.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1

    validation = validate_run(
        CASES.items,
        RECORDER.calls,
        DEGRADATIONS.items,
        budget_cny=RECORDER.budget_cny,
        spent_cny=RECORDER.spent_cny,
    )
    summary = {
        "status": "passed" if validation["ok"] else "failed",
        "verdict": "PASS" if validation["ok"] else "FAIL",
        "run_id": RUN_ID,
        "duration_s": round(time.time() - started, 1),
        **result,
        "validation": validation,
        "degradations": list(DEGRADATIONS.items),
        "recorder": RECORDER.summary(),
    }
    (WORK_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (WORK_DIR / "T8-usage.json").write_text(
        json.dumps(RECORDER.summary(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failed = [item["case"] for item in CASES.items if item.get("status") != "pass"]
    print("run_id", RUN_ID)
    print("cases", len(CASES.items), "failed", failed)
    print("model_requests", len(RECORDER.calls))
    print("spent_cny_conservative", round(RECORDER.spent_cny, 6))
    print("work_dir", WORK_DIR)
    print("verdict", summary["verdict"])
    if not validation["ok"]:
        print(
            "failure_reasons",
            json.dumps(validation["reasons"], ensure_ascii=False),
        )
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
