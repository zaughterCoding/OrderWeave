"""Online performance monitoring for agents and tools.

The monitor samples the stats already maintained by the orchestrator and tool
manager, raises threshold alerts, feeds routing penalties back to the
orchestrator, and optionally exposes Prometheus metrics or a webhook.
"""
import asyncio
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import httpx
from prometheus_client import Counter, Gauge, start_http_server

logger = logging.getLogger(__name__)


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity: Severity
    metric: str
    message: str
    value: float
    threshold: float
    ts: str = field(default_factory=lambda: datetime.now().isoformat())
    resolved: bool = False


@dataclass
class Suggestion:
    """An actionable optimization suggestion."""

    title: str
    detail: str
    action: str
    priority: int


class AnomalyDetector:
    """Sliding-window Z-score anomaly detection."""

    def __init__(self, window: int = 60, sensitivity: float = 2.5):
        self._window = window
        self._sensitivity = sensitivity
        self._history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """Record one point and return anomaly details when the z-score exceeds the threshold."""
        buf = self._history[metric]
        buf.append(value)

        if len(buf) < self._window // 2:
            return None

        mean = statistics.mean(buf)
        stdev = statistics.stdev(buf) if len(buf) > 1 else 0.0
        if stdev == 0:
            return None

        z = abs(value - mean) / stdev
        if z > self._sensitivity:
            return {
                "metric": metric,
                "value": value,
                "mean": mean,
                "z_score": round(z, 2),
                "severity": "high" if z > self._sensitivity * 1.5 else "medium",
            }
        return None


class PerformanceMonitor:
    """Sample online agent and tool statistics and feed routing penalties back."""

    THRESHOLDS = {
        "agent_success_rate": (0.90, Severity.ERROR, "less_than"),
        "tool_success_rate": (0.95, Severity.WARNING, "less_than"),
        "agent_avg_ms": (3000, Severity.WARNING, "greater_than"),
        "tool_avg_ms": (5000, Severity.ERROR, "greater_than"),
    }

    def __init__(
        self,
        orchestrator,
        tool_manager,
        interval_s: float = 10.0,
        webhook_url: Optional[str] = None,
        prometheus_port: Optional[int] = None,
        prometheus_registry: Optional[Any] = None,
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._interval = interval_s
        self._webhook = webhook_url
        self._detector = AnomalyDetector()
        self._prometheus_registry = prometheus_registry
        self._execution_totals: Dict[str, int] = {}

        self._alerts: List[Alert] = []
        self._suggestions: List[Suggestion] = []
        self._active = False
        self._task: Optional[asyncio.Task] = None

        self._prom: Dict[str, Any] = {}
        if prometheus_port:
            self._setup_prometheus(prometheus_port)

    def _setup_prometheus(self, port: int) -> None:
        registry = self._prometheus_registry
        metric_kwargs: Dict[str, Any] = (
            {"registry": registry} if registry is not None else {}
        )
        self._prom = {
            "agent_success_rate": Gauge(
                "agent_success_rate",
                "Agent success rate",
                ["agent"],
                **metric_kwargs,
            ),
            # Only a current average is available per agent, so a Gauge avoids
            # pretending that polling produces request latency samples.
            "agent_avg_latency_ms": Gauge(
                "agent_avg_latency_ms",
                "Average agent latency in milliseconds",
                ["agent"],
                **metric_kwargs,
            ),
            "tool_success_rate": Gauge(
                "tool_success_rate",
                "Tool success rate",
                ["tool"],
                **metric_kwargs,
            ),
            "agent_executions_total": Counter(
                "agent_executions_total",
                "Cumulative agent executions observed by the monitor",
                ["agent"],
                **metric_kwargs,
            ),
        }
        if registry is not None:
            start_http_server(port, registry=registry)
        else:
            start_http_server(port)
        logger.info(f"Prometheus 已启动: :{port}")

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"Monitor 已启动，采集间隔 {self._interval}s")

    async def stop(self) -> None:
        self._active = False
        task = self._task
        self._task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while self._active:
            try:
                await self._collect()
            except Exception as ex:
                logger.error(f"Monitor 采集异常: {ex}")
            await asyncio.sleep(self._interval)

    async def _collect(self) -> None:
        """Read current stats, detect anomalies, raise alerts and update routing."""
        agent_stats = self._orchestrator.get_stats()
        tool_stats = self._tool_manager.get_stats()
        routing_penalties: Dict[str, float] = {}

        for agent_key, stats in agent_stats.items():
            total = stats.get("total", 0) or 0
            try:
                total = int(total)
            except (TypeError, ValueError):
                total = 0
            if "agent_executions_total" in self._prom:
                delta = max(0, total - self._execution_totals.get(agent_key, 0))
                if delta:
                    self._prom["agent_executions_total"].labels(
                        agent=agent_key
                    ).inc(delta)
                self._execution_totals[agent_key] = total

            success_rate = stats.get("success_rate")
            avg_ms = stats.get("avg_ms")
            if avg_ms is None:
                avg_ms = stats.get("avg_latency_ms")
            if success_rate is None or avg_ms is None:
                logger.warning(f"忽略字段不完整的 Agent 统计: {agent_key}")
                continue
            success_rate = float(success_rate)
            avg_ms = float(avg_ms)

            for metric, value in [
                ("agent_success_rate", success_rate),
                ("agent_avg_ms", avg_ms),
            ]:
                anomaly = self._detector.record(f"{metric}:{agent_key}", value)
                if anomaly:
                    logger.warning(
                        f"异常检测 [{agent_key}] {metric}={value:.3f} "
                        f"z={anomaly['z_score']}"
                    )

            self._check_threshold("agent_success_rate", success_rate, agent_key)
            self._check_threshold("agent_avg_ms", avg_ms, agent_key)

            if "agent_success_rate" in self._prom:
                self._prom["agent_success_rate"].labels(agent=agent_key).set(
                    success_rate
                )
                self._prom["agent_avg_latency_ms"].labels(agent=agent_key).set(
                    avg_ms
                )

            routing_penalties[agent_key] = self._routing_penalty(
                success_rate, avg_ms
            )

        for tool_name, stats in tool_stats.items():
            success_rate = stats.get("success_rate")
            avg_ms = stats.get("avg_latency_ms")
            if avg_ms is None:
                avg_ms = stats.get("avg_ms")
            if success_rate is None or avg_ms is None:
                logger.warning(f"忽略字段不完整的工具统计: {tool_name}")
                continue
            success_rate = float(success_rate)
            avg_ms = float(avg_ms)
            consecutive_fails = int(stats.get("consecutive_fails", 0) or 0)
            circuit_state = stats.get("circuit_state", "unknown")

            self._check_threshold("tool_success_rate", success_rate, tool_name)
            self._check_threshold("tool_avg_ms", avg_ms, tool_name)

            if "tool_success_rate" in self._prom:
                self._prom["tool_success_rate"].labels(tool=tool_name).set(
                    success_rate
                )

            if consecutive_fails >= 3:
                self._add_suggestion(
                    Suggestion(
                        title=f"工具 {tool_name} 连续失败 {consecutive_fails} 次",
                        detail=(
                            f"成功率 {success_rate:.1%}，平均延迟 {avg_ms:.0f}ms，"
                            f"熔断状态: {circuit_state}"
                        ),
                        action=(
                            "1. 检查工具依赖服务是否正常\n"
                            "2. 查看错误日志\n"
                            "3. 考虑增加超时时间或降级策略"
                        ),
                        priority=9,
                    )
                )

        updater = getattr(self._orchestrator, "update_routing_penalties", None)
        if updater:
            updater(routing_penalties)
        self._generate_routing_suggestions(agent_stats)

    @staticmethod
    def _routing_penalty(success_rate: float, avg_ms: float) -> float:
        """Map online performance to a routing penalty in the range 0-0.9."""
        penalty = 0.0
        if success_rate < 0.90:
            penalty += min(0.5, (0.90 - success_rate) * 2)
        if avg_ms > 3000:
            penalty += min(0.4, (avg_ms - 3000) / 10000)
        return min(penalty, 0.9)

    def _check_threshold(self, metric: str, value: float, label: str) -> None:
        if metric not in self.THRESHOLDS:
            return
        threshold, severity, operator = self.THRESHOLDS[metric]
        triggered = (
            operator == "less_than" and value < threshold
        ) or (operator == "greater_than" and value > threshold)
        alert_key = f"{metric}:{label}"
        if triggered:
            message = f"{label} 的 {metric} = {value:.3f}，阈值 {threshold}"
            existing = next(
                (
                    alert
                    for alert in self._alerts
                    if not alert.resolved and alert.metric == alert_key
                ),
                None,
            )
            if existing is not None:
                existing.severity = severity
                existing.message = message
                existing.value = value
                existing.threshold = threshold
                existing.ts = datetime.now().isoformat()
                logger.warning(f"[{severity.value.upper()}] {message}")
                return

            alert = Alert(
                severity=severity,
                metric=alert_key,
                message=message,
                value=value,
                threshold=threshold,
            )
            self._alerts.append(alert)
            logger.warning(f"[{severity.value.upper()}] {alert.message}")
            if self._webhook:
                asyncio.create_task(self._send_webhook(alert))
            return

        for alert in self._alerts:
            if not alert.resolved and alert.metric == alert_key:
                alert.resolved = True

    def _generate_routing_suggestions(
        self, agent_stats: Dict[str, Any]
    ) -> None:
        for agent_key, stats in agent_stats.items():
            success_rate = stats.get("success_rate")
            total = stats.get("total")
            if success_rate is None or total is None:
                continue
            if float(success_rate) < 0.85 and int(total) > 10:
                routing_score = float(stats.get("routing_score", 0.0) or 0.0)
                self._add_suggestion(
                    Suggestion(
                        title=f"Agent {agent_key} 成功率偏低",
                        detail=(
                            f"成功率 {float(success_rate):.1%}，"
                            f"路由评分 {routing_score:.3f}"
                        ),
                        action=(
                            "Orchestrator 的 _best_agent() 已自动降低该 Agent 的路由权重。\n"
                            "建议：1. 检查 system_prompt 是否需要优化\n"
                            "      2. 检查该类型问题的复杂度是否超出 Agent 能力\n"
                            "      3. 考虑增加同类型 Agent 实例"
                        ),
                        priority=8,
                    )
                )

    def _add_suggestion(self, suggestion: Suggestion) -> None:
        if not any(item.title == suggestion.title for item in self._suggestions):
            self._suggestions.append(suggestion)
            logger.info(f"优化建议 [P{suggestion.priority}]: {suggestion.title}")

    async def _send_webhook(self, alert: Alert) -> None:
        try:
            payload = asdict(alert)
            payload["severity"] = alert.severity.value
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._webhook, json=payload)  # type: ignore[arg-type]
        except Exception as ex:
            logger.error(f"Webhook 发送失败: {ex}")

    def summary(self) -> Dict[str, Any]:
        """Return the current monitoring summary for the API layer."""
        return {
            "agent_stats": self._orchestrator.get_stats(),
            "tool_stats": self._tool_manager.get_stats(),
            "active_alerts": [
                {**asdict(alert), "severity": alert.severity.value}
                for alert in self._alerts
                if not alert.resolved
            ][-10:],
            "suggestions": [
                {
                    "title": suggestion.title,
                    "action": suggestion.action,
                    "priority": suggestion.priority,
                }
                for suggestion in sorted(
                    self._suggestions, key=lambda item: -item.priority
                )[:5]
            ],
        }
