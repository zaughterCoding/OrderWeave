"""Tests for performance monitoring and response evaluation."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentType, OrchestratorResult
from core.intent_recognizer import IntentCategory
from evaluation.evaluator import EndToEndEvaluator, LLMJudge
from mcp.tool_manager import MCPToolManager, Tool, ToolStats
from monitor.performance_monitor import PerformanceMonitor


class FakeOrchestrator:
    def __init__(self, stats=None):
        self.stats = stats or {}
        self.penalties = None
        self.calls = 0

    def get_stats(self):
        return self.stats

    def update_routing_penalties(self, penalties):
        self.penalties = penalties

    async def run(self, request):
        self.calls += 1
        return OrchestratorResult(
            request_id=f"req-{self.calls}",
            response=f"已收到问题：{request.message}",
            agent_type=AgentType.GENERAL,
            intent=IntentCategory.QUERY,
        )


class FakeToolManager:
    def __init__(self, stats=None):
        self.stats = stats or {}

    def get_stats(self):
        return self.stats


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    async def create(self, **kwargs):
        text = self._responses.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)]
        )


class FakeAnthropic:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def test_monitor_collects_stats_and_feeds_routing_penalties():
    orchestrator = FakeOrchestrator(
        {
            "general_0": {
                "total": 20,
                "success": 16,
                "success_rate": 0.8,
                "avg_ms": 4000.0,
                "routing_score": 0.5,
            }
        }
    )
    tool_manager = FakeToolManager(
        {
            "search": {
                "total": 10,
                "success_rate": 0.8,
                "avg_latency_ms": 6000.0,
                "consecutive_fails": 3,
                "circuit_state": "open",
            }
        }
    )
    monitor = PerformanceMonitor(orchestrator, tool_manager, interval_s=3600)

    asyncio.run(monitor._collect())

    # 0.2 from success rate and 0.1 from the latency excess over 3000 ms.
    assert orchestrator.penalties == {"general_0": pytest.approx(0.3)}

    summary = monitor.summary()
    assert summary["agent_stats"] == orchestrator.stats
    assert summary["tool_stats"] == tool_manager.stats
    metrics = {alert["metric"] for alert in summary["active_alerts"]}
    assert metrics == {
        "agent_success_rate:general_0",
        "agent_avg_ms:general_0",
        "tool_success_rate:search",
        "tool_avg_ms:search",
    }
    titles = [suggestion["title"] for suggestion in summary["suggestions"]]
    assert any("连续失败 3 次" in title for title in titles)
    assert any("成功率偏低" in title for title in titles)
    assert all(isinstance(alert["severity"], str) for alert in summary["active_alerts"])


def test_monitor_accepts_field_aliases_and_skips_incomplete_stats():
    orchestrator = FakeOrchestrator(
        {
            "general_0": {"success_rate": 0.99, "avg_latency_ms": 100.0},
            "incomplete": {"success_rate": 0.5},
        }
    )
    tool_manager = FakeToolManager(
        {
            "search": {"success_rate": 0.99, "avg_ms": 200.0},
            "incomplete": {"avg_ms": 1.0},
        }
    )
    monitor = PerformanceMonitor(orchestrator, tool_manager, interval_s=3600)

    asyncio.run(monitor._collect())

    assert orchestrator.penalties == {"general_0": 0.0}
    assert monitor.summary()["active_alerts"] == []


def test_monitor_start_and_stop():
    async def scenario():
        monitor = PerformanceMonitor(
            FakeOrchestrator(), FakeToolManager(), interval_s=3600
        )
        await monitor.start()
        assert monitor._active is True
        assert monitor._task is not None
        await monitor.stop()
        assert monitor._active is False
        assert monitor._task is None

    asyncio.run(scenario())


def test_judge_requires_every_score_and_reports_failure():
    async def scenario():
        good_client = FakeAnthropic(
            [
                json.dumps(
                    {
                        "relevance": 0.8,
                        "accuracy": 0.8,
                        "completeness": 0.8,
                        "helpfulness": 0.8,
                    }
                )
            ]
        )
        good = await LLMJudge(good_client, "fake-model").judge("q", "a")
        assert good.judge_failed is False
        assert good.overall == pytest.approx(0.8)

        bad_client = FakeAnthropic(
            ['{"relevance": 0.9, "accuracy": 0.9, "completeness": 0.9}']
        )
        bad = await LLMJudge(bad_client, "fake-model").judge("q", "a")
        assert bad.judge_failed is True
        assert bad.error == "judge response missing helpfulness"
        assert (bad.relevance, bad.accuracy, bad.completeness, bad.helpfulness) == (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        assert bad.overall == 0.0

    asyncio.run(scenario())


def test_evaluator_counts_grader_errors_without_fake_scores(tmp_path):
    orchestrator = FakeOrchestrator()
    judge = LLMJudge(
        FakeAnthropic(
            [
                json.dumps(
                    {
                        "relevance": 0.8,
                        "accuracy": 0.8,
                        "completeness": 0.8,
                        "helpfulness": 0.8,
                    }
                ),
                "not json",
            ]
        ),
        "fake-model",
    )
    evaluator = EndToEndEvaluator(
        orchestrator=orchestrator,
        recognizer=None,
        model="fake-model",
        report_dir=str(tmp_path / "reports"),
        judge=judge,
    )

    report = asyncio.run(
        evaluator.run(
            dialog_cases=[
                {"question": "订单什么时候到？"},
                {"question": "怎么申请退款？"},
            ]
        )
    )

    assert report.total == 2
    assert report.passed == 1
    assert report.grader_errors == 1
    assert report.avg_scores["relevance"] == pytest.approx(0.8)
    assert any("评分失败" in item for item in report.recommendations)

    failed = [result for result in report.results if not result.passed]
    assert len(failed) == 1
    assert failed[0].scores == {}
    assert "GRADER_ERROR" in failed[0].metadata
    assert failed[0].metadata["GRADER_ERROR"] == (
        "judge response contains no JSON object"
    )


def test_evaluator_keeps_baseline_bytes_and_writes_new_report(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00",
                "total": 1,
                "passed": 1,
                "pass_rate": 1.0,
                "avg_scores": {"relevance": 0.8},
                "regressions": [],
                "recommendations": [],
                "results": [],
            }
        ),
        encoding="utf-8",
    )
    before = baseline.read_bytes()

    scores = json.dumps(
        {
            "relevance": 0.8,
            "accuracy": 0.8,
            "completeness": 0.8,
            "helpfulness": 0.8,
        }
    )
    evaluator = EndToEndEvaluator(
        orchestrator=FakeOrchestrator(),
        recognizer=None,
        model="fake-model",
        baseline_path=str(baseline),
        judge=LLMJudge(FakeAnthropic([scores, scores]), "fake-model"),
    )

    async def scenario():
        await evaluator.run(dialog_cases=[{"question": "第一个问题"}])
        await evaluator.run(dialog_cases=[{"question": "第二个问题"}])

    asyncio.run(scenario())

    assert baseline.read_bytes() == before
    reports = sorted((tmp_path / "reports").glob("report_*.json"))
    assert len(reports) == 2
    assert len({path.name for path in reports}) == 2
    for path in reports:
        assert "grader_errors" in json.loads(path.read_text(encoding="utf-8"))


def test_tool_stats_average_uses_recorded_latency_samples():
    stats = ToolStats(
        total=5,
        success=2,
        total_latency_ms=240.0,
        latency_samples=2,
    )
    assert stats.avg_latency_ms == pytest.approx(120.0)
    assert ToolStats().avg_latency_ms == 0.0


def test_tool_manager_exposes_latency_sample_count():
    async def handler(params, context):
        return {"ok": True}

    manager = MCPToolManager(api_key="test", model="fake-model")
    manager.register(
        Tool(
            name="echo",
            description="Echo",
            handler=handler,
            schema={"type": "object", "properties": {}},
        )
    )

    async def scenario():
        first = await manager.call("echo", {})
        second = await manager.call("echo", {})
        assert first.success is True
        assert second.success is True
        stats = manager.get_stats()["echo"]
        assert stats["total"] == 2
        assert stats["latency_samples"] == 2
        assert stats["avg_latency_ms"] >= 0.0

    asyncio.run(scenario())
