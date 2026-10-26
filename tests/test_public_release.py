"""无私有数据的公开入口、离线汇总与运行资产边界。"""
import hashlib
import io
import sys
from dataclasses import replace

import pytest

from evaluation import prepare_assets
from evaluation.score_results import score


def test_assets_are_opt_in_and_unknown_files_are_not_overwritten(tmp_path, monkeypatch):
    payload = b"external-runtime-asset"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(prepare_assets, "ASSETS", {"asset": ("asset", digest)})
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(prepare_assets, "urlopen", download)
    with pytest.raises(FileNotFoundError):
        prepare_assets.prepare(tmp_path)
    assert calls == []
    assert prepare_assets.prepare(tmp_path, download=True) == {"asset": digest}
    assert len(calls) == 1
    prepare_assets.prepare(tmp_path)
    assert len(calls) == 1
    (tmp_path / "asset").write_bytes(b"unknown")
    with pytest.raises(ValueError, match="哈希"):
        prepare_assets.prepare(tmp_path, download=True)
    assert (tmp_path / "asset").read_bytes() == b"unknown"


def test_task_summary_keeps_invalid_and_not_run_in_planned_denominator():
    rows = [{"id": "a", "status": "valid", "success": True},
            {"id": "b", "status": "invalid", "success": False},
            {"id": "c", "status": "not_run", "success": False}]
    result = score(rows, "task", 3)
    assert result["effective"] == result["successful"] == 1
    assert result["success_effective"] == 1
    assert result["success_planned"] == result["effective_planned"] == 1 / 3
    assert result["invalid"] == result["not_run"] == 1
    with pytest.raises(ValueError, match="槽位"):
        score(rows[:1], "task", 3)
    with pytest.raises(ValueError, match="唯一"):
        score([rows[0], rows[0]], "task", 2)


def test_failed_prediction_and_failed_recovery_cannot_be_dropped():
    result = score([{"id": "a", "gold": "refund", "prediction": "refund"},
                    {"id": "b", "gold": "query", "prediction": "query", "is_error": True}], "intent", 2)
    assert result["correct"] == 1 and result["accuracy"] == 0.5
    assert "cases" not in result
    assert score([{"id": "a", "gold": "billing", "prediction": None}], "route", 1)["accuracy"] == 0
    assert score([{"id": "a", "passed": False}], "recovery", 1)["recovery_passed"] == 0


def test_disabled_knowledge_does_not_require_private_assets(tmp_path, monkeypatch):
    from agents import retail_runtime as runtime

    for key, value in {"ANTHROPIC_API_KEY": "offline-fake", "LLM_MODEL": "fake", "RETAIL_INPUT_CNY_PER_MTOK": "1",
                       "RETAIL_OUTPUT_CNY_PER_MTOK": "1", "RETAIL_SUPPORT_ENABLED": "false",
                       "RETAIL_ENABLE_MEMORY": "false", "RETAIL_ENABLE_KNOWLEDGE": "false"}.items():
        monkeypatch.setenv(key, value)
    seed, policy = tmp_path / "seed", tmp_path / "policy"
    seed.write_text("external", encoding="utf-8")
    policy.write_text("external", encoding="utf-8")
    monkeypatch.setattr(runtime, "EXPECTED_POLICY_SHA256", hashlib.sha256(policy.read_bytes()).hexdigest())
    config = replace(runtime.RetailRuntimeConfig.from_env(), seed_path=seed, policy_path=policy,
                     knowledge_path=tmp_path / "not-distributed.jsonl", enable_knowledge=False)

    class ReachedBudgetBoundary(Exception):
        pass

    def stop_before_clients(**kwargs):
        raise ReachedBudgetBoundary

    monkeypatch.setattr(runtime, "RetailLedger", stop_before_clients)
    with pytest.raises(ReachedBudgetBoundary):
        runtime._build_retail_runtime(config, None)
    with pytest.raises(RuntimeError, match="冻结知识文件不存在"):
        runtime._build_retail_runtime(replace(config, enable_knowledge=True), None)


def test_skills_use_public_brand_and_keep_legacy_setting_compatible(tmp_path, monkeypatch):
    from core.skill_loader import SkillManager
    monkeypatch.setenv("ORDERWEAVE_SKILLS_DIR", str(tmp_path))
    monkeypatch.setenv("ECHOMIND_SKILLS_DIR", str(tmp_path / "legacy"))
    assert SkillManager().root_dir == tmp_path


@pytest.mark.skipif(sys.platform != "win32", reason="Windows内存接口")
def test_reranker_memory_stats_do_not_import_experimental_code():
    from mcp.qwen_reranker import memory_peak
    stats = memory_peak()
    assert stats["rss_peak_mib"] > 0
    assert stats["private_peak_mib"] > 0
