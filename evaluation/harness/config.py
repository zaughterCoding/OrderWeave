"""评测路径、模型参数与被测系统 profile。

路径默认指向本地外部资料目录，均可用环境变量覆盖；仓库不分发这些资料。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else default


TAU_DATA = _path("HARNESS_TAU_DATA", ROOT / "evaluation" / "data" / "tau_retail" / "upstream_data")
TAU_RETAIL_DIR = TAU_DATA / "tau2" / "domains" / "retail"
TAU_TASKS = TAU_RETAIL_DIR / "tasks.json"
SEED_PATH = _path("HARNESS_SEED_PATH", TAU_RETAIL_DIR / "db.json")
POLICY_PATH = _path("HARNESS_POLICY_PATH", TAU_RETAIL_DIR / "policy.md")
TAU_PYTHON = _path("HARNESS_TAU_PYTHON", ROOT / ".venv-tau" / "Scripts" / "python.exe")
USER_WORKER = ROOT / "evaluation" / "tau_user_worker.py"
EVAL_WORKER = ROOT / "evaluation" / "tau_eval_worker.py"
DATASETS = _path("HARNESS_DATASETS", ROOT / "evaluation" / "data" / "harness")
ENV_FILE = _path("HARNESS_ENV_FILE", ROOT / ".env")
RUNS = _path("HARNESS_RUNS_DIR", ROOT / "data" / "evaluation" / "harness")

GREETING = "Hi! How can I help you today?"
SEED = 300
MAX_STEPS = 200
MAX_ERRORS = 10
MAX_MESSAGE_CHARS = 4000
MAX_CONVERSATION_CHARS = 120000
USER_MAX_TOKENS = 512
JUDGE_MAX_TOKENS = 4096
USER_TIMEOUT_S = 180.0
EVAL_TIMEOUT_S = 300.0
# 单会话账本上限只防失控；全局预算由调度器在启动新会话前核对。
SESSION_BUDGET_CNY = 0.5
# DeepSeek 官方价 2026-11-01 核实：峰时未命中输入 2、输出 8 元／百万 token 作为预留上界。
INPUT_CNY_PER_MTOK = 2.0
OUTPUT_CNY_PER_MTOK = 8.0


@dataclass(frozen=True)
class Profile:
    """被测系统配置；新增阶段方案以新 profile 名对照，不改旧 profile。"""

    name: str
    agent_mode: str = "multi"
    workflow_mode: str = "langgraph"
    enable_memory: bool = False
    max_tokens: int = 2048
    # 与历史 τ 结果口径一致：所有阶段（含模拟用户与 Judge）关闭思考。
    thinking: str = "disabled"
    # window 为旧两轮窗口；compiled 为 S1 完整消息日志＋上下文编排。
    context_mode: str = "window"
    # inline 为整份政策常驻；on_demand 为 S2b 按需 Skill。
    skill_mode: str = "inline"
    # S3 会话事实账本。
    task_facts: bool = False
    # S4 发布前校验与修订。
    publish_revision: bool = False
    # 会话请求清单（多动作规划核对）。
    request_plan: bool = False


PROFILES = {
    "legacy-multi": Profile("legacy-multi"),
    "legacy-single": Profile("legacy-single", agent_mode="single"),
    "ctx-multi": Profile("ctx-multi", context_mode="compiled"),
    "ctx-single": Profile("ctx-single", agent_mode="single", context_mode="compiled"),
    "skill-multi": Profile("skill-multi", context_mode="compiled", skill_mode="on_demand"),
    "skill-single": Profile("skill-single", agent_mode="single", context_mode="compiled", skill_mode="on_demand"),
    "check-multi": Profile("check-multi", context_mode="compiled", skill_mode="on_demand", publish_revision=True),
    "check-single": Profile("check-single", agent_mode="single", context_mode="compiled", skill_mode="on_demand",
                            publish_revision=True),
    "plan-multi": Profile("plan-multi", context_mode="compiled", skill_mode="on_demand", publish_revision=True,
                          request_plan=True),
    "facts-multi": Profile("facts-multi", context_mode="compiled", skill_mode="on_demand", task_facts=True),
    "facts-single": Profile("facts-single", agent_mode="single", context_mode="compiled", skill_mode="on_demand",
                            task_facts=True),
}


def load_env() -> None:
    """项目 .env 覆盖外层进程变量，并移除不属于被测服务的令牌，防止凭证发往错误端点。"""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


def model_settings() -> dict:
    model = os.getenv("LLM_MODEL", "").strip()
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if not model or not os.getenv("ANTHROPIC_API_KEY", "").strip():
        raise RuntimeError("需要在 .env 配置 ANTHROPIC_API_KEY 与 LLM_MODEL")
    settlement = "deepseek_flash_cache_20261024" if model == "deepseek-flash" else "legacy"
    return {"model": model, "base_url": base_url, "settlement_policy": settlement,
            "user_model": os.getenv("HARNESS_USER_MODEL", "").strip() or model,
            "judge_model": os.getenv("HARNESS_JUDGE_MODEL", "").strip() or model}
