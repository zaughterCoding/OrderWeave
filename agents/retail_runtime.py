"""零售模式的运行时装配。

API 与 CLI 共用同一条装配路径：固定公开种子、独立 SQLite、隔离的记忆键与
Chroma 集合、仅含冻结证据的知识集合、受预算账本约束的客户端，以及单个
零售 Agent。旧客服模式不经过本模块。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.retail_agent import RetailAgent, RetailService
from agents.retail_tools import (
    retail_knowledge_tool,
    retail_tool_scope,
    retail_tools,
)
from core.skill_loader import SkillManager
from evaluation.retail_ledger import (
    BudgetedClient,
    RetailLedger,
    build_budgeted_client,
)
from mcp.knowledge_base import KnowledgeBase
from mcp.retail_client import RetailMCPClient
from mcp.retail_store import RetailStore
from mcp.tool_manager import MCPToolManager, Tool
from memory.conversation_memory import MemoryManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_POLICY_SHA256 = (
    "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467"
)
EXPECTED_KNOWLEDGE_SHA256 = (
    "e7360f9d22531ed836da12fe34d90c971a9300a3ed943031d75f8f6dc3b734c0"
)
EXPECTED_KNOWLEDGE_SEGMENTS = 29
RETRIEVAL_TOP_K = 5

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$")


def retail_mode_enabled() -> bool:
    raw = os.getenv("RETAIL_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_text(name: str, default: str = "") -> str:
    raw = os.getenv(name, "")
    return raw.strip() if raw.strip() else default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    return path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_positive_float(name: str) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        raise RuntimeError(
            f"未配置 {name}。真实运行前必须核对当前官方模型价格并显式写入；"
            "无法核实时不要启动付费运行。"
        )
    try:
        value = float(raw)
    except ValueError as ex:
        raise RuntimeError(f"{name} 不是有效数字: {raw!r}") from ex
    if value <= 0:
        raise RuntimeError(f"{name} 必须大于 0")
    return value


def _env_positive_seconds(name: str, default: float) -> float:
    """MCP 超时必须有限且为正数。"""
    raw = _env_text(name, str(default))
    try:
        value = float(raw)
    except ValueError as ex:
        raise RuntimeError(f"{name} 不是有效数字: {raw!r}") from ex
    if value <= 0:
        raise RuntimeError(f"{name} 必须大于 0")
    return value


def _env_optional_path(name: str) -> Optional[Path]:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_collection_name(name: str, env_name: str) -> str:
    value = str(name or "").strip()
    if not _COLLECTION_RE.match(value):
        raise RuntimeError(
            f"{env_name} 不是合法的 Chroma 集合名: {value!r}；"
            "只允许字母、数字、点、下划线、短横线，长度 3-63。"
        )
    return value


def _memory_suffix(name: str) -> str:
    """把外部配置压成 Chroma 集合可用的后缀。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(name or "").strip())
    cleaned = re.sub(r"^[^A-Za-z0-9]+", "_", cleaned)
    return cleaned or "_retail"


def _load_knowledge(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as ex:
                raise RuntimeError(
                    f"知识文件第 {line_number} 行不是合法 JSON: {ex}"
                ) from ex
            if not isinstance(row, dict):
                raise RuntimeError(f"知识文件第 {line_number} 行必须是对象")
            rows.append(row)
    if len(rows) != EXPECTED_KNOWLEDGE_SEGMENTS:
        raise RuntimeError(
            "冻结知识片段数量不符: "
            f"{len(rows)} != {EXPECTED_KNOWLEDGE_SEGMENTS}"
        )
    return rows


def _register_knowledge_tool(
    tool_manager: MCPToolManager,
    knowledge_base: KnowledgeBase,
) -> None:
    """注册零售使用的唯一检索入口，沿用既有改写与重排链路。"""

    def knowledge_fallback(
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> List[Dict[str, Any]]:
        query = params.get("query", "")
        return [
            {
                "title": "知识库降级结果",
                "content": (
                    f"知识库暂时不可用，未能完成对“{query}”的语义检索。"
                    "请说明无法核实该政策，并建议用户稍后重试。"
                ),
                "score": 0.0,
                "fallback": True,
                "error": error,
            }
        ]

    tool_manager.register(
        Tool(
            name="knowledge_search",
            description="检索冻结零售知识集合",
            handler=knowledge_base.search_handler,
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
            fallback=knowledge_fallback,
        )
    )


@dataclass(frozen=True)
class RetailRuntimeConfig:
    db_path: Path
    seed_path: Path
    policy_path: Path
    knowledge_path: Path
    knowledge_collection: str
    ledger_path: Path
    event_log_path: Path
    budget_cny: float
    input_price_cny_per_mtok: float
    output_price_cny_per_mtok: float
    model: str
    base_url: str
    redis_url: str
    chroma_host: str
    chroma_port: int
    chroma_path: Path
    case_id: str
    run_id: str
    profile_enabled: bool
    memory_key_prefix: str
    memory_collection_suffix: str
    max_tokens: int
    max_message_chars: int
    max_context_chars: int
    mcp_timeout_seconds: float
    mcp_stderr_log: Optional[Path]

    @classmethod
    def from_env(cls) -> "RetailRuntimeConfig":
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未设置 ANTHROPIC_API_KEY")
        model = (
            _env_text("LLM_MODEL")
            or _env_text("ANTHROPIC_MODEL")
        )
        if not model:
            raise RuntimeError("未设置 LLM_MODEL")
        collection = _validate_collection_name(
            _env_text("RETAIL_COLLECTION", "retail_knowledge_v1"),
            "RETAIL_COLLECTION",
        )
        suffix = _memory_suffix(
            _env_text("RETAIL_MEMORY_COLLECTION_SUFFIX", "_retail")
        )
        return cls(
            db_path=_env_path(
                "RETAIL_DB_PATH", _PROJECT_ROOT / "data" / "retail" / "retail.db"
            ),
            seed_path=_env_path(
                "RETAIL_SEED_PATH",
                _PROJECT_ROOT / "evaluation" / "data" / "retail_sandbox" / "db.json",
            ),
            policy_path=_env_path(
                "RETAIL_POLICY_PATH",
                _PROJECT_ROOT / "evaluation" / "data" / "retail_sandbox" / "policy.md",
            ),
            knowledge_path=_env_path(
                "RETAIL_KNOWLEDGE_PATH",
                _PROJECT_ROOT
                / "evaluation"
                / "data"
                / "retail_sandbox"
                / "knowledge.jsonl",
            ),
            knowledge_collection=collection,
            ledger_path=_env_path(
                "RETAIL_BUDGET_LEDGER",
                _PROJECT_ROOT / "data" / "retail" / "budget_ledger.db",
            ),
            event_log_path=_env_path(
                "RETAIL_EVENT_LOG",
                _PROJECT_ROOT / "data" / "retail" / "events.jsonl",
            ),
            budget_cny=float(_env_text("RETAIL_BUDGET_CNY", "5")),
            input_price_cny_per_mtok=_env_positive_float(
                "RETAIL_INPUT_CNY_PER_MTOK"
            ),
            output_price_cny_per_mtok=_env_positive_float(
                "RETAIL_OUTPUT_CNY_PER_MTOK"
            ),
            model=model,
            base_url=_env_text("ANTHROPIC_BASE_URL"),
            redis_url=_env_text(
                "REDIS_URL", "redis://:agentecommerce123@127.0.0.1:6380/0"
            ),
            chroma_host=_env_text("CHROMA_HOST", "127.0.0.1"),
            chroma_port=int(_env_text("CHROMA_PORT", "8002")),
            chroma_path=_env_path(
                "CHROMA_PERSIST_DIRECTORY", _PROJECT_ROOT / "data" / "chroma"
            ),
            case_id=_env_text("RETAIL_CASE_ID"),
            run_id=_env_text("RETAIL_RUN_ID"),
            profile_enabled=_env_bool("RETAIL_PROFILE_ENABLED", False),
            memory_key_prefix=_env_text(
                "RETAIL_MEMORY_KEY_PREFIX", "retail:"
            ),
            memory_collection_suffix=suffix,
            max_tokens=max(256, int(_env_text("RETAIL_AGENT_MAX_TOKENS", "2048"))),
            max_message_chars=max(
                200, int(_env_text("RETAIL_MAX_MESSAGE_CHARS", "4000"))
            ),
            max_context_chars=max(
                2000, int(_env_text("RETAIL_MAX_CONTEXT_CHARS", "12000"))
            ),
            mcp_timeout_seconds=_env_positive_seconds(
                "RETAIL_MCP_TIMEOUT_SECONDS", 30.0
            ),
            mcp_stderr_log=(
                _env_optional_path("RETAIL_MCP_STDERR_LOG")
                or _PROJECT_ROOT / "data" / "retail" / "mcp_stderr.log"
            ),
        )


@dataclass
class RetailRuntime:
    config: RetailRuntimeConfig
    ledger: RetailLedger
    client: BudgetedClient
    store: RetailStore
    memory: MemoryManager
    tool_manager: MCPToolManager
    knowledge_base: KnowledgeBase
    skill_manager: SkillManager
    agent: RetailAgent
    service: RetailService
    mcp_client: RetailMCPClient
    _started: bool = False
    _closed: bool = False

    async def start(self) -> None:
        """启动工具子进程并在同一事件循环内装入 MCP 工具白名单。

        失败时关闭本次已打开的资源；未启动时不会回退到进程内直调。
        """
        if self._closed:
            raise RuntimeError("零售运行时已关闭，不能重新启动")
        if self._started:
            return
        try:
            await self.mcp_client.start()
            discovered = self.mcp_client.tools()
            expected = set(retail_tool_scope(has_knowledge=False))
            if set(discovered) != expected:
                raise RuntimeError(
                    "MCP 工具白名单不一致: "
                    f"{sorted(discovered)} != {sorted(expected)}"
                )
            # RAG 仍在主程序原链路执行，其余九个业务工具只通过 MCP 调用。
            tools = dict(discovered)
            tools["search_knowledge_base"] = retail_knowledge_tool(
                self.tool_manager
            )
            self.agent.set_shared_tools(tools)
        except Exception:
            await self.close()
            raise
        self._started = True

    async def close(self) -> None:
        """按 MCP、记忆、模型客户端的顺序关闭，单步失败不跳过后续清理。"""
        if self._closed:
            return
        self._closed = True
        self._started = False
        try:
            await self.mcp_client.close()
        except Exception:
            logger.exception("关闭 MCP 工具服务失败")
        try:
            await self.memory.close()
        except Exception:
            logger.exception("关闭记忆服务失败")
        try:
            await self.client.aclose()
        except Exception:
            logger.exception("关闭模型客户端失败")


def build_retail_runtime(
    config: Optional[RetailRuntimeConfig] = None,
) -> RetailRuntime:
    """按环境配置构造零售运行时；不会自动重置已有业务库。"""
    config = config or RetailRuntimeConfig.from_env()

    for path, label in (
        (config.seed_path, "公开种子"),
        (config.policy_path, "固定政策"),
        (config.knowledge_path, "冻结知识"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label}文件不存在: {path}")

    policy_sha = _sha256_file(config.policy_path)
    if policy_sha != EXPECTED_POLICY_SHA256:
        raise RuntimeError(
            f"固定政策哈希不一致: {policy_sha} != {EXPECTED_POLICY_SHA256}"
        )
    knowledge_sha = _sha256_file(config.knowledge_path)
    if knowledge_sha != EXPECTED_KNOWLEDGE_SHA256:
        raise RuntimeError(
            f"冻结知识哈希不一致: {knowledge_sha} != {EXPECTED_KNOWLEDGE_SHA256}"
        )
    knowledge_rows = _load_knowledge(config.knowledge_path)

    ledger = RetailLedger(
        path=config.ledger_path,
        budget_cny=config.budget_cny,
        input_price_cny_per_mtok=config.input_price_cny_per_mtok,
        output_price_cny_per_mtok=config.output_price_cny_per_mtok,
        run_id=config.run_id,
        model=config.model,
    )
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    client = build_budgeted_client(
        api_key=api_key,
        base_url=config.base_url or None,
        ledger=ledger,
        stage="agent",
    )

    store = RetailStore(config.db_path, config.seed_path)
    language = store.language
    mcp_client = RetailMCPClient(
        db_path=config.db_path,
        seed_path=config.seed_path,
        language=language,
        expected_specs=retail_tools(store, None, language=language),
        project_root=_PROJECT_ROOT,
        python_executable=sys.executable,
        startup_timeout=config.mcp_timeout_seconds,
        call_timeout=config.mcp_timeout_seconds,
        stderr_log_path=config.mcp_stderr_log,
    )
    memory = MemoryManager(
        redis_url=config.redis_url,
        chroma_host=config.chroma_host,
        chroma_port=config.chroma_port,
        chroma_path=str(config.chroma_path),
        client=client.for_stage("memory"),
        model=config.model,
        key_prefix=config.memory_key_prefix,
        collection_suffix=config.memory_collection_suffix,
        cross_conv_history=False,
        profile_enabled=config.profile_enabled,
    )

    knowledge_base = KnowledgeBase(
        chroma_host=config.chroma_host,
        chroma_port=config.chroma_port,
        chroma_path=str(config.chroma_path),
        collection_name=config.knowledge_collection,
        load_demo_documents=False,
    )
    added = knowledge_base.upsert_evidence(knowledge_rows)
    total = knowledge_base.doc_count
    if total != EXPECTED_KNOWLEDGE_SEGMENTS:
        raise RuntimeError(
            "零售知识集合必须只包含冻结片段，当前数量不符: "
            f"{total} != {EXPECTED_KNOWLEDGE_SEGMENTS}；"
            f"请为新运行使用独立 RETAIL_COLLECTION，不自动清理旧集合。"
        )
    logger.info(
        "零售知识集合 %s: %s 个片段，本次写入 %s 个",
        config.knowledge_collection,
        total,
        added,
    )

    tool_manager = MCPToolManager(
        api_key=api_key,
        base_url=config.base_url or None,
        model=config.model,
        client=client.for_stage("retrieval"),
    )
    _register_knowledge_tool(tool_manager, knowledge_base)

    skills_dir = _env_text(
        "ECHOMIND_SKILLS_DIR", str(_PROJECT_ROOT / "skills")
    )
    skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(
            _env_text("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000")
        ),
    )
    skill_manager.load()

    policy_text = config.policy_path.read_text(encoding="utf-8")
    agent = RetailAgent(
        client=client.for_stage("agent"),
        model=config.model,
        store=store,
        tool_manager=tool_manager,
        skill_manager=skill_manager,
        policy_text=policy_text,
        policy_source=(
            "evaluation/data/retail_sandbox/policy.md "
            f"sha256={EXPECTED_POLICY_SHA256}"
        ),
        max_tokens=config.max_tokens,
    )

    service = RetailService(
        store=store,
        agent=agent,
        memory=memory,
        event_log_path=str(config.event_log_path),
        case_id=config.case_id,
        run_id=ledger.run_id,
        ledger_summary=ledger.summary,
        profile_enabled=config.profile_enabled,
        max_message_chars=config.max_message_chars,
        max_context_chars=config.max_context_chars,
    )

    return RetailRuntime(
        config=config,
        ledger=ledger,
        client=client,
        store=store,
        memory=memory,
        tool_manager=tool_manager,
        knowledge_base=knowledge_base,
        skill_manager=skill_manager,
        agent=agent,
        service=service,
        mcp_client=mcp_client,
    )
