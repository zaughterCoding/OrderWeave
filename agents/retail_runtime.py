"""零售模式的运行时装配。

API 与 CLI 共用同一条装配路径：固定公开种子、独立 SQLite、隔离的记忆键与
Chroma 集合、仅含冻结证据的知识集合、受预算账本约束的客户端，以及按需
协作的零售角色。旧客服模式不经过本模块。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.agent_orchestrator import AgentOrchestrator, ResponseComposer
from agents.retail_agent import RetailAgent, RetailService
from agents.retail_support import (
    MAX_HANDOFF_CHARS,
    SUPPORT_EVIDENCE_IDS,
    SUPPORT_ISSUE,
    SUPPORT_PRODUCT_ID,
    SUPPORT_STEP_IDS,
)
from agents.retail_multi_agent import (
    RetailBillingAgent,
    RetailEscalationAgent,
    RetailGeneralAgent,
    RetailMultiAgent,
    RetailTechnicalAgent,
)
from agents.retail_tools import (
    retail_knowledge_tool,
    retail_tool_scope,
    retail_tools,
)
from core.intent_recognizer import IntentRecognizer
from core.skill_loader import SkillManager
from evaluation.retail_ledger import (
    BudgetedClient,
    RetailLedger,
    build_budgeted_client,
)
from mcp.knowledge_base import KnowledgeBase
from mcp.qwen_embedding import QwenEmbeddingFunction
from mcp.qwen_reranker import QwenReranker
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
EXPECTED_SUPPORT_KNOWLEDGE_SHA256 = (
    "ca15b525e749a1f0b89e7fe87bfe5fe4b42dfb929558c7997222e4f1fac8c044"
)
EXPECTED_SUPPORT_SEGMENTS = 6
COMBINED_KNOWLEDGE_SEGMENTS = (
    EXPECTED_KNOWLEDGE_SEGMENTS + EXPECTED_SUPPORT_SEGMENTS
)
SUPPORT_MANIFEST_FORMAT = "agent_ecommerce_retail_support_manifest"
DEFAULT_SUPPORT_COLLECTION = "retail_earbuds_support_v1"
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


def _load_support_knowledge(path: Path) -> List[Dict[str, Any]]:
    """加载项目自编 6 段；旧 29 段始终由固定文件单独校验。"""
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
                    f"支持知识第 {line_number} 行不是合法 JSON: {ex}"
                ) from ex
            if not isinstance(row, dict):
                raise RuntimeError(f"支持知识第 {line_number} 行必须是对象")
            rows.append(row)
    if len(rows) != EXPECTED_SUPPORT_SEGMENTS:
        raise RuntimeError(
            "支持知识片段数量不符: "
            f"{len(rows)} != {EXPECTED_SUPPORT_SEGMENTS}"
        )
    evidence_ids = [str(row.get("evidence_id") or "") for row in rows]
    if tuple(evidence_ids) != SUPPORT_EVIDENCE_IDS:
        raise RuntimeError(
            "支持知识 evidence_id 不符: "
            f"{evidence_ids} != {list(SUPPORT_EVIDENCE_IDS)}"
        )
    for row in rows:
        if str(row.get("product_id") or "") != SUPPORT_PRODUCT_ID:
            raise RuntimeError(
                f"支持知识商品不匹配: {row.get('evidence_id')}"
            )
        if str(row.get("issue") or "") != SUPPORT_ISSUE:
            raise RuntimeError(
                f"支持知识问题不匹配: {row.get('evidence_id')}"
            )
        step_id = str(row.get("step_id") or "")
        if step_id and step_id not in SUPPORT_STEP_IDS:
            raise RuntimeError(
                f"支持知识步骤不在固定集合: {row.get('evidence_id')} {step_id}"
            )
        if str(row.get("source_kind") or "") != "project_authored_sandbox":
            raise RuntimeError(
                f"支持知识来源不是项目自编沙箱: {row.get('evidence_id')}"
            )
        if str(row.get("language") or "") != "zh":
            raise RuntimeError(
                f"支持知识必须是中文: {row.get('evidence_id')}"
            )
    return rows


def _load_support_manifest(
    path: Path,
    support_knowledge_path: Path,
) -> Dict[str, Any]:
    """校验清单格式、资产哈希、旧 29 段与新 35 段口径。"""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as ex:
        raise RuntimeError(f"无法读取支持清单: {ex}") from ex
    if not isinstance(manifest, dict):
        raise RuntimeError("支持清单顶层必须是对象")
    if manifest.get("format") != SUPPORT_MANIFEST_FORMAT:
        raise RuntimeError(
            f"支持清单格式不符: {manifest.get('format')!r}"
        )
    if manifest.get("nature") != "project_authored_sandbox_not_vendor_documentation":
        raise RuntimeError("支持清单必须声明项目自编沙箱性质")
    product = manifest.get("product") or {}
    if (
        str(product.get("product_id") or "") != SUPPORT_PRODUCT_ID
        or str(product.get("issue") or "") != SUPPORT_ISSUE
    ):
        raise RuntimeError("支持清单商品或问题不在固定范围")

    for entry in (manifest.get("files") or {}).values():
        if not isinstance(entry, dict):
            raise RuntimeError("支持清单 files 条目必须是对象")
        relative = str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "")
        if not relative or not expected:
            raise RuntimeError("支持清单 files 缺少 path 或 sha256")
        asset_path = Path(relative)
        if not asset_path.is_absolute():
            asset_path = _PROJECT_ROOT / asset_path
        if not asset_path.is_file():
            raise RuntimeError(f"支持资产不存在: {asset_path}")
        actual = _sha256_file(asset_path)
        if actual != expected:
            raise RuntimeError(
                f"支持资产哈希不一致: {asset_path} {actual} != {expected}"
            )

    support = manifest.get("support_knowledge") or {}
    if int(support.get("count") or 0) != EXPECTED_SUPPORT_SEGMENTS:
        raise RuntimeError("支持清单片段数量不是 6")
    if tuple(support.get("evidence_ids") or ()) != SUPPORT_EVIDENCE_IDS:
        raise RuntimeError("支持清单 evidence_id 列表不符")
    if tuple(support.get("step_ids") or ()) != SUPPORT_STEP_IDS:
        raise RuntimeError("支持清单 step_id 列表不符")
    declared_support_path = Path(str(support.get("path") or ""))
    if not declared_support_path.is_absolute():
        declared_support_path = _PROJECT_ROOT / declared_support_path
    if declared_support_path.resolve() != support_knowledge_path.resolve():
        raise RuntimeError("支持清单知识路径与运行时配置不一致")
    if str(support.get("sha256") or "") != EXPECTED_SUPPORT_KNOWLEDGE_SHA256:
        raise RuntimeError("支持清单知识哈希不是冻结值")

    legacy = manifest.get("legacy_knowledge") or {}
    if int(legacy.get("count") or 0) != EXPECTED_KNOWLEDGE_SEGMENTS:
        raise RuntimeError("支持清单旧知识数量不是 29")
    if str(legacy.get("sha256") or "") != EXPECTED_KNOWLEDGE_SHA256:
        raise RuntimeError("支持清单旧知识哈希与冻结值不一致")
    combined = manifest.get("combined") or {}
    if int(combined.get("count") or 0) != COMBINED_KNOWLEDGE_SEGMENTS:
        raise RuntimeError("支持清单合并数量不是 35")
    if str(combined.get("collection_default") or "") != DEFAULT_SUPPORT_COLLECTION:
        raise RuntimeError("支持清单默认集合名不符")
    scenarios = manifest.get("scenarios") or {}
    if len(scenarios.get("case_ids") or []) != 8:
        raise RuntimeError("支持清单必须有 8 个固定验收案例")
    return manifest


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
    # 正式英文对照关闭旧中文记忆与 RAG，保证两组可见信息一致。
    agent_mode: str = "multi"
    enable_memory: bool = True
    enable_knowledge: bool = True
    # T17 默认关闭；开启时只加独立支持集合与工作记录，不改旧 29 段集合。
    language: str = "zh"
    support_enabled: bool = False
    support_collection: str = DEFAULT_SUPPORT_COLLECTION
    support_knowledge_path: Path = field(
        default_factory=lambda: (
            _PROJECT_ROOT
            / "evaluation"
            / "data"
            / "retail_support"
            / "knowledge.jsonl"
        )
    )
    support_manifest_path: Path = field(
        default_factory=lambda: (
            _PROJECT_ROOT
            / "evaluation"
            / "data"
            / "retail_support"
            / "manifest.json"
        )
    )
    support_max_chars: int = MAX_HANDOFF_CHARS
    embedding_model: str = "qwen"
    retrieval_mode: str = "hybrid"
    reranker_model: str = "qwen"
    retrieval_strategy: str = "relevance"
    workflow_mode: str = "langgraph"
    checkpoint_path: Optional[Path] = None
    semantic_routing: bool = True
    semantic_model_path: Optional[Path] = None

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
        agent_mode = _env_text("RETAIL_AGENT_MODE", "multi").lower()
        if agent_mode not in {"single", "multi"}:
            raise RuntimeError(
                "RETAIL_AGENT_MODE 只能是 single 或 multi: " + agent_mode
            )
        language = _env_text("RETAIL_LANGUAGE", "zh").lower()
        if language not in {"zh", "en"}:
            raise RuntimeError("RETAIL_LANGUAGE 只能是 zh 或 en: " + language)
        support_enabled = _env_bool("RETAIL_SUPPORT_ENABLED", False)
        support_collection = _validate_collection_name(
            _env_text("RETAIL_SUPPORT_COLLECTION", DEFAULT_SUPPORT_COLLECTION),
            "RETAIL_SUPPORT_COLLECTION",
        )
        enable_knowledge = _env_bool("RETAIL_ENABLE_KNOWLEDGE", True)
        if support_enabled:
            if not retail_mode_enabled():
                raise RuntimeError(
                    "RETAIL_SUPPORT_ENABLED=true 需要 RETAIL_MODE=true"
                )
            if language != "zh":
                raise RuntimeError(
                    "RETAIL_SUPPORT_ENABLED 只支持中文零售 profile（RETAIL_LANGUAGE=zh）"
                )
            if not enable_knowledge:
                raise RuntimeError(
                    "RETAIL_SUPPORT_ENABLED=true 需要 RETAIL_ENABLE_KNOWLEDGE=true"
                )
            if support_collection == collection:
                raise RuntimeError(
                    "RETAIL_SUPPORT_COLLECTION 不能与 RETAIL_COLLECTION 同名"
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
            agent_mode=agent_mode,
            workflow_mode=_env_text("RETAIL_WORKFLOW_MODE", "langgraph"),
            checkpoint_path=_env_optional_path("RETAIL_CHECKPOINT_PATH"),
            semantic_routing=_env_bool("RETAIL_SEMANTIC_ROUTING", True),
            semantic_model_path=_env_optional_path("RETAIL_SEMANTIC_MODEL_PATH"),
            enable_memory=_env_bool("RETAIL_ENABLE_MEMORY", True),
            enable_knowledge=enable_knowledge,
            embedding_model=_env_text("RETAIL_EMBEDDING_MODEL", "qwen").lower(),
            retrieval_mode=_env_text("RETAIL_RETRIEVAL_MODE", "hybrid").lower(),
            reranker_model=_env_text("RETAIL_RERANKER_MODEL", "qwen").lower(),
            retrieval_strategy=_env_text("RETAIL_RETRIEVAL_STRATEGY", "relevance").lower(),
            language=language,
            support_enabled=support_enabled,
            support_collection=support_collection,
            support_knowledge_path=_env_path(
                "RETAIL_SUPPORT_KNOWLEDGE_PATH",
                _PROJECT_ROOT
                / "evaluation"
                / "data"
                / "retail_support"
                / "knowledge.jsonl",
            ),
            support_manifest_path=_env_path(
                "RETAIL_SUPPORT_MANIFEST_PATH",
                _PROJECT_ROOT
                / "evaluation"
                / "data"
                / "retail_support"
                / "manifest.json",
            ),
            support_max_chars=max(
                1000,
                min(
                    MAX_HANDOFF_CHARS,
                    int(_env_text("RETAIL_SUPPORT_MAX_CHARS", str(MAX_HANDOFF_CHARS))),
                ),
            ),
        )


@dataclass
class RetailRuntime:
    config: RetailRuntimeConfig
    ledger: RetailLedger
    client: BudgetedClient
    store: RetailStore
    memory: Optional[MemoryManager]
    tool_manager: Optional[MCPToolManager]
    knowledge_base: Optional[KnowledgeBase]
    skill_manager: Optional[SkillManager]
    agent: Any
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
            if self.config.enable_knowledge and self.config.embedding_model == "qwen":
                await asyncio.to_thread(self.knowledge_base._embedding_function, ["模型预热"])
            if self.tool_manager is not None and self.tool_manager._reranker is not None:
                await asyncio.to_thread(self.tool_manager._reranker.score, "模型预热", [{"content": "政策帮助"}])
            await self.mcp_client.start()
            discovered = self.mcp_client.tools()
            expected = set(retail_tool_scope(has_knowledge=False))
            if set(discovered) != expected:
                raise RuntimeError(
                    "MCP 工具白名单不一致: "
                    f"{sorted(discovered)} != {sorted(expected)}"
                )
            # RAG 仍在主程序原链路执行，其余九个业务工具只通过 MCP 调用。
            # 正式英文对照关闭 RAG，两组都只保留同样的九个业务工具。
            tools = dict(discovered)
            if self.config.enable_knowledge:
                if self.tool_manager is None:
                    raise RuntimeError("启用知识检索但未构建工具管理器")
                tools["search_knowledge_base"] = retail_knowledge_tool(
                    self.tool_manager,
                    support_enabled=self.config.support_enabled,
                )
            self.agent.set_shared_tools(tools)
            if self.service.workflow is not None:
                await self.service.workflow.start()
        except Exception:
            await self.close()
            raise
        self._started = True

    async def close(self) -> None:
        """关闭工具、记忆、向量进程与模型客户端，单步失败不跳过后续清理。"""
        if self._closed:
            return
        self._closed = True
        self._started = False
        router = getattr(getattr(getattr(self, "agent", None), "orchestrator", None), "_retail_semantic_router", None)
        if router is not None:
            try:
                await asyncio.to_thread(router.close)
            except Exception:
                logger.exception("关闭语义路由编码器失败")
        workflow = getattr(getattr(self, "service", None), "workflow", None)
        if workflow is not None:
            try:
                await workflow.close()
            except Exception:
                logger.exception("关闭工作图 checkpoint 失败")
        try:
            await self.mcp_client.close()
        except Exception:
            logger.exception("关闭 MCP 工具服务失败")
        if self.memory is not None:
            try:
                await self.memory.close()
            except Exception:
                logger.exception("关闭记忆服务失败")
        if self.knowledge_base is not None:
            try:
                await asyncio.to_thread(self.knowledge_base.close)
            except Exception:
                logger.exception("关闭知识向量进程失败")
        if self.tool_manager is not None and self.tool_manager._reranker is not None:
            try:
                await asyncio.to_thread(self.tool_manager._reranker.close)
            except Exception:
                logger.exception("关闭重排进程失败")
        try:
            await self.client.aclose()
        except Exception:
            logger.exception("关闭模型客户端失败")


def build_retail_runtime(
    config: Optional[RetailRuntimeConfig] = None,
) -> RetailRuntime:
    """按环境配置构造零售运行时；不会自动重置已有业务库。"""
    config = config or RetailRuntimeConfig.from_env()
    if config.workflow_mode not in {"langgraph", "legacy"}:
        raise ValueError("RETAIL_WORKFLOW_MODE 只能是 langgraph 或 legacy")
    if config.embedding_model not in {"qwen", "minilm"}:
        raise ValueError("RETAIL_EMBEDDING_MODEL 只能是 qwen 或 minilm")
    if config.retrieval_mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError("RETAIL_RETRIEVAL_MODE 只能是 dense、bm25 或 hybrid")
    if config.reranker_model not in {"llm", "qwen"}:
        raise ValueError("RETAIL_RERANKER_MODEL 只能是 llm 或 qwen")
    from mcp.evidence_coverage import STRATEGIES
    if config.retrieval_strategy not in STRATEGIES:
        raise ValueError("未知RETAIL_RETRIEVAL_STRATEGY")
    if config.retrieval_strategy != "relevance" and config.reranker_model != "qwen":
        raise ValueError("实验覆盖策略需要本地Qwen重排")
    encoder = QwenEmbeddingFunction() if config.enable_knowledge and config.embedding_model == "qwen" else None
    try:
        return _build_retail_runtime(config, encoder)
    except Exception:
        if encoder is not None:
            encoder.close()
        raise


def _build_retail_runtime(config: RetailRuntimeConfig, encoder) -> RetailRuntime:

    if config.support_enabled:
        if str(config.language).lower() != "zh":
            raise RuntimeError("有限支持只支持中文零售 profile")
        if not config.enable_knowledge:
            raise RuntimeError("有限支持需要启用知识检索")
        if config.support_collection == config.knowledge_collection:
            raise RuntimeError("支持集合不能与旧知识集合同名")
        _validate_collection_name(
            config.support_collection, "RETAIL_SUPPORT_COLLECTION"
        )

    required_files = [
        (config.seed_path, "公开种子"),
        (config.policy_path, "固定政策"),
    ]
    if config.enable_knowledge:
        required_files.append((config.knowledge_path, "冻结知识"))
    for path, label in required_files:
        if not path.is_file():
            raise RuntimeError(f"{label}文件不存在: {path}")

    policy_sha = _sha256_file(config.policy_path)
    if policy_sha != EXPECTED_POLICY_SHA256:
        raise RuntimeError(
            f"固定政策哈希不一致: {policy_sha} != {EXPECTED_POLICY_SHA256}"
        )
    knowledge_rows = []
    if config.enable_knowledge:
        knowledge_sha = _sha256_file(config.knowledge_path)
        if knowledge_sha != EXPECTED_KNOWLEDGE_SHA256:
            raise RuntimeError(
                f"冻结知识哈希不一致: {knowledge_sha} != {EXPECTED_KNOWLEDGE_SHA256}"
            )
        knowledge_rows = _load_knowledge(config.knowledge_path)

    support_rows: List[Dict[str, Any]] = []
    if config.support_enabled:
        for path, label in (
            (config.support_knowledge_path, "支持知识"),
            (config.support_manifest_path, "支持清单"),
        ):
            if not path.is_file():
                raise RuntimeError(f"{label}文件不存在: {path}")
        support_sha = _sha256_file(config.support_knowledge_path)
        if support_sha != EXPECTED_SUPPORT_KNOWLEDGE_SHA256:
            raise RuntimeError(
                "支持知识哈希不一致: "
                f"{support_sha} != {EXPECTED_SUPPORT_KNOWLEDGE_SHA256}"
            )
        _load_support_manifest(
            config.support_manifest_path,
            config.support_knowledge_path,
        )
        support_rows = _load_support_knowledge(config.support_knowledge_path)
        if len(knowledge_rows) + len(support_rows) != COMBINED_KNOWLEDGE_SEGMENTS:
            raise RuntimeError("旧 29 段与支持 6 段合并后不是 35 段")

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

    store = RetailStore(
        config.db_path, config.seed_path, language=config.language
    )
    language = store.language
    mcp_client = RetailMCPClient(
        db_path=config.db_path,
        seed_path=config.seed_path,
        language=language,
        expected_specs=retail_tools(
            store, None, language=language, support_enabled=config.support_enabled
        ),
        project_root=_PROJECT_ROOT,
        python_executable=sys.executable,
        startup_timeout=config.mcp_timeout_seconds,
        call_timeout=config.mcp_timeout_seconds,
        stderr_log_path=config.mcp_stderr_log,
    )
    memory = None
    if config.enable_memory:
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

    knowledge_base = None
    tool_manager = None
    if config.enable_knowledge:
        collection_name = (
            config.support_collection
            if config.support_enabled
            else config.knowledge_collection
        )
        expected_segments = (
            COMBINED_KNOWLEDGE_SEGMENTS
            if config.support_enabled
            else EXPECTED_KNOWLEDGE_SEGMENTS
        )
        knowledge_base = KnowledgeBase(
            chroma_host=config.chroma_host,
            chroma_port=config.chroma_port,
            chroma_path=str(config.chroma_path),
            collection_name=collection_name,
            load_demo_documents=False,
            embedding_function=encoder,
        )
        incoming_rows = (
            list(knowledge_rows) + list(support_rows)
            if config.support_enabled
            else list(knowledge_rows)
        )
        added = knowledge_base.upsert_evidence(incoming_rows)
        knowledge_base.configure_retrieval(incoming_rows, config.retrieval_mode)
        total = knowledge_base.doc_count
        if total != expected_segments:
            raise RuntimeError(
                "零售知识集合数量不符: "
                f"{total} != {expected_segments}；"
                f"请为新运行使用独立集合，不自动清理已有集合 {collection_name}。"
            )
        if config.support_enabled:
            expected_ids = {
                str(row.get("evidence_id") or "") for row in incoming_rows
            }
            actual_ids = set(knowledge_base.list_ids())
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise RuntimeError(
                    "支持集合证据 ID 不完整或包含额外片段；不自动清理。"
                    f" 缺少={missing} 额外={extra}"
                )
        logger.info(
            "零售知识集合 %s: %s 个片段，本次写入 %s 个",
            collection_name,
            total,
            added,
        )
        tool_manager = MCPToolManager(
            api_key=api_key,
            base_url=config.base_url or None,
            model=config.model,
            client=client.for_stage("retrieval"),
            reranker=QwenReranker() if config.reranker_model == "qwen" else None,
            retrieval_strategy=config.retrieval_strategy,
        )
        _register_knowledge_tool(tool_manager, knowledge_base)

    policy_text = config.policy_path.read_text(encoding="utf-8")
    skill_manager = None
    if config.agent_mode == "multi":
        from core.retail_semantic_router import DEFAULT_MODEL, RetailSemanticRouter

        # 四个角色、识别与 Composer 共用同一预算客户端，并按阶段计数。
        orchestrator = AgentOrchestrator(
            client=client.for_stage("roles", count_turn=True),
            model=config.model,
            intent_recognizer=IntentRecognizer(
                model=config.model,
                client=client.for_stage("intent", count_turn=True),
                language=language,
            ),
            composer=ResponseComposer(
                client.for_stage("composer", count_turn=True),
                config.model,
                None,
                language=language,
                retail_mode=True,
            ),
            language=language,
            retail_mode=True,
            policy_text=policy_text,
            retail_role_classes={
                "general": RetailGeneralAgent,
                "technical": RetailTechnicalAgent,
                "billing": RetailBillingAgent,
                "escalation": RetailEscalationAgent,
            },
            retail_support_enabled=config.support_enabled,
            retail_semantic_router=(RetailSemanticRouter(config.semantic_model_path or DEFAULT_MODEL,
                                                         encoder=encoder, owns_encoder=encoder is None)
                                    if config.semantic_routing and language == "zh" else None),
        )
        agent = RetailMultiAgent(
            orchestrator,
            ledger=ledger,
            turn_label=config.case_id or ledger.run_id or "B",
            max_requests_per_turn=32,
            language=language,
        )
    else:
        skills_dir = _env_text(
            "ORDERWEAVE_SKILLS_DIR", _env_text("ECHOMIND_SKILLS_DIR", str(_PROJECT_ROOT / "skills"))
        )
        skill_manager = SkillManager(
            root_dir=skills_dir,
            max_prompt_chars=int(
                _env_text("ORDERWEAVE_SKILLS_MAX_PROMPT_CHARS", _env_text("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", "5000"))
            ),
        )
        skill_manager.load()
        agent = RetailAgent(
            client=client.for_stage("agent"),
            model=config.model,
            store=store,
            tool_manager=tool_manager,
            skill_manager=skill_manager,
            policy_text=policy_text,
            policy_source=(
                f"{config.policy_path} "
                f"sha256={EXPECTED_POLICY_SHA256}"
            ),
            max_tokens=config.max_tokens,
            language=language,
            support_enabled=config.support_enabled,
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
        language=language,
        support_enabled=config.support_enabled,
        support_max_chars=config.support_max_chars,
        workflow_mode=config.workflow_mode,
        checkpoint_path=config.checkpoint_path,
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
