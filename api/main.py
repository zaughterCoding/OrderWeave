"""OrderWeave 客服系统的 FastAPI 入口。

核心组件在 lifespan 处理器中创建。导入本模块不会连接 Redis、
ChromaDB 或付费模型端点。
"""
import asyncio
import json
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

load_dotenv(pathlib.Path(_ROOT) / ".env")

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
  ==========================================
   OrderWeave Customer Service Agent
  ==========================================
"""

_orchestrator: Any = None
_memory: Any = None
_tool_manager: Any = None
_monitor: Any = None
_evaluator: Any = None
_skill_manager: Any = None
_knowledge_base: Any = None
_retail_runtime: Any = None
_background_tasks: set[asyncio.Task] = set()


def _retail_mode() -> bool:
    """零售模式采用显式开关；关闭时旧客服路径保持不变。"""
    return os.getenv("RETAIL_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _project_path(*parts: str) -> pathlib.Path:
    return pathlib.Path(_ROOT).joinpath(*parts)


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    model = (
        os.getenv("LLM_MODEL", "").strip()
        or os.getenv("ANTHROPIC_MODEL", "").strip()
    )
    if not model:
        raise RuntimeError("未设置 LLM_MODEL")
    cfg: Dict[str, Any] = {"api_key": key, "model": model}
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _build_knowledge_tooling(
    cfg: Dict[str, Any],
    chroma_host: str,
    chroma_port: int,
    chroma_path: str,
):
    """构造 API 与 CLI 共用的知识检索工具链。"""
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool

    tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    knowledge_base = KnowledgeBase(
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=chroma_path,
    )

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
                    "请稍后重试，或转人工客服确认。"
                ),
                "score": 0.0,
                "fallback": True,
                "error": error,
            }
        ]

    tool_manager.register(
        Tool(
            name="knowledge_search",
            description="搜索知识库（基于 ChromaDB 向量检索）",
            handler=knowledge_base.search_handler,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            cache_ttl=300.0,
            supports_rerank=True,
            fallback=knowledge_fallback,
        )
    )
    return tool_manager, knowledge_base


def _spawn_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_done)


def _background_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    try:
        error = task.exception()
    except Exception:
        return
    if error:
        logger.warning(f"后台任务失败: {error}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor
    global _evaluator, _skill_manager, _knowledge_base, _retail_runtime

    print(BANNER, flush=True)
    if _retail_mode():
        from agents.retail_runtime import build_retail_runtime

        try:
            _retail_runtime = build_retail_runtime()
            await _retail_runtime.start()
            _skill_manager = _retail_runtime.skill_manager
            _tool_manager = _retail_runtime.tool_manager
            _knowledge_base = _retail_runtime.knowledge_base
            logger.info(
                "零售模式已就绪: model=%s db=%s collection=%s budget=%.2f CNY",
                _retail_runtime.config.model,
                _retail_runtime.config.db_path,
                (_retail_runtime.knowledge_base.collection_name
                 if _retail_runtime.knowledge_base is not None else "disabled"),
                _retail_runtime.config.budget_cny,
            )
            yield
        finally:
            if _retail_runtime is not None:
                await _retail_runtime.close()
            pending = list(_background_tasks)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            _retail_runtime = None
            _tool_manager = None
            _skill_manager = None
            _knowledge_base = None
            logger.info("OrderWeave 零售模式已关闭")
        return

    try:
        from agents.agent_orchestrator import AgentOrchestrator
        from core.skill_loader import SkillManager
        from evaluation.evaluator import EndToEndEvaluator
        from memory.conversation_memory import MemoryManager
        from monitor.performance_monitor import PerformanceMonitor
        from core.intent_recognizer import IntentRecognizer

        cfg = _anthropic_cfg()
        logger.info(
            f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}"
        )

        recognizer = IntentRecognizer(
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
        )

        skills_dir = (os.getenv("ORDERWEAVE_SKILLS_DIR") or os.getenv("ECHOMIND_SKILLS_DIR", "")).strip() or str(
            _project_path("skills")
        )
        _skill_manager = SkillManager(
            root_dir=skills_dir,
            max_prompt_chars=_env_int("ORDERWEAVE_SKILLS_MAX_PROMPT_CHARS", _env_int("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", 5000)),
        )
        _skill_manager.load()

        redis_url = os.getenv("REDIS_URL", "").strip() or (
            "redis://:agentecommerce123@127.0.0.1:6380/0"
        )
        chroma_host = os.getenv("CHROMA_HOST", "").strip() or "127.0.0.1"
        chroma_port = _env_int("CHROMA_PORT", 8002)
        chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip() or str(
            _project_path("data", "chroma")
        )

        _memory = MemoryManager(
            redis_url=redis_url,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            chroma_path=chroma_path,
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
        )

        _tool_manager, _knowledge_base = _build_knowledge_tooling(
            cfg, chroma_host, chroma_port, chroma_path
        )
        logger.info(
            f"知识库已加载: {await _knowledge_base.doc_count_async()} 个文档片段"
        )

        _orchestrator = AgentOrchestrator(
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
            skill_manager=_skill_manager,
            rag_tool_manager=_tool_manager,
        )

        monitor_interval = _env_float(
            "MONITOR_INTERVAL_S",
            _env_float("MONITOR_INTERVAL", 10.0),
        )
        prometheus_port = _env_int(
            "MONITOR_PROMETHEUS_PORT",
            _env_int("PROMETHEUS_PORT", 0),
        ) or None
        webhook_url = (
            os.getenv("MONITOR_WEBHOOK_URL", "").strip()
            or os.getenv("ALERT_WEBHOOK_URL", "").strip()
            or None
        )
        _monitor = PerformanceMonitor(
            orchestrator=_orchestrator,
            tool_manager=_tool_manager,
            interval_s=monitor_interval,
            webhook_url=webhook_url,
            prometheus_port=prometheus_port,
        )
        await _monitor.start()

        baseline_path = os.getenv("EVAL_BASELINE_PATH", "").strip() or str(
            _project_path("data", "eval", "baseline.json")
        )
        report_dir = os.getenv("EVAL_REPORT_DIR", "").strip() or str(
            _project_path("data", "eval", "reports")
        )
        _evaluator = EndToEndEvaluator(
            orchestrator=_orchestrator,
            recognizer=recognizer,
            api_key=cfg["api_key"],
            base_url=cfg.get("base_url"),
            model=cfg["model"],
            baseline_path=baseline_path,
            report_dir=report_dir,
        )

        logger.info("OrderWeave 已就绪")
        yield
    finally:
        if _monitor is not None:
            await _monitor.stop()
        pending = list(_background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if _memory is not None:
            await _memory.close()

        _orchestrator = None
        _memory = None
        _tool_manager = None
        _monitor = None
        _evaluator = None
        _skill_manager = None
        _knowledge_base = None
        logger.info("OrderWeave 已关闭")


app = FastAPI(
    title="OrderWeave 智能客服",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None


class ChatResponse(BaseModel):
    conv_id: str
    request_id: str = ""
    response: str
    intent: str
    intent_group: str = "other"
    agent_type: str
    agent_types: List[str] = Field(default_factory=list)
    primary_agent: str = ""
    supporting_agents: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    escalated: bool
    latency_ms: float
    knowledge_used: bool = False
    entities: Dict[str, List[str]] = Field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_source_scores: Dict[str, float] = Field(default_factory=dict)
    support_enabled: bool = False
    support_scope: str = ""
    support_stage: str = ""
    support_state_error: str = ""
    support_context_limit: bool = False


class ToolTraceResponse(BaseModel):
    request_id: str
    found: bool
    trace: Dict[str, Any] = Field(default_factory=dict)


class RecentToolTracesResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)


class DocInput(BaseModel):
    title: str
    content: str


class BatchDocInput(BaseModel):
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


def _require_orchestrator() -> None:
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")


@app.get("/health")
async def health():
    if _retail_runtime is not None:
        summary = _retail_runtime.service.summary()
        body = {
            "status": "ok",
            "mode": "retail",
            "agent_mode": summary.get("agent_mode", _retail_runtime.config.agent_mode),
            "agents": summary["agent"],
            "store": _retail_runtime.store.stats(),
            "budget": _retail_runtime.ledger.summary(),
            "embedding_model": (_retail_runtime.config.embedding_model
                                if _retail_runtime.knowledge_base is not None else "disabled"),
            "retrieval_mode": (_retail_runtime.config.retrieval_mode
                               if _retail_runtime.knowledge_base is not None else "disabled"),
            "reranker_model": (_retail_runtime.config.reranker_model
                               if _retail_runtime.knowledge_base is not None else "disabled"),
            "retrieval_strategy": getattr(_retail_runtime.config, "retrieval_strategy", "relevance"),
        }
        if _retail_runtime.config.support_enabled:
            body["support"] = {
                "enabled": True,
                "collection": _retail_runtime.knowledge_base.collection_name,
                "language": _retail_runtime.config.language,
            }
        return body
    _require_orchestrator()
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _retail_runtime is not None:
        conv_id = req.conv_id or str(uuid.uuid4())
        try:
            result = await _retail_runtime.service.handle_turn(
                req.message, req.user_id, conv_id
            )
        except ValueError as ex:
            raise HTTPException(400, str(ex))
        return ChatResponse(
            conv_id=conv_id,
            request_id=result.request_id,
            response=result.response,
            intent="retail_support",
            intent_group="retail",
            agent_type=result.agent_type.value,
            agent_types=result.agent_types or [result.agent_type.value],
            primary_agent=result.primary_agent or result.agent_type.value,
            supporting_agents=result.supporting_agents,
            tools_used=result.tools_used,
            escalated=result.escalated,
            latency_ms=result.latency_ms,
            knowledge_used=result.knowledge_hits > 0,
            entities={},
            intent_confidence=1.0,
            intent_source_scores={},
            support_enabled=result.support_enabled,
            support_scope=result.support_scope,
            support_stage=result.support_stage,
            support_state_error=result.support_state_error,
            support_context_limit=result.support_context_limit,
        )

    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcRequest
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
    history = (
        [
            {"role": message.role.value, "content": message.content}
            for message in mem_ctx.recent_messages[-5:]
        ]
        if mem_ctx.recent_messages
        else None
    )

    intent_result = await _orchestrator.recognize_intent(
        req.message, history=history
    )
    orch_request = OrcRequest(
        message=req.message,
        user_id=req.user_id,
        conv_id=conv_id,
        context=mem_ctx.to_prompt_text(),
        history=history,
        entities=intent_result.entities,
        intent=intent_result.intent,
        intent_group=intent_result.intent_group,
        urgency=intent_result.urgency,
        intent_confidence=intent_result.confidence,
    )
    result = await _orchestrator.run(orch_request)

    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
    await _memory.add_message(
        req.user_id, conv_id, MsgRole.ASSISTANT, result.response
    )
    _spawn_background(_memory.update_profile(req.user_id, conv_id))

    return ChatResponse(
        conv_id=conv_id,
        request_id=result.request_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        intent_group=intent_result.intent_group,
        agent_type=result.agent_type.value,
        agent_types=[agent.value for agent in result.agent_types],
        primary_agent=(
            result.primary_agent.value
            if result.primary_agent
            else result.agent_type.value
        ),
        supporting_agents=[agent.value for agent in result.supporting_agents],
        tools_used=result.tools_used,
        routing_reason=result.routing_reason,
        routing_confidence=result.routing_confidence,
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=result.knowledge_hits > 0,
        entities=intent_result.entities,
        intent_confidence=round(intent_result.confidence, 4),
        intent_source_scores=intent_result.source_scores,
    )


@app.get("/monitor")
async def monitor_summary():
    if _retail_runtime is not None:
        summary = _retail_runtime.service.summary()
        summary["recent_traces"] = _retail_runtime.service.get_recent_tool_traces(
            limit=20
        )
        return summary
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    if _retail_runtime is not None:
        trace = _retail_runtime.service.get_tool_trace(request_id)
        return ToolTraceResponse(
            request_id=request_id,
            found=trace is not None,
            trace=trace or {},
        )
    _require_orchestrator()
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    if _retail_runtime is not None:
        return RecentToolTracesResponse(
            items=_retail_runtime.service.get_recent_tool_traces(limit=limit)
        )
    _require_orchestrator()
    return RecentToolTracesResponse(
        items=_orchestrator.get_recent_tool_traces(limit=limit)
    )


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    extra = {}
    if _retail_runtime is not None:
        top_k = max(1, min(int(top_k), 10))
        from agents.retail_tools import retail_evidence_applicable
        from agents.retail_support import analyze_message, build_support_packet
        packet = None
        if _retail_runtime.config.support_enabled:
            analysis = analyze_message(query, None, None, False)
            packet = build_support_packet(None, analysis, query, 0, None, [])
        extra = {
            "item_filter": lambda item: retail_evidence_applicable(item, packet, query),
            # 独立搜索保留原候选数；会话支持工具的20条预算由工具适配器控制。
            "expand_filtered_recall": False,
        }
        if getattr(_tool_manager, "retrieval_strategy", "relevance") != "relevance":
            extra.update(original_query=query, coverage_context=packet)
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite(
        "knowledge_search", query, top_k=top_k, **extra
    )
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
    }


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    if _retail_runtime is not None:
        raise HTTPException(
            409, "零售模式使用冻结知识集合，禁止在线导入或修改"
        )
    if _knowledge_base is None:
        raise HTTPException(503, "知识库未初始化")
    count = await _knowledge_base.add_documents_async(
        [
            {"title": document.title, "content": document.content}
            for document in body.documents
        ]
    )
    total = await _knowledge_base.doc_count_async()
    return {
        "message": f"成功导入 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    if _retail_runtime is not None:
        raise HTTPException(
            409, "零售模式使用冻结知识集合，禁止在线导入或修改"
        )
    if _knowledge_base is None:
        raise HTTPException(503, "知识库未初始化")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.lower().endswith(".json"):
        try:
            documents = json.loads(text)
        except json.JSONDecodeError as ex:
            raise HTTPException(400, f"JSON 解析失败: {ex}")
        if not isinstance(documents, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("title"), str)
            and isinstance(item.get("content"), str)
            for item in documents
        ):
            raise HTTPException(
                400,
                "JSON 文件应为数组格式: [{title, content}, ...]",
            )
    else:
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        documents = [{"title": title, "content": text}]

    count = await _knowledge_base.add_documents_async(documents)
    total = await _knowledge_base.doc_count_async()
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": total,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    if _knowledge_base is None:
        raise HTTPException(503, "知识库未初始化")
    total = await _knowledge_base.doc_count_async()
    if _retail_runtime is not None:
        config = _retail_runtime.config
        if not config.support_enabled:
            return {
                "total_chunks": total,
                "collection": _knowledge_base.collection_name,
                "embedding_model": config.embedding_model,
                "retrieval_mode": config.retrieval_mode,
                "reranker_model": config.reranker_model,
                "frozen": True,
                "mode": "retail",
            }
        return {
            "total_chunks": total,
            "collection": _knowledge_base.collection_name,
            "embedding_model": config.embedding_model,
            "retrieval_mode": config.retrieval_mode,
            "reranker_model": config.reranker_model,
            "frozen": True,
            "mode": "retail",
            "support_enabled": True,
            "support_material": "project_authored_sandbox",
        }
    return {"total_chunks": total}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    if _retail_runtime is not None:
        raise HTTPException(
            409, "零售模式不运行旧评测、模型用户模拟器或付费 Judge"
        )
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")

    from evaluation.evaluator import (
        DEFAULT_DIALOG_CASES,
        DEFAULT_INTENT_CASES,
        IntentTestCase,
    )

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=case.message,
                expected_intent=case.expected_intent,
                context=case.context,
            )
            for case in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            case.model_dump(exclude_none=True) for case in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate": report.pass_rate,
        "total": report.total,
        "passed": report.passed,
        "grader_errors": report.grader_errors,
        "judge_summary": report.judge_summary,
        "avg_scores": report.avg_scores,
        "regressions": report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": result.test_id,
                "passed": result.passed,
                "scores": result.scores,
                "detail": result.detail,
                "metadata": result.metadata,
            }
            for result in report.results
        ],
    }


async def _retail_cli() -> None:
    """零售模式 CLI；与 API 共用 RetailService 和同一账本。"""
    from agents.retail_runtime import build_retail_runtime

    runtime = build_retail_runtime()
    user_id = os.getenv("RETAIL_CLIENT_USER", "").strip() or "cli_user"
    conv_id = os.getenv("RETAIL_CONV_ID", "").strip() or str(uuid.uuid4())
    print(
        f"零售模式：会话 {conv_id}，客户端用户 {user_id}；"
        f"Agent 模式 {runtime.config.agent_mode}；数据库 {runtime.config.db_path}；"
        f"有限支持 {'开启' if runtime.config.support_enabled else '关闭'}"
    )
    try:
        await runtime.start()
        while True:
            try:
                message = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                break
            if not message or message.lower() in ("quit", "exit", "退出"):
                print("再见")
                break
            try:
                result = await runtime.service.handle_turn(
                    message, user_id, conv_id
                )
            except ValueError as ex:
                print(f"\n系统: {ex}\n")
                continue
            roles = ",".join(result.agent_types) or "retail"
            print(f"\nOrderWeave [retail:{roles}]: {result.response}\n")
    finally:
        await runtime.close()


async def _cli() -> None:
    print(BANNER)
    print("OrderWeave CLI — 输入 quit 退出\n")

    if _retail_mode():
        await _retail_cli()
        return

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.skill_loader import SkillManager
    from memory.conversation_memory import MemoryManager, MsgRole

    cfg = _anthropic_cfg()
    skills_dir = (os.getenv("ORDERWEAVE_SKILLS_DIR") or os.getenv("ECHOMIND_SKILLS_DIR", "")).strip() or str(
        _project_path("skills")
    )
    skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=_env_int("ORDERWEAVE_SKILLS_MAX_PROMPT_CHARS", _env_int("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", 5000)),
    )
    skill_manager.load()

    redis_url = os.getenv("REDIS_URL", "").strip() or (
        "redis://:agentecommerce123@127.0.0.1:6380/0"
    )
    chroma_host = os.getenv("CHROMA_HOST", "").strip() or "127.0.0.1"
    chroma_port = _env_int("CHROMA_PORT", 8002)
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "").strip() or str(
        _project_path("data", "chroma")
    )
    # CLI 需要与 API 相同的知识链，否则手工验证时工具调用会静默
    # 退化为“找不到工具”。
    tool_manager, _knowledge_base = _build_knowledge_tooling(
        cfg, chroma_host, chroma_port, chroma_path
    )

    orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
        rag_tool_manager=tool_manager,
    )
    memory = MemoryManager(
        redis_url=redis_url,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        chroma_path=chroma_path,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())
    try:
        while True:
            try:
                message = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见")
                break
            if not message or message.lower() in ("quit", "exit", "退出"):
                print("再见")
                break

            context = await memory.get_context(
                user_id, conv_id, query=message
            )
            history = (
                [
                    {"role": item.role.value, "content": item.content}
                    for item in context.recent_messages[-5:]
                ]
                if context.recent_messages
                else None
            )
            request = Request(
                message=message,
                user_id=user_id,
                conv_id=conv_id,
                context=context.to_prompt_text(),
                history=history,
            )
            result = await orchestrator.run(request)

            await memory.add_message(
                user_id, conv_id, MsgRole.USER, message
            )
            await memory.add_message(
                user_id, conv_id, MsgRole.ASSISTANT, result.response
            )
            print(
                f"\nOrderWeave [{result.agent_type.value}]: "
                f"{result.response}\n"
            )
    finally:
        await memory.close()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
