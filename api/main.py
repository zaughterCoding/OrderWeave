"""AgentEcommerce 客服系统的 FastAPI 入口。

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

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
  ==========================================
   AgentEcommerce Customer Service Agent
  ==========================================
"""

_orchestrator: Any = None
_memory: Any = None
_tool_manager: Any = None
_monitor: Any = None
_evaluator: Any = None
_skill_manager: Any = None
_knowledge_base: Any = None
_background_tasks: set[asyncio.Task] = set()


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
    global _evaluator, _skill_manager, _knowledge_base

    print(BANNER, flush=True)
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

        skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", "").strip() or str(
            _project_path("skills")
        )
        _skill_manager = SkillManager(
            root_dir=skills_dir,
            max_prompt_chars=_env_int("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", 5000),
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

        logger.info("AgentEcommerce 已就绪")
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
        logger.info("AgentEcommerce 已关闭")


app = FastAPI(
    title="AgentEcommerce 智能客服",
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
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/trace/tool/{request_id}", response_model=ToolTraceResponse)
async def get_tool_trace(request_id: str):
    _require_orchestrator()
    trace = _orchestrator.get_tool_trace(request_id)
    return ToolTraceResponse(
        request_id=request_id,
        found=trace is not None,
        trace=trace or {},
    )


@app.get("/trace/tools", response_model=RecentToolTracesResponse)
async def list_recent_tool_traces(limit: int = 20):
    _require_orchestrator()
    return RecentToolTracesResponse(
        items=_orchestrator.get_recent_tool_traces(limit=limit)
    )


@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite(
        "knowledge_search", query, top_k=top_k
    )
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
    }


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
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
    return {"total_chunks": await _knowledge_base.doc_count_async()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
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


async def _cli() -> None:
    print(BANNER)
    print("AgentEcommerce CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.skill_loader import SkillManager
    from memory.conversation_memory import MemoryManager, MsgRole

    cfg = _anthropic_cfg()
    skills_dir = os.getenv("ECHOMIND_SKILLS_DIR", "").strip() or str(
        _project_path("skills")
    )
    skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=_env_int("ECHOMIND_SKILLS_MAX_PROMPT_CHARS", 5000),
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
                f"\nAgentEcommerce [{result.agent_type.value}]: "
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
