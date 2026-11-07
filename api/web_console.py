"""OrderWeave Web 控制台后端：独立会话库、实时链路事件（SSE）与数据库快照差异。

每个 Web 会话复制一份 τ² retail 种子库并启动独立运行时（英文评测同款 profile，
不需要 Redis／Chroma）。一轮对话中的工作图节点、模型调用、工具调用、守卫等
事件经 monitor.live_events 实时推送；每次工具调用后对“相关实体”取快照并
给出字段级差异。所有事件同时写入会话目录下的 events.jsonl，可离线回放。

启动：uv run python -m api.web_console（默认 http://127.0.0.1:8010）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import APIRouter, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from monitor import live_events  # noqa: E402

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path(os.getenv("ORDERWEAVE_WEB_SESSIONS", str(ROOT / "data" / "web-sessions"))).resolve()
BUILTIN_REPLAYS = ROOT / "web" / "replays"
WEB_DIST = ROOT / "web" / "dist"
MAX_SESSIONS = max(1, int(os.getenv("ORDERWEAVE_WEB_MAX_SESSIONS", "4")))
# 单会话账本上限只防失控；与评测单会话上限一致。
SESSION_BUDGET_CNY = float(os.getenv("ORDERWEAVE_WEB_SESSION_BUDGET_CNY", "0.5"))
GREETING = "Hi! How can I help you today?"

# 预置演示场景：身份与订单来自公开 τ² retail 沙箱数据。
SCENARIOS: List[Dict[str, Any]] = [
    {"id": "cancel", "kind": "cancel_pending_order",
     "title": {"en": "Cancel a pending order", "zh": "取消待发货订单"},
     "persona": {"name": "Ivan Santos", "zip": "75277", "email": "ivan.santos3158@example.com"},
     "focus": "#W8770097",
     "messages": ["Hi, I'm Ivan Santos, zip code 75277. I'd like to cancel my order #W8770097, "
                  "the red office chair. I ordered it by mistake.",
                  "Yes, please go ahead and cancel it."]},
    {"id": "return", "kind": "return_delivered_order",
     "title": {"en": "Return a delivered item", "zh": "退回已送达商品"},
     "persona": {"name": "Sofia Li", "zip": "78260", "email": "sofia.li7352@example.com"},
     "focus": "#W4689314",
     "messages": ["Hello, my email is sofia.li7352@example.com. I want to return the digital camera "
                  "from order #W4689314.",
                  "Please refund it to the original credit card.",
                  "Yes, please proceed with the return."]},
    {"id": "exchange", "kind": "exchange_delivered_order",
     "title": {"en": "Exchange for another variant", "zh": "换成其他款式"},
     "persona": {"name": "Yusuf Hernandez", "zip": "43175", "email": "yusuf.hernandez6086@example.com"},
     "focus": "#W1633718",
     "messages": ["Hi, this is Yusuf Hernandez, zip 43175. The coffee maker in order #W1633718 is too small; "
                  "can I exchange it for a version with a larger capacity?",
                  "Any available option with a larger capacity is fine; keep the other features the same if possible, "
                  "and use my original payment method for any price difference.",
                  "Yes, please proceed with the exchange."]},
    {"id": "address", "kind": "modify_user_address",
     "title": {"en": "Update the default address", "zh": "修改默认地址"},
     "persona": {"name": "Emma Smith", "zip": "10192", "email": "emma.smith3991@example.com"},
     "focus": "emma_smith_8564",
     "messages": ["Hi, I'm Emma Smith (emma.smith3991@example.com). I moved; please change my default address "
                  "to 120 Park Avenue, Apt 5B, New York, NY 10017, USA.",
                  "Yes, please proceed with the address update."]},
]

# 工作图骨架；节点名与 agents/retail_workflow.py 编译图一致（测试校验）。
_COMMON_EDGES = [
    ("__start__", "prepare_state", ""), ("__start__", "recover_result", "recovering"),
    ("prepare_state", "route_agent", ""), ("route_agent", "cancel_return", "cancel"),
    ("route_agent", "leave_return", "human"), ("cancel_return", "update_result", ""),
    ("run_agent", "update_result", ""), ("update_result", "wait_user", "awaiting confirmation"),
    ("update_result", "__end__", ""), ("wait_user", "resume_input", "user reply"),
    ("resume_input", "prepare_state", ""), ("recover_result", "__end__", ""),
]
_ROLES = ("general", "technical", "billing", "escalation")


def graph_spec(mode: str) -> Dict[str, Any]:
    edges = list(_COMMON_EDGES)
    if mode == "single":
        edges += [("route_agent", "run_agent", ""), ("leave_return", "run_agent", "")]
    else:
        edges += [("route_agent", "select_role", ""), ("route_agent", "run_agent", "support blocked"),
                  ("leave_return", "select_role", ""), ("select_role", "update_result", "clarify"),
                  ("handoff_role", "compose_roles", "queue done"), ("handoff_role", "update_result", "error"),
                  ("compose_roles", "update_result", "")]
        for role in _ROLES:
            edges += [("select_role", role, ""), (role, "handoff_role", ""), ("handoff_role", role, "next role")]
    nodes = sorted({n for edge in edges for n in edge[:2]})
    return {"mode": mode, "nodes": nodes,
            "edges": [{"source": s, "target": t, "label": label} for s, t, label in edges]}


# ---------------------------------------------------------------- 数据库快照

_STATE_KEYS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "sessions": lambda r: "session",
    "pending_actions": lambda r: str(r["op_id"]),
    "confirmation_decisions": lambda r: f"turn {r['turn']} · {r['op_id']}",
    "request_plan": lambda r: "plan",
    "task_facts": lambda r: "facts",
    "context_state": lambda r: "context",
    "retail_events": lambda r: str(r["event_id"]),
}
# 每轮必变、对理解无帮助的字段不进入差异。
_DIFF_IGNORE = {"updated_at"}


def _connect_ro(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection


def _decode_row(row: sqlite3.Row) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in row.keys():
        if key in ("client_user_id", "conv_id"):
            continue
        value = row[key]
        if isinstance(value, str) and (key.endswith("_json") or key == "payload"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            key = key[:-5] if key.endswith("_json") else key
        out[key] = value
    return out


def _payloads(connection: sqlite3.Connection, sql: str, args: Iterable[Any]) -> List[Dict[str, Any]]:
    return [json.loads(row[0]) for row in connection.execute(sql, tuple(args)).fetchall()]


def read_business(path: Path, user_id: Optional[str], order_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """当前会话相关的业务实体：绑定用户、其订单、被查询过的订单及订单内商品。"""
    result: Dict[str, Dict[str, Any]] = {"users": {}, "orders": {}, "products": {}}
    with closing(_connect_ro(path)) as connection:
        if user_id:
            for user in _payloads(connection, "SELECT payload FROM users WHERE user_id=?", [user_id]):
                result["users"][user["user_id"]] = user
            for order in _payloads(connection, "SELECT payload FROM orders WHERE user_id=?", [user_id]):
                result["orders"][order["order_id"]] = order
        extra = [o for o in order_ids if o not in result["orders"]]
        if extra:
            marks = ",".join("?" * len(extra))
            for order in _payloads(connection, f"SELECT payload FROM orders WHERE order_id IN ({marks})", extra):
                result["orders"][order["order_id"]] = order
        product_ids = sorted({str(item.get("product_id")) for order in result["orders"].values()
                              for item in order.get("items") or [] if item.get("product_id")})
        if product_ids:
            marks = ",".join("?" * len(product_ids))
            for product in _payloads(connection, f"SELECT payload FROM products WHERE product_id IN ({marks})",
                                     product_ids):
                result["products"][product["product_id"]] = product
    return result


def read_state(path: Path, client_id: str, conv_id: str) -> Dict[str, Dict[str, Any]]:
    """本会话的内部状态表（待确认动作、确认判定、请求清单、事实、上下文、业务事件）。"""
    result: Dict[str, Dict[str, Any]] = {}
    with closing(_connect_ro(path)) as connection:
        for table, key_of in _STATE_KEYS.items():
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE client_user_id=? AND conv_id=?", (client_id, conv_id)).fetchall()
            result[table] = {key_of(row): _decode_row(row) for row in rows}
    return result


def bound_user(path: Path, client_id: str, conv_id: str) -> Optional[str]:
    with closing(_connect_ro(path)) as connection:
        row = connection.execute("SELECT retail_user_id FROM sessions WHERE client_user_id=? AND conv_id=?",
                                 (client_id, conv_id)).fetchone()
    return row[0] if row and row[0] else None


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, dict) and value:
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _DIFF_IGNORE:
                continue
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, list) and value:
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def field_diff(before: Any, after: Any) -> List[Dict[str, Any]]:
    old, new = _flatten(before), _flatten(after)
    return [{"path": path, "before": old.get(path), "after": new.get(path)}
            for path in sorted(set(old) | set(new)) if old.get(path) != new.get(path)]


def diff_snapshots(previous: Dict[str, Any], current: Dict[str, Any],
                   baseline: Callable[[str, str], Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """业务实体新进入视图时与种子库比较（τ 工具不新建业务行），状态表按增删改比较。"""
    changes: List[Dict[str, Any]] = []
    for group in ("business", "state"):
        old_group, new_group = previous.get(group) or {}, current.get(group) or {}
        for table in sorted(set(old_group) | set(new_group)):
            old_rows, new_rows = old_group.get(table) or {}, new_group.get(table) or {}
            for key in sorted(set(old_rows) | set(new_rows), key=str):
                before, after = old_rows.get(key), new_rows.get(key)
                if group == "business":
                    if after is None:
                        continue  # 离开视图不是数据变化
                    if before is None:
                        before = baseline(table, key)
                        if before is None:
                            continue
                elif before is None or after is None:
                    changes.append({"group": group, "table": table, "key": key,
                                    "op": "added" if before is None else "removed"})
                    continue
                fields = field_diff(before, after)
                if fields:
                    changes.append({"group": group, "table": table, "key": key, "op": "changed",
                                    "fields": fields})
    return changes


def redact_paths(text: str) -> str:
    """录制文件可能被公开分享：提示词里的本机项目路径替换为占位符。"""
    root = str(ROOT)
    for variant in (json.dumps(root)[1:-1], root, root.replace("\\", "/")):
        text = text.replace(variant, "<project>")
    return text


# ---------------------------------------------------------------- 会话

@dataclass
class WebSession:
    id: str
    mode: str
    language: str
    folder: Path
    runtime: Any
    client_id: str
    conv_id: str
    created: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    turn: int = 0
    seq: int = 0
    touched_orders: set = field(default_factory=set)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    turn_snapshots: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    task: Optional[asyncio.Task] = None
    template: Optional[Path] = None

    @property
    def db_path(self) -> Path:
        return self.folder / "retail.sqlite"

    @property
    def log_path(self) -> Path:
        return self.folder / "events.jsonl"

    def read_snapshot(self) -> Dict[str, Any]:
        user = bound_user(self.db_path, self.client_id, self.conv_id)
        return {"bound_user": user,
                "business": read_business(self.db_path, user, sorted(self.touched_orders)),
                "state": read_state(self.db_path, self.client_id, self.conv_id)}

    def baseline(self, table: str, key: str) -> Optional[Dict[str, Any]]:
        if self.template is None:
            return None
        key_column = {"users": "user_id", "orders": "order_id", "products": "product_id"}[table]
        with closing(_connect_ro(self.template)) as connection:
            row = connection.execute(f"SELECT payload FROM {table} WHERE {key_column}=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def log(self, event: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(redact_paths(json.dumps(event, ensure_ascii=False, default=str)) + "\n")


class SessionManager:
    def __init__(self, root: Path = SESSIONS_DIR, runtime_factory: Optional[Callable[..., Any]] = None):
        self.root = root
        self.sessions: Dict[str, WebSession] = {}
        self._factory = runtime_factory or build_session_runtime
        self._create_lock = asyncio.Lock()

    def template_for(self, language: str) -> Path:
        """首次使用时由种子导入一份模板库，之后每个会话直接复制文件。"""
        from evaluation.harness import config as cfg
        from mcp_tools.retail_store import RetailStore

        path = self.root / f"_template-{language}" / "retail.sqlite"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            RetailStore(path, cfg.SEED_PATH, language=language)
        return path

    async def create(self, mode: str, language: str) -> WebSession:
        async with self._create_lock:
            while len(self.sessions) >= MAX_SESSIONS:
                oldest = min(self.sessions.values(), key=lambda s: s.created)
                await self.close(oldest.id)
            session_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            folder = self.root / session_id
            folder.mkdir(parents=True)
            template = await asyncio.to_thread(self.template_for, language)
            shutil.copyfile(template, folder / "retail.sqlite")
            runtime = await self._factory(folder, mode, language)
            session = WebSession(id=session_id, mode=mode, language=language, folder=folder, runtime=runtime,
                                 client_id=f"web-{session_id}", conv_id=f"conv-{session_id}", template=template)
            session.snapshot = session.read_snapshot()
            session.turn_snapshots[0] = session.snapshot
            session.log({"type": "session", "id": session_id, "mode": mode, "language": language,
                         "greeting": GREETING, "graph": graph_spec(mode), "ts": time.time(),
                         "snapshot": session.snapshot})
            self.sessions[session_id] = session
            return session

    def get(self, session_id: str) -> WebSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(404, "session not found")
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        if session.task is not None and not session.task.done():
            # 不取消进行中的写操作，等待本轮结束再关闭。
            try:
                await asyncio.wait_for(asyncio.shield(session.task), timeout=120)
            except Exception:
                logger.warning("会话 %s 关闭前本轮未正常结束", session_id)
        try:
            await session.runtime.close()
        except Exception:
            logger.exception("关闭会话运行时失败")

    async def close_all(self) -> None:
        for session_id in list(self.sessions):
            await self.close(session_id)


async def build_session_runtime(folder: Path, mode: str, language: str) -> Any:
    """与评测 plan profile 同款配置：compiled 上下文、按需 Skill、发布校验、请求清单。"""
    from agents.retail_runtime import RetailRuntimeConfig, build_retail_runtime
    from evaluation.harness import config as cfg

    model = cfg.model_settings()
    config = RetailRuntimeConfig(
        db_path=folder / "retail.sqlite", seed_path=cfg.SEED_PATH, policy_path=cfg.POLICY_PATH,
        knowledge_path=folder / "unused-knowledge.jsonl", knowledge_collection="web_unused",
        ledger_path=folder / "ledger.sqlite", event_log_path=folder / "service-events.jsonl",
        budget_cny=SESSION_BUDGET_CNY, input_price_cny_per_mtok=cfg.INPUT_CNY_PER_MTOK,
        output_price_cny_per_mtok=cfg.OUTPUT_CNY_PER_MTOK, model=model["model"], base_url=model["base_url"],
        redis_url="", chroma_host="127.0.0.1", chroma_port=0, chroma_path=folder / "chroma",
        case_id=folder.name, run_id=f"web-{folder.name}", profile_enabled=False,
        memory_key_prefix=f"web:{folder.name}:", memory_collection_suffix="_web",
        max_tokens=2048, max_message_chars=cfg.MAX_MESSAGE_CHARS, max_context_chars=12000,
        mcp_timeout_seconds=60.0, mcp_stderr_log=folder / "mcp.log", agent_mode=mode,
        enable_memory=False, enable_knowledge=False, language=language,
        workflow_mode="langgraph", checkpoint_path=folder / "checkpoints.sqlite",
        semantic_routing=False, ledger_settlement_policy=model["settlement_policy"],
        model_thinking="disabled", context_mode="compiled", skill_mode="on_demand",
        task_facts=False, publish_revision=True, request_plan=True)
    runtime = build_retail_runtime(config)
    await runtime.start()
    return runtime


def _order_ids_in(value: Any) -> List[str]:
    text = json.dumps(value, ensure_ascii=False, default=str)
    import re

    return sorted(set(re.findall(r"#W\d{7}", text)))


async def run_turn(session: WebSession, message: str) -> asyncio.Queue:
    """启动一轮对话，返回事件队列；None 表示本轮事件结束。客户端断开不影响本轮执行。"""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    session.turn += 1
    turn = session.turn

    def push(event: Dict[str, Any]) -> None:
        session.seq += 1
        event = {**event, "seq": session.seq, "turn": turn}
        event.setdefault("ts", time.time())
        session.log(event)
        if threading.get_ident() == loop_thread:
            queue.put_nowait(event)
        else:
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def refresh(trigger: Dict[str, Any]) -> None:
        current = session.read_snapshot()
        changes = diff_snapshots(session.snapshot, current, session.baseline)
        entered = current["bound_user"] != session.snapshot.get("bound_user") or any(
            set((current["business"].get(t) or {})) != set((session.snapshot.get("business", {}).get(t) or {}))
            for t in ("users", "orders", "products"))
        session.snapshot = current
        if changes or entered:
            push({"type": "db", "trigger": trigger, "changes": changes, "snapshot": current})

    def sink(event: Dict[str, Any]) -> None:
        push(event)
        if event.get("type") == "point" and event.get("kind") == "tool":
            data = event.get("data") or {}
            session.touched_orders.update(_order_ids_in(data.get("input")))
            refresh({"tool": event.get("name"), "event": event.get("id")})

    async def work() -> None:
        started = time.monotonic()
        spent_before = session.runtime.ledger.summary()["spent_cny"]
        push({"type": "turn_start", "message": message})
        try:
            with live_events.capture(sink):
                result = await session.runtime.service.handle_turn(message, session.client_id, session.conv_id)
            refresh({"turn_end": True})
            session.turn_snapshots[turn] = session.snapshot
            summary = session.runtime.ledger.summary()
            push({"type": "turn_end", "response": result.response, "success": result.success,
                  "tools_used": list(result.tools_used or []), "escalated": bool(result.escalated),
                  "workflow": live_events.clip(getattr(result, "workflow", {}) or {}),
                  "response_guard": live_events.clip(getattr(result, "response_guard", {}) or {}),
                  "latency_ms": round((time.monotonic() - started) * 1000, 1),
                  "cost_cny": round(summary["spent_cny"] - spent_before, 6),
                  "session_cost_cny": round(summary["spent_cny"], 6)})
        except Exception as ex:
            logger.exception("Web 会话本轮失败")
            session.turn_snapshots[turn] = session.snapshot
            push({"type": "turn_error", "error": f"{type(ex).__name__}: {ex}"[:1000]})
        finally:
            if session.lock.locked():
                session.lock.release()
            loop.call_soon(queue.put_nowait, None)

    session.task = asyncio.create_task(work())
    return queue


# ---------------------------------------------------------------- 接口

class SessionInput(BaseModel):
    mode: str = Field("multi", pattern="^(single|multi)$")
    language: str = Field("en", pattern="^(en|zh)$")


class TurnInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


def _search_business(path: Path, query: str, limit: int = 30) -> Dict[str, Any]:
    pattern = f"%{query.lower()}%"
    with closing(_connect_ro(path)) as connection:
        users = _payloads(connection, "SELECT payload FROM users WHERE lower(user_id) LIKE ? OR email_lower LIKE ? "
                          "OR first_lower || ' ' || last_lower LIKE ? LIMIT ?", [pattern, pattern, pattern, limit])
        orders = _payloads(connection, "SELECT payload FROM orders WHERE lower(order_id) LIKE ? "
                           "OR lower(user_id) LIKE ? LIMIT ?", [pattern, pattern, limit])
        products = _payloads(connection, "SELECT payload FROM products WHERE lower(payload) LIKE ? LIMIT ?",
                             [pattern, limit])
        counts = {t: connection.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("users", "orders", "products")}
    return {"counts": counts, "users": users, "orders": orders,
            "products": [{"product_id": p["product_id"], "name": p.get("name"),
                          "variants": len(p.get("variants") or {})} for p in products]}


def _replay_files(sessions_root: Path) -> List[Dict[str, Any]]:
    items = []
    for source, folder, pattern in (("builtin", BUILTIN_REPLAYS, "*.jsonl"),
                                    ("recorded", sessions_root, "*/events.jsonl")):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob(pattern)):
            if path.parent.name.startswith("_"):
                continue
            name = path.stem if source == "builtin" else path.parent.name
            turns = sum(1 for line in path.open(encoding="utf-8") if '"type": "turn_start"' in line)
            if turns:
                items.append({"source": source, "name": name, "turns": turns,
                              "modified": path.stat().st_mtime})
    # 内置回放按文件名编号排序，录制的会话最新在前。
    return sorted(items, key=lambda item: (item["source"] != "builtin",
                                           item["name"] if item["source"] == "builtin" else -item["modified"]))


def _replay_path(sessions_root: Path, source: str, name: str) -> Path:
    if not name.replace("-", "").replace("_", "").isalnum():
        raise HTTPException(400, "invalid replay name")
    path = BUILTIN_REPLAYS / f"{name}.jsonl" if source == "builtin" else sessions_root / name / "events.jsonl"
    if source not in ("builtin", "recorded") or not path.is_file():
        raise HTTPException(404, "replay not found")
    return path


def build_router(manager: SessionManager) -> APIRouter:
    router = APIRouter(prefix="/api/web")

    @router.get("/scenarios")
    async def scenarios():
        return {"greeting": GREETING, "items": SCENARIOS}

    @router.get("/graph")
    async def graph(mode: str = "multi"):
        if mode not in ("single", "multi"):
            raise HTTPException(400, "mode must be single or multi")
        return graph_spec(mode)

    @router.post("/sessions")
    async def create_session(body: SessionInput):
        try:
            session = await manager.create(body.mode, body.language)
        except RuntimeError as ex:
            raise HTTPException(503, str(ex))
        return {"id": session.id, "mode": session.mode, "language": session.language, "greeting": GREETING,
                "graph": graph_spec(session.mode), "snapshot": session.snapshot}

    @router.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        await manager.close(session_id)
        return {"closed": session_id}

    @router.post("/sessions/{session_id}/turns")
    async def post_turn(session_id: str, body: TurnInput):
        session = manager.get(session_id)
        if session.lock.locked():
            raise HTTPException(409, "a turn is already running")
        await session.lock.acquire()
        try:
            queue = await run_turn(session, body.message.strip())
        except Exception:
            session.lock.release()
            raise

        async def stream():
            # 锁由本轮任务结束时释放；客户端中途断开不中断执行。
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield "data: " + json.dumps(event, ensure_ascii=False, default=str) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.get("/sessions/{session_id}/db")
    async def database(session_id: str, q: str = ""):
        session = manager.get(session_id)
        if q.strip():
            return await asyncio.to_thread(_search_business, session.db_path, q.strip())
        return session.snapshot

    @router.get("/sessions/{session_id}/snapshots")
    async def snapshots(session_id: str):
        return {"turns": sorted(manager.get(session_id).turn_snapshots)}

    @router.get("/sessions/{session_id}/snapshots/{turn}")
    async def snapshot(session_id: str, turn: int):
        session = manager.get(session_id)
        if turn not in session.turn_snapshots:
            raise HTTPException(404, "snapshot not found")
        return session.turn_snapshots[turn]

    @router.get("/replays")
    async def replays():
        return {"items": await asyncio.to_thread(_replay_files, manager.root)}

    @router.get("/replays/{source}/{name}")
    async def replay(source: str, name: str):
        return FileResponse(_replay_path(manager.root, source, name), media_type="application/x-ndjson")

    return router


def create_app(manager: Optional[SessionManager] = None) -> FastAPI:
    manager = manager or SessionManager()

    @asynccontextmanager
    async def lifespan(_app):
        from evaluation.harness import config as cfg

        cfg.load_env()
        manager.root.mkdir(parents=True, exist_ok=True)
        yield
        await manager.close_all()

    app = FastAPI(title="OrderWeave Console", lifespan=lifespan, docs_url="/api/docs")
    app.state.sessions = manager
    app.include_router(build_router(manager))
    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    uvicorn.run(app, host=os.getenv("ORDERWEAVE_WEB_HOST", "127.0.0.1"),
                port=int(os.getenv("ORDERWEAVE_WEB_PORT", "8010")))
