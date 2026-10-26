"""零售真实调用的跨进程预算账本。

API 与 CLI 进程共用同一个 SQLite 账本：请求前按输入上界和 max_tokens 预留，
响应后按 usage 结算；失败或缺少 usage 时按预留上界计费。一旦实际用量超过
预留上界，账本永久拒绝后续请求，不清理、不重置额度。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from evaluation.retail_budget import (
    INPUT_COUNT_BASIS,
    REQUEST_FORMAT_OVERHEAD_TOKENS,
    BudgetStopExceeded,
    TurnRequestLimitExceeded,
    _json_default,
    request_input_upper_tokens,
)


class LedgerError(RuntimeError):
    """账本参数不一致或状态不允许继续。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RetailReservation:
    request_id: str
    stage: str
    model: str
    input_tokens_upper: int
    max_tokens: int
    reserved_cny: float
    created_at: str = ""


class RetailLedger:
    """单文件 SQLite 预算账本，可在多个进程间共享。"""

    def __init__(
        self,
        path: str | Path,
        budget_cny: float,
        input_price_cny_per_mtok: float,
        output_price_cny_per_mtok: float,
        run_id: str = "",
        model: str = "",
        *,
        settlement_policy: str = "legacy",
    ):
        if budget_cny <= 0:
            raise LedgerError("预算必须大于 0")
        if input_price_cny_per_mtok <= 0 or output_price_cny_per_mtok <= 0:
            raise LedgerError(
                "真实调用价格必须大于 0；无法核实时不要启动付费运行"
            )
        self.path = Path(path).expanduser().resolve()
        if settlement_policy not in {"legacy", "deepseek_flash_cache_20261024"}:
            raise LedgerError("未知结算策略")
        if settlement_policy != "legacy" and (model != "deepseek-flash" or input_price_cny_per_mtok != 2 or output_price_cny_per_mtok != 8):
            raise LedgerError("缓存分价策略须固定deepseek-flash及峰时2／8预留上界")
        self.settlement_policy = settlement_policy
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # 最近一次在发送前被拒绝的原因；调用方据此区分预算停止与普通模型失败。
        self._last_rejection_reason = ""
        # 单轮请求上限只在当前进程内计数，由评测入口在每个外部用户轮次开始时重置。
        self._turn_label = ""
        self._turn_requests = 0
        self._turn_max_requests = 0
        self._initialize(
            float(budget_cny),
            float(input_price_cny_per_mtok),
            float(output_price_cny_per_mtok),
            run_id.strip(),
            model.strip(),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(
        self,
        budget: float,
        input_price: float,
        output_price: float,
        run_id: str,
        model: str,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    request_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_cny REAL NOT NULL,
                    actual_cny REAL,
                    input_tokens_upper INTEGER NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    input_tokens_main INTEGER,
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    usage_present INTEGER NOT NULL DEFAULT 0,
                    bound_violation TEXT,
                    request_sha256 TEXT NOT NULL,
                    request_bytes INTEGER NOT NULL,
                    latency_ms REAL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    settled_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_entries_status
                    ON entries(status);
                """
            )
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM meta"
                ).fetchall()
            }
            if not meta:
                self._run_id = run_id or f"t13-{uuid.uuid4().hex[:10]}"
                values = {
                    "budget_cny": repr(budget),
                    "input_cny_per_mtok": repr(input_price),
                    "output_cny_per_mtok": repr(output_price),
                    "run_id": self._run_id,
                    "model": model,
                    "input_count_basis": INPUT_COUNT_BASIS,
                    "format_overhead_tokens": str(REQUEST_FORMAT_OVERHEAD_TOKENS),
                    "overrun_reason": "",
                    "created_at": _now(),
                    "settlement_policy": self.settlement_policy,
                }
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    list(values.items()),
                )
                self._pricing = {
                    "budget_cny": budget,
                    "input_cny_per_mtok": input_price,
                    "output_cny_per_mtok": output_price,
                }
                return

            stored_budget = float(meta.get("budget_cny", "nan"))
            if meta.get("settlement_policy", "legacy") != self.settlement_policy:
                raise LedgerError("结算策略不一致，拒绝重用旧账本")
            stored_input = float(meta.get("input_cny_per_mtok", "nan"))
            stored_output = float(meta.get("output_cny_per_mtok", "nan"))
            if (
                abs(stored_budget - budget) > 1e-12
                or abs(stored_input - input_price) > 1e-12
                or abs(stored_output - output_price) > 1e-12
            ):
                raise LedgerError(
                    "账本预算或价格与当前配置不一致，拒绝重置或混用: "
                    f"ledger={stored_budget}/{stored_input}/{stored_output}, "
                    f"config={budget}/{input_price}/{output_price}"
                )
            stored_run_id = meta.get("run_id", "")
            if run_id and stored_run_id and run_id != stored_run_id:
                raise LedgerError(
                    f"run_id 不一致，拒绝混用账本: {run_id} != {stored_run_id}"
                )
            stored_model = meta.get("model", "")
            if model and stored_model and model != stored_model:
                raise LedgerError(
                    f"账本模型不一致，拒绝混用: {model} != {stored_model}"
                )
            self._run_id = stored_run_id or run_id or "t13"
            self._pricing = {
                "budget_cny": stored_budget,
                "input_cny_per_mtok": stored_input,
                "output_cny_per_mtok": stored_output,
            }

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def last_rejection_reason(self) -> str:
        """最近一次发送前被预算拒绝的原因；空字符串表示尚未拒绝。"""
        return self._last_rejection_reason

    def begin_turn(
        self, label: str = "", max_requests: Optional[int] = None
    ) -> None:
        """开始一个外部用户轮次；该轮的角色、识别与 Composer 请求共用上限。

        User／Judge 等评测侧调用通过 count_turn=False 不进入该计数。
        """
        with self._lock:
            self._turn_label = str(label or "")
            self._turn_requests = 0
            if max_requests is not None:
                self._turn_max_requests = max(0, int(max_requests))

    def consume_turn_request(self, stage: str) -> int:
        """发送前计入当前轮请求数；达到冻结上限时抛出 TurnRequestLimitExceeded。"""
        with self._lock:
            limit = self._turn_max_requests
            current = self._turn_requests
            if limit > 0 and current >= limit:
                raise TurnRequestLimitExceeded(
                    "单个外部用户轮次的模型请求数达到上限: "
                    f"limit={limit}, turn={self._turn_label or '-'}, stage={stage}"
                )
            self._turn_requests = current + 1
            return self._turn_requests

    def turn_state(self) -> Dict[str, Any]:
        """返回当前轮计数，供轨迹与汇总记录。"""
        with self._lock:
            return {
                "label": self._turn_label,
                "requests": self._turn_requests,
                "max_requests": self._turn_max_requests,
            }

    def estimate(
        self, input_tokens_upper: int, max_tokens: int
    ) -> float:
        return (
            max(0, int(input_tokens_upper))
            * self._pricing["input_cny_per_mtok"]
            / 1_000_000
            + max(0, int(max_tokens))
            * self._pricing["output_cny_per_mtok"]
            / 1_000_000
        )

    def reserve(
        self,
        stage: str,
        request_kwargs: Dict[str, Any],
    ) -> RetailReservation:
        input_upper = request_input_upper_tokens(request_kwargs)
        max_tokens = int(request_kwargs.get("max_tokens") or 0)
        upper = self.estimate(input_upper, max_tokens)
        payload = json.dumps(
            request_kwargs,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
        request_sha = hashlib.sha256(payload).hexdigest()
        model = str(request_kwargs.get("model") or "")

        with self._lock, self._tx() as connection:
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM meta"
                ).fetchall()
            }
            overrun = meta.get("overrun_reason", "")
            if overrun:
                self._last_rejection_reason = (
                    "账本已因超出预留上界永久停止: " + overrun
                )
                raise BudgetStopExceeded(self._last_rejection_reason)
            totals = connection.execute(
                "SELECT"
                " COALESCE(SUM(CASE WHEN status IN ('success','failed','crashed')"
                " THEN actual_cny ELSE 0 END), 0) AS spent,"
                " COALESCE(SUM(CASE WHEN status='reserved'"
                " THEN reserved_cny ELSE 0 END), 0) AS reserved"
                " FROM entries"
            ).fetchone()
            spent = float(totals["spent"] or 0.0)
            reserved = float(totals["reserved"] or 0.0)
            budget = self._pricing["budget_cny"]
            if spent + reserved + upper > budget + 1e-12:
                self._last_rejection_reason = (
                    "预算不足，已在请求前停止: "
                    f"已花 {spent:.6f} + 在途 {reserved:.6f} "
                    f"+ 本次上界 {upper:.6f} > 预算 {budget:.6f} 元"
                )
                raise BudgetStopExceeded(self._last_rejection_reason)
            request_id = f"req-{uuid.uuid4().hex[:12]}"
            created_at = _now()
            connection.execute(
                "INSERT INTO entries"
                "(request_id, stage, model, status, reserved_cny, actual_cny,"
                " input_tokens_upper, max_tokens, input_tokens, output_tokens,"
                " input_tokens_main, cache_read_tokens, cache_write_tokens,"
                " usage_present, bound_violation, request_sha256, request_bytes,"
                " latency_ms, error, created_at, settled_at)"
                " VALUES (?, ?, ?, 'reserved', ?, NULL, ?, ?, NULL, NULL,"
                " NULL, NULL, NULL, 0, NULL, ?, ?, NULL, NULL, ?, NULL)",
                (
                    request_id,
                    str(stage),
                    model,
                    upper,
                    input_upper,
                    max_tokens,
                    request_sha,
                    len(payload),
                    created_at,
                ),
            )
        return RetailReservation(
            request_id=request_id,
            stage=str(stage),
            model=model,
            input_tokens_upper=input_upper,
            max_tokens=max_tokens,
            reserved_cny=upper,
            created_at=created_at,
        )

    def settle_success(
        self,
        reservation: RetailReservation,
        usage: Any,
        latency_ms: float,
    ) -> float:
        input_main, output, cache_read, cache_write = self._usage_values(usage)
        usage_present = (
            input_main is not None and output is not None
            and min(input_main, output, cache_read, cache_write) >= 0
        )
        if not usage_present:
            charged = reservation.reserved_cny
            self._settle(
                reservation,
                status="success",
                actual_cny=charged,
                input_tokens=None,
                output_tokens=None,
                input_main=None,
                cache_read=0,
                cache_write=0,
                usage_present=False,
                latency_ms=latency_ms,
                error="usage 缺失，按预留上界计费",
                violation=None,
            )
            return charged

        input_total = int(input_main) + int(cache_read) + int(cache_write)
        actual = (
            input_total * self._pricing["input_cny_per_mtok"] / 1_000_000
            + int(output) * self._pricing["output_cny_per_mtok"] / 1_000_000
        )
        if self.settlement_policy == "deepseek_flash_cache_20261024":
            # 按请求发起时刻估价；未知法定节假日仍按工作日高峰上界处理。
            stamp = datetime.fromisoformat(reservation.created_at).astimezone(timezone(timedelta(hours=8)))
            hour = stamp.hour + stamp.minute / 60
            peak = stamp.weekday() < 5 and (9 <= hour < 12 or 14 <= hour < 18)
            scale = 1.0 if peak else 0.5
            actual = ((int(input_main) + int(cache_write)) * 2 + int(cache_read) * 0.04 + int(output) * 8) * scale / 1_000_000
        violation: Dict[str, Any] = {}
        if input_total > reservation.input_tokens_upper:
            violation["input_tokens"] = {
                "actual": input_total,
                "reserved_upper": reservation.input_tokens_upper,
            }
        if int(output) > reservation.max_tokens:
            violation["output_tokens"] = {
                "actual": int(output),
                "reserved_max": reservation.max_tokens,
            }
        if actual > reservation.reserved_cny + 1e-12:
            violation["charged_cny"] = {
                "actual": actual,
                "reserved_upper": reservation.reserved_cny,
            }
        self._settle(
            reservation,
            status="success",
            actual_cny=actual,
            input_tokens=input_total,
            output_tokens=int(output),
            input_main=int(input_main),
            cache_read=int(cache_read),
            cache_write=int(cache_write),
            usage_present=True,
            latency_ms=latency_ms,
            error="",
            violation=violation or None,
        )
        return actual

    def settle_failure(
        self,
        reservation: RetailReservation,
        error: str,
        latency_ms: float,
    ) -> float:
        charged = reservation.reserved_cny
        self._settle(
            reservation,
            status="failed",
            actual_cny=charged,
            input_tokens=None,
            output_tokens=None,
            input_main=None,
            cache_read=0,
            cache_write=0,
            usage_present=False,
            latency_ms=latency_ms,
            error=f"请求失败，按预留上界计费: {str(error)[:400]}",
            violation=None,
        )
        return charged

    def _settle(
        self,
        reservation: RetailReservation,
        *,
        status: str,
        actual_cny: float,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        input_main: Optional[int],
        cache_read: int,
        cache_write: int,
        usage_present: bool,
        latency_ms: float,
        error: str,
        violation: Optional[Dict[str, Any]],
    ) -> None:
        with self._lock, self._tx() as connection:
            updated = connection.execute(
                "UPDATE entries SET status=?, actual_cny=?, input_tokens=?,"
                " output_tokens=?, input_tokens_main=?, cache_read_tokens=?,"
                " cache_write_tokens=?, usage_present=?, bound_violation=?,"
                " latency_ms=?, error=?, settled_at=?"
                " WHERE request_id=? AND status='reserved'",
                (
                    status,
                    float(actual_cny),
                    input_tokens,
                    output_tokens,
                    input_main,
                    int(cache_read),
                    int(cache_write),
                    1 if usage_present else 0,
                    json.dumps(violation, ensure_ascii=False, sort_keys=True)
                    if violation
                    else None,
                    round(float(latency_ms), 1),
                    error or None,
                    _now(),
                    reservation.request_id,
                ),
            )
            if updated.rowcount != 1:
                raise LedgerError(
                    f"预留记录不存在或已结算: {reservation.request_id}"
                )
            if violation:
                reason = (
                    "实际用量或费用超过预留上界: "
                    f"request_id={reservation.request_id}, "
                    f"stage={reservation.stage}, "
                    f"details={json.dumps(violation, ensure_ascii=False, sort_keys=True)}"
                )
                connection.execute(
                    "UPDATE meta SET value=? WHERE key='overrun_reason'",
                    (reason,),
                )

    @staticmethod
    def _usage_values(usage: Any) -> Tuple[
        Optional[int], Optional[int], int, int
    ]:
        def read(name: str) -> Optional[int]:
            if usage is None:
                return None
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            return int(value) if value is not None else None

        input_main = read("input_tokens")
        output = read("output_tokens")
        cache_read = read("cache_read_input_tokens") or 0
        cache_write = read("cache_creation_input_tokens") or 0
        return input_main, output, cache_read, cache_write

    def summary(self) -> Dict[str, Any]:
        with closing(self._connect()) as connection:
            totals = connection.execute(
                "SELECT"
                " COALESCE(SUM(CASE WHEN status IN ('success','failed','crashed')"
                " THEN actual_cny ELSE 0 END), 0) AS spent,"
                " COALESCE(SUM(CASE WHEN status='reserved'"
                " THEN reserved_cny ELSE 0 END), 0) AS reserved,"
                " COUNT(*) AS total,"
                " SUM(CASE WHEN status='reserved' THEN 1 ELSE 0 END) AS unfinished,"
                " SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failures,"
                " SUM(CASE WHEN status='success' AND usage_present=0"
                " THEN 1 ELSE 0 END) AS usage_missing,"
                " SUM(CASE WHEN bound_violation IS NOT NULL"
                " THEN 1 ELSE 0 END) AS violations"
                " FROM entries"
            ).fetchone()
            meta = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM meta"
                ).fetchall()
            }
        budget = self._pricing["budget_cny"]
        spent = float(totals["spent"] or 0.0)
        reserved = float(totals["reserved"] or 0.0)
        return {
            "ledger_path": str(self.path),
            "run_id": self._run_id,
            "budget_cny": budget,
            "spent_cny": spent,
            "reserved_cny": reserved,
            "remaining_cny": budget - spent - reserved,
            "requests": int(totals["total"] or 0),
            "unfinished_reservations": int(totals["unfinished"] or 0),
            "failures": int(totals["failures"] or 0),
            "usage_missing": int(totals["usage_missing"] or 0),
            "reservation_bound_violations": int(totals["violations"] or 0),
            "cost_stop_reason": meta.get("overrun_reason") or None,
            "input_count_basis": meta.get("input_count_basis", INPUT_COUNT_BASIS),
            "format_overhead_tokens": int(
                meta.get("format_overhead_tokens", REQUEST_FORMAT_OVERHEAD_TOKENS)
            ),
            "pricing": {
                "input_cny_per_mtok": self._pricing["input_cny_per_mtok"],
                "output_cny_per_mtok": self._pricing["output_cny_per_mtok"],
                "settlement_policy": self.settlement_policy,
            },
            "spend_basis": ("usage_cache_time_estimate_holiday_upper" if self.settlement_policy != "legacy" else "legacy_conservative_bookkeeping"),
            "provider_statement_verified": False,
            "model": meta.get("model", ""),
            "turn": self.turn_state(),
        }

    def entries(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM entries ORDER BY created_at, request_id"
            ).fetchall()
        return [dict(row) for row in rows]


class _BudgetedMessages:
    def __init__(
        self,
        inner: Any,
        ledger: RetailLedger,
        stage: str,
        count_turn: bool = False,
    ):
        self._inner = inner
        self._ledger = ledger
        self._stage = stage
        self._count_turn = count_turn

    async def create(self, **kwargs: Any) -> Any:
        # 先检查单轮请求上限，再预留费用；被拒绝时不发出请求。
        if self._count_turn:
            self._ledger.consume_turn_request(self._stage)
        reservation = self._ledger.reserve(self._stage, kwargs)
        started = time.monotonic()
        try:
            response = await self._inner.create(**kwargs)
        except Exception as ex:
            self._ledger.settle_failure(
                reservation,
                str(ex),
                (time.monotonic() - started) * 1000,
            )
            raise
        self._ledger.settle_success(
            reservation,
            getattr(response, "usage", None),
            (time.monotonic() - started) * 1000,
        )
        return response


class BudgetedClient:
    """只暴露 messages.create，并按阶段写入同一个账本。"""

    def __init__(
        self,
        inner: Any,
        ledger: RetailLedger,
        stage: str,
        count_turn: bool = False,
    ):
        self._inner = inner
        self._ledger = ledger
        self.stage = stage
        self.count_turn = count_turn
        self.messages = _BudgetedMessages(
            inner.messages, ledger, stage, count_turn=count_turn
        )

    def for_stage(
        self, stage: str, count_turn: Optional[bool] = None
    ) -> "BudgetedClient":
        """派生阶段包装器；未指定时沿用当前包装器的单轮计数设置。"""
        return BudgetedClient(
            self._inner,
            self._ledger,
            stage,
            count_turn=self.count_turn if count_turn is None else count_turn,
        )

    async def aclose(self) -> None:
        """关闭共享的底层异步客户端；各阶段包装器不重复关闭。"""
        closer = getattr(self._inner, "close", None)
        if closer is None:
            return
        result = closer()
        if hasattr(result, "__await__"):
            await result


def build_budgeted_client(
    api_key: str,
    base_url: Optional[str],
    ledger: RetailLedger,
    stage: str,
    count_turn: bool = False,
) -> BudgetedClient:
    """创建禁用 SDK 隐式重试的真实客户端。"""
    from anthropic import AsyncAnthropic

    kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        kwargs["base_url"] = base_url
    return BudgetedClient(
        AsyncAnthropic(**kwargs), ledger, stage, count_turn=count_turn
    )
