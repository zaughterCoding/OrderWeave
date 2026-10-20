"""付费运行前的保守预算检查。

只做费用上界、并发预留和结算记录，不发送请求。价格必须由调用者按
当前供应商价目显式传入，不内置任何历史价格。
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 输入上界口径：完整请求体的 UTF-8 JSON 字节数，加固定的协议格式开销。
INPUT_COUNT_BASIS = "utf8_json_request_bytes_plus_format_overhead"
REQUEST_FORMAT_OVERHEAD_TOKENS = 512


def _json_default(value: Any) -> Any:
    """让 SDK 内容块也能计入请求体上界。"""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")
    raise TypeError(f"无法序列化请求字段: {type(value).__name__}")


def request_input_upper_tokens(request_kwargs: Dict[str, Any]) -> int:
    """返回一次请求输入 token 的保守上界。

    目标模型使用字节级 BPE，token 至少覆盖一个 UTF-8 字节，因此输入 token
    数不会超过请求体字节数。这里序列化实际发送的全部字段（消息、system、
    model、max_tokens 等），再计入角色标记、分隔符和协议包装的固定开销。
    JSON 键名等非计费内容也进入计数，只会进一步放大上界。
    """
    payload = json.dumps(
        request_kwargs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return len(payload.encode("utf-8")) + REQUEST_FORMAT_OVERHEAD_TOKENS


class BudgetExceeded(RuntimeError):
    """在请求发出前预算已不足，或费用上界已经失效。"""


class BudgetStopExceeded(BudgetExceeded):
    """持久化账本的硬停止；业务层不得把它降级为普通识别失败。"""


class TurnRequestLimitExceeded(RuntimeError):
    """单个外部用户轮次的模型请求数达到冻结上限。"""


@dataclass
class Reservation:
    request_id: str
    stage: str
    upper_bound_cny: float
    input_tokens_upper: int
    max_tokens: int
    created_at: float = field(default_factory=time.time)


class BudgetGuard:
    """按费用上界预留、按 usage 结算，失败按上界保守计费。"""

    def __init__(
        self,
        budget_cny: float,
        input_price_cny_per_mtok: float,
        output_price_cny_per_mtok: float,
    ):
        if budget_cny <= 0:
            raise BudgetExceeded("预算必须大于 0")
        if input_price_cny_per_mtok < 0 or output_price_cny_per_mtok < 0:
            raise ValueError("价格不能为负")
        self.budget_cny = float(budget_cny)
        self.input_price = float(input_price_cny_per_mtok)
        self.output_price = float(output_price_cny_per_mtok)
        self._lock = threading.Lock()
        self._reserved_cny = 0.0
        self._spent_cny = 0.0
        self._reservations: Dict[str, Reservation] = {}
        self._usage: List[Dict[str, Any]] = []
        self._counter = 0
        # 一旦实际用量超过预留上界，永久停止，避免异常或降级路径继续发送。
        self._cost_stop_reason: Optional[str] = None
        self._bound_violations = 0

    def estimate(self, input_tokens: int, max_tokens: int) -> float:
        """按输入 token 上界和输出 max_tokens 计满费用上界。"""
        input_count = max(1, int(input_tokens))
        output_tokens = max(0, int(max_tokens))
        return (
            input_count * self.input_price / 1_000_000
            + output_tokens * self.output_price / 1_000_000
        )

    def reserve(self, stage: str, input_tokens: int, max_tokens: int) -> Reservation:
        upper = self.estimate(input_tokens, max_tokens)
        with self._lock:
            if self._cost_stop_reason:
                raise BudgetExceeded(
                    "费用上界此前已失效，拒绝后续请求: "
                    + self._cost_stop_reason
                )
            if self._spent_cny + self._reserved_cny + upper > self.budget_cny + 1e-12:
                raise BudgetExceeded(
                    "预算不足，已在请求前停止: "
                    f"已花 {self._spent_cny:.6f} + 在途 {self._reserved_cny:.6f} "
                    f"+ 本次上界 {upper:.6f} > 预算 {self.budget_cny:.6f} 元"
                )
            self._counter += 1
            request_id = f"req-{self._counter:05d}"
            reservation = Reservation(
                request_id=request_id,
                stage=stage,
                upper_bound_cny=upper,
                input_tokens_upper=int(input_tokens),
                max_tokens=int(max_tokens),
            )
            self._reservations[request_id] = reservation
            self._reserved_cny += upper
            return reservation

    def settle_success(
        self,
        reservation: Reservation,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        latency_ms: float,
    ) -> float:
        """按实际 usage 结算；usage 缺失时按预留上界保守计费。

        实际输入 token、输出 token 或费用任一超过预留上界时，保留真实费用，
        记录显式错误并永久停止后续请求；不把超出部分压回预算。
        """
        with self._lock:
            self._reserved_cny -= reservation.upper_bound_cny
            missing_usage = input_tokens is None or output_tokens is None
            violation: Optional[Dict[str, Any]] = None
            if missing_usage:
                charged = reservation.upper_bound_cny
            else:
                actual_input = max(0, int(input_tokens))
                actual_output = max(0, int(output_tokens))
                charged = (
                    actual_input * self.input_price / 1_000_000
                    + actual_output * self.output_price / 1_000_000
                )
                violation = self._bound_violation(
                    reservation, actual_input, actual_output, charged
                )
            row = {
                "request_id": reservation.request_id,
                "stage": reservation.stage,
                "status": "success",
                "model_usage_present": not missing_usage,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_upper": reservation.input_tokens_upper,
                "max_tokens": reservation.max_tokens,
                "upper_bound_cny": reservation.upper_bound_cny,
                "charged_cny": charged,
                "reservation_bound_violated": violation is not None,
                "latency_ms": latency_ms,
            }
            if violation is not None:
                self._bound_violations += 1
                self._cost_stop_reason = (
                    "实际用量或费用超过预留上界: "
                    f"request_id={reservation.request_id}, stage={reservation.stage}, "
                    f"input_tokens={input_tokens}/{reservation.input_tokens_upper}, "
                    f"output_tokens={output_tokens}/{reservation.max_tokens}, "
                    f"charged={charged:.6f}/{reservation.upper_bound_cny:.6f} CNY"
                )
                row["bound_violation"] = violation
                row["cost_stop_reason"] = self._cost_stop_reason
            self._spent_cny += charged
            self._reservations.pop(reservation.request_id, None)
            self._usage.append(row)
            return charged

    @staticmethod
    def _bound_violation(
        reservation: Reservation,
        actual_input: int,
        actual_output: int,
        charged: float,
    ) -> Optional[Dict[str, Any]]:
        """返回首个或全部上界失效细节；无失效时返回 None。"""
        details: Dict[str, Any] = {}
        if actual_input > reservation.input_tokens_upper:
            details["input_tokens"] = {
                "actual": actual_input,
                "reserved_upper": reservation.input_tokens_upper,
            }
        if actual_output > reservation.max_tokens:
            details["output_tokens"] = {
                "actual": actual_output,
                "reserved_max": reservation.max_tokens,
            }
        if charged > reservation.upper_bound_cny + 1e-12:
            details["charged_cny"] = {
                "actual": charged,
                "reserved_upper": reservation.upper_bound_cny,
            }
        return details or None

    def settle_failure(
        self,
        reservation: Reservation,
        error: str,
        latency_ms: float,
    ) -> float:
        """失败也占用预留上界，避免重试绕过预算。"""
        with self._lock:
            self._reserved_cny -= reservation.upper_bound_cny
            charged = reservation.upper_bound_cny
            self._spent_cny += charged
            self._reservations.pop(reservation.request_id, None)
            self._usage.append(
                {
                    "request_id": reservation.request_id,
                    "stage": reservation.stage,
                    "status": "failed",
                    "model_usage_present": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "input_tokens_upper": reservation.input_tokens_upper,
                    "max_tokens": reservation.max_tokens,
                    "upper_bound_cny": reservation.upper_bound_cny,
                    "charged_cny": charged,
                    "reservation_bound_violated": False,
                    "latency_ms": latency_ms,
                    "error": str(error)[:500],
                }
            )
            return charged

    @property
    def cost_stop_reason(self) -> Optional[str]:
        with self._lock:
            return self._cost_stop_reason

    @property
    def usage(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._usage)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "budget_cny": self.budget_cny,
                "spent_cny": self._spent_cny,
                "reserved_cny": self._reserved_cny,
                "remaining_cny": self.budget_cny - self._spent_cny - self._reserved_cny,
                "requests": len(self._usage),
                "failures": sum(1 for item in self._usage if item["status"] == "failed"),
                "usage_missing": sum(
                    1
                    for item in self._usage
                    if item["status"] == "success" and not item["model_usage_present"]
                ),
                "reservation_bound_violations": self._bound_violations,
                "cost_stop_reason": self._cost_stop_reason,
                "input_count_basis": INPUT_COUNT_BASIS,
                "format_overhead_tokens": REQUEST_FORMAT_OVERHEAD_TOKENS,
                "pricing": {
                    "input_cny_per_mtok": self.input_price,
                    "output_cny_per_mtok": self.output_price,
                },
            }
