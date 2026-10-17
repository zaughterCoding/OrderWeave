"""Tool execution with query rewriting, reranking, caching and fallback.

The manager handles the retrieval optimization chain:
query rewriting -> parallel recall -> merge/deduplicate -> rerank -> top-k.
It is an in-process tool registry, not an MCP protocol server.
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ToolResult:
    success: bool
    data: Any
    tool_name: str
    error: Optional[str] = None
    cached: bool = False
    latency_ms: float = 0.0
    reranked: bool = False
    # True when data came from a fallback path and must not count as a real hit.
    degraded: bool = False


@dataclass
class ToolStats:
    """Runtime counters exposed to the monitor."""

    total: int = 0
    success: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    latency_samples: int = 0
    consecutive_fails: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        """Average over calls that actually recorded a latency sample."""
        if not self.latency_samples:
            return 0.0
        return self.total_latency_ms / self.latency_samples


class CircuitBreaker:
    """Three-state circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold = failure_threshold
        self.recovery_s = recovery_s
        self.state = CircuitState.CLOSED
        self.fail_count = 0
        self.opened_at: Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore[operator]
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable
    schema: Dict[str, Any]
    cache_ttl: float = 0.0
    timeout_s: float = 30.0
    supports_rerank: bool = False
    fallback: Optional[Callable] = None

    stats: ToolStats = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


class MCPToolManager:
    """In-process tool registry with cache, circuit breaker and fallback."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if not model:
            raise ValueError("model is required")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,
    ) -> ToolResult:
        """Execute a tool through cache, breaker, handler, rerank and fallback."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            self._validate_params(tool, params)
            data = await asyncio.wait_for(
                self._run_handler(tool, params, context),
                timeout=tool.timeout_s,
            )
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.stats.latency_samples += 1
            tool.breaker.record_success()

            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k)

            # Cache the final result so a later hit cannot bypass reranking.
            if tool.cache_ttl > 0:
                self._set_cache(
                    name,
                    params,
                    data,
                    tool.cache_ttl,
                    cache_rerank_top_k,
                    reranked,
                )

            return ToolResult(
                success=True,
                data=data,
                tool_name=name,
                latency_ms=latency,
                reranked=reranked,
            )

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """Return a degraded result; fallback content is never a successful hit."""
        if tool.fallback is None:
            return ToolResult(
                success=False,
                data=None,
                tool_name=tool.name,
                error=error,
                degraded=True,
            )
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=False,
                data=data,
                tool_name=tool.name,
                error=error,
                degraded=True,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(
                success=False,
                data=None,
                tool_name=tool.name,
                error=f"{error}; fallback失败: {ex}",
                degraded=True,
            )

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """Run async handlers directly and sync handlers in a worker thread."""
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """Expand a query into n sub-queries while keeping the original query."""
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            start, end = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[start:end])
            if not isinstance(queries, list):
                return [query]
            return list(dict.fromkeys([query] + [str(item) for item in queries]))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Run rewrite, parallel recall, merge/deduplicate and rerank."""
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        recall_k = max(top_k, 5)
        tasks = [
            self.call(tool_name, {"query": q, "top_k": recall_k}, context, use_cache=True)
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen: Dict[Tuple[str, ...], int] = {}
        merged: List[Any] = []
        degraded = False
        for result in results:
            if isinstance(result, Exception):
                degraded = True
                logger.warning(f"子查询执行异常: {result}")
                continue
            if not isinstance(result, ToolResult):
                degraded = True
                continue
            degraded = degraded or result.degraded
            # Fallback payloads are informative text, not retrieval hits.
            if result.success and not result.degraded and isinstance(result.data, list):
                for item in result.data:
                    # Scores differ per sub-query, so identity must exclude
                    # dynamic fields; keep the highest score for one fragment.
                    key = self._fragment_key(item)
                    index = seen.get(key)
                    if index is None:
                        seen[key] = len(merged)
                        merged.append(item)
                    elif self._fragment_score(item) > self._fragment_score(merged[index]):
                        merged[index] = item

        if not merged:
            error = "知识库检索降级，未获得有效结果" if degraded else "所有子查询均无结果"
            return ToolResult(
                success=False,
                data=[],
                tool_name=tool_name,
                error=error,
                degraded=degraded,
            )

        reranked, was_reranked = await self._rerank(query, merged, top_k)
        return ToolResult(
            success=True,
            data=reranked,
            tool_name=tool_name,
            reranked=was_reranked,
            degraded=degraded,
        )

    @staticmethod
    def _fragment_key(item: Any) -> Tuple[str, ...]:
        """Identify a fragment by stable fields only."""
        if isinstance(item, dict):
            fragment_id = item.get("id")
            if fragment_id not in (None, ""):
                return ("id", str(fragment_id))
            return (
                "content",
                str(item.get("title", "")),
                str(item.get("content", "")),
                str(item.get("chunk", item.get("chunk_index", ""))),
            )
        return ("raw", str(item))

    @staticmethod
    def _fragment_score(item: Any) -> float:
        if not isinstance(item, dict):
            return 0.0
        try:
            return float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _rerank(
        self,
        query: str,
        items: List[Any],
        top_k: int,
    ) -> Tuple[List[Any], bool]:
        """Rerank items with the LLM; the second value reports whether it ran."""
        if len(items) <= top_k:
            return items, False

        items_text = "\n".join(
            f"{i}. {json.dumps(item, ensure_ascii=False)[:200]}"
            for i, item in enumerate(items)
        )
        prompt = f"""根据用户查询，对以下检索结果按相关性打分（0-10），返回 JSON 数组。
用户查询: "{query}"
检索结果:
{items_text}

返回格式（按相关性降序排列的索引列表）: [最相关的索引, ..., 最不相关的索引]
只返回 JSON 数组，不要其他文字。"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            start, end = raw.find("["), raw.rfind("]") + 1
            order: List[int] = json.loads(raw[start:end])
            if not isinstance(order, list):
                return items[:top_k], False
            reranked = [items[i] for i in order if isinstance(i, int) and 0 <= i < len(items)]
            if not reranked:
                return items[:top_k], False
            return reranked[:top_k], True
        except Exception as ex:
            logger.warning(f"重排失败，返回原始顺序: {ex}")
            return items[:top_k], False

    def _cache_key(self, name: str, params: Dict, rerank_top_k: int = 0) -> str:
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(
        self,
        name: str,
        params: Dict,
        rerank_top_k: int = 0,
    ) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params, rerank_top_k)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            for key in list(self._cache)[:1250]:
                del self._cache[key]
        self._cache[self._cache_key(name, params, rerank_top_k)] = (
            data,
            time.monotonic() + ttl,
            reranked,
        )

    _TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """Validate required and typed fields from the tool JSON Schema."""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field_name in required:
            if field_name not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field_name}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: "
                            f"期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """Remove surrogate characters that break LLM request encoding."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": tool.stats.total,
                "success_rate": round(tool.stats.success_rate, 3),
                "avg_latency_ms": round(tool.stats.avg_latency_ms, 1),
                "latency_samples": tool.stats.latency_samples,
                "consecutive_fails": tool.stats.consecutive_fails,
                "circuit_state": tool.breaker.state.value,
            }
            for name, tool in self._tools.items()
        }
