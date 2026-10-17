"""Three-tier conversation memory.

Working memory uses Redis, episodic memory and user profiles use ChromaDB.
Compression keeps the latest messages in time order and never drops them when
summary generation or episodic storage fails.
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import redis.asyncio as redis
from anthropic import AsyncAnthropic
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EMBEDDING_CACHE = _PROJECT_ROOT / "data" / "chroma_cache"


class _MemoryEmbeddingFunction(ONNXMiniLM_L6_V2):
    """Keep the ONNX model cache inside the project instead of the user profile."""

    DOWNLOAD_PATH = (
        Path(
            os.getenv("CHROMA_EMBEDDING_CACHE") or str(_DEFAULT_EMBEDDING_CACHE)
        )
        .expanduser()
        .resolve()
        / "onnx_models"
        / ONNXMiniLM_L6_V2.MODEL_NAME
    )


class MsgRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    role: MsgRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """Memory data passed to an agent for one request."""

    recent_messages: List[Message]
    relevant_history: List[str]
    user_profile: Dict[str, Any]
    summary: str

    @staticmethod
    def _clean(text: str) -> str:
        """Remove surrogate characters before prompt construction."""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """Format the memory context as prompt text."""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append(
                "[相关历史]\n"
                + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3])
            )
        if self.user_profile:
            parts.append(
                f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}"
            )
        if self.recent_messages:
            parts.append("[最近对话]")
            for message in self.recent_messages[-8:]:
                parts.append(f"{message.role.value}: {self._clean(message.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """Redis working memory plus ChromaDB episodic memory and profiles."""

    WORKING_MAX = 20
    COMPRESS_AT = 15
    HISTORY_TOP_K = 5
    SUMMARY_MAX_CHARS = 800
    PROFILE_DOC_PREFIX = "user_profile:"

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        api_key: str = "",
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

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}

        self._use_server = False
        try:
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()
            self._use_server = True
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        embedding_function = _MemoryEmbeddingFunction()
        self._episodic = chroma.get_or_create_collection(
            "episodic",
            embedding_function=embedding_function,
        )
        self._profile = chroma.get_or_create_collection(
            "user_profile",
            embedding_function=embedding_function,
        )

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role: MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a message and compress while holding the conversation lock."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(key): self._safe_metadata_value(value)
            for key, value in (metadata or {}).items()
        }
        message = Message(
            role=role,
            content=self._safe_text(content),
            metadata=clean_metadata,
        )
        key = self._wm_key(user_id, conv_id)

        async with self._lock_for(user_id, conv_id):
            await self._redis.lpush(
                key,
                json.dumps(
                    {
                        "role": message.role.value,
                        "content": message.content,
                        "ts": message.timestamp.isoformat(),
                        "metadata": message.metadata,
                    }
                ),
            )
            await self._redis.expire(key, 86400)

            if await self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress_locked(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """Extract profile data from recent messages and store it in ChromaDB."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        current_profile = await self._get_profile(user_id)

        text = self._safe_text(
            "\n".join(f"{message.role.value}: {message.content}" for message in messages[-10:])
        )
        profile_ctx = (
            json.dumps(current_profile, ensure_ascii=False) if current_profile else "{}"
        )
        prompt = f"""从以下对话和已有用户画像中提炼或更新用户偏好和关键实体，返回 JSON。
对话:
{text}

已有画像:
{profile_ctx}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            start, end = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[start:end])

            doc_id = self._profile_doc_id(user_id)
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            try:
                await asyncio.to_thread(self._profile.delete, ids=[doc_id])
            except Exception:
                pass

            await asyncio.to_thread(
                self._profile.add,
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[
                    {
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "updated_at": datetime.now().isoformat(),
                    }
                ],
            )
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    async def get_context(
        self,
        user_id: str,
        conv_id: str,
        query: str = "",
    ) -> MemoryContext:
        """Build the memory context used by one agent request."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)
        history = await self._search_episodic(
            user_id,
            conv_id,
            query or (recent[-1].content if recent else ""),
        )
        profile = await self._get_profile(user_id)
        summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """Compress one conversation under its lock."""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        async with self._lock_for(user_id, conv_id):
            await self._compress_locked(user_id, conv_id)

    async def _compress_locked(self, user_id: str, conv_id: str) -> None:
        """Compress working memory while the caller holds the conversation lock."""
        # Compression removes the whole list, so it must inspect every pending
        # message, including a backlog left by earlier summary failures.
        messages = await self._read_working_messages(
            user_id, conv_id, limit=None
        )
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]
        keep = messages[-5:]

        text = self._safe_text(
            "\n".join(
                f"{message.role.value}: {message.content}" for message in to_compress
            )
        )
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
        except Exception as ex:
            logger.warning(f"生成摘要失败，保留工作记忆: {ex}")
            return

        if not summary:
            logger.warning("生成摘要为空，保留工作记忆")
            return

        summary_key = self._summary_key(user_id, conv_id)
        old_summary = await self._redis.get(summary_key) or ""
        new_summary = await self._merge_summary(old_summary, summary)

        stored = await self._store_episodic(user_id, conv_id, text, summary)
        if stored is False:
            logger.warning("存储情景记忆失败，保留工作记忆")
            return

        await self._redis.setex(summary_key, 86400, new_summary)

        key = self._wm_key(user_id, conv_id)
        await self._redis.delete(key)
        for message in keep:
            await self._redis.lpush(
                key,
                json.dumps(
                    {
                        "role": message.role.value,
                        "content": message.content,
                        "ts": message.timestamp.isoformat(),
                        "metadata": message.metadata,
                    }
                ),
            )
        await self._redis.expire(key, 86400)
        logger.info(
            f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字"
        )

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        """Return the latest messages used as the normal context window."""
        return await self._read_working_messages(
            user_id, conv_id, limit=self.WORKING_MAX
        )

    async def _read_working_messages(
        self,
        user_id: str,
        conv_id: str,
        limit: Optional[int],
    ) -> List[Message]:
        key = self._wm_key(user_id, conv_id)
        end = -1 if limit is None else limit - 1
        raws = await self._redis.lrange(key, 0, end)
        messages = []
        for raw in reversed(raws):
            data = json.loads(raw)
            messages.append(
                Message(
                    role=MsgRole(data["role"]),
                    content=data["content"],
                    timestamp=datetime.fromisoformat(data["ts"]),
                    metadata=data.get("metadata", {}),
                )
            )
        return messages

    async def _search_episodic(
        self,
        user_id: str,
        conv_id: str,
        query: str,
    ) -> List[str]:
        """Search the current conversation first, then the same user's history."""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            results = await self._query_episodic(
                query_text,
                n_results=self.HISTORY_TOP_K,
                where={
                    "$and": [
                        {"user_id": self._safe_text(user_id)},
                        {"conv_id": self._safe_text(conv_id)},
                    ]
                },
            )
            docs = self._extract_docs(results)
            if len(docs) < self.HISTORY_TOP_K:
                fallback = await self._query_episodic(
                    query_text,
                    n_results=self.HISTORY_TOP_K,
                    where={"user_id": self._safe_text(user_id)},
                )
                docs.extend(self._extract_docs(fallback))
            return self._dedupe_texts(docs)[: self.HISTORY_TOP_K]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(
        self,
        user_id: str,
        conv_id: str,
        text: str,
        summary: str,
    ) -> bool:
        """Store a compressed conversation; return False so callers keep messages."""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(
                f"{user_id}{conv_id}{time.time()}".encode()
            ).hexdigest()
            await asyncio.to_thread(
                self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[
                    {
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "ts": datetime.now().isoformat(),
                        "full_text": self._safe_text(text[:500]),
                    }
                ],
            )
            return True
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")
            return False

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """Return the latest stored profile for a user."""
        try:
            doc_id = self._profile_doc_id(user_id)
            direct = await asyncio.to_thread(self._profile.get, ids=[doc_id])
            if direct.get("documents"):
                return json.loads(direct["documents"][0])

            results = await asyncio.to_thread(
                self._profile.get,
                where={"user_id": user_id},
            )
            return self._latest_profile_from_results(results)
        except Exception:
            pass
        return {}

    async def close(self) -> None:
        """Close the async Redis connection."""
        await self._redis.aclose()

    def _lock_for(self, user_id: str, conv_id: str) -> asyncio.Lock:
        key = (user_id, conv_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @classmethod
    def _profile_doc_id(cls, user_id: str) -> str:
        return f"{cls.PROFILE_DOC_PREFIX}{user_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """Convert a value to plain UTF-8 text acceptable to Redis and ChromaDB."""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """Recursively clean metadata strings."""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {
                cls._safe_text(key): cls._safe_metadata_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._safe_metadata_value(item) for item in value]
        return value

    async def _query_episodic(
        self,
        query_text: str,
        n_results: int,
        where: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._episodic.query,
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

    @staticmethod
    def _extract_docs(results: Dict[str, Any]) -> List[str]:
        docs = results.get("documents") or []
        if not docs:
            return []
        first = docs[0] if isinstance(docs[0], list) else docs
        return [doc for doc in first if isinstance(doc, str) and doc.strip()]

    @staticmethod
    def _dedupe_texts(values: List[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for value in values:
            text = value.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            deduped.append(text)
        return deduped

    async def _merge_summary(self, old_summary: str, new_summary: str) -> str:
        old_summary = self._safe_text(old_summary).strip()
        new_summary = self._safe_text(new_summary).strip()
        if not old_summary:
            return new_summary[: self.SUMMARY_MAX_CHARS]
        if not new_summary:
            return old_summary[: self.SUMMARY_MAX_CHARS]

        prompt = self._safe_text(
            f"""你是对话摘要器。请把下面两段摘要合并为一段不超过 {self.SUMMARY_MAX_CHARS} 个中文字符的摘要。
保留：用户偏好、关键实体、待办事项、约束条件、未解决问题。
只输出摘要正文，不要编号，不要解释。

旧摘要:
{old_summary}

新增摘要:
{new_summary}
"""
        )
        try:
            resp = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            merged = self._safe_text(extract_text_content(resp.content)).strip()
            if merged:
                return merged[: self.SUMMARY_MAX_CHARS]
        except Exception as ex:
            logger.warning(f"合并摘要失败，回退为截断拼接: {ex}")

        merged = self._safe_text(f"{old_summary}\n{new_summary}").strip()
        return merged[-self.SUMMARY_MAX_CHARS :]

    @staticmethod
    def _latest_profile_from_results(results: Dict[str, Any]) -> Dict[str, Any]:
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        if not documents:
            return {}

        candidates: List[tuple[str, Dict[str, Any], str]] = []
        for idx, doc in enumerate(documents):
            if not doc:
                continue
            metadata = (
                metadatas[idx]
                if idx < len(metadatas) and isinstance(metadatas[idx], dict)
                else {}
            )
            ts = str(metadata.get("updated_at") or metadata.get("ts") or "")
            candidates.append((ts, metadata, doc))

        if not candidates:
            return {}

        candidates.sort(key=lambda item: item[0], reverse=True)
        latest_doc = candidates[0][2]
        try:
            return json.loads(latest_doc)
        except Exception:
            return {}
