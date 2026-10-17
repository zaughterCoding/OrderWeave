"""ChromaDB-backed knowledge base for document storage and semantic search.

The same ChromaDB service is also used for conversation memory later, but
this module keeps its documents in a separate collection.
"""
import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_EMBEDDING_CACHE = _PROJECT_ROOT / "data" / "chroma_cache"


class _ProjectEmbeddingFunction(ONNXMiniLM_L6_V2):
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


class KnowledgeBase:
    """RAG knowledge base backed by a ChromaDB server or local persistence."""

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        collection_name: Optional[str] = None,
    ):
        # Prefer an independent ChromaDB server. Its embedding model runs on the
        # server, so the client does not need to download one.
        self._use_server = False
        try:
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        self._collection_name = collection_name or self.COLLECTION_NAME
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "AgentEcommerce RAG 知识库"},
            embedding_function=_ProjectEmbeddingFunction(),
        )

        if self._collection.count() == 0:
            self._load_default_docs()

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """Chunk and upsert documents; returns the number of written chunks."""
        ids, docs, metas = [], [], []

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            chunks = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append(
                    {"title": title, "chunk_index": i, "total_chunks": len(chunks)}
                )

        if ids:
            # Stable IDs make repeated imports idempotent and allow updates.
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库写入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """Run the synchronous ChromaDB write in a worker thread."""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the most relevant chunks for a query."""
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        if documents:
            for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                items.append(
                    {
                        "id": doc_id,
                        "title": meta.get("title", ""),
                        "content": doc,
                        "score": round(1.0 - dist, 4),
                        "chunk": meta.get("chunk_index", 0),
                    }
                )

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run the synchronous ChromaDB query in a worker thread."""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """Return the number of stored chunks without blocking the event loop."""
        return await asyncio.to_thread(self._collection.count)

    async def search_handler(
        self,
        params: Dict[str, Any],
        context: Any,
    ) -> List[Dict[str, Any]]:
        """Handler used when this knowledge base is registered as a tool."""
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split long text by sentence-like boundaries."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        sentences = text.replace("\n", "。").split("。")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current) + len(sentence) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current}。{sentence}" if current else sentence

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """Seed a new collection with common customer-service documents."""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
