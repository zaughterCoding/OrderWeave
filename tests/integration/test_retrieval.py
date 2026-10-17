import asyncio
import os
import uuid

from mcp.knowledge_base import KnowledgeBase

CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8002"))


def test_real_chroma_import_search_and_idempotency(tmp_path):
    collection_name = f"retrieval_test_{uuid.uuid4().hex}"
    kb = KnowledgeBase(
        chroma_host=CHROMA_HOST,
        chroma_port=CHROMA_PORT,
        chroma_path=str(tmp_path / "chroma"),
        collection_name=collection_name,
    )

    try:
        assert kb._use_server is True, (
            "independent Chroma server is required at "
            f"{CHROMA_HOST}:{CHROMA_PORT}; run compose.yml first"
        )

        documents = [
            {
                "title": "合成退款文档A",
                "content": "火星退款条款：订单号 MARS-ALPHA 在七天内可以申请退款。",
            },
            {
                "title": "合成配送文档B",
                "content": "月球配送说明：订单号 LUNA-BETA 使用星际加急配送。",
            },
        ]

        assert kb.add_documents(documents) == 2
        count_after_first_import = kb.doc_count
        assert asyncio.run(kb.add_documents_async(documents)) == 2
        assert kb.doc_count == count_after_first_import
        assert asyncio.run(kb.doc_count_async()) == count_after_first_import

        refund_ids = set(kb._collection.get(where={"title": "合成退款文档A"})["ids"])
        refund_results = kb.search("火星退款 MARS-ALPHA", top_k=3)
        assert refund_ids
        assert any(
            item["id"] in refund_ids and item["title"] == "合成退款文档A"
            for item in refund_results
        )

        delivery_ids = set(kb._collection.get(where={"title": "合成配送文档B"})["ids"])
        delivery_results = asyncio.run(
            kb.search_handler({"query": "月球配送 LUNA-BETA", "top_k": 3}, None)
        )
        assert delivery_ids
        assert any(
            item["id"] in delivery_ids and item["title"] == "合成配送文档B"
            for item in delivery_results
        )
    finally:
        try:
            kb._client.delete_collection(collection_name)
        except Exception:
            pass
