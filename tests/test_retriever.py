"""retriever 编排:embed→search→阈值→回查 MySQL→回排;孤儿向量天然跳过。"""

import pytest


async def test_retrieve_orders_thresholds_and_skips_orphans(monkeypatch):
    from types import SimpleNamespace

    from app.rag import retriever

    st = SimpleNamespace(milvus_uri="http://fake", milvus_collection="knowledge",
                         embedding_dimensions=1024, embedding_batch_size=10,
                         embedding_model="m", openai_api_key="k", openai_api_base=None,
                         openai_base_url="http://fake/v1", rag_top_k=5, rag_score_threshold=0.3)
    async def fake_embed_query(self, text):
        return [0.1, 0.2]
    monkeypatch.setattr(retriever.EmbeddingClient, "embed_query", fake_embed_query)
    # search 结果:10 高分、11 低于阈值、12 是孤儿(MySQL 无行)
    monkeypatch.setattr(retriever, "_search_sync", lambda vec, s: [(10, 0.91), (11, 0.2), (12, 0.8)])

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False
    async def fake_by_ids(session, ids):
        assert sorted(ids) == [10, 12]  # 11 被阈值滤掉
        return [SimpleNamespace(id=10, questions="q10", answer="a10", category="c10")]
    monkeypatch.setattr(retriever.crud, "fetch_chunks_by_ids", fake_by_ids)
    monkeypatch.setattr(retriever, "get_session_factory", lambda: (lambda: Sess()))

    hits = await retriever.retrieve_hits("邮费是多少", settings=st)
    assert [h.id for h in hits] == [10]  # 12 孤儿被回查过滤;10 在 11 之前(相似度序)
