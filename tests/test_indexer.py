"""indexer 单测:假 milvus/假 embedder/猴补丁 crud,验证编排(§5 两段与注入)。"""

from types import SimpleNamespace

import pytest


def _make_pending(n):
    return [
        SimpleNamespace(id=i, category=f"c{i}", questions=f"q{i}", answer=f"a{i}" * 3,
                        content_type=None)  # v2 行组装读 r.content_type(None→"")
        for i in range(1, n + 1)
    ]


async def test_vectorize_batches_and_marks(monkeypatch):
    from app.rag import indexer

    st = SimpleNamespace(milvus_uri="http://fake", milvus_collection="knowledge",
                         embedding_dimensions=8, embedding_batch_size=10)
    monkeypatch.setattr(indexer, "get_settings", lambda: st)
    # build_embeddings 走真 Settings 字段,SimpleNamespace 喂不动——单测只验编排,直接换假:
    monkeypatch.setattr(indexer, "build_embeddings", lambda s: object())
    calls = {"upsert": [], "mark": [], "ensure": []}

    class Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    emb_box = {}

    async def fake_embed_texts(self, texts):  # EmbeddingClient.embed_texts 替身
        emb_box.setdefault("texts", []).extend(texts)
        return [[0.1] for _ in texts]

    monkeypatch.setattr(indexer.EmbeddingClient, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(indexer.milvus_store, "get_client", lambda uri, timeout=10.0: object())
    monkeypatch.setattr(indexer.milvus_store, "health_ok", lambda c: True)

    def ensure(c, name, dim): calls["ensure"].append((name, dim))
    def upsert(c, name, rows): calls["upsert"].append(rows); return len(rows)

    monkeypatch.setattr(indexer.milvus_store, "ensure_collection", ensure)
    monkeypatch.setattr(indexer.milvus_store, "upsert_rows", upsert)
    monkeypatch.setattr(indexer.milvus_store, "flush", lambda c, n: None)
    async def fake_pending(session): return _make_pending(25)
    monkeypatch.setattr(indexer.crud, "fetch_pending_chunks", fake_pending)
    async def mark(session, ids): calls["mark"].append(list(ids))
    monkeypatch.setattr(indexer.crud, "mark_chunks_vectorized", mark)
    monkeypatch.setattr(indexer, "get_session_factory", lambda: (lambda: Sess()))

    done = await indexer.vectorize_pending()
    assert done == 25 and calls["ensure"] == [("knowledge", 8)]
    assert [len(p) for p in calls["upsert"]] == [10, 10, 5]
    assert [len(m) for m in calls["mark"]] == [10, 10, 5]
    assert len(emb_box["texts"]) == 25
    assert emb_box["texts"][0] == "c1\nq1\na1a1a1"  # 向量文本=三格拼接(§4-6)


async def test_vectorize_fault_after(monkeypatch):
    from app.rag import indexer

    st = SimpleNamespace(milvus_uri="http://fake", milvus_collection="knowledge",
                         embedding_dimensions=8, embedding_batch_size=10)
    monkeypatch.setattr(indexer, "get_settings", lambda: st)
    monkeypatch.setattr(indexer, "build_embeddings", lambda s: object())
    monkeypatch.setattr(indexer.milvus_store, "get_client", lambda uri, timeout=10.0: object())
    monkeypatch.setattr(indexer.milvus_store, "health_ok", lambda c: True)
    monkeypatch.setattr(indexer.milvus_store, "ensure_collection", lambda *a: None)
    monkeypatch.setattr(indexer.milvus_store, "upsert_rows", lambda c, n, r: len(r))

    async def fake_embed(self, texts): return [[0.1] for _ in texts]
    monkeypatch.setattr(indexer.EmbeddingClient, "embed_texts", fake_embed)

    marked = []
    async def mark(session, ids): marked.extend(ids)
    monkeypatch.setattr(indexer.crud, "mark_chunks_vectorized", mark)

    async def pending(session): return _make_pending(25)
    monkeypatch.setattr(indexer.crud, "fetch_pending_chunks", pending)

    class Sess:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
    monkeypatch.setattr(indexer, "get_session_factory", lambda: (lambda: Sess()))

    with pytest.raises(SystemExit) as exc:
        await indexer.vectorize_pending(fault_after=3)
    assert exc.value.code == 42
    assert len(marked) == 10  # 批粒度:第一批(10)完成即触发,前 10 行已翻 done


