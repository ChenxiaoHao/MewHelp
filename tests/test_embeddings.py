def test_build_embeddings_pins_compat_knobs(fake_settings):
    from app.rag.embeddings import build_embeddings

    emb = build_embeddings(fake_settings)
    assert emb.model == "text-embedding-v4"
    assert emb.dimensions == 1024
    assert emb.chunk_size == fake_settings.embedding_batch_size
    assert emb.check_embedding_ctx_length is False  # 非 OpenAI 提供方必须关 tiktoken 路径
    assert emb.max_retries == 3                     # spec §5:批内指数退避 3 次


def test_dimensions_droppable(fake_settings):
    """核对点③的逃生门:提供方拒收 dimensions 时,置 EMBEDDING_DIMENSIONS=0 即不发送该参数。"""
    from app.rag.embeddings import build_embeddings

    fake_settings.embedding_dimensions = 0
    emb = build_embeddings(fake_settings)
    assert emb.dimensions is None


async def test_client_delegates(fake_settings):
    from app.rag.embeddings import EmbeddingClient

    class FakeEmb:
        def __init__(self):
            self.doc_calls = []

        async def aembed_documents(self, texts):
            self.doc_calls.append(texts)
            return [[0.5] for _ in texts]

        async def aembed_query(self, text):
            return [0.25]

    f = FakeEmb()
    c = EmbeddingClient(f)
    assert await c.embed_texts(["a", "b"]) == [[0.5], [0.5]]
    assert f.doc_calls == [["a", "b"]]
    assert await c.embed_query("邮费是多少") == [0.25]
