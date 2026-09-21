"""ch03 嵌入客户端(spec §10):DashScope OpenAI 兼容端点 + text-embedding-v4。

两个已知坑(核对点③):
- check_embedding_ctx_length=False:跳过 tiktoken 计数路径(兼容端点没有 OpenAI 的 encoding)
- dimensions 直传兼容模式;若服务端拒参,设 EMBEDDING_DIMENSIONS=0 走模型默认 1024 维(spec §3.2 维度不变)
"""

from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


def build_embeddings(st: Settings) -> OpenAIEmbeddings:
    kwargs = {"dimensions": st.embedding_dimensions} if st.embedding_dimensions else {}
    return OpenAIEmbeddings(
        model=st.embedding_model,
        api_key=st.openai_api_key,
        base_url=st.openai_base_url,
        chunk_size=st.embedding_batch_size,
        max_retries=3,
        check_embedding_ctx_length=False,
        **kwargs,
    )


class EmbeddingClient:
    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        self._e = embeddings

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._e.aembed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        return await self._e.aembed_query(text)
