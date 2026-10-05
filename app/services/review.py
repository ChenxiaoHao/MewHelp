"""ch09 T9 核准回写:通过条目 → knowledge_chunks + Milvus 双落(indexer 原语直写)。

Focus 5 半成功顺序即全设计:pending 行先单独 commit(重试自愈面),
embed+upsert 在置态之前,done 回填+置「通过」由 crud.approve_review 同 commit
收口——任何一步崩,状态都不可能被置。指纹=(category,questions,answer) 三元
精确匹配(chunk_fingerprint 同口径):已 done 命中=KB 零重做只置态;
pending 命中=补写向量(上次半崩自愈)。
"""

from __future__ import annotations

import logging

from app.core.config import get_settings
from app.db import crud
from app.rag.chunking import ChunkDraft

logger = logging.getLogger(__name__)

CATEGORY = "客服对话问答"  # spec 定死;与 ch03 语料类目同名单一路


async def default_writer(chunk_id: int, text: str, settings) -> None:
    """单块向量化+upsert(pk=chunk_id 幂等覆写)+flush——vectorize_pending 同律。"""
    from app.rag import milvus_store
    from app.rag.embeddings import EmbeddingClient, build_embeddings

    client = milvus_store.get_client(settings.milvus_uri)
    if not milvus_store.health_ok(client):
        raise RuntimeError("Milvus 不可达,核准回写失败(状态留待审,重试自愈)")
    milvus_store.ensure_collection(client, settings.milvus_collection,
                                   settings.embedding_dimensions)
    emb = EmbeddingClient(build_embeddings(settings))
    vectors = await emb.embed_texts([text])
    milvus_store.upsert_rows(client, settings.milvus_collection, [{
        "chunk_id": chunk_id, "text": text, "embedding": vectors[0],
        "category": CATEGORY, "content_type": "qa"}])
    milvus_store.flush(client, settings.milvus_collection)


async def publish_approved(session, row, approved_answer: str, *,
                          settings=None, writer=None) -> int:
    """核准双落并置「通过」;失败抛(端点转 502,状态不动)。返回 chunk id。

    writer 注入位:活库测免 Milvus;默认路走真 embed+upsert。
    """
    st = settings or get_settings()
    w = writer or default_writer
    existing = await crud.find_chunk_by_qa(session, CATEGORY,
                                           row.normalized_question, approved_answer)
    if existing is not None and existing.vectorize_status == "done":
        await crud.approve_review(session, row.id, approved_answer, existing.id)
        return existing.id
    if existing is None:
        draft = ChunkDraft(category=CATEGORY, questions=row.normalized_question,
                           answer=approved_answer, section_path=f"review/{row.id}",
                           content_type="qa", is_key_clause=False)
        rows = await crud.add_chunk_drafts(session, [draft], commit=True)
        chunk = rows[0]
    else:
        chunk = existing
    text = "\n".join([chunk.category, chunk.questions, chunk.answer])  # §3.1 三格同源
    await w(chunk.id, text, st)
    await crud.approve_review(session, row.id, approved_answer, chunk.id)
    return chunk.id
