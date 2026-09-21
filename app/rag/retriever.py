"""在线检索(spec §7):embed_query → Milvus Top-K → 阈值过滤 → 回查 MySQL 组装。

Milvus 只存 id+向量:回查即权威;孤儿向量(MySQL 已无行)在 by_id 映射时天然丢弃。
集合不存在 → RuntimeError 上抛,由 ch02 executor 统一包成错误帧(§8),工具链自愈转工单。
"""

from __future__ import annotations

import asyncio

from app.core.config import Settings, get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.rag import milvus_store
from app.rag.embeddings import EmbeddingClient, build_embeddings

_clients: dict[str, object] = {}  # uri → MilvusClient 进程级缓存(grpc 通道不宜每请求新建)


def _search_sync(vector, st: Settings) -> list[tuple[int, float]]:
    client = _clients.get(st.milvus_uri)
    if client is None:
        client = _clients[st.milvus_uri] = milvus_store.get_client(st.milvus_uri, timeout=5.0)
    if not client.has_collection(st.milvus_collection):
        raise RuntimeError(f"知识库集合 {st.milvus_collection} 不存在,先跑 python -m app.jobs.build_knowledge")
    return milvus_store.search_vectors(client, st.milvus_collection, vector, st.rag_top_k)


async def retrieve_hits(query: str, settings: Settings | None = None) -> list:
    st = settings or get_settings()
    emb = EmbeddingClient(build_embeddings(st))
    vec = await emb.embed_query(query)
    scored = await asyncio.to_thread(_search_sync, vec, st)  # pymilvus 同步门面不阻塞事件循环
    ordered = [(cid, s) for cid, s in scored if s >= st.rag_score_threshold]
    if not ordered:
        return []
    async with get_session_factory()() as session:
        rows = await crud.fetch_chunks_by_ids(session, [cid for cid, _ in ordered])
    by_id = {r.id: r for r in rows}
    return [by_id[cid] for cid, _ in ordered if cid in by_id]
