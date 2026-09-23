"""ch04 在线检索核心(spec §4/§5):query 理解 → 策略化召回(四臂同一代码路径)→ MySQL 回查 → 重排 → 闸1。

strategy 是唯一开关:评估四臂=在线一路,差异只在配置——rag_score_threshold 只作用纯 dense 腿
(§4.2-3);闸1 只在 hybrid_rerank 且重排成功时生效(§4.3)。
集合 v2 起 Milvus 带 text(BM25 服务端分词必需,§0-7),原文权威仍是 MySQL:命中 id 回查整行,
孤儿向量被回查天然过滤(ch03 语义)。
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.rag import milvus_store, reranker
from app.rag.embeddings import EmbeddingClient, build_embeddings
from app.rag.query_understanding import UnderstandResult, understand_query

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
_clients: dict[str, Any] = {}  # uri → MilvusClient 进程级缓存(grpc 通道不宜每请求新建,ch03 同款)

# §4.2-4 白名单:品类值实测若含全角括号致集成红,只允许加「（）」两个字符并记 dev-notes,其余照 spec 原样
_ALLOWED_FILTER = re.compile(r"^[\w一-鿿 >()×/,-]+$")


def build_category_expr(category: str | None) -> str | None:
    """白名单校验通过才拼等值 expr——引号/反斜杠进不来,表达式不可注入。"""
    if category is None:
        return None
    if not _ALLOWED_FILTER.match(category):
        raise ValueError(f"非法 category 过滤值: {category!r}")
    return f'category == "{category}"'


@dataclass
class ScoredRow:
    chunk_id: int
    score: float
    row: Any  # KnowledgeChunk ORM(原文权威在 MySQL;score 语义随策略:COSINE/BM25/RRF/relevance)


@dataclass
class RetrieveResult:
    chunks: list[ScoredRow]  # 相关性降序(重排/RRF 序),未过首尾排布
    refused: bool = False
    note: str = ""


def _get_client(st: Settings):
    client = _clients.get(st.milvus_uri)
    if client is None:
        client = _clients[st.milvus_uri] = milvus_store.get_client(st.milvus_uri, timeout=5.0)
    if not client.has_collection(st.milvus_collection):
        raise RuntimeError(f"知识库集合 {st.milvus_collection} 不存在,先跑 python -m app.jobs.build_knowledge")
    return client


def vector_text(row) -> str:
    """与 embed 输入、集合 text 列同源三格拼接(§3.1)。"""
    return f"{row.category}\n{row.questions}\n{row.answer}"


def _dense_sync(client, st: Settings, vec: list[float], expr: str | None):
    return milvus_store.search_vectors(client, st.milvus_collection, vec, st.hybrid_recall_k, expr=expr)


def _bm25_sync(client, st: Settings, text: str, expr: str | None):
    return milvus_store.bm25_search(client, st.milvus_collection, text, st.hybrid_recall_k, expr=expr)


def _hybrid_sync(client, st: Settings, vec, text, expr):
    return milvus_store.hybrid_search(client, st.milvus_collection, vec, text,
                                      limit=st.hybrid_recall_k, recall_k=st.hybrid_recall_k,
                                      rrf_k=st.rrf_k, expr=expr)


async def _embed(text: str, st: Settings) -> list[float]:
    return (await EmbeddingClient(build_embeddings(st)).embed_texts([text]))[0]


async def _fetch_rows(ids: list[int]) -> list[Any]:
    async with get_session_factory()() as session:
        return await crud.fetch_chunks_by_ids(session, ids)


async def retrieve(query: str, *, strategy: str = "hybrid_rerank", category: str | None = None,
                   settings: Settings | None = None,
                   understood: UnderstandResult | None = None) -> RetrieveResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略: {strategy}")
    st = settings or get_settings()
    expr = build_category_expr(category)
    if understood is None:
        understood = await understand_query(query, st)
    client = _get_client(st)
    pairs: list[tuple[int, float]] = []
    if strategy == "bm25":
        pairs = await asyncio.to_thread(_bm25_sync, client, st, understood.bm25_text, expr)
    elif strategy == "dense":
        vec = await _embed(understood.standard_query, st)
        pairs = await asyncio.to_thread(_dense_sync, client, st, vec, expr)
        pairs = [(cid, s) for cid, s in pairs if s >= st.rag_score_threshold]  # §4.2-3 只作用这条腿
    else:
        vec = await _embed(understood.standard_query, st)
        pairs = await asyncio.to_thread(_hybrid_sync, client, st, vec, understood.bm25_text, expr)
        if strategy == "hybrid_rerank":
            return await _rerank_stage(st, pairs, understood)
    return await _attach(st, pairs)


async def _attach(st: Settings, pairs) -> RetrieveResult:
    if not pairs:
        return RetrieveResult(chunks=[])
    rows = {r.id: r for r in await _fetch_rows([cid for cid, _ in pairs])}
    chunks = [ScoredRow(cid, s, rows[cid]) for cid, s in pairs if cid in rows]
    return RetrieveResult(chunks=chunks)


async def _rerank_stage(st: Settings, pairs, understood: UnderstandResult) -> RetrieveResult:
    res = await _attach(st, pairs)
    if not res.chunks:
        return RetrieveResult(chunks=[], refused=True, note="知识库无命中")
    cands = res.chunks[: st.hybrid_recall_k]
    scores = await reranker.rerank(understood.standard_query,
                                   [vector_text(c.row) for c in cands], st)
    if scores is None:  # §4.3-3:重排不可用 → RRF 前 N 序、闸1 跳过(WARN 已在 reranker 内,note 不外泄运维噪声)
        return RetrieveResult(chunks=cands[: st.rerank_top_n])
    reranked = [ScoredRow(cands[i].chunk_id, s, cands[i].row)
                for i, s in scores if 0 <= i < len(cands)][: st.rerank_top_n]
    if not reranked:
        return RetrieveResult(chunks=[], refused=True, note="知识库无命中")
    top1 = reranked[0].score
    if top1 < st.retrieval_low_conf_threshold:  # 闸1(§5.2)
        return RetrieveResult(chunks=[], refused=True,
                              note=f"证据置信度不足(top1={top1:.2f} < 阈值 {st.retrieval_low_conf_threshold})")
    return RetrieveResult(chunks=reranked)


# ---- §4.4 首尾排布纯函数(query_faq 与评估共用,[n] = 排布后 1-based 位置) ----

def head_tail_indices(n: int) -> list[int]:
    """奇数升序铺前段、偶数降序铺后段:n=10 → [1,3,5,7,9,10,8,6,4,2],首尾最相关。"""
    odds = list(range(1, n + 1, 2))
    evens = list(range(n if n % 2 == 0 else n - 1, 1, -2))
    return odds + evens


def apply_head_tail(items: list) -> list:
    return [items[i - 1] for i in head_tail_indices(len(items))]


async def retrieve_hits(query: str, settings: Settings | None = None) -> list:
    """ch03 兼容 shim(返回 ORM 行列表):T7 切 query_faq v2 后即删,勿增新调用方。"""
    res = await retrieve(query, strategy="hybrid", settings=settings)
    return [c.row for c in res.chunks]
