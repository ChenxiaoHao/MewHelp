"""SiliconFlow /v1/rerank 门面(spec §4.3)。BAAI/bge-reranker-v2-m3 云托管,httpx 直连零新依赖。

降级语义(§8):无 key/超时/HTTP 错/响应不合形状 → None,调用方用 RRF 前序续跑并跳过闸1;
WARN 每进程只响一次(评估批量跑 300 题时防刷屏)。请求体字段 documents/top_n 形状经 Context7
预核(核对点③),集成用例现场销账。
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_warned = False


def _warn_degraded(exc: Exception) -> None:
    global _warned
    if not _warned:
        logger.warning("rerank 不可用,降级 RRF 序(WARN 只响一次): %s", exc)
        _warned = True
    else:
        logger.info("rerank 再次失败(已降级): %s", exc)


async def rerank(query: str, texts: list[str], st: Settings, *,
                 transport: httpx.AsyncBaseTransport | None = None
                 ) -> list[tuple[int, float]] | None:
    if not st.rerank_api_key or not texts:
        return None
    try:
        kw = {"transport": transport} if transport else {}
        async with httpx.AsyncClient(timeout=st.rerank_timeout_seconds, **kw) as client:
            resp = await client.post(
                f"{st.rerank_api_base.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {st.rerank_api_key}"},
                json={"model": st.rerank_model, "query": query, "documents": texts,
                      "top_n": min(st.rerank_top_n, len(texts)), "return_documents": False},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]
    except Exception as exc:  # noqa: BLE001 —— §8:重排永不炸检索链路
        _warn_degraded(exc)
        return None
