"""ch03 两段双写(spec §5)。

Stage1 ingest_docs:默认全量重建(清 MySQL 表 + drop Milvus 集合 → 重灌 pending);
  --skip-existing:不清表,sha1 指纹跳过重复,只追加新块。
Stage2 vectorize_pending:扫 pending → embed_batch → upsert(chunk_id=pk) → 回填 done。
  任一点崩溃:行仍 pending 或已 done 但向量同 pk 可覆写,重跑即自愈——「按主键幂等」。
fault_after:N 之后(批粒度)SystemExit(42),验收 2 的注入。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.rag import milvus_store
from app.rag.chunking import split_markdown
from app.rag.embeddings import EmbeddingClient, build_embeddings


async def ingest_docs(docs_dir: str, *, skip_existing: bool = False) -> int:
    st = get_settings()
    files = sorted(Path(docs_dir).glob("*.md"))
    if not files:
        raise SystemExit(f"{docs_dir} 下没有 .md,语料放对位置了吗?")
    client = milvus_store.get_client(st.milvus_uri)
    if not milvus_store.health_ok(client):
        raise SystemExit("Milvus 不可达:离线建库 fail-fast,不起半库(§8)")
    total = 0
    async with get_session_factory()() as session:
        if skip_existing:
            seen = await crud.existing_chunk_fingerprints(session)
        else:
            await crud.truncate_knowledge_chunks(session)
            milvus_store.drop_collection(client, st.milvus_collection)
            seen = set()
        for f in files:
            drafts = split_markdown(
                f.read_text(encoding="utf-8"),
                chunk_size=st.chunk_size,
                chunk_overlap=st.chunk_overlap,
            )
            fresh = [
                d for d in drafts
                if crud.chunk_fingerprint(d.category, d.questions, d.answer) not in seen
            ]
            if fresh:
                await crud.add_chunk_drafts(session, fresh)
                total += len(fresh)
            print(f"[ingest] {f.name}: {len(drafts)} 块,新增 {len(fresh)}(skip_existing={skip_existing})")
    return total


async def vectorize_pending(fault_after: int | None = None) -> int:
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    if not milvus_store.health_ok(client):
        raise SystemExit("Milvus 不可达,向量化终止(pending 行未动,恢复后重跑自捡)")
    milvus_store.ensure_collection(client, st.milvus_collection, st.embedding_dimensions)
    emb = EmbeddingClient(build_embeddings(st))
    async with get_session_factory()() as session:
        pending = await crud.fetch_pending_chunks(session)
    print(f"[vectorize] pending={len(pending)}")
    done_total = 0
    for i in range(0, len(pending), st.embedding_batch_size):
        batch = pending[i : i + st.embedding_batch_size]
        ids = [r.id for r in batch]
        texts = [f"{r.category}\n{r.questions}\n{r.answer}" for r in batch]
        vectors = await emb.embed_texts(texts)  # OpenAIEmbeddings 内置 max_retries=3 退避
        rows = [
            {"chunk_id": cid, "text": t, "embedding": vec,
             "category": r.category, "content_type": r.content_type or ""}
            for cid, t, vec, r in zip(ids, texts, vectors, batch)
        ]  # text 与 embed 输入同源三格拼接(§3.1);r 来自 batch(ORM 行)
        milvus_store.upsert_rows(client, st.milvus_collection, rows)
        async with get_session_factory()() as session:  # 每批独立事务(§5)
            await crud.mark_chunks_vectorized(session, ids)
        done_total += len(ids)
        print(f"[vectorize] {ids[0]}..{ids[-1]} done ({done_total}/{len(pending)})")
        if fault_after is not None and done_total >= fault_after:
            print(f"[fault-after] 已向量化 {done_total} ≥ {fault_after},注入退出 rc=42")
            raise SystemExit(42)
    if pending:
        milvus_store.flush(client, st.milvus_collection)  # 核对点①:全批写完必须 flush,check/在线 search 才可见
    return done_total


async def check() -> int:
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    if not milvus_store.health_ok(client):
        print("[check] Milvus 不可达,无法对账")
        return 2
    async with get_session_factory()() as session:
        counts = await crud.count_chunks_by_status(session)
        done_ids = set(await crud.fetch_done_ids(session))
    has = client.has_collection(st.milvus_collection)
    milvus_ids = set(milvus_store.all_ids(client, st.milvus_collection)) if has else set()
    diff = done_ids ^ milvus_ids
    rc = 0 if counts.get("pending", 0) == 0 and not diff else 1
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows GBK 控制台防崩(显示层,非业务)
    print(f"[check] pending={counts.get('pending', 0)} done={len(done_ids)} "
          f"milvus={len(milvus_ids)} 差集={sorted(diff) if diff else '∅'} → {'OK' if rc == 0 else 'MISMATCH'}")
    return rc
