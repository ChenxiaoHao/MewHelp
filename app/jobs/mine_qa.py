"""挖 QA CLI 两阶段(spec §6):extract(LLM→staging)→ dedup(三道闸→入库→顺带向量化)。

uv run python -m app.jobs.mine_qa                   # 全流程
uv run python -m app.jobs.mine_qa --dedup-only      # 只跑去重入库
uv run python -m app.jobs.mine_qa --reprocess-kept  # 全量重建后找回 mined 知识:kept→extracted 再走闸
uv run python -m app.jobs.mine_qa --clear-staging   # 物理清暂存表
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.prompts.qa_mining import mining_messages
from app.rag import dedupe, indexer, milvus_store
from app.rag.chunking import ChunkDraft
from app.rag.embeddings import EmbeddingClient, build_embeddings
from app.schemas.qa_mining import MinedQA

QA_CATEGORY = "客服对话问答"  # §4-5 固定值


def _build_model():
    from langchain_openai import ChatOpenAI  # 与 routes dep_chat_model 同源配置

    st = get_settings()
    model = ChatOpenAI(model=st.model_name, temperature=0,
                       api_key=st.openai_api_key, base_url=st.openai_base_url)
    return model.with_structured_output(MinedQA)  # 核对点④


async def _extract_one_conversation(structured, cid: int) -> int:
    async with get_session_factory()() as session:
        transcript = await crud.conversation_transcript(session, cid)
    if not transcript:
        return 0
    result = None
    for attempt in (1, 2):  # 不合 schema 重试 1 次,再败整通不落账(§6)
        try:
            result = await structured.ainvoke(mining_messages(transcript))
            break
        except Exception as exc:  # noqa: BLE001 —— 单通失败不拖垮批次
            print(f"[extract] conv:{cid} 第 {attempt} 次失败: {exc}")
    if result is None:
        return 0
    items = [(i.question.strip(), i.answer.strip())
             for i in result.items if i.question.strip() and i.answer.strip()]
    batch_no = f"{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    async with get_session_factory()() as session:
        if items:
            n = await crud.add_qa_staging_rows(session, batch_no, f"conv:{cid}", items)
        else:  # 无可抽会话也要记账(写一条直接 discarded 的占位行,防重抽)
            await crud.add_qa_staging_rows(
                session, batch_no, f"conv:{cid}", [("(无可复用知识)", "(无)")], status="discarded")
            n = 0
    print(f"[extract] conv:{cid} → {len(items)} 条入 staging")
    return n


async def extract_phase(batch_size: int | None = None) -> int:
    st = get_settings()
    structured = _build_model()
    async with get_session_factory()() as session:
        ids = await crud.unmined_conversation_ids(session, batch_size or st.qa_mine_batch_conversations)
    written = 0
    for cid in ids:
        written += await _extract_one_conversation(structured, cid)
    return written


async def dedup_phase() -> int:
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    if not milvus_store.health_ok(client):
        raise SystemExit("Milvus 不可达:闸3 无法执行,挖 QA fail-fast(§8)")
    milvus_store.ensure_collection(client, st.milvus_collection, st.embedding_dimensions)
    emb = EmbeddingClient(build_embeddings(st))
    async with get_session_factory()() as session:
        rows = await crud.fetch_extracted_staging(session)
    if not rows:
        print("[dedup] 无 extracted 行,跳过")
        return 0
    by_id = {r.id: r for r in rows}
    groups = dedupe.group_exact([(r.id, r.question) for r in rows])              # 闸1
    keys = list(range(len(groups)))
    rep_row = {k: by_id[groups[k][0]] for k in keys}
    texts = [f"{QA_CATEGORY}\n{rep_row[k].question}\n{rep_row[k].answer}" for k in keys]  # 与库内 chunk 同构拼接(§4-6)
    rep_vecs = dict(zip(keys, await emb.embed_texts(texts)))
    clusters = dedupe.merge_semantic(keys, rep_vecs, st.qa_dedup_threshold)      # 闸2

    def search_fn(vec, k):
        return milvus_store.search_vectors(client, st.milvus_collection, vec, k)   # 闸3(同步 client,批量小,直接调)

    kept_clusters, dead_clusters = dedupe.gate3_split(clusters, rep_vecs, search_fn, st.qa_dedup_threshold)
    kept: list[tuple[ChunkDraft, list[int]]] = []
    for cluster in kept_clusters:
        seen, qs, member_ids = set(), [], []
        for g in cluster:
            for sid in groups[g]:
                member_ids.append(sid)
                q = by_id[sid].question.strip()
                key = dedupe.normalize_question(q)
                if key not in seen:
                    seen.add(key)
                    qs.append(q)
        draft = ChunkDraft(category=QA_CATEGORY, questions="\n".join(qs),
                           answer=rep_row[cluster[0]].answer, section_path=None,
                           content_type="qa_mined", is_key_clause=False)
        kept.append((draft, member_ids))
    discarded_ids = [sid for c in dead_clusters for g in c for sid in groups[g]]
    async with get_session_factory()() as session:
        n = await crud.finalize_qa(session, kept, discarded_ids)  # 单事务:断→回滚→仍 extracted
    print(f"[dedup] extracted {len(rows)} 行 → kept {n} chunk,discarded {len(discarded_ids)} 行")
    await indexer.vectorize_pending()  # §6 尾:新知即刻可检索
    return n


async def _run(args: argparse.Namespace) -> int:
    try:
        if args.clear_staging:
            async with get_session_factory()() as session:
                print(f"[clear-staging] 物理清空 {await crud.clear_staging(session)} 行")
        if args.reprocess_kept:
            async with get_session_factory()() as session:
                print(f"[reprocess-kept] {await crud.reprocess_kept_staging(session)} 行 kept→extracted")
        if not args.dedup_only:
            await extract_phase(args.batch_size)
        await dedup_phase()
        return 0
    finally:
        await dispose_engine()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mine_qa")
    parser.add_argument("--batch-size", type=int, default=None, help="本次抽取会话数(默认 Settings.qa_mine_batch_conversations)")
    parser.add_argument("--dedup-only", action="store_true", help="跳过抽取,只跑去重入库")
    parser.add_argument("--clear-staging", action="store_true", help="物理清空 staging(§6 保留行可追溯的对立面)")
    parser.add_argument("--reprocess-kept", action="store_true", help="把 kept 行翻回 extracted 重走三道闸(全量重建清空 qa_mined 后的找回路径,§5 副作用闭环)")
    args = parser.parse_args(argv)
    init_engine(get_settings())
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
