import pytest

from app.core.config import get_settings

pytestmark = pytest.mark.integration


async def test_full_build_then_fault_and_resume():
    """真实链路(需 Milvus up + .env 真 key + 已迁移库):本测试跑「干净链」(全量重建→向量化→check=0);
    fault-after 崩溃→skip-existing 捡漏的「中断链」由下方 CLI 演练完成(验收 2 的正式载体),两链输出都进 dev-notes。
    TRUNCATE 首次执行即核对点⑤销账:若 03 表带自引用 FK,不包 FK 开关会 1701——
    truncate_knowledge_chunks 走通即证明包裹有效(dev-notes 记录首跑是否触发过 1701)。
    """
    from app.db.engine import dispose_engine, init_engine
    from app.rag import indexer

    init_engine(get_settings())
    try:
        total = await indexer.ingest_docs("knowledge")
        assert total > 0
        done = await indexer.vectorize_pending()
        assert done == total
        assert await indexer.check() == 0
    finally:
        await dispose_engine()


@pytest.mark.integration
async def test_corpus_block_count_pinned():
    """实测锚(附录 C 重锚定):当前老师语料全量重建后总块数 = EXPECTED_TOTAL。
    语料再被替换时此测试红 → 人工确认后更新数字并在 dev-notes 说明(回归绊线,非可调阈值)。"""
    from app.core.config import get_settings
    from app.db import crud
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.rag import milvus_store

    EXPECTED_TOTAL = 51  # ← Step 4 实测输出 N 替换本行数字(实施时唯一运行时填值,来源已在 Step 4 说明)
    st = get_settings()
    init_engine(st)
    try:
        async with get_session_factory()() as session:
            counts = await crud.count_chunks_by_status(session)
        assert counts.get("done", 0) == EXPECTED_TOTAL and counts.get("pending", 0) == 0
        client = milvus_store.get_client(st.milvus_uri)
        assert milvus_store.count_rows(client, st.milvus_collection) == EXPECTED_TOTAL
    finally:
        await dispose_engine()
