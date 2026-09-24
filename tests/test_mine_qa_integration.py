import pytest

pytestmark = pytest.mark.integration


async def test_mine_two_runs_idempotent():
    """确定性断言放幂等与状态机上;附录 D 的闸靶子判定交给 qa_mining eval + 一次人工观察(打印),
    防 LLM 措辞漂移把 CI 化断言变成抛硬币。"""
    from sqlalchemy import select, func

    from app.core.config import get_settings
    from app.db import crud
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.db.models import KnowledgeChunk, QaExtractionStaging
    from app.jobs import mine_qa
    from app.rag import indexer

    init_engine(get_settings())
    try:
        # 可重入化(Task 11 现场修):计划默认"首跑后状态",但 Task 11/12 要求纯文档库、必然发生重建,
        # 一次性状态会让本测试重建后必挂(kept1=0 在第 32/33 行断言前无从恢复)。
        # 入口先走本测试自己钉死的官方找回路径:全量重建(清 qa_mined)→ kept 翻回 extracted → 再跑原断言链。
        await indexer.ingest_docs("knowledge")
        async with get_session_factory()() as session:
            await crud.reprocess_kept_staging(session)
        await mine_qa.extract_phase(batch_size=100)
        kept1 = await mine_qa.dedup_phase()
        async with get_session_factory()() as session:
            staging_total_1 = (await session.execute(select(func.count()).select_from(QaExtractionStaging))).scalar()
            mined = (await session.execute(
                select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.content_type == "qa_mined")
            )).scalar()
            ins = (await session.execute(
                select(KnowledgeChunk.questions, KnowledgeChunk.answer).where(
                    KnowledgeChunk.content_type == "qa_mined")
            )).all()
        assert mined >= 4 and all(q.strip() and a.strip() for q, a in ins)  # 机制锚:条数+完整性;话题词锚随语料换代废止(附录C)
        assert kept1 >= 4
        # 第二轮:抽取阶段应零新行(source_ref 记账幂等),dedup 零变化
        assert await mine_qa.extract_phase(batch_size=100) == 0
        async with get_session_factory()() as session:
            assert (await session.execute(select(func.count()).select_from(QaExtractionStaging))).scalar() == staging_total_1
        # 全量重建后找回路径(§5 副作用闭环)。顺序必须是 先重建、后翻档:
        # 不重建直接 reprocess,闸3 会撞上自家旧 qa_mined chunk(cosine=1.0)全部被判重——这个坑由本测试钉死
        await indexer.ingest_docs("knowledge")          # 全量重建:清表+drop 集合+文档块 pending(旧 qa_mined 清零)
        async with get_session_factory()() as session:
            n = await crud.reprocess_kept_staging(session)
        assert n >= 4
        kept2 = await mine_qa.dedup_phase()             # 找回 kept 块,尾部 vectorize_pending 顺带补齐全部 pending
        assert kept2 >= 4
    finally:
        await dispose_engine()
