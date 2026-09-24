"""检索在线集成:需要 T3 全量重灌完成的 knowledge 集合 + 已配 embedding key。
(混合检索臂不依赖 rerank key:无 key 走 RRF 降级序,断言用 bm25/hybrid 腿不受影响。

T6 重锚:旧「邮费→运费说明」锚(ch03 自造语料节名,老师语料无此节)换为老师语料的
「运费族」(billing-shipping「运费与包邮」/product-faq「运费怎么算」)与「MH-LP100」型号锚。
两处对 brief 的实测修正(brief 内部矛盾,按「断言语义不动」红线收口,记 dev-notes ④):
① ensure_knowledge_built 里 `asyncio.run(bk.main([]))` 不成立——build_knowledge.main(argv)
   本体自带 init_engine+asyncio.run+dispose 并返回 int,外层再 run 会 TypeError;brief 注释
   明示「以现有 CLI 函数为准,选定后删另一支」→ 直调 `bk.main([])`,删 asyncio.run 支。
② retriever.retrieve 的 _attach 要 MySQL 回查(get_session_factory),brief 全文无 init_engine
   (旧锚是测试体内自管的)。补法先试会话级 init+teardown dispose:实测一红一绿交替炸
   (pool_pre_ping 的连接不能跨 pytest-asyncio 每测新 loop 复用)→ 改回仓库既有约定
   (test_indexer_integration 同款):引擎生命周期进每个测试自身的事件循环。4 个断言零改动。
③ brief 自举条件「库空则重建」欠「库混」:mine_qa 集成(字典序在前)跑完留 qa_mined 块
   (section_path=None)入 MySQL+Milvus,「运费族 top3 含运费节」断言前提=纯文档库(附录 C,
   旧锚入口注释同款自给)。实测混库下 mined 块挤进 top3 → TypeError(None 不可 in)。
   → 空或混(存在 qa_mined)= 均现场全量重建,断言零改动;本文件为字典序最后的集成文件,
   重建后留给下一会话的也是纯库(顺带修复性)。
"""

import asyncio

import pytest

from app.core.config import get_settings
from app.db.engine import dispose_engine, init_engine
from app.rag import milvus_store, retriever

pytestmark = pytest.mark.integration


def _has_mined_chunks() -> bool:
    """会话级同步探一下 MySQL 是否存在 qa_mined 块(附录 C 纯文档库前提的混库信号)。"""
    from sqlalchemy import func, select

    from app.db import models
    from app.db.engine import dispose_engine, get_session_factory, init_engine

    st = get_settings()
    init_engine(st)

    async def _q():
        async with get_session_factory()() as session:
            n = (await session.execute(
                select(func.count()).select_from(models.KnowledgeChunk)
                .where(models.KnowledgeChunk.content_type == "qa_mined"))).scalar()
        await dispose_engine()
        return n

    return asyncio.run(_q()) > 0


@pytest.fixture(scope="session", autouse=True)
def ensure_knowledge_built():
    """库空「或混」则现场全量重建(自举;CI 单跑此文件也成立)。混=存在 qa_mined 块,
    mine_qa 集成(字典序在前)跑完留下的,断言前提为纯文档库(附录 C;docstring 修正③)。
    本夹具为同步函数:引擎生命周期各归其 asyncio.run,不与任何测试 loop 交叉。"""
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    empty = not (client.has_collection(st.milvus_collection)
                 and milvus_store.count_rows(client, st.milvus_collection) > 0)
    if not empty and not _has_mined_chunks():
        return
    from app.jobs import build_knowledge as bk

    bk.main([])  # 真实 CLI 入口:自带 init_engine+asyncio.run+dispose(见 docstring 修正①)


@pytest.fixture(autouse=True)
async def engine_in_test_loop():
    """每测引擎生命周期同 loop(修正②,仓库集成惯例)。"""
    init_engine(get_settings())
    yield
    await dispose_engine()


async def test_bm25_arm_hits_model_number():
    """验收2 地基:型号题 BM25 腿 top-1 就是该型号节。"""
    res = await retriever.retrieve("MH-LP100 的废砂盒多久倒一次", strategy="bm25")
    assert res.chunks and "MH-LP100" in res.chunks[0].row.section_path


async def test_hybrid_ship_family_top3():
    """口语「邮费」经 hybrid 腿命中运费族章节(ch03 锚「运费说明」→ 老师语料族)。"""
    res = await retriever.retrieve("邮费是多少", strategy="hybrid")
    assert any("运费" in c.row.section_path for c in res.chunks[:3])


async def test_category_filter_restricts_results():
    res = await retriever.retrieve("MH-LP100", strategy="hybrid", category="商品规格手册")
    assert res.chunks, "该品类下应命中型号内容"
    assert all(c.row.category == "商品规格手册" for c in res.chunks)


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="闸1 需 rerank 分数,无 key 时降级路径不判拒答")
async def test_gate1_refuses_absent_topic():
    """验收4(检索侧):知识库没有的问题 → 闸1 refused+note。"""
    res = await retriever.retrieve("喵星人太空电梯门票多少钱")
    assert res.refused and res.chunks == [] and res.note
