import pytest

pytestmark = pytest.mark.integration


async def test_freight_query_hits_ship_section():
    """验收 1 的断言形态:「邮费是多少」top-1 必须是运费说明块。"""
    from app.core.config import get_settings
    from app.db.engine import dispose_engine, init_engine
    from app.rag import retriever

    init_engine(get_settings())
    try:
        # 顺序无关化:mine_qa 集成(字典序在前)跑完库变混库;本断言定义在纯文档库上(附录 C 同款前提),
        # 入口处自给纯文档状态——重建是既有 CLI 路径,非新语义。
        from app.rag import indexer

        await indexer.ingest_docs("knowledge")
        await indexer.vectorize_pending()
        hits = await retriever.retrieve_hits("邮费是多少")
        assert hits and "运费说明" in hits[0].section_path
        assert "99" in hits[0].answer
    finally:
        await dispose_engine()
