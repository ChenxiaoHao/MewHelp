"""验收4 集成:无 key 时闸1 不判 → 用例 skipif;有 key:问知识库没有的 → query_faq refused + 池行数 +1。"""

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.engine import dispose_engine, get_engine, init_engine
from app.db.models import LowConfidenceQuestion
from app.tools import definitions as d

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="闸1 依赖 rerank 分数")
async def test_absent_question_refused_and_pooled():
    st = get_settings()
    init_engine(st)
    try:
        async with get_engine().connect() as conn:
            before = (await conn.execute(select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        out = await d.query_faq.ainvoke({"keyword": "请问月球基地的喵星人会员费多少钱一个月"},
                                        config={"configurable": {"conversation_id": None}})
        assert out["refused"] is True and out["hits"] == []
        async with get_engine().connect() as conn:
            after = (await conn.execute(select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        assert after == before + 1
    finally:
        await dispose_engine()
