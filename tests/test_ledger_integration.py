"""台账活库往返:中文 ENUM 读写(核对点⑥)、uk_eval_id upsert、JSON citations、复发流转。
每次用随机 eval_id 避免重跑撞 uk;结束自清理。"""

import uuid
from datetime import datetime

import pytest

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine

pytestmark = pytest.mark.integration


async def test_ledger_roundtrip_live():
    init_engine(get_settings())
    eid = f"IT{uuid.uuid4().hex[:10]}"
    try:
        async with get_session_factory()() as session:
            assert await crud.upsert_faith_case(
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开",
                reason="应拒答未拒", citations=[{"n": 1, "chunk_id": 7, "section_path": "s",
                                                "question": "q", "answer": "a"}],
                judge_model="judge-x") == "created"
            assert await crud.upsert_faith_case(
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开2",
                reason="复发", citations=None, judge_model="judge-x") == "updated"
            row = (await crud.list_faith_cases(session, status="未解决", bucket="D_absent"))
            mine = [r for r in row if r.eval_id == eid][0]
            # 计划缺陷修订(实施者自判;计划回写由 controller 补做,commit 21267a3):重判行照 brief 原文(citations=None);
            # spec §3.2/附录 A「重判→更新 citations 快照」为无条件覆写,None 亦覆写(旧值不留),
            # 故 brief 原文断言 mine.citations[0]["n"] == 1 不成立——按修订改钉 None
            # (此路由 FakeSession 单测同向加钉,见 test_faith_crud_ch04 末案;断言只增不减)。
            assert mine.seen_count == 2 and mine.citations is None
            await crud.set_faith_case_status(session, mine.id, "已解决", "老师标注出入,语料可答")
            assert await crud.upsert_faith_case(  # 复发 → reactivated
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开3",
                reason="再判编造", citations=[], judge_model="judge-x") == "reactivated"
            again = await crud.get_faith_case(session, mine.id)
            assert again.status == "未解决" and again.resolution is None
            assert again.resolved_at is not None and again.seen_count == 3
            assert isinstance(again.first_seen_at, datetime)
    finally:
        from sqlalchemy import delete

        from app.db.models import FaithCase

        async with get_session_factory()() as session:
            await session.execute(delete(FaithCase).where(FaithCase.eval_id == eid))
            await session.commit()
        await dispose_engine()
