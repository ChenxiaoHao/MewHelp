"""ch10 T9 Step3:integration——真模型活库 --limit 3 真写 + 幂等(演示资产保留,dev-notes 记 id)。

跑法:uv run pytest tests/test_topic_classify_live_ch10.py -m integration -v
前置:models/ch10_topic 训练资产已落盘(缺则 skip);job 走子进程避免与测试引擎互踩 dispose。
幂等口径:同池再 --limit 3 只会选中别的未归类行——用 dry-run selected 数钉「已归类不被重选」,
--rerun 钉总数不变(一人一行 uk,T1 已钉 IntegrityError)。
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from app.core.config import get_settings
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import LowConfidenceQuestion, TopicClassification

MODEL_DIR = Path("models/ch10_topic")
POOL = 138  # 现状事实:ch09 池全量(spec 现状节);变动即红,强制对账


def _counts() -> tuple[int, int]:
    async def _q():
        init_engine(get_settings())
        try:
            async with get_session_factory()() as s:
                total = (await s.execute(
                    select(func.count()).select_from(LowConfidenceQuestion))).scalar()
                cls = (await s.execute(
                    select(func.count()).select_from(TopicClassification))).scalar()
                return total, cls
        finally:
            await dispose_engine()
    return asyncio.run(_q())


def _run_job(*flags) -> str:
    r = subprocess.run([sys.executable, "-m", "app.jobs.topic_classify", *flags],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-400:] + r.stderr[-400:]
    return r.stdout


@pytest.mark.integration
def test_real_model_writes_three_then_idempotent():
    if not (MODEL_DIR / "topic_config.json").exists():
        pytest.skip("训练资产未就绪")
    total0, cls0 = _counts()
    assert cls0 <= 6, "先清场:演示行应只来自本章小批真写"
    _run_job("--limit", "3")
    _, cls1 = _counts()
    assert cls1 == cls0 + 3                       # 首跑 insert 3
    out = _run_job("--dry-run")  # 全池 dry-run:已归类行不出现在选池(只剩 138-已归类)
    assert int(re.search(r"selected=(\d+)", out).group(1)) == total0 - cls1
    _run_job("--rerun", "--limit", "3")
    _, cls2 = _counts()
    assert cls2 == cls1                           # rerun 只刷新不新增(uk 幂等)


@pytest.mark.integration
def test_demo_row_ids_printed_for_devnotes():
    if not (MODEL_DIR / "topic_config.json").exists():
        pytest.skip("训练资产未就绪")

    async def _q():
        init_engine(get_settings())
        try:
            async with get_session_factory()() as s:
                return (await s.execute(text(
                    "SELECT id, question_id, labels FROM topic_classifications "
                    "ORDER BY id DESC LIMIT 8"))).all()
        finally:
            await dispose_engine()
    for r in asyncio.run(_q()):
        print(f"[demo] tc_id={r[0]} qid={r[1]} labels={r[2]}")
