"""ch10 验收三钉:integration——报告落盘 / 多诉求句双命中 / 后台分布 API 有数。

跑法:uv run pytest tests/e2e/test_ch10_acceptance.py -m integration -v
前置:T7 训练 + T8 评测 + T9 真写(演示行)已完成。
"""

from pathlib import Path

import pytest

from finetune.glossary import CLASSES

pytestmark = pytest.mark.integration


def test_accept1_eval_report_exists():
    p = Path("finetune/reports/ch10_eval_report.md")
    assert p.exists(), "T8 封存首跑未落报告"
    md = p.read_text(encoding="utf-8")
    assert "| 类目 | TP | FP | FN | TN |" in md and "micro-F1" in md


def test_accept3_multi_demand_hits_multiple_classes():
    from app.services.topic_classifier import TopicClassifier

    clf = TopicClassifier()
    (labels,) = clf.classify_batch(["买大了想退"])
    assert "尺码" in labels and "退换货" in labels, f"双命中失败: {labels}"


async def test_accept2_distribution_api_has_data():
    from httpx import ASGITransport, AsyncClient

    from app.api import routes as routes_mod
    from app.core.config import get_settings
    from app.db.engine import dispose_engine, init_engine
    from app.main import app

    init_engine(get_settings())
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.get("/api/topics/distribution")
        assert r.status_code == 200
        body = r.json()
        assert len(body) >= 1, "演示归类行未进分布(先跑 T9 真写)"
        assert all(item["label"] in CLASSES for item in body)
        assert body == sorted(body, key=lambda i: -i["count"])  # count 降序
        assert Path("static/topic.html").exists()  # 终审 I-3:页与入口在分支面上
    finally:
        await dispose_engine()
