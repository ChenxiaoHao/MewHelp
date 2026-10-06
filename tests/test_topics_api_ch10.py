"""ch10 T10 码面钉:topic_distribution 展开计数/pct/降序;端点 days 校验+503+shape。

真库有归类行的计数核对是集成验证(T9 演示行就绪后跑),不在本单测。
"""

from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

import app.db.crud as crud_mod
from app.api import routes as routes_mod
from app.main import app


class _Res:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSession:
    def __init__(self, rows):
        self.sqls: list[str] = []
        self._rows = rows

    async def execute(self, stmt):
        self.sqls.append(str(stmt.compile()))
        return _Res(self._rows)


def _rows():
    now = datetime(2026, 10, 6, 12, 0, 0)
    return [(["发票"], now), (["物流", "运费"], now), (["物流"], now)]


@pytest.mark.asyncio
async def test_topic_distribution_counts_pct_desc_order():
    out = await crud_mod.topic_distribution(FakeSession(_rows()))
    d = {item["label"]: item for item in out}
    assert d["物流"]["count"] == 2 and d["发票"]["count"] == 1
    assert d["运费"]["count"] == 1
    assert out[0]["label"] == "物流"  # count 降序领头
    assert abs(sum(i["pct"] for i in out) - 100.0) < 0.1  # 按标签出现总次数归一


@pytest.mark.asyncio
async def test_topic_distribution_days_window_in_sql_and_empty_safe():
    s = FakeSession(_rows())
    await crud_mod.topic_distribution(s, days=30)
    assert "created_at >=" in s.sqls[0]
    assert await crud_mod.topic_distribution(FakeSession([])) == []


async def _get(days=None):
    app.dependency_overrides[routes_mod.dep_db_session] = lambda: FakeSession(_rows())
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            url = "/api/topics/distribution" + (f"?days={days}" if days is not None else "")
            return await ac.get(url)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_endpoint_shape_and_validation():
    r = await _get()
    assert r.status_code == 200
    body = r.json()
    assert body[0] == {"label": "物流", "count": 2, "pct": 50.0}
    assert {"label", "count", "pct"} <= set(body[0])

    r422 = await _get(days=0)
    assert r422.status_code == 422

    app.dependency_overrides[routes_mod.dep_db_session] = lambda: None
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as ac:
            r503 = await ac.get("/api/topics/distribution")
        assert r503.status_code == 503
    finally:
        app.dependency_overrides.clear()
