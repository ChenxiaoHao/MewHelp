"""三端点单测:fake crud 转发/404/503/PATCH 校验 422(核对点⑤)。"""

import pytest

from app.api import routes as routes_mod
from app.main import app


def _override(app_obj, dep, val):
    app_obj.dependency_overrides[dep] = lambda: val


class FakeCrud:
    def __init__(self, chunk=None, cases=None, patched="ROW"):
        self.chunk, self.cases, self.patched = chunk, cases or [], patched
        self.calls = []

    async def get_chunk(self, session, chunk_id):
        self.calls.append(("chunk", chunk_id))
        return self.chunk

    async def list_faith_cases(self, session, *, status, bucket, limit=200):
        self.calls.append(("list", status, bucket))
        return self.cases

    async def set_faith_case_status(self, session, case_id, status, resolution):
        self.calls.append(("patch", case_id, status, resolution))
        return self.patched


def _faith_row(case_id=1, status="未解决"):
    from datetime import datetime

    from app.db.models import FaithCase

    return FaithCase(id=case_id, eval_id="A22", bucket="A_policy", query="银卡打折吗",
                     strategy="hybrid_rerank", answer="a", reason="r", citations=[{"n": 1}],
                     judge_model="m", status=status, seen_count=1,
                     first_seen_at=datetime(2026, 9, 23), last_seen_at=datetime(2026, 9, 23),
                     resolution=None, resolved_at=None)


def _chunk_row():
    from app.db.models import KnowledgeChunk

    # 核对点纠偏:brief 逐字写 status="done",但 ORM 列名是 vectorize_status(T1 定,
    # 构造器拒未知 kwarg → TypeError)。语义保真:该行仍是「已建向量」chunk。
    return KnowledgeChunk(id=7, category="商品规格手册", questions="q", answer="a",
                          vector_id="7", vectorize_status="done", section_path="商品规格手册 > 节",
                          content_type="policy", prev_chunk_id=None, next_chunk_id=8)


async def test_get_chunk_ok_404_503(client, monkeypatch):
    fake = FakeCrud(chunk=_chunk_row())
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.get("/api/chunks/7")
    assert r.status_code == 200
    assert r.json() == {"id": 7, "section_path": "商品规格手册 > 节", "category": "商品规格手册",
                        "questions": "q", "answer": "a", "content_type": "policy",
                        "prev_chunk_id": None, "next_chunk_id": 8}
    monkeypatch.setattr(routes_mod, "crud", FakeCrud(chunk=None))
    assert (await client.get("/api/chunks/999999")).status_code == 404
    _override(app, routes_mod.dep_db_session, None)
    assert (await client.get("/api/chunks/7")).status_code == 503
    app.dependency_overrides.clear()


async def test_faith_list_and_filter(client, monkeypatch):
    fake = FakeCrud(cases=[_faith_row()])
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.get("/api/faith_cases", params={"status": "未解决", "bucket": "A_policy"})
    assert r.status_code == 200 and r.json()[0]["eval_id"] == "A22"
    assert fake.calls[-1] == ("list", "未解决", "A_policy")
    app.dependency_overrides.clear()


async def test_faith_patch_validation_422(client, monkeypatch):
    # 核对点纠偏:brief 逐字用 patched="ROW" 哨兵,但路由 response_model=FaithCaseOut
    # 会真序列化返回值,裸字符串 → ResponseValidationError 500(实测),自身 assert 200
    # 即不可能成立。改用真实 ORM 行——断言一字未动且更有力(200 路径同时钉 FaithCaseOut
    # 序列化);404 路径仍由显式 patched=None 表达(下方)。
    fake = FakeCrud(patched=_faith_row())
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决"})  # 缺处置说明
    assert r.status_code == 422
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决", "resolution": "  "})
    assert r.status_code == 422  # 空白同缺
    r = await client.patch("/api/faith_cases/1", json={"status": "随便"})
    assert r.status_code == 422  # Literal 枚举外
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决", "resolution": "老师标注出入"})
    assert r.status_code == 200
    assert fake.calls[-1] == ("patch", 1, "已解决", "老师标注出入")
    monkeypatch.setattr(routes_mod, "crud", FakeCrud(patched=None))
    assert (await client.patch("/api/faith_cases/2",
                               json={"status": "未解决"})).status_code == 404
    app.dependency_overrides.clear()
