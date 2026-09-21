"""门面纯逻辑测试:假 client 记录调用形状,断言门面语义(不碰真 Milvus)。
真实 API 形状以 Task 2 冒烟 + 下方集成测试为准——这是核对点①的双保险。"""

import pytest


class FakeClient:
    def __init__(self, *, has=False, search_out=None, query_out=None, fail=False):
        self.calls = []
        self.has = has
        self.search_out = search_out if search_out is not None else [[]]
        self.query_out = query_out if query_out is not None else []
        self.fail = fail

    def _guard(self):
        self.calls.append(("guard",))
        if self.fail:
            raise RuntimeError("boom")

    def list_collections(self):
        self._guard()
        return ["knowledge"]

    def has_collection(self, collection_name):
        self.calls.append(("has", collection_name))
        return self.has

    def create_collection(self, **kw):
        self.calls.append(("create", kw))

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))
        return type("R", (), {"upsert_count": len(kw["data"])})()

    def search(self, **kw):
        self.calls.append(("search", kw))
        return self.search_out

    def query(self, **kw):
        self.calls.append(("query", kw))
        return self.query_out

    def drop_collection(self, collection_name):
        self.calls.append(("drop", collection_name))


def test_health_ok():
    from app.rag.milvus_store import health_ok

    assert health_ok(FakeClient()) is True
    assert health_ok(FakeClient(fail=True)) is False


def test_ensure_collection_creates_only_when_missing():
    from app.rag.milvus_store import ensure_collection

    c = FakeClient(has=False)
    ensure_collection(c, "knowledge", 1024)
    kind, kw = c.calls[1]
    assert kind == "create"
    assert kw["primary_field_name"] == "chunk_id" and kw["auto_id"] is False
    assert kw["metric_type"] == "COSINE" and kw["dimension"] == 1024
    c2 = FakeClient(has=True)
    ensure_collection(c2, "knowledge", 1024)
    assert not any(k == "create" for k, *_ in c2.calls)


def test_upsert_shapes_and_empty_guard():
    from app.rag.milvus_store import upsert_vectors

    assert upsert_vectors(FakeClient(), "knowledge", []) == 0
    c = FakeClient()
    n = upsert_vectors(c, "knowledge", [(5, [0.1]), (6, [0.2])])
    assert n == 2
    _, kw = [call for call in c.calls if call[0] == "upsert"][0]
    assert kw["data"] == [{"chunk_id": 5, "embedding": [0.1]}, {"chunk_id": 6, "embedding": [0.2]}]


def test_search_maps_hits_to_pairs():
    from app.rag.milvus_store import search_vectors

    c = FakeClient(search_out=[[{"chunk_id": 2, "distance": 0.9}, {"chunk_id": 1, "distance": 0.31}]])  # 键名=实测PK回显形状(Task2冒烟),非计划预核的id
    assert search_vectors(c, "knowledge", [0.0], 2) == [(2, 0.9), (1, 0.31)]
    assert search_vectors(FakeClient(search_out=[]), "knowledge", [0.0], 2) == []


def test_count_and_ids():
    from app.rag.milvus_store import all_ids, count_rows

    assert count_rows(FakeClient(query_out=[{"count(*)": 5}]), "knowledge") == 5
    assert count_rows(FakeClient(query_out=[]), "knowledge") == 0
    assert all_ids(FakeClient(query_out=[{"chunk_id": 1}, {"chunk_id": 3}]), "knowledge") == [1, 3]


def test_drop_only_when_exists():
    from app.rag.milvus_store import drop_collection

    c = FakeClient(has=True)
    drop_collection(c, "knowledge")
    assert ("drop", "knowledge") in c.calls
    drop_collection(FakeClient(has=False), "knowledge")
