"""门面纯逻辑测试 v2:假 client 记录 schema/调用形状(不碰真 Milvus)。
真实形状以集成冒烟为准——核对点①双保险(同 ch03 惯例)。"""


class FakeSchema:
    def __init__(self):
        self.fields = []      # [(name, data_type, kwargs)]
        self.functions = []

    def add_field(self, name, data_type, **kw):
        self.fields.append((name, data_type, kw))

    def add_function(self, fn):
        self.functions.append(fn)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []     # [(field, kwargs)]

    def add_index(self, field, **kw):
        self.indexes.append((field, kw))


class FakeClient:
    def __init__(self, *, has=False, search_out=None, query_out=None, hybrid_out=None, fail=False):
        self.calls = []
        self.has = has
        self.search_out = search_out if search_out is not None else [[]]
        self.query_out = query_out if query_out is not None else []
        self.hybrid_out = hybrid_out if hybrid_out is not None else [[]]
        self.fail = fail
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()

    def list_collections(self):
        if self.fail:
            raise RuntimeError("boom")
        return ["knowledge"]

    def has_collection(self, collection_name):
        self.calls.append(("has", collection_name))
        return self.has

    def create_schema(self, **kw):
        self.calls.append(("create_schema", kw))
        return self.schema

    def prepare_index_params(self):
        return self.index_params

    def create_collection(self, **kw):
        self.calls.append(("create", kw))

    def drop_collection(self, collection_name):
        self.calls.append(("drop", collection_name))

    def flush(self, collection_name):
        self.calls.append(("flush", collection_name))

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))
        return type("R", (), {"upsert_count": len(kw["data"])})()

    def search(self, **kw):
        self.calls.append(("search", kw))
        return self.search_out

    def query(self, **kw):
        self.calls.append(("query", kw))
        return self.query_out

    def hybrid_search(self, **kw):
        self.calls.append(("hybrid", kw))
        return self.hybrid_out


def _field_kw(client):
    return {name: kw for name, _dt, kw in client.schema.fields}


def test_ensure_collection_declares_bm25_schema():
    from pymilvus import DataType, FunctionType

    from app.rag.milvus_store import ensure_collection

    c = FakeClient(has=False)
    ensure_collection(c, "knowledge", 1024)
    names = [n for n, _dt, _kw in c.schema.fields]
    assert names == ["chunk_id", "text", "sparse", "embedding", "category", "content_type"]
    text = _field_kw(c)["text"]
    assert text["enable_analyzer"] is True and text["analyzer_params"] == {"type": "chinese"}
    assert text["max_length"] == 8192
    assert c.schema.fields[0][1] is DataType.INT64 and c.schema.fields[0][2]["is_primary"] is True
    assert c.schema.fields[2][1] is DataType.SPARSE_FLOAT_VECTOR
    assert _field_kw(c)["embedding"]["dim"] == 1024
    assert _field_kw(c)["category"]["max_length"] == 765
    fn = c.schema.functions[0]
    assert fn.type is FunctionType.BM25 and fn.name == "bm25_fn"  # 实测:公开访问器是 .type(计划预核的 .function_type 仅构造参数名,Context7 复核)
    assert list(fn.input_field_names) == ["text"] and list(fn.output_field_names) == ["sparse"]
    assert (c.index_params.indexes[0][0], c.index_params.indexes[0][1]["index_type"],
            c.index_params.indexes[0][1]["metric_type"]) == ("embedding", "AUTOINDEX", "COSINE")
    assert (c.index_params.indexes[1][0], c.index_params.indexes[1][1]["index_type"],
            c.index_params.indexes[1][1]["metric_type"]) == ("sparse", "SPARSE_INVERTED_INDEX", "BM25")
    kw = [call for call in c.calls if call[0] == "create"][0][1]
    assert kw["collection_name"] == "knowledge" and kw["schema"] is c.schema
    c2 = FakeClient(has=True)
    ensure_collection(c2, "knowledge", 1024)
    assert not any(k == "create" for k, *_ in c2.calls)


def test_upsert_rows_passthrough_and_empty_guard():
    from app.rag.milvus_store import upsert_rows

    assert upsert_rows(FakeClient(), "knowledge", []) == 0
    c = FakeClient()
    rows = [{"chunk_id": 5, "text": "t5", "embedding": [0.1], "category": "c", "content_type": "policy"}]
    assert upsert_rows(c, "knowledge", rows) == 1
    kw = [call for call in c.calls if call[0] == "upsert"][0][1]
    assert kw["data"] == rows  # 仅 {chunk_id,text,embedding,category,content_type};sparse 服务端函数生成
    assert all("sparse" not in r for r in kw["data"])


def test_search_dense_and_bm25_shapes():
    from app.rag.milvus_store import bm25_search, search_vectors

    c = FakeClient(search_out=[[{"chunk_id": 2, "distance": 0.9}, {"chunk_id": 1, "distance": 0.31}]])
    assert search_vectors(c, "knowledge", [0.0], 2) == [(2, 0.9), (1, 0.31)]
    kw = [call for call in c.calls if call[0] == "search"][0][1]
    assert kw["data"] == [[0.0]] and "filter" not in kw
    assert kw["anns_field"] == "embedding"  # 实测:双向量列集合缺省 anns_field 报 1100,门面必须显式传
    c2 = FakeClient(search_out=[[{"chunk_id": 7, "distance": 12.5}]])
    assert bm25_search(c2, "knowledge", "MH-LP100", 3, expr='category == "商品参数"') == [(7, 12.5)]
    kw2 = [call for call in c2.calls if call[0] == "search"][0][1]
    assert kw2["data"] == ["MH-LP100"] and kw2["anns_field"] == "sparse"
    assert kw2["filter"] == 'category == "商品参数"'


def test_hybrid_search_builds_two_legs_with_rrf():
    from pymilvus import RRFRanker

    from app.rag.milvus_store import hybrid_search

    c = FakeClient(hybrid_out=[[{"chunk_id": 3, "distance": 0.032}]])
    out = hybrid_search(c, "knowledge", [0.1, 0.2], "猫砂盆 清理",
                        limit=10, recall_k=50, rrf_k=60, expr='category == "x"')
    assert out == [(3, 0.032)]
    kw = [call for call in c.calls if call[0] == "hybrid"][0][1]
    reqs = kw["reqs"]
    assert (reqs[0].anns_field, reqs[0].limit, reqs[0].expr) == ("embedding", 50, 'category == "x"')
    assert (reqs[1].anns_field, reqs[1].limit, reqs[1].data) == ("sparse", 50, ["猫砂盆 清理"])
    assert isinstance(kw["ranker"], RRFRanker)
    assert kw["ranker"].dict()["params"] == {"k": 60}  # 实测:RRFRanker 无公开 .k 属性,文档口径=dict() 序列化(Context7 复核)
    assert kw["collection_name"] == "knowledge" and kw["limit"] == 10


def test_empty_results_map_to_empty_lists():
    """三检索出口空结果契约:res=[[]](零命中)与 res=[](空回包)都必须归一为 [],不炸索引。"""
    from app.rag.milvus_store import bm25_search, hybrid_search, search_vectors

    for kw in ({}, {"search_out": [], "hybrid_out": []}):
        c = FakeClient(**kw)
        assert search_vectors(c, "knowledge", [0.0], 3) == []
        assert bm25_search(c, "knowledge", "猫砂盆", 3) == []
        assert hybrid_search(c, "knowledge", [0.0], "猫砂盆", limit=3, recall_k=5, rrf_k=60) == []


def test_health_drop_flush_count_ids():
    from app.rag import milvus_store as ms

    assert ms.health_ok(FakeClient()) is True
    assert ms.health_ok(FakeClient(fail=True)) is False
    c = FakeClient(has=True)
    ms.drop_collection(c, "knowledge")
    assert ("drop", "knowledge") in c.calls
    ms.drop_collection(FakeClient(has=False), "knowledge")  # 不存在不报错
    ms.flush(FakeClient(), "knowledge")
    assert ms.count_rows(FakeClient(query_out=[{"count(*)": 5}]), "knowledge") == 5
    assert ms.all_ids(FakeClient(query_out=[{"chunk_id": 1}, {"chunk_id": 3}]), "knowledge") == [1, 3]
