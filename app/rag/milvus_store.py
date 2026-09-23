"""Milvus 同步门面 v2(spec §3.1)。ch03「只存 id+向量」已推翻:原生 BM25 必须存 text(§0-7),
sparse 由服务端 BM25 Function 生成、客户端不写;原文权威仍是 MySQL,回查/孤儿过滤语义不变。

ch03 实测语义沿用:PK 回显键名=主键字段名 `chunk_id`;COSINE distance 越大越相似;写完必 flush。
ch04 核对点①②(建集确切写法/VARCHAR 字节上限/中文分词)由集成冒烟现场销账。
"""

from __future__ import annotations

TEXT_MAX_LENGTH = 8192  # 核对点②:三格拼接 chunk ≈ ≤2.4KB UTF-8,3 倍冗余;超限实测后再调


def get_client(uri: str, timeout: float = 10.0):
    from pymilvus import MilvusClient  # 延迟 import:纯为启动快,本项目已装

    return MilvusClient(uri=uri, timeout=timeout)


def health_ok(client) -> bool:
    try:
        client.list_collections()
        return True
    except Exception:  # noqa: BLE001 —— 探活语义:任何异常都算不健康
        return False


def ensure_collection(client, name: str, dim: int) -> None:
    """集合 v2:chunk_id PK + text(chinese analyzer)+ sparse(BM25 函数输出)
    + embedding(FLOAT_VECTOR)+ category/content_type(标量过滤)。建集 schema 不可改,迁移=drop 重建。"""
    if client.has_collection(name):
        return
    from pymilvus import DataType, Function, FunctionType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("category", DataType.VARCHAR, max_length=765)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_function(Function(
        name="bm25_fn", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["sparse"],
    ))
    index_params = client.prepare_index_params()
    index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def drop_collection(client, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)


def flush(client, name: str) -> None:
    """核对点(ch03 实测):upsert 是缓冲的,search/count 前必须 flush。"""
    client.flush(name)


def upsert_rows(client, name: str, rows: list[dict]) -> int:
    """rows 每项 {chunk_id,text,embedding,category,content_type};sparse 缺席(函数列服务端生成)。
    按主键幂等 upsert(§5 语义核心):同 chunk_id 再写 = 覆盖。"""
    if not rows:
        return 0
    res = client.upsert(collection_name=name, data=rows)
    return getattr(res, "upsert_count", None) or len(rows)


def search_vectors(client, name: str, vector: list[float], top_k: int,
                   expr: str | None = None) -> list[tuple[int, float]]:
    """dense 腿。COSINE 相似度(越大越相似)。expr=标量过滤(需求3)。
    实测(集成冒烟):v2 集合双向量列,dense search 必须显式 anns_field,否则服务端 1100。"""
    kw = dict(collection_name=name, data=[vector], anns_field="embedding", limit=top_k)
    if expr:
        kw["filter"] = expr
    res = client.search(**kw)
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def bm25_search(client, name: str, text: str, top_k: int,
                expr: str | None = None) -> list[tuple[int, float]]:
    """BM25 腿:查询文本原样进 data(服务端 analyzer 分词),distance=BM25 分数(越大越相关)。"""
    kw = dict(collection_name=name, data=[text], anns_field="sparse", limit=top_k)
    if expr:
        kw["filter"] = expr
    res = client.search(**kw)
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def hybrid_search(client, name: str, vector: list[float], bm25_text: str, *,
                  limit: int, recall_k: int, rrf_k: int,
                  expr: str | None = None) -> list[tuple[int, float]]:
    """§4.2-1:双腿各 Top-recall_k → RRFRanker(k=rrf_k) 融合,返回 [(chunk_id, rrf_score)]。"""
    from pymilvus import AnnSearchRequest, RRFRanker

    reqs = [
        AnnSearchRequest(data=[vector], anns_field="embedding",
                         param={"metric_type": "COSINE"}, limit=recall_k, expr=expr),
        AnnSearchRequest(data=[bm25_text], anns_field="sparse",
                         param={}, limit=recall_k, expr=expr),
    ]
    res = client.hybrid_search(collection_name=name, reqs=reqs, ranker=RRFRanker(k=rrf_k),
                               limit=limit, output_fields=["chunk_id"])
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def count_rows(client, name: str) -> int:
    res = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    return int(res[0]["count(*)"]) if res else 0


def all_ids(client, name: str) -> list[int]:
    """demo 规模一次性拉全 id,供 --check 对账差集。"""
    res = client.query(collection_name=name, filter="chunk_id > 0", output_fields=["chunk_id"])
    return sorted(r["chunk_id"] for r in res)
