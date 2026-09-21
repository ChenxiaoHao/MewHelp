"""Milvus 同步门面(spec §3.2)。集合极简化:只存 chunk_id + embedding 两列,
原文权威在 MySQL,在线检索命中 id 后回查——孤儿向量天然被回查过滤(§7)。

核对点①已销账(Task 2 冒烟实测 + 本文件集成往返双保险):
- 命中 dict 的 PK 回显键名 = 主键字段名 `chunk_id`(不是 `id`);COSINE distance 即相似度,越大越近
- 简化 API 写入是缓冲的:写完立刻 search/count 前必须 flush → 提供 flush(),调用方自行决定时机
方法签名与预核不符处只改本文件适配,门面契约(返回 pair 列表等)对上层不变。
"""

from __future__ import annotations


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
    if client.has_collection(name):
        return
    client.create_collection(
        collection_name=name,
        dimension=dim,
        primary_field_name="chunk_id",
        id_type="int",
        vector_field_name="embedding",
        metric_type="COSINE",
        auto_id=False,
    )


def drop_collection(client, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)


def flush(client, name: str) -> None:
    """核对点①: upsert/insert 后缓冲数据对 search/query 不可见,显式 flush 才定案。"""
    client.flush(name)


def upsert_vectors(client, name: str, pairs: list[tuple[int, list[float]]]) -> int:
    """按主键幂等 upsert(§5 语义核心):同 chunk_id 再写 = 覆盖,不产生重复。"""
    if not pairs:
        return 0
    data = [{"chunk_id": cid, "embedding": vec} for cid, vec in pairs]
    res = client.upsert(collection_name=name, data=data)
    return getattr(res, "upsert_count", None) or len(pairs)


def search_vectors(client, name: str, vector: list[float], top_k: int) -> list[tuple[int, float]]:
    """返回 [(chunk_id, score)],score 按 COSINE 相似度(越大越相似,集成测试断言方向)。"""
    res = client.search(collection_name=name, data=[vector], limit=top_k)
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]  # 实测:PK 回显键名=chunk_id


def count_rows(client, name: str) -> int:
    res = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    return int(res[0]["count(*)"]) if res else 0


def all_ids(client, name: str) -> list[int]:
    """demo 规模(数百 entity)一次性拉全 id,供 --check 对账差集。"""
    res = client.query(collection_name=name, filter="chunk_id > 0", output_fields=["chunk_id"])
    return sorted(r["chunk_id"] for r in res)
