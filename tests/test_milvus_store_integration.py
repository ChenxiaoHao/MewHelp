import pytest

from app.core.config import get_settings
from app.rag import milvus_store

COLL = "knowledge_it"  # 独立测试集合,不碰演示用 knowledge


@pytest.mark.integration
def test_roundtrip_direction_count_idempotent():
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    assert milvus_store.health_ok(client)
    milvus_store.drop_collection(client, COLL)
    try:
        milvus_store.ensure_collection(client, COLL, 4)
        n = milvus_store.upsert_vectors(
            client, COLL, [(1, [1.0, 0.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0, 0.0]), (3, [0.7, 0.7, 0.0, 0.0])]
        )
        assert n == 3
        milvus_store.flush(client, COLL)  # 核对点①: upsert是缓冲的, 不flush则search/count看不到
        hits = milvus_store.search_vectors(client, COLL, [1.0, 0.0, 0.0, 0.0], 3)
        assert [cid for cid, _ in hits] == [1, 3, 2], "COSINE 方向/排序与门面假设不符 → 改门面"
        assert hits[0][1] > hits[1][1] > hits[2][1] and 0.0 <= hits[2][1] < 1.01
        milvus_store.upsert_vectors(client, COLL, [(1, [1.0, 0.0, 0.0, 0.0])])  # 同 pk 再 upsert
        assert milvus_store.count_rows(client, COLL) == 3, "upsert 幂等假设被破坏(变 4 条即 insert 语义)"
        assert milvus_store.all_ids(client, COLL) == [1, 2, 3]
    finally:
        milvus_store.drop_collection(client, COLL)
