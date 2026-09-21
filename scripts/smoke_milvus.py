"""ch03 Milvus 连通冒烟:health + 建删测试集合往返。用法 uv run python scripts/smoke_milvus.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目惯例同 evals/run_*.py:脚本目录外导入需补根路径

from app.core.config import get_settings

# milvus_store 在 Task 7 才存在——本任务刻意用 pymilvus 裸调用,不反向依赖门面:
# (TODO:Task 7 落地后把本脚本的健康探测改走 app.rag.milvus_store.health_ok)


def main() -> int:
    from pymilvus import MilvusClient

    st = get_settings()
    client = MilvusClient(uri=st.milvus_uri, timeout=10)
    assert client.list_collections() is not None
    tmp = "ch03_smoke_tmp"
    if client.has_collection(tmp):
        client.drop_collection(tmp)
    client.create_collection(
        collection_name=tmp,
        dimension=8,
        primary_field_name="chunk_id",
        id_type="int",
        vector_field_name="embedding",
        metric_type="COSINE",
        auto_id=False,
    )
    client.insert(collection_name=tmp, data=[{"chunk_id": 1, "embedding": [0.1] * 8}])
    # 核对点①实测:pymilvus 3.0 简化 API 里 insert/upsert 为缓冲写,不 flush 直接 search 返回空 data:[[]];
    # flush 后命中形态 [{'chunk_id': 1, 'distance': 1.0, 'entity': {}}] —— 主键回显键名是 PK 字段名而非 "id"。
    client.flush(tmp)
    res = client.search(collection_name=tmp, data=[[0.1] * 8], limit=1)
    assert res and res[0] and res[0][0]["chunk_id"] == 1, f"冒烟 search 未命中: {res}"
    print("milvus OK: server lists", client.list_collections(), "| smoke hit", res)
    client.drop_collection(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
