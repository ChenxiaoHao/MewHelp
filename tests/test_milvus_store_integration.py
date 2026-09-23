"""集合 v2 集成冒烟(spec §4.2-6 硬断言):chinese analyzer 裸 BM25 命中型号/中文词、
标量过滤、hybrid_search 形状、flush 语义——实测形状记 dev-notes,核对点①②在此销账。"""

import pytest

from app.core.config import get_settings
from app.rag import milvus_store

pytestmark = pytest.mark.integration

COLL = "knowledge_it"  # 独立测试集合,不碰演示用 knowledge
DIM = 4

ROWS = [
    {"chunk_id": 1, "text": "智能猫砂盆 Pro(MH-LP100)支持 App 远程监控,废砂盒建议 5 至 7 天清理一次",
     "embedding": [1.0, 0.0, 0.0, 0.0], "category": "商品参数", "content_type": "policy"},
    {"chunk_id": 2, "text": "冻干猫粮喂食量:幼猫每天 3-4 次,按体重计算",
     "embedding": [0.0, 1.0, 0.0, 0.0], "category": "商品参数", "content_type": "policy"},
    {"chunk_id": 3, "text": "退货政策:签收后 7 天内无理由退货,不影响二次销售",
     "embedding": [0.9, 0.1, 0.0, 0.0], "category": "退货政策", "content_type": "policy"},
]


def test_collection_v2_bm25_hybrid_roundtrip():
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    assert milvus_store.health_ok(client)
    milvus_store.drop_collection(client, COLL)
    try:
        milvus_store.ensure_collection(client, COLL, DIM)
        assert milvus_store.upsert_rows(client, COLL, ROWS) == 3
        milvus_store.flush(client, COLL)
        # 硬断言①:型号 token 裸 BM25 命中 = chinese analyzer 对拉丁型号可用(验收2 地基)
        hits = milvus_store.bm25_search(client, COLL, "MH-LP100", 3)
        assert hits and hits[0][0] == 1
        # 硬断言②:中文词命中(分词生效)
        zh = milvus_store.bm25_search(client, COLL, "废砂盒", 3)
        assert zh and zh[0][0] == 1
        dense = milvus_store.search_vectors(client, COLL, [1.0, 0.0, 0.0, 0.0], 3)
        assert dense[0][0] == 1
        # 标量过滤(需求3):品类 expr 只放行该品类块。
        # 计划 fixture 缺陷修正(dev-notes④,断言语义不变):原「猫→只 {3}」物理不成立——
        # row3 文本无「猫」token,BM25 零重叠永不命中,任何门面都过不了;改用 1/3 行共有的「7 天」,
        # 无过滤先证 {1,3} 都在、过滤后只放行 3,才真正测到「排除其它品类」语义。
        assert {cid for cid, _ in milvus_store.bm25_search(client, COLL, "7 天", 3)} == {1, 3}
        filt = milvus_store.bm25_search(client, COLL, "7 天", 3, expr='category == "退货政策"')
        assert {cid for cid, _ in filt} == {3}
        # hybrid:dense 腿偏 2、BM25 腿偏 3 → 融合含两者且形状 [(int, float)]
        hyb = milvus_store.hybrid_search(client, COLL, [0.0, 1.0, 0.0, 0.0], "7 天无理由退货",
                                         limit=3, recall_k=3, rrf_k=60)
        assert hyb and all(isinstance(cid, int) and isinstance(s, float) for cid, s in hyb)
        # 计划 fixture 缺陷修正(dev-notes④):原「== {2,3}」不成立——recall_k=3 在 3 行小库=全库进
        # 融合,chunk1 双腿中游(dense 第3+BM25 第2)RRF 反超单腿冠军 chunk2。实测顺序 [3,1,2] 与
        # RRFRanker(k=60) 公式 1/(k+rank) 逐项吻合(0.0325/0.0320/0.0164),按「融合含两者」本意改 ⊇。
        assert {2, 3} <= {cid for cid, _ in hyb}
        assert [cid for cid, _ in hyb] == [3, 1, 2]
        # 同 PK 覆写幂等 + flush 语义沿用
        assert milvus_store.upsert_rows(client, COLL, [dict(ROWS[0])]) == 1
        milvus_store.flush(client, COLL)
        assert milvus_store.count_rows(client, COLL) == 3
    finally:
        milvus_store.drop_collection(client, COLL)
