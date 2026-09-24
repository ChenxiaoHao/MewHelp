"""retriever v2 单测:假腿/假回查/假重排,断言策略分派、阈值作用域、闸1、降级、排布纯函数。"""

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.rag import retriever as r
from app.rag.query_understanding import UnderstandResult

ST = Settings(_env_file=None, openai_base_url="http://f/v1", openai_api_key="f", model_name="f")
U = UnderstandResult(standard_query="标准问", synonyms=["同1", "同2"])


def _row(i):
    return SimpleNamespace(id=i, category=f"c{i}", questions=f"q{i}\nq{i}b", answer=f"a{i}",
                           section_path=f"手册 > 节{i}", content_type="policy")


@pytest.fixture
def legs(monkeypatch):
    calls = {}
    monkeypatch.setattr(r, "_get_client", lambda st: object())

    async def fake_embed(text, st):
        calls["embed"] = text
        return [0.1, 0.2]

    async def fake_fetch(ids):
        return [_row(i) for i in ids]

    monkeypatch.setattr(r, "_embed", fake_embed)
    monkeypatch.setattr(r, "_fetch_rows", fake_fetch)

    def dense(c, st, vec, expr):
        calls["dense"] = (vec, expr)
        return [(1, 0.9), (2, 0.2)]

    def bm25(c, st, text, expr):
        calls["bm25"] = (text, expr)
        return [(2, 5.0), (3, 1.0)]

    def hyb(c, st, vec, text, expr):
        calls["hyb"] = (vec, text, expr)
        return [(3, 0.03), (1, 0.02), (2, 0.01)]

    monkeypatch.setattr(r, "_dense_sync", dense)
    monkeypatch.setattr(r, "_bm25_sync", bm25)
    monkeypatch.setattr(r, "_hybrid_sync", hyb)

    async def no_rerank(query, texts, st):
        calls["rerank_texts"] = texts
        return None

    monkeypatch.setattr(r.reranker, "rerank", no_rerank)
    return calls


async def test_dense_arm_uses_threshold_only_here(legs):
    res = await r.retrieve("q", strategy="dense", settings=ST, understood=U)
    assert legs["embed"] == "标准问"
    assert [c.chunk_id for c in res.chunks] == [1]  # 0.2 < 0.3 被纯 dense 腿阈值砍掉
    assert not res.refused


async def test_bm25_arm_sends_built_text_no_threshold(legs):
    res = await r.retrieve("q", strategy="bm25", settings=ST, understood=U)
    assert legs["bm25"][0] == "标准问 同1 同2"
    assert [c.chunk_id for c in res.chunks] == [2, 3]  # 低分不砍:bm25 分数量级与 COSINE 无关


async def test_hybrid_arm_legs_and_no_refusal(legs):
    res = await r.retrieve("q", strategy="hybrid", settings=ST, understood=U)
    assert legs["hyb"] == ([0.1, 0.2], "标准问 同1 同2", None)
    assert [c.chunk_id for c in res.chunks] == [3, 1, 2]
    assert not res.refused


async def test_hybrid_rerank_degrades_to_rrf_when_rerank_none(legs):
    res = await r.retrieve("q", settings=ST, understood=U)  # legs 里 rerank → None
    assert [c.chunk_id for c in res.chunks] == [3, 1, 2]
    assert not res.refused and res.note == ""
    assert legs["rerank_texts"][0] == "c3\nq3\nq3b\na3"  # 重排候选文本 = 三格拼接(vector_text 同源)


async def test_hybrid_rerank_success_orders_by_rerank(legs, monkeypatch):
    async def ok_rerank(query, texts, st):
        return [(2, 0.95), (0, 0.80)]  # cands=[3,1,2] → id2 最相关、id3 次之

    monkeypatch.setattr(r.reranker, "rerank", ok_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert [(c.chunk_id, round(c.score, 2)) for c in res.chunks] == [(2, 0.95), (3, 0.8)]
    assert not res.refused


async def test_gate1_low_top1_refuses(legs, monkeypatch):
    async def weak_rerank(query, texts, st):
        return [(1, 0.05)]

    monkeypatch.setattr(r.reranker, "rerank", weak_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert res.refused and res.chunks == [] and "置信度不足" in res.note


async def test_gate1_zero_hits_refuses(legs, monkeypatch):
    monkeypatch.setattr(r, "_hybrid_sync", lambda c, st, vec, text, expr: [])
    res = await r.retrieve("q", settings=ST, understood=U)
    assert res.refused and "无命中" in res.note


async def test_gate1_boundary_top1_equals_threshold_not_refused(legs, monkeypatch):
    """I1(a):闸1 是严格 `<`——top1 恰等于阈值不得拒(边界值放行)。"""
    thr = ST.retrieval_low_conf_threshold

    async def edge_rerank(query, texts, st):
        return [(1, thr), (0, thr + 0.1)]  # cands=[3,1,2]:id1 恰在阈值、id3 高于

    monkeypatch.setattr(r.reranker, "rerank", edge_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert not res.refused and res.note == ""
    assert [c.chunk_id for c in res.chunks] == [1, 3]


async def test_rerank_indices_all_out_of_range_refuses_empty(legs, monkeypatch):
    """I1(b):重排返回的下标全越界 → 过滤后空 → 走「知识库无命中」空分支(refused)。"""
    async def oob_rerank(query, texts, st):
        return [(99, 0.9), (-1, 0.8), (3, 0.7)]  # cands 只有 0..2

    monkeypatch.setattr(r.reranker, "rerank", oob_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert res.refused and res.chunks == [] and "无命中" in res.note


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
async def test_zero_hits_on_non_rerank_arms_never_refused(legs, monkeypatch, strategy):
    """I1(c):refused 旗标是 hybrid_rerank 臂专属——其余三臂零命中=普通空结果。"""
    monkeypatch.setattr(r, "_dense_sync", lambda c, st, vec, expr: [])
    monkeypatch.setattr(r, "_bm25_sync", lambda c, st, text, expr: [])
    monkeypatch.setattr(r, "_hybrid_sync", lambda c, st, vec, text, expr: [])
    res = await r.retrieve("q", strategy=strategy, settings=ST, understood=U)
    assert res.chunks == [] and not res.refused and res.note == ""


# ---- T6 评审 Important 折入(I1 边界 + I2 接线钉死,dev-notes ②) ----

# 假 store 门面(比 legs 低一层:让真实 _dense_sync/_bm25_sync/_hybrid_sync 跑起来,
# 钉死 st.hybrid_recall_k/rrf_k 与 expr 确实传进 milvus_store 调用,而非只进 sync 壳)。
@pytest.fixture
def store(monkeypatch):
    calls = {}
    monkeypatch.setattr(r, "_get_client", lambda st: object())

    async def fake_embed(text, st):
        return [0.1]

    async def fake_fetch(ids):
        return [_row(i) for i in ids]

    monkeypatch.setattr(r, "_embed", fake_embed)
    monkeypatch.setattr(r, "_fetch_rows", fake_fetch)

    def dense(c, name, vec, top_k, expr=None):
        calls["dense"] = (top_k, expr)
        return [(1, 0.9)]

    def bm25(c, name, text, top_k, expr=None):
        calls["bm25"] = (top_k, expr)
        return [(1, 5.0)]

    def hyb(c, name, vec, text, *, limit, recall_k, rrf_k, expr=None):
        calls["hyb"] = (limit, recall_k, rrf_k, expr)
        return [(1, 0.02)]

    monkeypatch.setattr(r.milvus_store, "search_vectors", dense)
    monkeypatch.setattr(r.milvus_store, "bm25_search", bm25)
    monkeypatch.setattr(r.milvus_store, "hybrid_search", hyb)
    return calls


# 非默认值证明「透传」而非硬编码巧合(50/60 恰是默认值,用 7/9 才能钉死)
ST_KY = Settings(_env_file=None, openai_base_url="http://f/v1", openai_api_key="f",
                 model_name="f", hybrid_recall_k=7, rrf_k=9)


async def test_recall_k_and_rrf_k_passthrough(store):
    await r.retrieve("q", strategy="dense", settings=ST_KY, understood=U)
    assert store["dense"] == (7, None)  # dense 腿 limit=st.hybrid_recall_k
    await r.retrieve("q", strategy="bm25", settings=ST_KY, understood=U)
    assert store["bm25"] == (7, None)  # bm25 腿同上
    await r.retrieve("q", strategy="hybrid", settings=ST_KY, understood=U)
    assert store["hyb"] == (7, 7, 9, None)  # limit/recall_k=hybrid_recall_k、rrf_k 透传


async def test_category_expr_flows_into_all_legs(store):
    """改名不虚传:dense/bm25/hybrid 三条腿的真实 store 调用都收到 expr
    (hybrid_rerank 与 hybrid 共用 _hybrid_sync,同一路径)。"""
    await r.retrieve("q", strategy="hybrid", category="退货政策", settings=ST, understood=U)
    assert store["hyb"][3] == 'category == "退货政策"'
    await r.retrieve("q", strategy="dense", category="退货政策", settings=ST, understood=U)
    assert store["dense"][1] == 'category == "退货政策"'
    await r.retrieve("q", strategy="bm25", category="退货政策", settings=ST, understood=U)
    assert store["bm25"][1] == 'category == "退货政策"'


# T6 实测修正(verbatim 的 bad 列表含 "drop table",与同 brief 的 spec §4.2 逐字白名单
# `^[\w一-鿿 >()×/,-]+$` 互斥:纯字母+空格全在许可集内,任何合规实现都不可能对同值既放行
# "商品与购物 FAQ"(合法品类含空格)又拒 "drop table"——词组黑名单非白名单语义)。
# 真实注入面(引号破壳/分号/全角括号/空值)4 例照旧必拒,强度不降;另反向钉死该词组只能
# 成为无害等值过滤 expr。
@pytest.mark.parametrize("bad", ['x" or 1=1', "a;b", "括号（）测", ""])
def test_build_category_expr_whitelist(bad):
    assert r.build_category_expr(None) is None
    assert r.build_category_expr("商品参数") == 'category == "商品参数"'
    assert r.build_category_expr("drop table") == 'category == "drop table"'
    with pytest.raises(ValueError):
        r.build_category_expr(bad)


def test_head_tail_indices_pure():
    assert r.head_tail_indices(10) == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert r.head_tail_indices(0) == [] and r.head_tail_indices(1) == [1]
    assert r.head_tail_indices(4) == [1, 3, 4, 2]
    assert r.apply_head_tail(list("abcdef")) == ["a", "c", "e", "f", "d", "b"]
    assert sorted(map(str, r.apply_head_tail([i for i in range(1, 11)]))) == sorted(map(str, range(1, 11)))


async def test_unknown_strategy_raises(legs):
    with pytest.raises(ValueError):
        await r.retrieve("q", strategy="magic", settings=ST, understood=U)
