"""ch09 T4 evidence_confidence 闸核:三信号纯函数 + rule/sum 两形态 + 闸节点接线。

spec 置信度闸升级节钉死:位置与行为不变(拦下→REFUSAL+落池+fail);
判定 f(top1,n_eff,gap) 由 300 题校准定形定值(报告 evals/reports/);
RRF 降级带旁路语义原样(RRF_DEGRADED_MAX,过闸不落池);
缺 evidence_conf_* 键的旧夹具 → 回落 ch04 闸1 阈值(rule 形=0.161 等效零漂移)。
"""

from types import SimpleNamespace

import pytest
from app.rag import confidence
from app.services import refusals
from app.workflows.nodes import RRF_DEGRADED_MAX, make_confidence_gate_node

_FULL_CFG = dict(
    evidence_conf_form="rule", evidence_conf_floor_eff=0.161,
    evidence_conf_threshold=0.161, evidence_conf_n_eff_min=1,
    evidence_conf_gap_min=0.0, evidence_conf_w_top1=1.0,
    evidence_conf_w_n=0.0, evidence_conf_w_gap=0.0,
)


def _cfg(**over):
    return SimpleNamespace(**{**_FULL_CFG, **over})


def _ev(*scores, text="政策片段"):
    return [{"chunk_id": i + 1, "score": s, "text": text}
            for i, s in enumerate(scores)]


# ---- 三信号 ----

def test_signals_empty_evidence():
    assert confidence.signals_from_evidence([], floor_eff=0.161) == (0.0, 0, 0.0)


def test_signals_single_evidence_gap_is_top1():
    assert confidence.signals_from_evidence(_ev(0.9), floor_eff=0.161) == (0.9, 1, 0.9)


def test_signals_counts_n_eff_and_top1_minus_top2():
    top1, n, gap = confidence.signals_from_evidence(
        _ev(0.5, 0.2, 0.9, 0.1), floor_eff=0.161)
    assert top1 == 0.9 and n == 3
    assert gap == pytest.approx(0.4)  # 0.9-0.5


def test_signals_accepts_scored_row_objects():
    from types import SimpleNamespace as NS
    rows = [NS(chunk_id=1, score=0.7), NS(chunk_id=2, score=0.3)]
    top1, n, gap = confidence.signals_from_evidence(rows, floor_eff=0.4)
    assert top1 == 0.7 and n == 1 and gap == pytest.approx(0.4)


# ---- 判定两形态 ----

def test_evaluate_rule_placeholder_equiv_to_top1_only():
    # 未校准占位(rule, θ=0.161, n≥1, gap≥0)= 旧 ch04 语义逐例等值
    assert confidence.evaluate(_ev(0.05, 0.87), _cfg()).ok is True
    assert confidence.evaluate(_ev(0.05, 0.1), _cfg()).ok is False
    assert confidence.evaluate([], _cfg()).ok is False, "空证据必拦(spec 闸1 交互不变)"


def test_evaluate_detail_carries_three_signals():
    v = confidence.evaluate(_ev(0.05), _cfg())
    assert not v.ok
    assert "conf=0.0500<0.1610" in v.detail
    assert "top1=0.0500" in v.detail and "n=0" in v.detail and "gap=0.0500" in v.detail


def test_evaluate_rule_uses_n_eff_and_gap_knobs():
    cfg = _cfg(evidence_conf_n_eff_min=2, evidence_conf_gap_min=0.3)
    v = confidence.evaluate(_ev(0.9, 0.7, 0.01), cfg)  # n=2 ✓ gap=0.2 ✗
    assert v.ok is False and "gap=0.2000" in v.detail
    assert confidence.evaluate(_ev(0.9, 0.6, 0.5, 0.01), cfg).ok is True  # n=3 gap=0.3


def test_evaluate_sum_form_weighted():
    cfg = _cfg(evidence_conf_form="sum", evidence_conf_threshold=0.8,
               evidence_conf_w_top1=0.5, evidence_conf_w_n=0.3,
               evidence_conf_w_gap=0.2)
    # 0.5*0.9 + 0.3*(1/3) + 0.2*(0.1/0.5) = 0.45+0.1+0.04 = 0.59 < 0.8
    assert confidence.evaluate(_ev(0.9, 0.8), cfg).ok is False
    # 0.5*0.99 + 0.3*(3/3) + 0.2*(0.5/0.5) = 0.495+0.3+0.2 = 0.995 >= 0.8
    assert confidence.evaluate(_ev(0.99, 0.49, 0.3, 0.2), cfg).ok is True


def test_evaluate_legacy_cfg_without_keys_falls_back():
    # 旧夹具只有 retrieval_low_conf_threshold:零漂移回落 rule 形
    legacy = SimpleNamespace(retrieval_low_conf_threshold=0.161)
    assert confidence.evaluate(_ev(0.2), legacy).ok is True
    assert confidence.evaluate(_ev(0.1), legacy).ok is False
    bare = SimpleNamespace()
    assert confidence.evaluate(_ev(0.2), bare).ok is True  # 全缺键 → 默认 0.161


# ---- 闸节点接线(两实例共用同一判定核) ----

def _gate_settings(**over):
    base = dict(retrieval_low_conf_threshold=0.161,
                retrieval_snapshot_text_max=6)
    return SimpleNamespace(**{**base, **over})


async def _run_gate(node, evidence, monkeypatch, pool_target=None):
    calls = []

    async def fake_pool(cid, q, source, reason, retrieved_chunks=None):
        calls.append({"cid": cid, "q": q, "src": source, "reason": reason,
                      "snap": retrieved_chunks})
    monkeypatch.setattr(refusals, "pool_low_confidence", fake_pool)
    state = {"evidence": evidence, "resolved_query": "保修几年", "log": {"nodes": []}}
    out = await node(state, {"configurable": {"conversation_id": 9}})
    return out, calls


async def test_gate_blocks_refusal_pool_with_snapshot(monkeypatch):
    node = make_confidence_gate_node(_gate_settings())
    ev = _ev(0.05, text="一二三四五六七八九十")  # cap=6 → 截断
    out, calls = await _run_gate(node, ev, monkeypatch)
    assert out["gate_pass"] is False
    assert out["answer_text"] == refusals.REFUSAL_ANSWER
    assert out["suggestions"] == [{"action": "transfer_human", "label": "转人工"}]
    assert len(calls) == 1
    c = calls[0]
    assert c["src"] == "ch05_gate" and c["cid"] == 9
    assert "top1=0.0500" in c["reason"], "reason 须带三信号复盘面(spec)"
    assert c["snap"] == [{"chunk_id": 1, "score": 0.05, "text": "一二三四五六"}], \
        "落池带召回片段快照(T6 👎 回捞同款形制)"


async def test_gate_empty_evidence_pools_none_snapshot(monkeypatch):
    """M2-M-1(升级入批):空快照归 NULL——T5 裁决「空着与 [] 不两种表达」。

    闸1 清空的轮(缺无题主形态)落池快照若写 JSON [] ,T9 详情/T10 弹层须
    同时处理 []/NULL 两种「空」;与随行面 react/graph 两出口的 `or None` 同律。"""
    node = make_confidence_gate_node(_gate_settings())
    out, calls = await _run_gate(node, [], monkeypatch)
    assert out["gate_pass"] is False
    assert calls[0]["snap"] is None, "空证据落池 retrieved_chunks=NULL,不落 []"


async def test_gate_refund_instance_same_core(monkeypatch):
    node = make_confidence_gate_node(_gate_settings(), source="ch06_refund_gate",
                                     name="refund_gate")
    # 0.10:低于 θ=0.161 且高于降级带上限 0.04(0.02 会落旁路面)
    out, calls = await _run_gate(node, _ev(0.10), monkeypatch)
    assert out["gate_pass"] is False and calls[0]["src"] == "ch06_refund_gate"
    assert out["log"]["nodes"][-1] == "refund_gate"


async def test_gate_rrf_degraded_bypass_preserved(monkeypatch):
    """Review Focus 3:降级带旁路语义原样——RRF 分域内非空证据过闸不落池。"""
    node = make_confidence_gate_node(_gate_settings())
    out, calls = await _run_gate(node, _ev(RRF_DEGRADED_MAX), monkeypatch)
    assert out["gate_pass"] is True and out["log"]["gate_scale"] == "rrf_degraded"
    assert calls == [], "旁路不落池"
    # 降级带外(0.04<top1<θ)仍拦
    out2, calls2 = await _run_gate(
        make_confidence_gate_node(_gate_settings()),
        [{"chunk_id": 1, "score": 0.05, "text": "t"}], monkeypatch)
    assert out2["gate_pass"] is False and len(calls2) == 1


@pytest.mark.integration
def test_gate_live_chain_calibrated_verdicts():
    """T4 Step3 抽验:真实检索全链(rewrite+embed+Milvus+rerank)→ 同核判定。

    选题出自 300 题分数侧车的两端裕量(A36 top1≈0.999 必过 / D24
    top1≈0.0001 必拦),在线分数漂移 ±0.3 也翻不向;闸1 空证据拒回与
    节点闸拦下同一去向(REFUSAL+落池),这里只钉「活链判定方向」不重测落池。
    """
    import asyncio
    from app.core.config import get_settings
    from app.db.engine import dispose_engine, init_engine
    from app.rag import retriever

    st = get_settings()

    async def _chain(q):
        res = await retriever.retrieve(q, strategy="hybrid_rerank", settings=st)
        ev = [{"chunk_id": c.chunk_id, "score": c.score, "text": ""}
              for c in res.chunks]
        return res, confidence.evaluate(ev, st)

    async def _all():
        init_engine(st)  # 全链同一 loop:aiomysql 池跨 asyncio.run 复用会炸
        try:
            res_a, v_a = await _chain("免运费权益能不能抵掉偏远地区的附加运费")
            assert res_a.chunks, "答案题不该被闸1拒回(top1≈0.99)"
            assert v_a.ok, v_a.detail
            res_d, v_d = await _chain("你们跟宠物医院有合作吗,能约疫苗吗")
            assert not v_d.ok, "缺无题活链必须拦(空证据或低置信)"
        finally:
            await dispose_engine()

    asyncio.run(_all())
