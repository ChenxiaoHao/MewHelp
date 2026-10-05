"""池写入永不阻断拒答:成功转发 / 引擎未初始化 WARN 吞掉(spec §8)。"""

from app.services import refusals


async def test_pool_forwards_to_crud(monkeypatch):
    seen = {}

    class FakeSession:
        def __init__(self):
            self.committed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            self.committed = True

    async def fake_add(session, **kw):
        seen.update(kw)
        assert session.committed is False  # crud 里才 commit,这里传的是 session
    monkeypatch.setattr(refusals.crud, "add_low_confidence_question", fake_add)
    monkeypatch.setattr(refusals, "get_session_factory", lambda: (lambda: FakeSession()))
    await refusals.pool_low_confidence(5, "退货运费谁出", "retrieval_low_conf", "note-x")
    assert seen == {"conversation_id": 5, "raw_question": "退货运费谁出",
                    "source": "retrieval_low_conf", "reason": "note-x",
                    "retrieved_chunks": None}  # ch09 T4:快照参扩形,旧调用=None
    seen.clear()
    snap = [{"chunk_id": 1, "score": 0.05, "text": "t"}]
    await refusals.pool_low_confidence(5, "q2", "ch05_gate", "r2", retrieved_chunks=snap)
    assert seen["retrieved_chunks"] == snap


async def test_pool_swallows_engine_uninitialized():
    # 单测环境引擎没 init → get_session_factory 抛 RuntimeError → 必须被吞
    await refusals.pool_low_confidence(None, "问题", "self_check", "r")  # 不抛即过


def test_pool_source_enum_covers_graph_gate_sources():
    """闸池写入 source=ch05_gate/ch06_refund_gate;lcq_source ENUM 少值=生产静默丢行
    (ch06 T8 e2e 暴露的 ch05 遗留:MySQL 1265 Data truncated 被 pool 吞成 WARN)。"""
    from app.db.models import LowConfidenceQuestion
    vals = set(LowConfidenceQuestion.source.type.enums)
    assert {"ch05_gate", "ch06_refund_gate"} <= vals, vals
