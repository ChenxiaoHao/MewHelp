"""ch07 T6:后台摘要任务——非阻塞/防重入/失败只 WARN/段追加不回炉(spec「后台异步摘要」)。

FakeStore 不真连库:run_summary 的副作用全部钉在 crud 桩参数上(append 的
from/upto 边界推导、projection 重拼与追平);「不阻塞」用 gate 事件卡住任务,
schedule 同步返回 True 且 sleep(0) 后任务仍未完成来证。
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from app.context import summarizer
from app.db import crud

SETTINGS = SimpleNamespace(assistant_head_chars=60)


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _sf():
    return FakeSession()


def _row(rid, role, content=None, tool_calls=None, tool_call_id=None):
    return SimpleNamespace(id=rid, role=role, content=content,
                           tool_calls=tool_calls, tool_call_id=tool_call_id)


class FakeModel:
    def __init__(self, reply="用户咨询订单1001退货。", exc=None, gate=None):
        self.reply, self.exc, self.gate = reply, exc, gate
        self.msgs_seen = []

    async def ainvoke(self, msgs):
        if self.gate is not None:
            await self.gate.wait()
        if self.exc is not None:
            raise self.exc
        self.msgs_seen.append(msgs)
        return SimpleNamespace(content=self.reply)


@pytest.fixture(autouse=True)
def _clean_inflight():
    summarizer._IN_FLIGHT.clear()
    yield
    summarizer._IN_FLIGHT.clear()


def _batch_rows():
    return [_row(1, "user", "订单1001能退吗"),
            _row(2, "assistant", None, [{"id": "c1", "name": "query_order", "args": {}}]),
            _row(3, "tool", "订单1001已发货" * 40, "c1"),
            _row(4, "assistant", "已超七天无法退款"),
            _row(5, "user", "那转人工吧")]


async def test_success_appends_segment_and_rebuilds_projection(monkeypatch, caplog):
    seen = {}
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret((None, 0, 5)))
    monkeypatch.setattr(crud, "list_messages_after",
                        lambda s, cid, after, limit=500: _ret(_batch_rows()))

    async def fake_append(session, cid, *, from_msg_id, upto_msg_id, content):
        seen["append"] = (cid, from_msg_id, upto_msg_id, content)
        return 1
    async def fake_proj(session, cid, *, summary, upto_msg_id):
        seen["proj"] = (summary, upto_msg_id)
    async def fake_segs(session, cid):
        return ["早前一段", "用户咨询订单1001退货。"]
    monkeypatch.setattr(crud, "append_summary_segment", fake_append)
    monkeypatch.setattr(crud, "set_summary_projection", fake_proj)
    monkeypatch.setattr(summarizer, "_load_segments", fake_segs)

    model = FakeModel()
    with caplog.at_level(logging.INFO):
        await summarizer.run_summary(_sf, 42, SETTINGS, model)

    assert seen["append"] == (42, 1, 5, "用户咨询订单1001退货。")   # from=upto+1, upto=旧 layer1_from
    assert seen["proj"] == ("早前一段\n用户咨询订单1001退货。", 5)   # 全段按 seq 重拼+边界追平
    assert any("summary done 第1段 (0,5]" in r.getMessage() for r in caplog.records)
    human = model.msgs_seen[0][-1].content
    assert "订单1001能退吗" in human and "已折叠" in human          # 层2半压形态进 batch 位


async def test_model_raises_only_warns_anchor_untouched(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret((None, 0, 5)))
    monkeypatch.setattr(crud, "list_messages_after", lambda *a, **k: _ret(_batch_rows()))

    async def no_append(*a, **k):
        calls.append("append")
    monkeypatch.setattr(crud, "append_summary_segment", no_append)

    with caplog.at_level(logging.INFO):
        await summarizer.run_summary(_sf, 42, SETTINGS, FakeModel(exc=RuntimeError("boom")))
    assert not calls                                   # 失败不落段、不动锚
    assert any(r.levelno == logging.WARNING and "summary failed" in r.getMessage()
               for r in caplog.records)
    assert 42 not in summarizer._IN_FLIGHT             # finally 释放,下轮可重触发


def _patch_ctx_reaching_model(monkeypatch):
    """让任务能活着走到 model.ainvoke(前置 crud 全桩,落库桩 no-op)。"""
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret((None, 0, 5)))
    monkeypatch.setattr(crud, "list_messages_after", lambda *a, **k: _ret(_batch_rows()))
    async def noop(*a, **k):
        return 1
    monkeypatch.setattr(crud, "append_summary_segment", noop)
    monkeypatch.setattr(crud, "set_summary_projection", noop)
    async def segs(session, cid):
        return ["…"]
    monkeypatch.setattr(summarizer, "_load_segments", segs)


async def test_schedule_reentry_guard_skips_while_inflight(monkeypatch):
    _patch_ctx_reaching_model(monkeypatch)
    gate = asyncio.Event()
    model = FakeModel(gate=gate, reply="…")
    assert summarizer.schedule_summary(_sf, 42, SETTINGS, model) is True
    assert summarizer.schedule_summary(_sf, 42, SETTINGS, model) is False  # 防重入
    gate.set()
    await asyncio.sleep(0.02)
    assert 42 not in summarizer._IN_FLIGHT


async def test_schedule_returns_without_awaiting_task(monkeypatch):
    _patch_ctx_reaching_model(monkeypatch)
    gate = asyncio.Event()
    model = FakeModel(gate=gate)
    done_before = summarizer.schedule_summary(_sf, 42, SETTINGS, model)  # 同步返回
    await asyncio.sleep(0)                          # 任务刚起步、卡在 gate,绝不可能完成
    assert done_before is True and 42 in summarizer._IN_FLIGHT
    gate.set()
    await asyncio.sleep(0.02)
    assert 42 not in summarizer._IN_FLIGHT


async def test_empty_layer2_batch_skips_without_model_call(monkeypatch):
    """层2 无新批(layer1_from ≤ upto):直接回,模型零调用——防每轮空转排任务。"""
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret(("已有梗概", 5, 5)))
    monkeypatch.setattr(crud, "list_messages_after", lambda *a, **k: _ret(_batch_rows()))

    class SpyModel:
        n = 0

        async def ainvoke(self, msgs):
            SpyModel.n += 1
            return SimpleNamespace(content="x")

    await summarizer.run_summary(_sf, 42, SETTINGS, SpyModel())
    assert SpyModel.n == 0


def _ret(value):
    """crud 桩同步函数返回 awaitable(被测代码 await 它)。"""
    async def _v():
        return value
    return _v()
