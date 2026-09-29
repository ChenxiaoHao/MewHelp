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
    monkeypatch.setattr(summarizer, "_latest_segment", lambda s, cid: _ret(None))

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
    monkeypatch.setattr(summarizer, "_latest_segment", lambda s, cid: _ret(None))

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
    monkeypatch.setattr(summarizer, "_latest_segment", lambda s, cid: _ret(None))


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


async def test_segment_already_in_table_repairs_projection_without_model(monkeypatch):
    """M2-I1:append 落段成功但投影写挂 → 重触发时同区间双段(投影重拼出重复内容)。
    表内最新段 upto 已追平本批 layer1_from ⇒ 本批已压过:不调模型、不再 append,
    直接用全段重修补投影(裂口自愈)。"""
    calls = []
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret(("旧投影", 0, 5)))
    monkeypatch.setattr(crud, "list_messages_after", lambda *a, **k: _ret(_batch_rows()))
    monkeypatch.setattr(crud, "append_summary_segment",
                        lambda *a, **k: calls.append("append") or _ret(9))

    async def fake_proj(session, cid, *, summary, upto_msg_id):
        calls.append(("proj", summary, upto_msg_id))
    monkeypatch.setattr(crud, "set_summary_projection", fake_proj)
    monkeypatch.setattr(summarizer, "_load_segments", lambda s, cid: _ret(["段一", "段二"]))
    monkeypatch.setattr(summarizer, "_latest_segment",
                        lambda s, cid: _ret(SimpleNamespace(upto_msg_id=5)), raising=False)

    model = FakeModel()
    await summarizer.run_summary(_sf, 42, SETTINGS, model)
    assert model.msgs_seen == []                                   # 同批不重压
    assert "append" not in calls                                   # 同区间不双段
    assert ("proj", "段一\n段二", 5) in calls                       # 投影补到段表真相


async def test_truncated_batch_never_advances_boundary(monkeypatch, caplog):
    """M2-I2:list limit 截断 → batch 尾部够不到 layer1_from 时也照压照 append,
    被截消息永久失联。取数必须显式大 limit(读侧同源 10000),且推进边界前校验
    覆盖完整;不完整 = 一条 WARN,不动模型不动边界(下轮重触发)。"""
    seen, calls = {}, []
    monkeypatch.setattr(crud, "get_conv_ctx", lambda *a: _ret((None, 0, 5)))

    def fake_list(s, cid, after, limit=None):
        seen["limit"] = limit
        return _ret(_batch_rows()[:3])          # 尾部只到 id=3 < layer1_from=5(模拟截断)
    monkeypatch.setattr(crud, "list_messages_after", fake_list)
    monkeypatch.setattr(crud, "append_summary_segment",
                        lambda *a, **k: calls.append("append") or _ret(1))

    model = FakeModel()
    with caplog.at_level(logging.INFO):
        await summarizer.run_summary(_sf, 42, SETTINGS, model)
    assert seen["limit"] == 10000                                 # 不再吃 crud 默认 500
    assert model.msgs_seen == [] and "append" not in calls        # 截断批不压不落
    assert any(r.levelno == logging.WARNING and "truncated" in r.getMessage()
               for r in caplog.records)


def _ret(value):
    """crud 桩同步函数返回 awaitable(被测代码 await 它)。"""
    async def _v():
        return value
    return _v()
