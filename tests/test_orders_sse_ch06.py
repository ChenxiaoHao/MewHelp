"""ch06 Task 5: SSE `orders` 帧契约(P7)——帧序红线:末 token 后、done 前。

适配层直测 stream_graph_turn(匿名线程,无跨测状态);帧序与 token/done/error
逐字符红线由 routes 既有测试继续守,这里只钉 orders 新帧的位置与形状。
"""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from app.schemas.chat import ChatMessage, SuggestionsEvent
from app.workflows.graph import stream_graph_turn
from app.workflows.state import REFUND_APPLY


@pytest.fixture
def fake_settings():
    return SimpleNamespace(tool_timeout_seconds=5.0, tool_max_retries=0,
                           retrieval_low_conf_threshold=0.161,
                           react_max_iterations=6, react_token_budget=8000,
                           history_token_budget=4000, demo_user_id="demo_user",
                           rerank_top_n=10, intent_small_model="",
                           intent_confidence_threshold=0.75,
                           openai_base_url="http://x/v1", openai_api_key="k",
                           model_name="m", temperature=0.7)


class ScriptModel:
    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, msgs):
        return self.script.pop(0)


async def test_orders_frame_after_last_token(fake_settings):
    model = ScriptModel([AIMessage(
        content='{"intent":"退款退货","confidence":0.95}')])
    frames = [ev async for ev in stream_graph_turn(
        [ChatMessage(role="user", content="我要退款")], fake_settings, model)]
    kinds = [k for k, _ in frames]
    assert "orders" in kinds
    assert kinds.index("orders") == max(
        i for i, k in enumerate(kinds) if k == "token") + 1   # 紧随末 token
    items = next(p for k, p in frames if k == "orders")["items"]
    assert [c["order_id"] for c in items] == ["1001", "1002", "1003"]
    assert all(isinstance(s, str) and "×" in s for c in items for s in c["items"])


async def test_no_orders_frame_when_payload_empty(fake_settings):
    model = ScriptModel([AIMessage(content='{"intent":"投诉","confidence":0.95}')])
    frames = [ev async for ev in stream_graph_turn(
        [ChatMessage(role="user", content="太差了我要投诉")], fake_settings, model)]
    assert all(k != "orders" for k, _ in frames)


def test_refund_apply_passes_suggestion_schema():
    ev = SuggestionsEvent(items=[REFUND_APPLY])   # 三枚举扩容后 refund_apply 可过
    assert ev.items[0].action == "refund_apply"
