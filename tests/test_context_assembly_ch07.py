"""ch07 T4:五段装配序 + 保底 trim + model_ctx/history_ctx 日志 + FileHandler(需求 3/4/6)。

需求 3 逐条:system 恒定打头 → 层2 半压 → 层1 原文 → 当前句 → 早期梗概+证据
合成**一条**挂当前句**之后**(梗概不占 System,上游 system 上提会作废前缀缓存)。
Review Focus 4:极端超限走保底 trim,当前句绝不丢。
"""

import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context.layers import (
    build_history_view,
    build_model_context,
    degrade_if_needed,
)

DEMO = dict(model_context_window=18000, max_output_tokens=2000, max_user_input_tokens=2000,
            max_agent_steps=3, tool_result_max_tokens=1200, rerank_top_k=5,
            turns_to_keep=20, steady_tokens_per_turn=500, assistant_head_chars=60,
            summary_inject_tokens=707, safety_margin_tokens=1000, history_view_messages=6,
            chunk_size=500)


def _settings(**over):
    return SimpleNamespace(**{**DEMO, **over})


def _row(rid, role, content=None, tool_calls=None, tool_call_id=None):
    return SimpleNamespace(id=rid, role=role, content=content,
                           tool_calls=tool_calls, tool_call_id=tool_call_id)


class FakeStore:
    def __init__(self, rows, ctx):
        self.cid = 42
        self._rows = rows
        self._ctx = list(ctx)
        self.sets = []

    async def load_ctx(self):
        return tuple(self._ctx)

    async def fetch_all_rows(self):
        return list(self._rows)

    async def fetch_layer1(self):
        return [r for r in self._rows if r.id > self._ctx[2]]

    async def set_layer1_from(self, value):
        self.sets.append(value)
        self._ctx[2] = value


def _typical_rows():
    """upto=0,layer1_from=3:层2=id1-3(一轮含工具链),层1=id4-6(user/assistant/tool 全文)。"""
    return [
        _row(1, "user", "帮我查库存" + "汉" * 100),
        _row(2, "assistant", None, [{"id": "c1", "name": "query_stock", "args": {}}]),
        _row(3, "tool", "库存" * 300, "c1"),
        _row(4, "user", "那物流呢"),
        _row(5, "assistant", "已发货" + "细" * 300),   # 层1 原文:一个字不压
        _row(6, "assistant", None, [{"id": "c2", "name": "query_order", "args": {}}]),
        _row(7, "tool", "订单轨迹" * 50, "c2"),
    ]


async def test_five_segment_order_and_injection_placement():
    store = FakeStore(_typical_rows(), ("第一段梗概", 0, 3))
    cur = HumanMessage("我要退款")
    msgs = await build_model_context(
        store, evidence=[{"text": "7天无理由条款"}], order_data={"order_id": "1001"},
        current_human=cur, settings=_settings())
    assert isinstance(msgs[0], SystemMessage)
    assert not any(isinstance(m, SystemMessage) for m in msgs[1:])   # 段2/3 间无 System
    assert msgs[-1] is not cur
    cur_idx = next(i for i, m in enumerate(msgs) if m is cur)
    assert isinstance(msgs[-1], HumanMessage)                       # 段5 合成一条 Human
    inj = msgs[-1].content
    assert inj.index("早前对话梗概") < inj.index("知识库证据") < inj.index("订单数据")
    assert cur_idx == len(msgs) - 2                                 # 段5 在段4 之后且唯一
    assert sum(1 for m in msgs if isinstance(m, HumanMessage) and "订单数据" in m.content) == 1


async def test_layer_split_semantics():
    store = FakeStore(_typical_rows(), ("第一段梗概\n第二段梗概", 0, 3))
    cur = HumanMessage("继续")
    msgs = await build_model_context(store, evidence=[], order_data=None,
                                     current_human=cur, settings=_settings())
    body = [m for m in msgs if not isinstance(m, SystemMessage)]
    assert any("已折叠" in m.content and isinstance(m, AIMessage) for m in body)   # 层2 工具链 marker
    assert any(m.content == "那物流呢" for m in body if isinstance(m, HumanMessage))  # 层1 user 原样
    long_assistant = [m for m in body if isinstance(m, AIMessage) and "细" in m.content]
    assert len(long_assistant) == 1 and len(long_assistant[0].content) > 300      # 层1 原文不压


async def test_injection_omitted_when_all_empty():
    store = FakeStore(_typical_rows(), (None, 0, 10**9))   # 无梗概;层2 空;证据空
    cur = HumanMessage("你好呀")
    msgs = await build_model_context(store, evidence=[], order_data=None,
                                     current_human=cur, settings=_settings())
    assert msgs[-1] is cur                                 # 三项全空整条不发


async def test_floor_trim_never_drops_current():
    rows = [_row(1, "user", "长" * 6000), _row(2, "assistant", "答" * 6000)]
    store = FakeStore(rows, (None, 0, 0))                  # 全史在层1,估算远超 max_tokens
    cur = HumanMessage("短问题")
    msgs = await build_model_context(store, evidence=[], order_data=None,
                                     current_human=cur, settings=_settings())
    assert any(m is cur for m in msgs)                     # 保底:当前句必在
    assert isinstance(msgs[0], SystemMessage)


async def test_degrade_moves_anchor_and_logs(caplog):
    rows = [_row(i * 3 + 1, "user", "问" + "汉" * 700) if i % 1 == 0 else None
            for i in range(3)]
    # 三轮 user 行,每行 ≈702 token;demo layer1 预算 3954 装不下 4 轮 → 造 6 轮
    rows = []
    for t in range(6):
        rows.append(_row(t * 2 + 1, "user", "问" + "汉" * 700))
        rows.append(_row(t * 2 + 2, "assistant", "答" * 40))
    store = FakeStore(rows, (None, 0, 0))                  # 层1=全批 ≈4452>3954
    with caplog.at_level(logging.INFO):
        res = await degrade_if_needed(store, _settings())
    assert res is not None and store.sets == [res[1]]
    assert any("层1 降级" in r.getMessage() for r in caplog.records)
    kept = [r for r in rows if r.id > res[1]]
    assert kept[0].role == "user"                          # 保留段首必 human


async def test_history_view_contains_projection_and_logs_every_round(caplog):
    store = FakeStore(_typical_rows(), ("第一段梗概", 0, 3))
    with caplog.at_level(logging.INFO):
        view = await build_history_view(store, _settings())
    assert "第一段梗概" in view and "那物流呢" in view
    assert any(r.getMessage().startswith("history_ctx cid=42") for r in caplog.records)


def test_main_file_handler_utf8():
    """M1 评审#3 修复:按 baseFilename 认准 log/app.log——只查类型会被 caplog/
    其他套件的 handler 混过(首版假绿教训);独立跑必须真绿。"""
    import app.main  # noqa: F401  —— 模块导入即挂 FileHandler
    from logging import FileHandler
    fhs = [h for h in logging.getLogger().handlers if isinstance(h, FileHandler)
           and h.baseFilename.replace("\\", "/").endswith("log/app.log")]
    assert fhs, "root 上没有指向 log/app.log 的 FileHandler(验收4 grep 锚)"
    assert all(h.encoding and h.encoding.lower().replace("-", "") == "utf8"
               for h in fhs)
