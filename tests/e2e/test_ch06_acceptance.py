"""ch06 Task 8: 端到端验收 B1–B3(spec「验收标准」行逐钉,真模型)。

共环模式逐字沿用 test_ch05_acceptance.py 头部(loop_scope=module:Windows
proactor 下 httpx keepalive 连接绑创建环,单环+环内关客户端=唯一稳态)。
B1–B3 不写库(persister=None、cid 用保留段假值,仅当线程 key 用),
B4 浏览器点选面归 T6 手测+终审演示。
  B1 三轮物流→退款→聊回物流:逐轮意图+消解正确(验收1)
  B2 怪问题归「其他」全程无异常(验收2)
  B3 「这个能退吗」走 refund 链且政策检索命中、Agent 出回答(验收3后端面)
"""

import logging

import pytest
import pytest_asyncio

from app.core.config import get_settings
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.schemas.chat import ChatMessage
from app.services.chat_service import get_model
from app.workflows.graph import reset_checkpointer, stream_graph_turn

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

# 仅做 checkpointer 线程 key,不落库(无 persister)
FAKE_CID = 900001


# 共引擎逐字沿用 ch05 头部:aiomysql 连接绑环;且检索链 _fetch_rows 需要引擎活着
@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def db():
    init_engine(get_settings())
    async with get_session_factory()() as s:
        yield s
    await dispose_engine()


@pytest.fixture(autouse=True)
def _fresh_threads():
    reset_checkpointer()
    yield
    reset_checkpointer()


async def _one_turn(messages, conversation_id=None):
    # ch08(T11 波及收口):同 ch05 e2e——钉死 MCP 死端口保内置面确定性
    settings = get_settings().model_copy(update={
        "mcp_logistics_url": "http://127.0.0.1:9599/mcp",
        "mcp_aftersale_url": "http://127.0.0.1:9598/mcp"})
    model = get_model(settings)
    frames = []
    try:
        async for frame in stream_graph_turn(
            messages, settings, model,
            conversation_id=conversation_id, persister=None,
        ):
            frames.append(frame)
    finally:
        ac = getattr(model, "async_client", None)
        if ac is not None and hasattr(ac, "close"):
            await ac.close()
    return frames


def _joined_tokens(frames):
    return "".join(p for k, p in frames if k == "token")


def _msg(content):
    return ChatMessage(role="user", content=content)


def _turn_lines(caplog):
    return [r.getMessage() for r in caplog.records if "ch05 graph turn" in r.getMessage()]


# ---- B1 多轮:物流 → 退款 → 聊回物流(验收 1 原句逐钉) ----------------------------

async def test_b1_three_turn_intent_and_coref(db, caplog):
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        f1 = await _one_turn([_msg("订单1001的物流到哪了")], conversation_id=FAKE_CID)
        assert _joined_tokens(f1).strip(), "B1 第1轮无回答"
        f2 = await _one_turn([_msg("这个订单我想退掉")], conversation_id=FAKE_CID)
        assert _joined_tokens(f2).strip(), "B1 第2轮无回答"
        f3 = await _one_turn([_msg("它的物流呢")], conversation_id=FAKE_CID)
        assert _joined_tokens(f3).strip(), "B1 第3轮无回答"
    lines = _turn_lines(caplog)
    assert len(lines) >= 3, f"日志行不足: {lines}"
    l1, l2, l3 = lines[-3], lines[-2], lines[-1]
    assert "'intent': '物流'" in l1, l1
    assert "'intent': '退款退货'" in l2 and "'nodes': ['coref', 'intent', 'refund_" in l2, l2
    assert "'intent': '物流'" in l3, l3
    assert ("1001" in l3 and "物流" in l3), f"第3轮消解未钉住 1001+物流: {l3}"


# ---- B2 怪问题归「其他」,全程无异常(验收 2) --------------------------------------

async def test_b2_weird_question_lands_other(db, caplog):
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        frames = await _one_turn(
            [_msg("asdfgh 我不知道我想问啥 你们软件好奇怪")], conversation_id=FAKE_CID + 1)
    assert not [k for k, _ in frames if k == "error"], "B2 出现 error 帧"
    lines = _turn_lines(caplog)
    assert lines and "'intent': '其他'" in lines[-1], lines[-1]
    assert _joined_tokens(frames).strip(), "B2 其他→知识路收口无回答"


# ---- B3 「这个能退吗」→ refund 链+政策检索命中+Agent 回答(验收 3 后端面) ----------

async def test_b3_this_one_refundable_runs_refund_chain(db, caplog):
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        await _one_turn([_msg("我买了订单1001的冻干猫粮")], conversation_id=FAKE_CID + 2)
        f2 = await _one_turn([_msg("这个能退吗")], conversation_id=FAKE_CID + 2)
    lines = _turn_lines(caplog)
    assert len(lines) >= 2
    turn = lines[-1]
    assert "1001" in turn, f"resolved 未钉住 1001: {turn}"
    for node in ("refund_fetch", "refund_expand", "refund_policy", "agent"):
        assert node in turn, f"nodes 缺 {node}: {turn}"
    import re
    m = re.search(r"'retrieve_hits': (\d+)", turn)
    assert m and int(m.group(1)) > 0, f"政策检索零命中: {turn}"
    assert _joined_tokens(f2).strip(), "B3 回答为空"
