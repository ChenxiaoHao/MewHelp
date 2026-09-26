"""ch05 Task 9: 端到端验收 1–5（spec「验收标准 → 责任节点映射」节，需求逐字）。

真模型(qwen-plus)+真 MySQL+活 Milvus——integration 标记,默认 deselect,
验收时 `-m integration` 显式跑。每条 = spec 验收一行:
  A1 政策类问题,日志可见强制检索节点被走到(logging_node 的 ch05 graph turn 行)
  A2 「订单 1001 的物流到哪了」Agent 自调工具(tool_call 帧)
  A3 「我要投诉」两个独立按钮 suggestions 帧+不点零动作(帧序+ticket 行数不变)
  A4 闲聊固定话术(逐字)
  A5 先查订单再查物流 ReAct 多步(tool_call ≥2 且日志 agent_steps)
真模型有随机性:话术逐字取自 spec 验收行,失败回任务修不降口径(brief Step 1)。
"""

import logging
import re

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import Conversation, Message, Ticket
from app.schemas.chat import ChatMessage
from app.workflows.nodes import CHITCHAT_FIXED
from app.services.chat_service import get_model
from app.services.persistence import DBChatPersister
from app.workflows.graph import reset_checkpointer, stream_graph_turn

# 整模块共环:httpx2/httpcore2 的代理隧道连接绑创建环,Windows proactor 下
# 「上一测的 keepalive 连接在下一测环上最终化」必炸 Event loop is closed
# (成对二分实证:A3→A4、A4→A5a 皆前测毒后测)。单环+模块结束统一收尾=唯一稳态。
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


# 引擎池与 httpx2 隧道同理:aiomysql 连接也绑环。整模块共用一个 engine/session,
# init/dispose 各一次,全在模块环上,不留跨环尾巴。
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


async def _one_turn(messages, persister=None, conversation_id=None):
    settings = get_settings()
    model = get_model(settings)
    frames = []
    try:
        async for frame in stream_graph_turn(
            messages, settings, model,
            conversation_id=conversation_id, persister=persister,
        ):
            frames.append(frame)
    finally:
        # pytest-asyncio 每测一换环:模型客户端若不显式关,keepalive 连接留在
        # 已关环上靠 GC 最终化,会在下一测的 httpx 分配处炸 "Event loop is
        # closed"(A4 位抖根因)。本测环内同步关净,不留跨环尾巴。
        ac = getattr(model, "async_client", None)
        if ac is not None and hasattr(ac, "close"):
            await ac.close()
    return frames


def _joined_tokens(frames):
    return "".join(p for k, p in frames if k == "token")


def _msg(content):
    return ChatMessage(role="user", content=content)


async def _cleanup_conv(db, conv_id):
    await db.execute(delete(Message).where(Message.conversation_id == conv_id))
    await db.execute(delete(Conversation).where(Conversation.id == conv_id))
    await db.commit()


# ---- A1 政策类问题:检索节点日志面 ------------------------------------------------

async def test_a1_policy_question_logs_retrieve_node(db, caplog):
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        frames = await _one_turn([_msg("退货政策是什么")])
    turn_lines = [r.getMessage() for r in caplog.records
                  if "ch05 graph turn" in r.getMessage()]
    assert turn_lines, "日志节点行未出现(验收1后端面)"
    assert "'nodes': ['coref', 'intent', 'retrieve'" in turn_lines[-1]
    assert "gate" in turn_lines[-1]
    assert _joined_tokens(frames).strip(), "政策类回答为空"


# ---- A2 订单物流问题:Agent 自调工具 ----------------------------------------------

async def test_a2_order_logistics_agent_calls_tool(db):
    frames = await _one_turn([_msg("订单 1001 的物流到哪了")])
    calls = [p for k, p in frames if k == "tool_call"]
    assert calls, "Agent 未自调工具(验收2)"
    names = {c["name"] for c in calls}
    assert names & {"query_order", "query_logistics"}, names
    assert _joined_tokens(frames).strip(), "工具链收尾无回答"


# ---- A3 投诉:两独立按钮帧 + 不点零动作 -------------------------------------------

async def test_a3_complaint_suggestions_frame_and_no_ticket_write(db):
    before = (await db.execute(select(func.count()).select_from(Ticket))).scalar_one()
    conv = await crud.create_or_get_conversation(db, None, "u_e2e_a3")
    await crud.add_message(db, conv.id, "user", content="我要投诉")
    await db.commit()
    try:
        frames = await _one_turn([_msg("我要投诉")],
                                 persister=DBChatPersister(db, conv.id),
                                 conversation_id=conv.id)
    finally:
        await _cleanup_conv(db, conv.id)
    kinds = [k for k, _ in frames]
    assert "suggestions" in kinds, "suggestions 帧缺失(验收3)"
    items = next(p for k, p in frames if k == "suggestions")["items"]
    assert [it["action"] for it in items] == ["transfer_human", "create_ticket"]  # 两独立、互不绑定
    last_token = max(i for i, k in enumerate(kinds) if k == "token")
    assert kinds.index("suggestions") > last_token  # RF4:末 token 之后
    assert kinds[-1] == "suggestions"  # 本轮投诉出口:帧后收口
    after = (await db.execute(select(func.count()).select_from(Ticket))).scalar_one()
    assert after == before, "未点建工单却写了 tickets(验收3红线)"


# ---- A4 闲聊固定话术 --------------------------------------------------------------

async def test_a4_chitchat_fixed_answer(db):
    frames = await _one_turn([_msg("你好呀")])
    assert _joined_tokens(frames) == CHITCHAT_FIXED  # 逐字红线
    assert not [k for k, _ in frames if k == "tool_call"], "闲聊走了工具(需求6违例)"


# ---- A5 先订单后物流:ReAct 多步(spec 字面话术 + 依赖式话术双腿) -------------------

async def test_a5a_spec_wording_two_tools(db):
    """spec 字面行:两工具都要自调(该话术两参均为 order_id,模型合法并发,只钉调用面)。"""
    frames = await _one_turn(
        [_msg("帮我先查一下订单 1001 买了什么，再看看物流到哪了")])
    calls = [p for k, p in frames if k == "tool_call"]
    assert len(calls) >= 2, f"工具调用数不足: {len(calls)}(验收5)"
    names = {c["name"] for c in calls}
    assert "query_order" in names and "query_logistics" in names, names
    assert _joined_tokens(frames).strip()


async def test_a5b_dependent_wording_multistep(db, caplog):
    """ReAct「多步」的强钉:第二步参数(FAQ 关键词)依赖第一步输出(商品名),并发不可达。"""
    with caplog.at_level(logging.INFO, logger="app.workflows.nodes"):
        frames = await _one_turn(
            [_msg("先查下订单 1001 买的是什么商品，再按这个商品名在 FAQ 里查下使用说明")])
    calls = [p for k, p in frames if k == "tool_call"]
    names = {c["name"] for c in calls}
    assert "query_order" in names and "query_faq" in names, names
    turn_lines = [r.getMessage() for r in caplog.records
                  if "ch05 graph turn" in r.getMessage()]
    assert turn_lines, "日志节点行未出现(验收5步数面)"
    m = re.search(r"'agent_steps': (\d+)", turn_lines[-1])
    assert m and int(m.group(1)) >= 2, f"日志步数不足(验收5): {turn_lines[-1]}"
    assert _joined_tokens(frames).strip()
