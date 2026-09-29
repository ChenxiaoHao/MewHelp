"""ch07 Task 11: 端到端验收 C1–C4(spec「测试与验收策略」集成 e2e 节,真模型)。

共引擎共环模式逐字沿用 ch05/06 头部(loop_scope=module:Windows proactor 下
httpx keepalive 与 aiomysql 连接皆绑创建环;单环+环内关客户端=唯一稳态)。
  C1 demo env(18000 组)级联:含一次工具轮的连续轮次 → 「层1 降级」→「summary trigger」
     → conversation_summaries ≥1 段 → 投影追平层1起点;末轮「最开始那个订单后来
     怎么说」靠梗概答对 oid、零 error 帧(验收2/4)
  C2 默认窗 20 轮:零降级零摘要、锚不动、段表 0 行——装得下就不压(验收3)
  C3 trigger 当轮生成器正常耗尽、答案帧达且无 error;下一轮在飞摘要不阻塞(验收4;
     done 帧属 routes SSE 适配件,不在 stream_graph_turn 帧面断言——run7 校准)
  C4 清 checkpointer(=重启/切换语义)→ 回填轮 coref 非 passthrough(resolved 被改写)
真模型有随机性:话术与断言口径逐字取自 spec 验收行,失败回任务修不降口径。
"""

import asyncio
import logging
import re

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select

from app.context.budget import estimate_items
from app.core.config import get_settings
from app.context.layers import ContextStore, rows_to_messages
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine
from app.db.models import (Conversation, ConversationSummary, LowConfidenceQuestion,
                            Message)
from app.schemas.chat import ChatMessage
from app.services.chat_service import get_model
from app.services.persistence import DBChatPersister
from app.workflows.graph import reset_checkpointer, stream_graph_turn

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

DEMO = dict(model_context_window=18000, max_agent_steps=3)  # spec 验收锚 demo 组(余键走默认=5650/3954/1695)


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


async def _turn(messages, settings, model, *, conversation_id=None, ctx_store=None):
    """生产同形:每轮短 session 绑 persister(单请求单会话),轮开始先落 user 行
    (routes bootstrap 语义)。run1/run2 教训:绑长活 db 会话=store 独立会话可见性
    时序不稳,级联面拿不到近史。"""
    frames = []
    q = messages[-1].content
    async with get_session_factory()() as s:
        await crud.add_message(s, conversation_id, "user", content=q)
        persister = DBChatPersister(s, conversation_id)
        try:
            async for frame in stream_graph_turn(
                messages, settings, model, conversation_id=conversation_id,
                persister=persister, ctx_store=ctx_store,
            ):
                frames.append(frame)
        finally:
            ac = getattr(model, "async_client", None)
            if ac is not None and hasattr(ac, "close"):
                await ac.close()
    return frames


def _msg(content):
    return ChatMessage(role="user", content=content)


def _joined(frames):
    return "".join(p for k, p in frames if k == "token")


async def _cleanup(db, conv_id):
    # lcq 有 FK→conversations(run4 实测拒删):拒答进池行须先清
    await db.execute(delete(LowConfidenceQuestion)
                     .where(LowConfidenceQuestion.conversation_id == conv_id))
    await db.execute(delete(Message).where(Message.conversation_id == conv_id))
    await db.execute(delete(ConversationSummary)
                     .where(ConversationSummary.conversation_id == conv_id))
    await db.execute(delete(Conversation).where(Conversation.id == conv_id))
    await db.commit()


async def _make_conv(db, user_id):
    conv = await crud.create_or_get_conversation(db, None, user_id)
    await db.commit()
    return conv


async def _wait_summary_done(db, conv_id, caplog, timeout=90.0):
    """轮询后台摘要落库(真模型宽限 90s)——非阻塞面=C3,此处只等终态。"""
    for _ in range(int(timeout / 0.5)):
        n = (await db.execute(
            select(func.count()).select_from(ConversationSummary)
            .where(ConversationSummary.conversation_id == conv_id))).scalar_one()
        if n >= 1:
            return n
        if any("summary failed" in r.getMessage() for r in caplog.records):
            pytest.fail("后台摘要 failed(见 WARN 栈)")
        await asyncio.sleep(0.5)
    return n


KNOW_Q = ["退货政策是什么", "多久内可以申请退款", "退货运费谁承担", "换货怎么操作",
          "优惠券使用范围", "发票怎么开", "会员积分规则", "物流时效是多久",
          "包装破损怎么办", "客服工作时间"]
# run3 实测:真模型轮均≈190 token(设计 steady=500),裸 21 轮到 3852 差一口不越层1
# 预算 3954——级联面(C1/C3)按 brief 容许上限 22 轮加真实买家料(轮均→≈400,早于
# 轮尽越线);C2 保持轻话术,守「装得下不压」的设计稳态前提不被加料反噬。
BEEF_KNOW = [f"我买的订单 100{i % 3 + 1} 的冻干猫粮好像不太对,{q}。"
             f"请结合我的情况完整讲讲,最好分点说清楚!" for i, q in enumerate(KNOW_Q)]
CHIT_Q = ["你好", "谢谢", "好的", "再见", "嗯嗯",
          "哈喽", "辛苦啦", "拜拜", "就这些", "明白了"]
# 级联面话术=用户长文:层2 触发判据是「半压后估算」(客服只留 head64≈64 token,用户
# 全量保留)——run4 实测轻话术 21 轮层2 恒<1695 不触发。用户句 200+ 字则单轮≈260,
# 6-7 轮降级批即越线,仍在 brief 容许的 21 轮内。C2 轻话术不受影响(其口径=装得下不压)。
CASCADE_KNOW_Q = [
    f"我上周在你们喵帮平台下的订单 100{i % 3 + 1},买的是那款冻干猫粮两公斤装的,签收"
    f"之后发现外包装有点破损,打开袋子颜色和气味都不太对,我家猫吃了一口就一直不肯吃"
    f"东西,精神也蔫蔫的,我当天就拍了照片和开箱视频存着,快递单号我也留好了,本来是"
    f"打算今天就处理完的,之前在线客服排队排了很久没轮到我,现在想认真问清楚:{q}。"
    f"麻烦你结合我说的这些情况,把政策依据、具体办理步骤、要准备哪些材料、大概的时限"
    f"都一条条给我讲明白,我照着一步步办,尽量别让我再去找人工,真是麻烦你了,谢谢啦!"
    for i, q in enumerate(KNOW_Q)]


# ---- C1 demo env 全级联 + 末轮梗概答对最早订单 ------------------------------------

async def test_c1_demo_env_cascade_and_summary_answer(db, caplog):
    caplog.set_level(logging.INFO)
    settings = get_settings().model_copy(update=DEMO)
    model = get_model(settings)
    conv = await _make_conv(db, "u_e2e_c1")
    store = ContextStore(get_session_factory(), conv.id, settings)
    try:
        script = ["订单 1001 的物流到哪了"] + [
            q for pair in zip(CASCADE_KNOW_Q, CHIT_Q) for q in pair] + [
            CASCADE_KNOW_Q[0]]      # 22 轮顶格(brief 容许 18–22),交错控稳态
        triggered = False
        for i, q in enumerate(script):
            frames = await _turn([_msg(q)], settings, model,
                                 conversation_id=conv.id, ctx_store=store)
            assert not [k for k, _ in frames if k == "error"], f"级联前 error:{q}"
            print(f"C1 r{i}: layer1 est≈"
                  f"{estimate_items(rows_to_messages(await store.fetch_layer1()))}")
            if any("summary trigger" in r.getMessage() for r in caplog.records):
                triggered = True
                break
        assert any("层1 降级" in r.getMessage() for r in caplog.records), \
            "demo env 未出现「层1 降级」锚(验收4 grep)"
        assert triggered, "轮次用尽仍未见 summary trigger 锚"
        assert await _wait_summary_done(db, conv.id, caplog) >= 1, "段表 0 行(摘要未落库)"
        row = (await db.execute(
            select(Conversation.summary, Conversation.summary_upto_msg_id,
                   Conversation.layer1_from_msg_id)
            .where(Conversation.id == conv.id))).one()
        assert row[0], "投影列未回写"
        assert row[1] is not None and row[1] == row[2] > 0, \
            f"summary_upto={row[1]} 未追平层1起点 {row[2]}(验收:边界追平)"
        assert any("summary done" in r.getMessage() for r in caplog.records)
        frames = await _turn([_msg("最开始那个订单后来怎么说")], settings, model,
                             conversation_id=conv.id, ctx_store=store)
        assert not [k for k, _ in frames if k == "error"], "级联后末轮出 error 帧"
        ans = _joined(frames)
        assert "1001" in ans, f"梗概面未答出最早订单号:{ans[:120]}"
    finally:
        await _cleanup(db, conv.id)


# ---- C2 默认窗 20 轮零降级零摘要(验收3:装得下就不压) ------------------------------

async def test_c2_default_window_20_turns_zero_degrade(db, caplog):
    caplog.set_level(logging.INFO)
    settings = get_settings()                     # 默认 32000 窗(sliding 10000/层1 6999/层2 3000)
    model = get_model(settings)
    conv = await _make_conv(db, "u_e2e_c2")
    store = ContextStore(get_session_factory(), conv.id, settings)
    turns = [q for pair in zip(KNOW_Q, CHIT_Q) for q in pair]     # 20 轮知识+闲聊交错
    assert len(turns) == 20
    try:
        for i, q in enumerate(turns):
            frames = await _turn([_msg(q)], settings, model,
                                 conversation_id=conv.id, ctx_store=store)
            assert not [k for k, _ in frames if k == "error"], f"第{i + 1}轮 error 帧"
        assert not [r for r in caplog.records if "层1 降级" in r.getMessage()], \
            "默认窗 20 轮出现降级(验收3 违例:装得下不许压)"
        assert not [r for r in caplog.records if "summary trigger" in r.getMessage()]
        l1 = (await db.execute(
            select(Conversation.layer1_from_msg_id).where(Conversation.id == conv.id)
        )).scalar_one()
        assert l1 in (0, None), f"锚被挪过: layer1_from={l1}"
        assert (await db.execute(
            select(func.count()).select_from(ConversationSummary)
            .where(ConversationSummary.conversation_id == conv.id))).scalar_one() == 0
    finally:
        await _cleanup(db, conv.id)


# ---- C3 trigger 当轮 done 正常关流(非阻塞=帧序不因后台任务变化) ---------------------

async def test_c3_trigger_turn_closes_stream_normally(db, caplog):
    caplog.set_level(logging.INFO)
    settings = get_settings().model_copy(update=DEMO)
    model = get_model(settings)
    conv = await _make_conv(db, "u_e2e_c3")
    store = ContextStore(get_session_factory(), conv.id, settings)
    try:
        script = ["订单 1001 的物流到哪了"] + [
            q for pair in zip(CASCADE_KNOW_Q, CHIT_Q) for q in pair] + [CASCADE_KNOW_Q[0]]
        for idx, q in enumerate(script):
            before = sum("summary trigger" in r.getMessage() for r in caplog.records)
            frames = await _turn([_msg(q)], settings, model,
                                 conversation_id=conv.id, ctx_store=store)
            after = sum("summary trigger" in r.getMessage() for r in caplog.records)
            if after > before:
                kinds = [k for k, _ in frames]
                # run7 校准:done 是 routes SSE 适配件(routes.py:124,生成器耗尽
                # 后才发),stream_graph_turn 设计上不外发(graph.py:151)——本层关流
                # 证据=_turn 正常返回(生成器完整耗尽)+ 答案帧已达 + 无 error。
                assert "token" in kinds and "error" not in kinds, \
                    f"trigger 轮帧序异常: {kinds}"
                # 非阻塞行为面:在飞摘要不得拖/断下一轮关流(原「done<trigger」
                # 断言有摘要快于轮尽的竞态假阳性,换此更强等价证据)
                nxt = script[idx + 1] if idx + 1 < len(script) else CHIT_Q[0]
                f2 = await _turn([_msg(nxt)], settings, model,
                                 conversation_id=conv.id, ctx_store=store)
                k2 = [k for k, _ in f2]
                assert "token" in k2 and "error" not in k2, \
                    "trigger 后一轮被在飞摘要阻塞/出错"
                assert not [r for r in caplog.records
                            if "summary failed" in r.getMessage()], "后台摘要 failed"
                return                      # 非阻塞面已证
        pytest.fail("demo env 下未走到 trigger 轮")
    finally:
        await _wait_summary_done(db, conv.id, caplog)   # 等终态再清,防 FK 竞态
        await _cleanup(db, conv.id)


# ---- C4 清线程=重启/切换:回填轮 coref 非 passthrough --------------------------------

async def test_c4_refill_after_restart_rewrites_resolve(db, caplog):
    caplog.set_level(logging.INFO)
    settings = get_settings()
    model = get_model(settings)
    conv = await _make_conv(db, "u_e2e_c4")
    store = ContextStore(get_session_factory(), conv.id, settings)
    try:
        for q in ["订单 1003 现在到哪了", "那大概多久能到"]:
            await _turn([_msg(q)], settings, model, conversation_id=conv.id,
                        ctx_store=store)
        reset_checkpointer()               # 进程重启语义:线程空、DB 有史
        frames = await _turn([_msg("它要是还没到怎么办")], settings, model,
                             conversation_id=conv.id, ctx_store=store)
        assert not [k for k, _ in frames if k == "error"]
        turn_lines = [r.getMessage() for r in caplog.records
                      if "ch05 graph turn" in r.getMessage()]
        assert turn_lines, "ch05 graph turn 行未出现"
        m = re.search(r"'resolved': '([^']*)'", turn_lines[-1])
        assert m and m.group(1) != "它要是还没到怎么办", \
            f"回填轮被当首轮 passthrough:{turn_lines[-1]}"
    finally:
        await _cleanup(db, conv.id)
