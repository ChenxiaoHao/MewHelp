"""ch07 T3/T4:层2 半压渲染 + 降级切割(只挪 id) + 五段装配 + ContextStore 读侧。

Review Focus 1 的形态红线:半压渲染**绝不出现 ToolMessage/带 tool_calls 的 AIMessage**;
层1 原文区按现 react 输入语义还原合法工具链(AIMessage.tool_calls → 紧随其 id 的
ToolMessage),链首 assistant 有文本则保留——两条路径形态都合法。
降级=边界 id 移动,不搬数据(spec「三层结构与边界语义」)。
"""

import json
import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import trim_messages

from app.context.budget import MSG_OVERHEAD, compute_budgets, estimate_items, estimate_text
from app.db import crud
from app.prompts.customer_service import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _head(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _fold_marker(tool_row, name: str) -> str:
    return f"[工具结果·{name}·≈{estimate_text(tool_row.content or '')} token 已折叠]"


def render_layer2(rows, settings) -> list[BaseMessage]:
    """三规则:user 原样 / assistant 留头几十字 / 工具链整链折一行标识。"""
    out: list[BaseMessage] = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if r.role == "user":
            out.append(HumanMessage(r.content or ""))
            i += 1
        elif r.role == "assistant" and r.tool_calls:
            parts = []
            if r.content:
                parts.append(_head(r.content, settings.assistant_head_chars))
            names = {
                tc.get("id"): tc.get("name", "?")
                for tc in r.tool_calls if isinstance(tc, dict)
            }
            j = i + 1
            while j < n and rows[j].role == "tool":
                parts.append(_fold_marker(rows[j], names.get(rows[j].tool_call_id, "?")))
                j += 1
            if not parts:  # 悬空链(tool 行落库失败被吞):fallback 标识,绝不发空 content
                joined = "+".join(nm for nm in dict.fromkeys(names.values()) if nm) or "?"
                parts.append(f"[工具调用·{joined}·已折叠]")
            out.append(AIMessage("\n".join(parts)))
            i = j
        elif r.role == "assistant":
            out.append(AIMessage(_head(r.content or "", settings.assistant_head_chars)))
            i += 1
        else:  # 孤儿 tool 行(写路径理论不产生):折标识,绝不还原 ToolMessage
            out.append(AIMessage(_fold_marker(r, "?")))
            i += 1
    return out


def pick_degrade_cut(layer1_rows, budget, est) -> int | None:
    """返回新 layer1_from(降级批末行 id);None=不动(未超/无可降粒度)。

    从最新往老累计,纯预算可保留的起点若落在轮中(assistant/tool),**向新**对齐到
    下一个轮首 user——降级批按完整轮出、保留段必 ≤ 预算(向老对齐会造出
    「保留段=整批仍超」的死循环,T4 真实预算场景实测暴露;Ruling 见 ledger)。
    连一整轮都装不下时整批降级(层2 摘要接盘),层1 绝不留超限原文。
    """
    if not layer1_rows or est(layer1_rows) <= budget:
        return None
    keep_from = len(layer1_rows)
    while keep_from > 0 and est(layer1_rows[keep_from - 1:]) <= budget:
        keep_from -= 1
    k = keep_from  # 纯预算不超的最大后缀起点(rows[k-1:] 才是首个超限后缀)
    while k < len(layer1_rows) and layer1_rows[k].role != "user":
        k += 1
    if k >= len(layer1_rows):
        # 向新找不到轮首(连一整轮都装不下,或尾行是残缺轮):整批降级进层2——
        # 留在层1 只会让当轮装配超限,交给层2 摘要压批才是出路(M1 评审#2)。
        return layer1_rows[-1].id
    return layer1_rows[k - 1].id


# ---------- ch07 T4:五段装配 + 降级动作 + 每轮留痕(需求 3/6) ----------

def rows_to_messages(rows) -> list[BaseMessage]:
    """层1 行集→LangChain 消息:**回执齐**的完整链才照形还原;未闭合半链/悬空链
    整链折标识,孤儿 tool 行折标识(M2-C1:open_ids 悬停使裸 AIMessage.tool_calls
    进严格兼容端 = 400,on_tool_result 落库失败被吞是现实触发面)。"""
    msgs: list[BaseMessage] = []
    i, n = 0, len(rows)
    while i < n:
        r = rows[i]
        if r.role == "user":
            msgs.append(HumanMessage(r.content or ""))
            i += 1
        elif r.role == "assistant" and r.tool_calls:
            tcs = [{"id": t.get("id"), "name": t.get("name"), "args": t.get("args") or {}}
                   for t in r.tool_calls if isinstance(t, dict)]
            tool_rows: list = []
            j = i + 1
            while j < n and rows[j].role == "tool":
                tool_rows.append(rows[j])
                j += 1
            answered = {tr.tool_call_id for tr in tool_rows}
            if tcs and all(t["id"] in answered for t in tcs):
                msgs.append(AIMessage(content=r.content or "", tool_calls=tcs))
                for tr in tool_rows:
                    msgs.append(ToolMessage(content=tr.content or "",
                                            tool_call_id=tr.tool_call_id))
            else:
                text = r.content or ""
                if tcs:
                    names = "+".join(dict.fromkeys(t.get("name") or "?" for t in tcs))
                    marker = f"[工具调用·{names}·已折叠]"
                    text = f"{text}\n{marker}" if text else marker
                msgs.append(AIMessage(text or "[工具调用·?·已折叠]"))
                for tr in tool_rows:  # 半链回执同折:不留无链首的 ToolMessage
                    msgs.append(AIMessage(_fold_marker(tr, "?")))
            i = j
        elif r.role == "assistant":
            msgs.append(AIMessage(r.content or ""))
            i += 1
        else:  # 孤儿 tool 行(写路径理论不产生):折标识,绝不还原 ToolMessage
            msgs.append(AIMessage(_fold_marker(r, "?")))
            i += 1
    return msgs


def _cap_summary(text: str, cap: int) -> str:
    """梗概投影按行截断**保尾部**(新段优先,需求 3 段5)。"""
    keep: list[str] = []
    used = 0
    for ln in reversed(text.split("\n")):
        est = estimate_text(ln)
        if keep and used + est > cap:
            break
        keep.append(ln)
        used += est
    return "\n".join(reversed(keep))


def _build_injection(summary, evidence, order_data, settings) -> HumanMessage | None:
    """段5:梗概+证据+订单数据合成**一条** HumanMessage;子项空即省,全空整条不发。"""
    parts = []
    if summary:
        parts.append("早前对话梗概:\n" + _cap_summary(summary, settings.summary_inject_tokens))
    if evidence:
        ev = "\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(evidence))
        parts.append("知识库证据:\n" + ev)
    if order_data:
        parts.append("订单数据:\n" + json.dumps(order_data, ensure_ascii=False))
    if not parts:
        return None
    return HumanMessage(content="\n\n".join(parts))


def _floor_trim(msgs: list[BaseMessage], max_tokens: int,
                current_human: HumanMessage) -> list[BaseMessage]:
    """保底 trim:与 chat_service.trim_history 同款(自算估算器版),绝不丢当前句。"""
    trimmed = trim_messages(
        msgs, strategy="last", token_counter=estimate_items,
        max_tokens=max_tokens, start_on="human", include_system=True,
    )
    if msgs and all(t is not current_human for t in trimmed):
        return [msgs[0], current_human]
    return trimmed


async def build_model_context(store, *, evidence, order_data, current_human,
                              settings) -> list[BaseMessage]:
    """五段固定序(需求 3):System → 层2 半压 → 层1 原文 → 当前句 → 合并注入(挂当前句后)。"""
    summary, upto, layer1_from = await store.load_ctx()
    rows = await store.fetch_all_rows()
    layer2_rows = [r for r in rows if upto < r.id <= layer1_from]
    layer1_rows = [r for r in rows if r.id > layer1_from]
    # routes bootstrap 先落当前 user 行(现状 ch04/06 语义),层1 尾部即它——
    # 剔除后段3 与段4 才不重复(集成面实测暴露;Ruling 见 ledger)。
    if (layer1_rows and layer1_rows[-1].role == "user"
            and (layer1_rows[-1].content or "") == current_human.content):
        layer1_rows = layer1_rows[:-1]
    msgs: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    msgs += render_layer2(layer2_rows, settings)
    msgs += rows_to_messages(layer1_rows)
    msgs.append(current_human)
    inject = _build_injection(summary, evidence, order_data, settings)
    if inject is not None:
        msgs.append(inject)
    b = compute_budgets(settings)
    s_tokens = estimate_text(SYSTEM_PROMPT) + MSG_OVERHEAD
    max_tokens = (b.sliding + s_tokens + settings.rerank_top_k * settings.chunk_size
                  + settings.summary_inject_tokens)
    return _floor_trim(msgs, max_tokens, current_human)


async def degrade_if_needed(store, settings) -> tuple[int, int] | None:
    """层1 超预算 → 挪锚(只 UPDATE id)+ 留痕;不动则 None(spec「降级只挪 id」)。"""
    b = compute_budgets(settings)
    _, _, old_l1 = await store.load_ctx()
    rows = await store.fetch_layer1()
    cut = pick_degrade_cut(rows, b.layer1, estimate_items)
    if cut is None:
        return None
    await store.set_layer1_from(cut)
    logger.info("层1 降级 %d→%d", old_l1, cut)
    return (old_l1, cut)


async def build_history_view(store, settings) -> str:
    """history_ctx(消解/意图共用):摘要行 + 层1 末 N 条;每轮必打(含闲聊轮,需求 6)。

    Ruling:tool 行折一行标识——「原文」指不过层2 的几十头截断;tool JSON 对
    coref 是无效输入,且会吞掉 history 面预算(现 ch06 面同样不含全量 tool 文本)。
    """
    summary, _, layer1_from = await store.load_ctx()
    rows = await store.fetch_all_rows()
    layer1 = [r for r in rows if r.id > layer1_from][-settings.history_view_messages:]
    lines = []
    if summary:
        lines.append("早前对话梗概:" + _cap_summary(summary, 200).replace("\n", " "))
    for r in layer1:
        if r.role == "user":
            lines.append(f"用户:{r.content}")
        elif r.role == "assistant":
            if (r.content or "").strip():
                lines.append(f"客服:{r.content}")
            elif r.tool_calls:
                names = "+".join(t.get("name", "?") for t in r.tool_calls if isinstance(t, dict))
                lines.append(f"客服:[调用工具 {names}]")
        else:
            lines.append("[工具结果·已折叠]")
    view = "\n".join(lines) or "(无)"
    logger.info("history_ctx cid=%s\n%s", store.cid, view)
    return view


def log_model_ctx(cid, msgs: list[BaseMessage], tokens: int) -> None:
    """验收 4 grep 锚:header 一条 + 逐条正文(单条截 2000 防爆盘,截断处标 …[截断])。"""
    def _text(m) -> str:
        c = getattr(m, "content", "") or ""
        t = c if isinstance(c, str) else str(c)
        return t[:2000] + ("…[截断]" if len(t) > 2000 else "")
    logger.info(
        "model_ctx cid=%s msgs=%d tokens≈%d\n%s", cid, len(msgs), tokens,
        "\n".join(f"[{i + 1}]{type(m).__name__}:{_text(m)}" for i, m in enumerate(msgs)),
    )


class ContextStore:
    """DB 权威读侧薄包(crud 活库往返已由 test_crud_context_ch07 钉过,T7 接图消费)。"""

    def __init__(self, session_factory, conversation_id: int, settings):
        self._sf = session_factory
        self.cid = conversation_id
        self.settings = settings

    @property
    def session_factory(self):
        """ctx 节点排后台摘要任务用(任务必须自开 session,Review Focus 2)。"""
        return self._sf

    async def load_ctx(self) -> tuple[str | None, int, int]:
        async with self._sf() as session:
            return await crud.get_conv_ctx(session, self.cid)

    async def fetch_all_rows(self, limit: int = 10000):
        async with self._sf() as session:
            return await crud.list_messages_after(session, self.cid, 0, limit=limit)

    async def fetch_layer1(self, limit: int = 10000):
        _, _, layer1_from = await self.load_ctx()
        async with self._sf() as session:
            return await crud.list_messages_after(session, self.cid, layer1_from, limit=limit)

    async def set_layer1_from(self, value: int) -> None:
        async with self._sf() as session:
            await crud.set_layer1_from(session, self.cid, value)
