# ch06 正式版分流器 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (用户已指定 ch05 同款 Native 速度模板). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 占位分流器换成正式版：LLM 指代消解+改写、四件套意图识别(八类+置信度)、「其他」兜底、退款/售后确定性子流程(取单→扩写→政策强制检索→判定)、缺单号弹订单选择器点选续跑、退款固定类别表单。

**Architecture:** LangGraph StateGraph 上重构 coref/intent 两节点并新增 refund 出口五节点链;扩写/合并/槽位全为 `routing.py` 纯函数;订单数据同播种 mock 单一来源;SSE 新增 `orders` 帧 + `refund_apply` 建议;跨轮槽位走 checkpointer 通道、coref 读后即清。

**Tech Stack:** langgraph 1.2.11 / langchain-core 1.4.x / fastapi 0.141.1 / sqlalchemy 2.0.54(全部 uv 锁版本,零新增依赖)

**Spec:** `docs/superpowers/specs/2026-09-27-ch06-query-router-design.md`

## Global Constraints

- python 3.12 + uv 锁版本;`PYTHONPATH=. uv run pytest -q` 基线 280 passed / 24 deselected,每任务收尾必须全绿且**ch05 单测只允许按本计划明示的重定向点改动**。
- token/done/error SSE 帧逐字符红线;新帧一律「末 token 后、done 前」;`suggestions` 帧既有形状不变(仅 action 枚举扩容)。
- 禁触三处保留改动:`app/rag/retriever.py`、`app/prompts/self_check.py`、`app/tools/executor.py`(工作区 M 状态原样)。
- 一任务一 commit(代码+测试+勾选+dev-notes 阶段追加同笔);dev-notes=`dev-notes/ch06.md` 按阶段实时追加。
- 纯 Prompt 质量不做单元断言,走 `tests/samples/*.csv` + `evals/smoke_ch06.py` 真模型评估(T7);**前端走 Vibe Coding 例外**(T6:描述效果直改,不套 TDD/评审)。
- spec 点名的技术选型定死;实现中冒出矛盾或走不通 → 停下问用户,不自行换方案。
- 评审里程碑批量跑:M1(T1–T2)、M2(T3–T4)、M3(T5–T6)、终审(T7–T9 后,含全分支 fresh reviewer)。

## Review Focus

计划测试不全覆盖、但真人使用会撞上的输入面(每行在归属任务补钉):

1. **「退货政策是什么」被四件套判成退款退货** → 劫进子流程、砸 ch05 验收 A1——T2 few-shot 第一对钉死 + T7 评估集带该行实测,T8 连同 ch05 e2e 全跑。
2. **用户无视卡片直接改口**(pending_flow 高挂后发新问题) → 下一轮 coref 读后即清、正常走意图,T4 补一条「pending 不劫持正常轮」用例。
3. **confidence 脏值**(`"0.9"` 字符串过、`"高"`/缺键/越界拒) → 拒=不合 schema=None→重试→「其他」,T2 解析用例钉。
4. **「其他」落 knowledge 且检索全空** → 必须走闸拒答+落池转人工,绝不进 Agent 裸答,T4 拓扑用例钉(复用 ch05 闸语义)。
5. **退款表单在无会话(匿名/DB 未起)提交** → 后端 503、前端可重试提示不崩,T5 端点用例 + T6 明示。

---

## 文件结构

| 动作 | 文件 | 职责 |
|------|------|------|
| 新建 | `app/prompts/coref.py` | COREF_PROMPT(消解+改写合一,纯文本输出) |
| 新建 | `app/prompts/query_expand.py` | EXPAND_PROMPT(queries 单数组强制 JSON) |
| 新建 | `app/schemas/refund.py` | RefundCreateRequest(四类 reason Literal) |
| 新建 | `tests/samples/ch06_intent_qa.csv`、`ch06_coref_qa.csv` | 标注样例(prompt 质量评估面) |
| 新建 | `evals/smoke_ch06.py` | 真模型冒烟评估(非 CI 断言) |
| 新建 | 单测 `test_coref_ch06.py`、`test_intent_v2_ch06.py`、`test_refund_pure_ch06.py`、`test_refund_flow_ch06.py`、`test_orders_sse_ch06.py`、`test_api_refunds_ch06.py`;e2e `test_ch06_acceptance.py` | 各任务 RED 起点 |
| 修改 | `app/workflows/{state,routing,nodes,graph}.py`、`app/prompts/intent.py`(原地 V2)、`app/agents/react.py`、`app/core/config.py`、`app/services/chat_service.py`、`app/api/routes.py`、`app/schemas/chat.py`、`app/tools/definitions.py`、`static/index.html`、`README.md`、受钉旧口径的两个 ch05 测试文件 | — |

---

### Task 1: coref 正式版节点(指代消解+改写合一)

**Files:**
- Create: `app/prompts/coref.py`
- Modify: `app/workflows/nodes.py`(`coref_node` → `make_coref_node(model)`,import HumanMessage)
- Modify: `app/workflows/graph.py`(`g.add_node("coref", N.make_coref_node(model))`)
- Test: `tests/test_coref_ch06.py`;重定向 `tests/test_graph_topology_ch05.py` 两处多轮用例

**Interfaces:**
- Consumes: `ChatState`(现有键);ScriptModel 模式(`tests/test_graph_topology_ch05.py:19`,测试内自带)
- Produces: `make_coref_node(model)`;`app/prompts/coref.py::COREF_PROMPT`(human 模板变量 `history`、`question`);log 新键 `coref: "done"|"passthrough"|"degraded"`;每轮复位集合含新键 `pending_flow/slot_order_id/order_data/expanded_queries/orders_payload/intent_confidence`(pending_flow 先读后清,Task 4 消费)

- [x] **Step 1: RED——新建 `tests/test_coref_ch06.py`**

```python
"""ch06 T1: coref 节点——首轮零调用透传/有历史走 LLM/失败降级透传/复位集合扩键。"""
from types import SimpleNamespace
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from app.workflows.nodes import make_coref_node


class ScriptModel:
    def __init__(self, replies):
        self.replies = list(replies); self.calls = 0
    async def ainvoke(self, msgs):
        self.calls += 1; self.last_msgs = list(msgs)
        return AIMessage(content=self.replies.pop(0))


def _state(human_contents, **extra):
    msgs = [HumanMessage(content=h) for h in human_contents]
    return {"messages": msgs, "user_query": msgs[-1].content, **extra}


@pytest.mark.asyncio
async def test_first_turn_passthrough_zero_calls():
    m = ScriptModel([])
    upd = await make_coref_node(m)(_state(["订单1001到哪了"]))
    assert upd["resolved_query"] == "订单1001到哪了"
    assert m.calls == 0 and upd["log"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_history_llm_completion():
    m = ScriptModel(["订单1001的猫粮能退吗"])
    st = _state(["我买了订单1001的猫粮", "到了", "这个能退吗"],
                evidence=[{"chunk_id": 1, "score": 0.9, "text": "旧证据"}])
    upd = await make_coref_node(m)(st)
    assert upd["resolved_query"] == "订单1001的猫粮能退吗"
    assert upd["log"]["coref"] == "done"
    assert upd["evidence"] == []          # 复位集合照常
    assert upd["log"]["nodes"] == ["coref"]


@pytest.mark.asyncio
async def test_llm_failure_degrades_to_passthrough():
    class Boom:
        async def ainvoke(self, msgs): raise RuntimeError("net")
    upd = await make_coref_node(Boom())(_state(["上轮", "这个呢"]))
    assert upd["resolved_query"] == "这个呢" and upd["log"]["coref"] == "degraded"


@pytest.mark.asyncio
async def test_reset_clears_ch06_keys_but_consumes_pending():
    m = ScriptModel(["它能退吗"])          # 非选择句:照常走 LLM,pending 读后即清
    st = _state(["上轮", "它能退吗"], pending_flow="refund",
                slot_order_id="1001", orders_payload=[{"x": 1}])
    upd = await make_coref_node(m)(st)
    assert upd["pending_flow"] == "" and upd["slot_order_id"] == ""
    assert upd["orders_payload"] == [] and upd["intent_confidence"] == 0.0
```

- [x] **Step 2: 跑 RED** → `PYTHONPATH=. uv run pytest tests/test_coref_ch06.py -q`;Expected: ImportError(make_coref_node/COREF_PROMPT 不存在)
- [x] **Step 3: 实现** `app/prompts/coref.py`:

```python
COREF_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商客服的问题理解器。根据对话历史,把用户本轮消息补全成一句自包含、可直接检索与判意图的标准问法。
规则(逐条硬性):
- 指代(它/这个/那个/刚才买的…)必须用历史中真实出现过的实体补全,历史里没有的实体绝不许编造;
- 本轮问题本来就自包含时,**原样输出**,不强行改写、不添加多余信息;
- 口语转标准问法:去掉语气词与口水话,保留型号/单号/金额/时限等全部关键信息;
- 只输出问法本身一行,无引号、无前缀、无解释。"""),
    ("human", "对话历史(可能为空):\n{history}\n\n本轮用户消息:{question}"),
])
```

`app/workflows/nodes.py` 替换占位 `coref_node`(保留 docstring 里 M1-I1 复位语义说明):

```python
def make_coref_node(model):
    async def coref_node(state: dict) -> dict:
        q = state.get("user_query", "")
        pending = state.get("pending_flow") or ""
        reset = {"log": {"nodes": ["coref"]}, "evidence": [], "suggestions": [],
                 "answer_text": "", "gate_pass": False, "pending_flow": "",
                 "slot_order_id": "", "order_data": {}, "expanded_queries": [],
                 "orders_payload": [], "intent_confidence": 0.0}
        hist = list(state.get("messages") or [])[:-1]      # 本轮 human 入图前已 append
        prior_user = [m for m in hist if isinstance(m, HumanMessage)]
        if not prior_user:                                  # 首轮:零调用透传
            return {**reset, "resolved_query": q}
        try:
            msgs = COREF_PROMPT.format_messages(
                history=_render_history(hist[-6:]), question=q)
            text = _text_of(await model.ainvoke(msgs)).strip()
            resolved = next((ln.strip() for ln in reversed(text.splitlines())
                             if ln.strip()), "") or q
        except Exception:  # noqa: BLE001 —— 理解层不拖垮主流程(spec「失败与降级」)
            reset["log"]["coref"] = "degraded"
            return {**reset, "resolved_query": q}
        reset["log"]["coref"] = "done"
        return {**reset, "resolved_query": resolved}
    return coref_node


def _render_history(msgs) -> str:
    lines = []
    for m in msgs:
        who = "用户" if isinstance(m, HumanMessage) else "客服"
        lines.append(f"{who}:{_text_of(m)}")
    return "\n".join(lines) or "(无)"
```

(Task 4 会在本函数顶端加 3 行续跑识别,勿提前实现。)
`graph.py` 入口节点行改 `g.add_node("coref", N.make_coref_node(model))`。
`state.py`:新增 6 键(`pending_flow/slot_order_id/order_data/expanded_queries/orders_payload/intent_confidence`,类型注释按 spec State 表)。

- [x] **Step 4: 跑 GREEN** → 新文件 4 测过;全量 `PYTHONPATH=. uv run pytest -q`,Expected: 仅 `test_graph_topology_ch05.py::test_per_turn_state_does_not_leak_across_turns` 与 `test_coref_passthrough_keeps_history` 红(多轮脚本被 coref 消耗)

- [x] **Step 5: 重定向两旧用例**(行为变化=有历史轮多一次 coref 调用,脚本按序插 coref 输出):
  - `test_per_turn_state…`:脚本改 `[投诉, "退款政策是什么", {商品咨询}, 政策A, "订单1001物流到哪了", {物流}, 包裹已到]`(JSON 消息补 `",\"confidence\":0.9"`,T2 前 parse 忽略该键不报错?——**否**:T1 时 parse 未升级,保持原 `{"intent":...}` 形状即可,coref 输出为人话字符串);断言 calls/链名不新增。
  - `test_coref_passthrough_keeps_history` 改名 `test_coref_completion_keeps_history`,第二轮脚本前插 coref 输出 AIMessage(内容为补全后问法)。
  这两处的 JSON→tuple 兼容在 T2 还会再过一遍(T2 给 fake 脚本统一补 confidence)。
- [x] **Step 6: 全量绿** → `PYTHONPATH=. uv run pytest -q`
- [x] **Step 7: Commit + dev-notes 追加 T1 行**

```bash
git add -A app tests && git commit -m "feat(ch06-t1): coref 正式版——LLM 指代消解+改写合一,首轮零调用/失败降级,ch06 复位键入 state"
```

---

### Task 2: 意图四件套 V2 + 降级路(小→大) + route 表 v2

**Files:**
- Modify: `app/prompts/intent.py`(原地 V2)、`app/workflows/routing.py`、`app/workflows/nodes.py`(`make_intent_node`、`logging_node`)、`app/workflows/graph.py`(工厂传 settings)、`app/core/config.py`(2 新键)、`app/services/chat_service.py`(`get_model` 加参)
- Test: `tests/test_intent_v2_ch06.py`;重定向 `tests/test_routing_ch05.py`、`tests/test_graph_topology_ch05.py` 受影响用例

**Interfaces:**
- Consumes: coref 的 `resolved_query`
- Produces: `INTENTS`(8 含「其他」)、`Route` 含 `"refund"`、`INTENT_ROUTES`(退款退货/售后→refund,其他→knowledge)、`parse_intent_json(raw)->tuple[str,float]|None`、`match_order_selection(text)->str|None`、`extract_order_id(*texts)->str|None`、`make_intent_node(model, settings)`、state 键 `intent_confidence`;`get_model(settings, *, model_name=None)`;settings 键 `intent_small_model=""`、`intent_confidence_threshold=0.75`;日志行含 `resolved`/`confidence`(nodes 仍是首键)

- [x] **Step 1: RED——`tests/test_intent_v2_ch06.py`(routing 纯函数面)**

```python
"""ch06 T2: 八类表/退款出口/置信度解析/槽位正则——纯函数,Review Focus 3。"""
import pytest
from app.workflows.routing import (
    INTENTS, INTENT_ROUTES, extract_order_id, match_order_selection,
    parse_intent_json, route_for_intent,
)

def test_eight_intents_and_refund_route():
    assert "其他" in INTENTS and len(INTENTS) == 8
    assert route_for_intent("退款退货") == "refund"
    assert route_for_intent("售后") == "refund"
    assert route_for_intent("其他") == "knowledge"
    assert route_for_intent("没这类") == "knowledge"      # 未知按「其他」处理(P1)

def test_parse_requires_intent_and_confidence():
    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)
    assert parse_intent_json('{"intent": "物流", "confidence": "0.8"}') == ("物流", 0.8)
    assert parse_intent_json('{"intent": "物流"}') is None                 # 缺 confidence
    assert parse_intent_json('{"intent": "物流", "confidence": "高"}') is None
    assert parse_intent_json('{"intent": "物流", "confidence": 1.5}') is None
    assert parse_intent_json('{"intent": "转账", "confidence": 0.9}') is None

def test_slot_regexes():
    assert match_order_selection("我选择订单 1002") == "1002"
    assert match_order_selection("我选 1002") is None
    assert extract_order_id("订单 1001 的物流", None) == "1001"
    assert extract_order_id("订单#1003：发货了吗") == "1003"
    assert extract_order_id("我想退款") is None
```

- [x] **Step 2: 跑 RED**(ImportError/断言失败);**Step 3: 实现 routing.py**(`INTENTS` 追加、`INTENT_ROUTES` v2、parse 返回 tuple:`float(data.get("confidence"))` try/except + `0.0<=c<=1.0` 校验,非法=continue 找下一个候选;两正则与两纯函数:

```python
_SELECTION_RE = re.compile(r"^我选择订单\s*(\d{3,})$")
_ORDER_IN_TEXT_RE = re.compile(r"订单\s*[#＃:：]?\s*(\d{3,})")

def match_order_selection(text): 
    m = _SELECTION_RE.fullmatch((text or "").strip()); return m and m.group(1)

def extract_order_id(*texts):
    for t in texts:
        m = _ORDER_IN_TEXT_RE.search(t or "")
        if m: return m.group(1)
    return None
```
)

- [x] **Step 4: RED(节点面)**——同文件追加意图节点用例(ScriptModel+SimpleNamespace settings 含两新键;工厂调用 `make_intent_node(model, settings)`):

```python
# 覆盖:①few-shot 契约在 prompt 里(渲染含「其他」「confidence」「退货政策是什么」边界例);
# ②解析失败×2 → ("其他",0.0) 且 route=knowledge;③正常 JSON → state 带 intent_confidence;
# ④快路闲聊零调用(user_query 原话匹配);⑤降级路:配小模型且 conf≥阈值→采纳小模型
#   结果(大模型脚本不被消费,calls==1);⑥降级路低置信→大模型复判被消费;
# ⑦小模型 ainvoke 抛异常→直接升大模型(Review Focus 等价未配置)。
# ⑤–⑦ monkeypatch app.workflows.nodes.get_model 返回第二台 ScriptModel。
```

(实现步骤按此七断言逐条写测试体——断言形状抄 T1 文件模式,不另造基建。)

- [x] **Step 5: 实现**
  - `app/prompts/intent.py` 原地 V2:system = 八类枚举(含「其他:拿不准归它,别硬塞业务意图」)+判类口径+边界 few-shot ≥4 对(**第一对钉**「退货政策是什么=商品咨询;这单我要退=退款退货」)+输出契约恰好 `{{"intent": "...", "confidence": 0.0到1.0}}`(双花括号转义,ch04 T4 坑位注记保留)。
  - `nodes.make_intent_node(model, settings)`:`_judge(m, q)->tuple|None`(format→ainvoke→parse,异常 None);流程=快路(user_query 原话)→ `_judge` 大模型×2 或降级路(小模型先判,`conf>=settings.intent_confidence_threshold` 采纳,否则升大模型一次;小模型 None 直接升)→ 全失败 `("其他", 0.0)`;返回含 `"intent_confidence": conf`、`route=route_for_intent(intent)`。
  - `get_model(settings, *, model_name=None)`:`model=model_name or settings.model_name`,默认路径逐字符不变。
  - config 两键;`logging_node` 输出键列表加 `"resolved", "confidence"`(截断 80 字符)——**dict 推导里 `nodes` 保持首位**(ch05 A1/A5b 字面断言的序)。
  - `graph.py`:`make_intent_node(model, settings)`。
- [x] **Step 6: 重定向旧钉**——`test_routing_ch05.py`:参数表退款退货→refund/售后→refund、`INTENTS` 八元组、parse 用例改 tuple 契约+Review Focus 3 脏值行、prompt 测试加「confidence」「其他」断言;`test_graph_topology_ch05.py`:JSON 脚本统一补 `,"confidence":0.9`、`test_p2_fallback` 断言 `out["intent"] == "其他"`(route 仍 knowledge)、原「退款退货走 knowledge」两例改「商品咨询」。
- [x] **Step 7: 全量绿 + Commit** `"feat(ch06-t2): 意图四件套V2+其他兜底+route表v2(refund出口)+小→大降级路(默认关)"`

---

### Task 3: 扩写/合并纯函数 + 订单数据面(_make_order 抽壳/list_user_orders)

**Files:**
- Modify: `app/workflows/routing.py`(+`parse_queries_json`、`merge_evidence`)、`app/tools/definitions.py`、`app/prompts/query_expand.py`(新)
- Test: `tests/test_refund_pure_ch06.py`;补强 `tests/test_tools.py` 一致性用例

**Interfaces:**
- Consumes: 无新面
- Produces: `parse_queries_json(raw)->list[str]|None`(≤4、去重、保序)、`merge_evidence(groups:list[list[dict]], cap:int)->list[dict]`(chunk_id 去重取最高分降序截断)、`definitions._make_order(order_id)->dict`、`definitions.list_user_orders(user_id)->list[dict]`(卡片:`{order_id,status,amount,created_at,items:["名 ×qty"]}`)、`EXPAND_PROMPT`(human 变量 `question`)

- [ ] **Step 1: RED**

```python
def test_parse_queries():
    assert parse_queries_json('{"queries": ["a", "b", "a", "c", "d", "e"]}') == ["a", "b", "c", "d"]
    assert parse_queries_json('```json\n{"queries":["x"]}\n```') == ["x"]
    assert parse_queries_json('{"queries": "x"}') is None
    assert parse_queries_json('{"q": ["x"]}') is None
    assert parse_queries_json('不是json') is None

def test_merge_evidence_dedup_and_cap():
    g1 = [{"chunk_id": 1, "score": 0.5, "text": "t1"}, {"chunk_id": 2, "score": 0.9, "text": "t2"}]
    g2 = [{"chunk_id": 2, "score": 0.7, "text": "t2"}, {"chunk_id": 3, "score": 0.6, "text": "t3"}]
    out = merge_evidence([g1, g2], cap=2)
    assert [(c["chunk_id"], c["score"]) for c in out] == [(2, 0.9), (3, 0.6)]

async def test_orders_single_source_consistency():
    from app.tools.definitions import _make_order, list_user_orders
    cards = list_user_orders("demo_user")
    assert [c["order_id"] for c in cards] == ["1001", "1002", "1003"]
    for c in cards:
        full = _make_order(c["order_id"])
        assert (c["status"], c["amount"]) == (full["status"], full["amount"])  # P4 逐字一致
```

- [ ] **Step 2: 跑 RED** → **Step 3: 实现**(definitions 抽壳:`query_order` 薄壳 `return _make_order(order_id)`,生成体逐字段搬移零语义改动——`tests/test_tools.py` 既有用例是它的回归闸);`EXPAND_PROMPT` system 钉:同一退款诉求拆 ≤3 条**侧重不同**的检索问法(资格时限/流程运费/凭证要求),字段就 `{{"queries": [...]}}` 一个数组、不输出原问法之外的解释。
- [ ] **Step 4: 全量绿**(query_order 行为不变,280 基线自然守住)+ Commit `"feat(ch06-t3): 扩写/合并纯函数+EXPAND_PROMPT+订单同播种数据面(_make_order/list_user_orders)"`

---

### Task 4: 退款确定性子流程五节点 + graph 接线 + Agent 注入

**Files:**
- Modify: `app/workflows/nodes.py`、`app/workflows/graph.py`、`app/workflows/state.py`(`REFUND_APPLY` 常量 + `SELECT_ORDER_ASK`)、`app/agents/react.py`(order_data 前置 SystemMessage)、`app/core/config.py`(`demo_user_id` 已有,`rerank_top_n` 已有)
- Test: `tests/test_refund_flow_ch06.py`(复用 T1 ScriptModel 模式 + ch05 `fake_env` monkeypatch 模式)

**Interfaces:**
- Consumes: T1 复位键、T2 `match_order_selection/extract_order_id`/route refund、T3 `_make_order/list_user_orders/parse_queries_json/merge_evidence/EXPAND_PROMPT`
- Produces: 节点名(入 log nodes):`refund_slot/refund_selector/refund_fetch/refund_expand/refund_policy/refund_gate`;`make_confidence_gate_node(settings, source="ch05_gate")`(默认参不动旧行为);agent 节点 route==refund 且 done.suggestions 空 → `[REFUND_APPLY]`;coref 顶端续跑识别(slot_order_id=oid、resolved 模板 `f"订单 {oid} 能不能申请退款？"`、log `resume:True`、intent 见 slot 直通零调用)

- [ ] **Step 1: RED——`tests/test_refund_flow_ch06.py`** 用例清单(逐条写实体,断言 nodes 链+键):
  1. 无单号「我要退款」→ 链 `[coref,intent,refund_slot,refund_selector,logging]`,`orders_payload` 3 卡、`pending_flow=="refund"`、answer==SELECT_ORDER_ASK、零工具/检索调用;
  2. 同 thread 续跑「我选择订单 1001」→ coref resume(不耗模型调用)+intent 零调用直通,链含 `[refund_fetch,refund_expand,refund_policy,refund_gate,agent]`,`order_data["order_id"]=="1001"`、`pending_flow==""`;expand 脚本 `{"queries":["多久内可退","退款运费谁承担"]}`,monkeypatch retrieve 回固定两 chunk → `expanded_queries==["订单 1001 能不能申请退款？","多久内可退","退款运费谁承担"]`(原问法在首位)、evidence 为 merge 后;
  3. expand 输出烂 JSON → 单路 `[resolved]` 照常检索(降级不阻断);
  4. 政策检索全空 → refund_gate fail → REFUSAL_ANSWER+transfer_human 建议、**无 refund_apply**、落池被 monkeypatch 吞(Review Focus 4 同族);
  5. gate 通过 agent 正常收尾 → suggestions==[REFUND_APPLY 两键];done 带 transfer_human(预算熔断)→ **不被覆盖**;
  6. pending 高挂 + 用户改口「你好」→ 快路闲聊,pending 读后即清不劫持(Review Focus 2);
  7. react.py:state 含 `order_data` 时 `model.last_msgs` 有「订单数据:」SystemMessage,不含时没有(回归 ch05 零影响);
  8. 单号内联「订单1001我要退」→ 首轮直连 fetch(selector 不弹)。
- [ ] **Step 2: 跑 RED** → **Step 3: 实现**
  - `refund_slot_node(state)` 纯同步:oid=`slot_order_id` or `extract_order_id(resolved,user)`;返回值不放分支——分支交给条件边 `_after_slot`→"fetch"/"selector";
  - `refund_selector_node(settings)`:orders_payload=list_user_orders(settings.demo_user_id)、answer/messages 固定话术、pending_flow="refund";
  - `refund_fetch_node`:order_data=_make_order(oid);
  - `make_refund_expand_node(model)`:EXPAND_PROMPT→parse→None 降级 [resolved];上限截 `1+3`;
  - `make_refund_policy_retrieve_node(settings)`:asyncio.gather 每 q `retriever.retrieve(q, strategy="hybrid_rerank", settings=settings)`(经模块属性,monkeypatch 兼容)→ chunk→dict(同 knowledge 节点形状)→ `merge_evidence(..., cap=settings.rerank_top_n)`;log retrieve_hits;
  - graph:`_dispatch` 增 `"refund": "refund_slot"`;链 `refund_slot→(cond)→selector|fetch→expand→policy→refund_gate→(cond pass→agent/fail→logging)`;`make_confidence_gate_node(settings, source="ch05_gate")` 落池 source 参数化,refund_gate 实例传 `"ch06_refund_gate"`;
  - agent 节点尾:route=="refund" 且 gate_pass 且 `not done["suggestions"]` → `REFUND_APPLY`(state.py 常量,label 发起退款申请);
  - coref 顶端(T4 预告过的 3 行):`oid = match_order_selection(q) if state.get("pending_flow")=="refund" else None` → 命中走模板+resume 日志,不进 LLM 分支。
- [ ] **Step 4: 全量绿**(ch05 拓扑用例不受 refund 边影响)+ Commit `"feat(ch06-t4): 退款确定性子流程五节点+续跑识别+gate source 参数化+refund_apply 建议+agent 订单数据注入"`

---

### Task 5: SSE `orders` 帧 + refund_apply 枚举 + POST /api/refunds

**Files:**
- Modify: `app/schemas/chat.py`、`app/workflows/graph.py`(适配层发帧)、`app/api/routes.py`
- Create: `app/schemas/refund.py`
- Test: `tests/test_orders_sse_ch06.py`、`tests/test_api_refunds_ch06.py`(端点夹具抄 `tests/test_tickets_endpoint_ch05.py` 模式)

**Interfaces:**
- Consumes: T4 `orders_payload`/`pending_flow`/REFUND_APPLY
- Produces: SSE `orders` 帧(`OrdersEvent{items:list[OrderCard]}`)、`Suggestion.action` 三枚举、`POST /api/refunds`(201→`TicketOut`;reason 非法 422;无会话 503;tickets 落行 ticket_type="售后")、适配层帧序:token…→(persist)→orders→suggestions→done

- [ ] **Step 1: RED**——适配层用例:同 T4 场景跑 `stream_graph_turn`,断言 kinds 中 `"orders"` 位于最后一个 `"token"` 之后、`"done"` 之前,payload 三项含 items 字符串「名 ×qty」;端点用例:合法四类 reason→201+落表描述含「订单 1001」「原因」;reason="想退就退"→422;session None→503。
- [ ] **Step 2 跑 RED → Step 3 实现**:`schemas/refund.py`(`RefundReason = Literal["七天无理由","商品质量问题","拍错多拍","其他"]`,request 三字段);routes elif + 新端点(描述拼装 `f"【退款申请】订单 {req.order_id}｜原因：{req.reason}"`,crud.create_ticket(ticket_type="售后"));graph 适配层 `if final.get("orders_payload"): yield ("orders", {"items": final["orders_payload"]})`(落位 suggestions 之前,仅非空才发——Review Focus 4 防御)。
- [ ] **Step 4: 全量绿 + Commit** `"feat(ch06-t5): orders 帧契约+refund_apply 枚举+POST /api/refunds 落 tickets(售后)"`

---

### Task 6: 前端配套(Vibe Coding 例外,无 TDD/评审)

**Files:**
- Modify: `static/index.html`(CSS 段、send() 帧分发、新函数 `addOrderCards`/`openRefundForm`)

**效果描述(用户验收 4 的操作面):**
1. `orders` 帧 → 气泡下方一组像素风订单卡(单号/状态/金额/商品行 + 「选这个」钮);点任一卡:整组钮禁用+卡片点亮选中态+自动以「我选择订单 {id}」走既有 send() 流程(入 history、出 user 气泡);
2. suggestions 新钮 `refund_apply` →「🧾 发起退款申请」:弹轻表单(单号输入预填=本会话最后一条「我选择订单」提取值,无则空必填;原因=四类固定下拉;提交 POST /api/refunds);成功→「🎫 退款申请 {ticket_no} 已提交」assistant 气泡;422/503/网络错→错误气泡可重试不锁死(Review Focus 5);
3. 未知 action 值仍按 ch05 兜底不渲染。
- [ ] **Step 1: 直改**(改完 `PYTHONPATH=. uv run pytest -q` 确认后端零回归)
- [ ] **Step 2: 手测清单**(uvicorn 起服务:无单号退款→卡片→点选→结论→表单提交拿票号;截图不要求,口头过)
- [ ] **Step 3: Commit** `"feat(ch06-t6-ui): 订单选择卡片+退款固定类别表单(Vibe)"`

---

### Task 7: Prompt 质量评估集 + 真模型冒烟(替代 TDD 的 prompt 步)

**Files:**
- Create: `tests/samples/ch06_intent_qa.csv`、`tests/samples/ch06_coref_qa.csv`、`evals/smoke_ch06.py`

- [ ] **Step 1: 样例落盘**——intent CSV(question,expected_intent)≥14 行:**含「退货政策是什么→商品咨询」(Review Focus 1 的评估面)**、八类各≥1、「你们app好怪我不知道该说啥」「asdf 心情不好」类其他带≥3;coref CSV(history|question|expected_contains)≥6 行:「它/这个」补全对+自包含原样对。
- [ ] **Step 2: 写 `evals/smoke_ch06.py`**(自包含:载入两 CSV→get_model→INTENT_PROMPT/COREF_PROMPT 直调→parse/包含判定→打印逐行对照+准确率;恒 exit 0,不进 CI)
- [ ] **Step 3: 真模型跑** `PYTHONPATH=. uv run python evals/smoke_ch06.py`,读输出;意图准确率过低(<85%)→ 只调 few-shot 表述重跑,**不动 route 表/代码**(选型定死);结果记 dev-notes「Prompt 质量」段
- [ ] **Step 4: Commit** `"test(ch06-t7): 意图/消解标注样例+smoke_ch06 真模型冒烟评估"`

---

### Task 8: e2e 验收 B1–B3 + 双章全回归

**Files:**
- Create: `tests/e2e/test_ch06_acceptance.py`(共环+共引擎 fixture 逐字沿用 `tests/e2e/test_ch05_acceptance.py` 头部模式)

- [ ] **Step 1: 写用例(integration)**
  - **B1** 同 conversation 三轮:「订单1001的物流到哪了」→「这个订单我想退掉」→「它的物流呢」;逐轮 caplog 行断言 `'intent': '物流'` / `'intent': '退款退货'`(nodes 含 refund 链)/ 第三轮 `'resolved': '订单 1001 的物流到哪了'`(容语义包含「1001」+「物流」)且 intent 物流——验收 1 原句逐钉。
  - **B2** 怪问题「asdfgh 我不知道我想问啥 你们软件好奇怪」→ 日志 `'intent': '其他'` + 全程无异常 + 帧流正常收口(验收 2);
  - **B3** 「我买了订单1001的冻干猫粮」→「这个能退吗」→ resolved 含 1001、nodes 含 `refund_fetch/refund_expand/refund_policy/agent`、`retrieve_hits >0`、回答非空(验收 3 后端面;B4 浏览器面归 T6 手测+终审演示)。
- [ ] **Step 2: 跑** `PYTHONPATH=. uv run pytest -m integration -q -p no:cacheprovider`(含 ch05 六条——双章同绿是回归红线;ch05 A1 若被「退货政策」重判劫持=回 T2 修 few-shot,不降口径)
- [ ] **Step 3: 全量绿 + Commit** `"test(ch06-t8): e2e 验收 B1–B3(真模型+双章全回归)"`

---

### Task 9: 交付收尾(README + dev-notes + 演示)

- [ ] **Step 1:** README 追加 ch06 节(功能 bullet、演示命令、验收话术 4 条、integration 跑法)
- [ ] **Step 2:** dev-notes「finish」段:演示命令/测试结果/spec/plan/dev-notes 四路径 + 遗留 minor 汇总
- [ ] **Step 3:** `PYTHONPATH=. uv run pytest -q` 终绿;Commit `"docs(ch06-t9): README+dev-notes 完结交付"`
- [ ] 之后:终审 whole-branch fresh reviewer(opus)→ 修批 → `finishing-a-development-branch` 菜单(默认选项 1 本地合并,照 ch05 惯例再问)。

---

## Self-Review(计划完稿后跑)

1. **Spec coverage**:需求 1/2→T1;4→T2;3→T3+T4;5→T4;6→T4+T6;7→T6;端点→T5;验收 1–4→T8+T6;P1–P8 全数落进对应任务;「非目标」无越权实现。✔
2. **占位扫描**:T2 Step 4 的七断言清单是唯一「按形状写」位——七条行为枚举完整、基建有出处,可接受。✔
3. **类型一致**:`parse_intent_json` 返回 tuple 自 T2 起全链统一;`orders_payload` 卡形状=T3 list_user_orders 输出=T5 OrderCard 键=前端渲染键,三处同名。✔
4. **Review Focus 5 行** → 分别钉在 T7/T4-6/T2-3/T4-5/T5+T6。✔
