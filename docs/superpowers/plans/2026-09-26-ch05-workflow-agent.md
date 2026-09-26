# ch05 Workflow + Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (选定) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph StateGraph 把 ch01–ch04 的对话/检索/工具能力升级为「确定性 Workflow 骨架 + ReAct 主力 Agent 核心节点」，含裸循环教学对照、七意图四出口分流、前置置信度闸、SSE 建议可选项帧、前端转人工/建工双按钮、create_ticket 副作用窄移除。

**Architecture:** 图链「指代消解(透传)→意图识别→分流→[知识类]检索→置信度闸→主力Agent→日志记录」，State 全程贯穿，checkpointer=InMemorySaver(仅进程内)。聊天入口 `/api/chat/stream` 的编排调用点从 `stream_chat_with_tools` 换成 Graph 流适配层，SSE 既有帧逐字符不动、只增 `suggestions` 帧。

**Tech Stack:** langgraph 1.2.11、langchain 1.4.2、langchain-core 1.6.3、langgraph-checkpoint 4.2.0、fastapi 0.141.1、sqlalchemy 2.0.54（全部已在 uv.lock，**零新增依赖**）。

**Spec:** `docs/superpowers/specs/2026-09-26-ch05-workflow-agent-design.md`（决策日志 D1–D8、四出口表、验收映射以 spec 为准）

## Global Constraints

- 不升级任何依赖；官方文档与实装版本冲突时**停止并问用户**（spec「API 与版本核对记录」）。
- 三处 ch04 保留改动（`app/prompts/self_check.py`、`app/rag/retriever.py`、`app/tools/executor.py`）只 import 复用，**不得修改**。
- SSE `token` / `done` / `error` 帧逐字符红线（ch01 起）；`tool_result` 帧沿用 `exclude_none` 兼容模式。
- **一任务一 commit**：代码+测试+本计划勾选+dev-notes 阶段追加同笔提交；commit message 尾附 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- **断言示例化（CLAUDE.md 规约 4）**：本计划中的具体断言值（计数、分数、文案）均为意图示例；实施与实测不符时直接在该任务 commit 内修正断言并留一行注释说明，**不发起文档订正回合**。
- 可测行为 TDD；纯 Prompt 任务（意图 prompt）用标注样例跑验证替代单测步骤。
- 评审里程碑制：Task 1–4 完成后一次、Task 5–7 完成后一次、Task 8–9 完结评审一次；口径限行为/正确性/安全。
- 拍板项按 spec 推荐默认先行；用户改判时只改受影响任务，全局重排不做。

## Review Focus

1. **意图 JSON 畸形**：模型返回 markdown 围栏/缺字段/非法类别 → 不抛穿、按兜底出口走。→ Task 2 钉死。
2. **上游模型不支持 tool_calls 或空返回**：Agent 节点必须在 max_iterations 内收敛且不吞流。→ Task 1 冒烟 + Task 5 钉死。
3. **conversation_id=None（无 MySQL 降级路径）**：routes 既有降级语义下 Graph 仍需可跑完一轮（thread key 兜底）。→ Task 6 钉死。
4. **suggestions 帧序**：必须在全部 token 之后、`done` 之前到达；未知事件旧前端不得崩。→ Task 6+8 钉死。
5. **低置信池写失败不阻断兜底话术**（对齐 `_persist` 只 WARN 语义）。→ Task 4 钉死。

---

## Task 1: 裸循环对照 + 上游模型冒烟（祛魅热身）

**Files:**
- Create: `app/workflows/__init__.py`、`app/workflows/naive_agent_loop.py`
- Create: `tests/workflows/test_naive_agent_loop.py`（新建目录含 `__init__.py` 若仓内模式要求）
- Create: `tests/integration/test_ch05_model_smoke.py`（标 integration，默认 deselect，手动跑）

**Interfaces:**
- Consumes: `app/tools/registry.get_tools()`、`app/tools/executor.execute_tool`、`app/core/config.Settings`
- Produces: `async def naive_agent_turn(model, messages, *, ctx, max_iters: int = 6) -> NaiveLoopResult`；`NaiveLoopResult(text: str, steps: int, tool_calls: list[ToolOutcome])`

- [ ] **Step 1: 冒烟——真实上游模型 tool_calls/JSON 支持**（spec「开放核对项」销账）。脚本式一次性调用 `model.bind_tools(get_tools())` 发「订单 1001 的物流到哪了」，打印 `tool_calls` 是否出现、JSON 是否合法、响应模式是否支持流式中间帧。**若 tool_calls 恒空 → 停止问用户**（选型矛盾，D6）。结论追记 dev-notes。
- [ ] **Step 2: 写失败测试**（fake model：脚本化返回「第一轮 AIMessage(tool_calls=[query_order]) → 第二轮 AIMessage(tool_calls=[query_logistics]) → 第三轮纯文本」）：

```python
async def test_two_step_loop_feeds_results_back(fake_model, tool_ctx):
    res = await naive_agent_turn(fake_model, [HumanMessage("订单1001物流到哪")], ctx=tool_ctx)
    assert res.steps == 2                      # 断言示例化：以 fake 脚本为准
    assert [t.name for t in res.tool_calls] == ["query_order", "query_logistics"]
    assert "到了" in res.text                   # 第三轮收敛文本
```

- [ ] **Step 3: 跑红** `pytest tests/workflows/test_naive_agent_loop.py -v` → ImportError。
- [ ] **Step 4: 最小实现**——核心循环就是这个程度（看清「Agent 就是带工具的循环」）：

```python
async def naive_agent_turn(model, messages, *, ctx, max_iters=6):
    bound = model.bind_tools(get_tools())
    tool_calls_out, steps = [], 0
    for _ in range(max_iters):
        ai = await bound.ainvoke(messages)      # 教学版非流式;流式在 Task 5
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            return NaiveLoopResult(text=ai.content, steps=steps, tool_calls=tool_calls_out)
        for tc in ai.tool_calls:
            outcome = await execute_tool(tc["name"], tc["args"], tc["id"], ctx)
            tool_calls_out.append(outcome)
            messages.append(ToolMessage(content=json.dumps(outcome.result, ensure_ascii=False),
                                        tool_call_id=tc["id"]))
        steps += 1
    return NaiveLoopResult(text="（已达最大轮数）…" + _last_text(messages), steps=steps, tool_calls=tool_calls_out)
```

- [ ] **Step 5: 跑绿**，然后 `git add -A && git commit -m "feat(ch05): T1 裸循环教学对照+上游模型tool_calls冒烟(核对项销账)"`（含 dev-notes 追加段与计划勾选）。

## Task 2: 分流纯函数 + 意图识别 Prompt

**Files:**
- Create: `app/workflows/routing.py`、`app/prompts/intent.py`
- Create: `tests/workflows/test_routing.py`
- Create: `tests/samples/ch05_intent_qa.csv`（标注样例 ≥14 条：七类×2，含 2 条明显寒暄）

**Interfaces:**
- Consumes: 无（纯函数层）
- Produces: `INTENTS: tuple[str, ...]`（七类中文名）、`def route_for_intent(intent) -> Literal["knowledge","data","complaint","chitchat"]`、`def chitchat_fast_path(text: str) -> bool`、`def parse_intent_json(raw: str) -> str | None`、`INTENT_PROMPT: ChatPromptTemplate`（system 输出要求 `{"intent": "<七类之一>"}`）

- [ ] **Step 1: 写失败测试**（四出口映射全枚举 + 兜底 + 快路）：

```python
@pytest.mark.parametrize("intent,route", [
    ("商品咨询","knowledge"),("退款退货","knowledge"),
    ("物流","data"),("订单","data"),("售后","data"),
    ("投诉","complaint"),("闲聊","chitchat")])
def test_seven_intents_map_to_four_routes(intent, route):
    assert route_for_intent(intent) == route

def test_parse_intent_json_tolerates_code_fence():
    assert parse_intent_json('```json\n{"intent": "物流"}\n```') == "物流"

def test_parse_intent_json_rejects_unknown():
    assert parse_intent_json('{"intent": "转账"}') is None      # 非法类别→None→调用方走兜底

def test_fast_path_hits_greeting_misses_question():
    assert chitchat_fast_path("你好") and not chitchat_fast_path("你好，订单1001到哪了")
```

- [ ] **Step 2: 跑红→实现→跑绿**。`route_for_intent` 未知 intent 返回 `"knowledge"`（拍板默认：带检索+闸的防幻觉出口）；快路词表放模块常量 `GREETING_PATTERNS`（纯问候/寒暄短句精确匹配，含标点变体）。
- [ ] **Step 3: 标注样例跑验证**（Prompt 类替代 TDD 步骤）：脚本 `python -m app.tools.dev_scripts intent-eval`（无则临时脚本，不入库）对 CSV 逐条真实调用意图 prompt，命中 ≥12/14 视为过（示例阈值，可实测修正）；结果追记 dev-notes。
- [ ] **Step 4: 一任务一 commit**（同上格式，`feat(ch05): T2 四出口分流纯函数+意图识别prompt(样例12/14)`）。

## Task 3: Graph 骨架（State/节点/边/checkpointer，知识闸与 Agent 先直答版）

**Files:**
- Create: `app/workflows/state.py`、`app/workflows/graph.py`、`app/workflows/nodes.py`
- Create: `app/agents/__init__.py`
- Create: `tests/workflows/test_graph_topology.py`

**Interfaces:**
- Consumes: Task 2 全部产出；`app/rag/retriever.retrieve`（只读，D7）；langgraph 1.2.11 `StateGraph/add_conditional_edges/compile(checkpointer=InMemorySaver())`（官方文档已核对，dev-notes 晨间记录）
- Produces: `class ChatState(TypedDict)`：`messages: list`、`user_query: str`、`resolved_query: str`、`intent: str`、`route: str`、`evidence: list[dict]`、`gate_pass: bool`、`suggestions: list[dict]`、`answer_text: str`、`log: dict`；`def build_graph(settings, model) -> CompiledStateGraph`（模块级 `get_graph()` 单例入口）；节点函数 `coref_node / intent_node / knowledge_retrieve_node / confidence_gate_node / agent_node / complaint_node / chitchat_node / logging_node`

- [ ] **Step 1: 写失败测试**（fake model + fake retrieve，monkeypatch 注入；断言四出口各一条）：

```python
async def test_knowledge_route_forces_retrieval_before_answer(fake_env):
    out = await fake_env.graph.ainvoke(init_state("退款政策是什么"), config=thread_cfg(1))
    assert out["log"]["nodes"] == ["coref","intent","retrieve","gate","agent","logging"]  # 顺序示例化
    assert out["log"]["retrieve_hits"] >= 0   # 验收1 的单测面:检索节点被走到

async def test_chitchat_makes_zero_model_calls(fake_env):
    out = await fake_env.graph.ainvoke(init_state("你好"), config=thread_cfg(2))
    assert fake_env.model.calls == 0 and out["answer_text"] == CHITCHAT_FIXED

async def test_complaint_emits_two_unbound_suggestions(fake_env):
    out = await fake_env.graph.ainvoke(init_state("我要投诉"), config=thread_cfg(3))
    assert [s["action"] for s in out["suggestions"]] == ["transfer_human", "create_ticket"]
    assert fake_env.model.calls == 1  # 仅意图分类;投诉话术固定
```

- [ ] **Step 2: 跑红→实现→跑绿**。要点：
  - `coref_node`：`resolved_query = user_query` 原样透传（本章最简，D1/需求 6）。
  - `intent_node`：先 `chitchat_fast_path` 直判闲聊（零模型调用，D5）；否则 INTENT_PROMPT 单次调用 → `parse_intent_json`；None → 兜底 `"商品咨询"`→knowledge。
  - `confidence_gate_node`（本任务直答版）：`gate_pass = bool(res.chunks)`；Task 4 换阈值版。
  - `agent_node`（本任务直答版）：把证据+问题交 Task 1 `naive_agent_turn`；Task 5 换流式 ReAct 版。**接口签名两任务间不变**，图拓扑因此不再动。
  - `logging_node`：汇总 `log` dict 并 `logger.info("ch05 graph turn %s", json.dumps(..., ensure_ascii=False))`（验收 1 的日志面）。
  - `compile(checkpointer=InMemorySaver())`；spec 明示重启即丢（D3）。条件边 `add_conditional_edges("route", lambda s: s["route"], {...四出口...})`。
- [ ] **Step 3: checkpointer 跨轮测试**：同 thread 两轮 invoke，第二轮 State 可见第一轮消息。
- [ ] **Step 4: 一任务一 commit** `feat(ch05): T3 StateGraph骨架四出口+InMemorySaver跨轮`。

## Task 4: 置信度闸（阈值版）+ 低置信池落池

**Files:**
- Modify: `app/workflows/nodes.py`（替换 `confidence_gate_node` 直答版）
- Create: `tests/workflows/test_confidence_gate.py`

**Interfaces:**
- Consumes: `ScoredRow.score`（`app/rag/retriever.py:40`）、`Settings.retrieval_low_conf_threshold`（ch04 既有键，复用不新造）、`crud.add_low_confidence_question`（`app/db/crud.py:297`）、`app/services/refusals.py` 的 `REFUSAL_ANSWER/pool_low_confidence` 模式
- Produces: `def evidence_gate_verdict(chunks: list[ScoredRow], threshold: float) -> tuple[bool, float]`（纯函数）；闸节点行为：弱证据 → `answer_text=REFUSAL_ANSWER`、`suggestions=[transfer_human]`、跳过 agent

- [ ] **Step 1: 写失败测试**：纯函数三组（高分过/低分拦/空证据拦）+ 节点级两组（业务类旁路不进闸——拓扑已保证、断言 route；**Review Focus 5**：monkeypatch `add_low_confidence_question` 抛异常 → 兜底话术照常返回、仅 logger.warning）。
- [ ] **Step 2: 跑红→实现→跑绿**。阈值判据取 `max(重排得分)` 与命中数双条件（示例，实测可调）；落池 `reason="ch05_gate"`，**唯一写方**：本路径不经 query_faq 工具，与闸 1 不重叠（spec「置信度闸」节）。
- [ ] **Step 3: 一任务一 commit** `feat(ch05): T4 置信度闸阈值版+兜底落低置信池(写失败不阻断)`。

**▶ 里程碑评审 M1（Task 1–4）**：起一个 fresh reviewer 子代理审 `git diff master..HEAD` 的 1–4 任务面，口径=行为/正确性/安全；纸面问题不立 finding（CLAUDE.md 规约 5/6）。评审结论追记 dev-notes 一段。

## Task 5: ReAct 主力 Agent 节点（流式版替换）

**Files:**
- Create: `app/agents/react.py`；Modify: `app/workflows/nodes.py`（`agent_node` 换实现）、`app/core/config.py`（新增 `react_max_iterations: int = 6`、`react_token_budget: int = 8000`——拍板默认值）
- Create: `tests/agents/test_react_node.py`

**Interfaces:**
- Consumes: `get_tools()/execute_tool/ToolContext`；langgraph 流式（节点内 `astream` + 图 `astream(stream_mode=...)`——**动手前官方文档核对本版本流式透出 API**，一次预算，冲突即停问）
- Produces: `async def react_agent_stream(state, settings, model) -> AsyncIterator[AgentEvent]`，`AgentEvent = ("token", str) | ("tool_call", dict) | ("tool_result", dict) | ("done", {"steps": int, "suggestions": list})`——事件元组形状与 `stream_chat_with_tools` 的 ToolEvent 对齐，routes 层零改动复用（Task 6）

- [ ] **Step 1: 写失败测试**（fake model 脚本化三步，含「缺信息追问」样例：模型输出无 tool_calls 的提问文本 → 事件流只有 token+done、steps==0）：

```python
async def test_complex_question_walks_multiple_steps(fake_model_stream):
    events = [e async for e in react_agent_stream(state_with_evidence(), st, fake_model_stream)]
    assert sum(1 for k, _ in events if k == "tool_call") >= 2      # 验收5 单测面:ReAct>1步
    assert events[-1][0] == "done" and events[-1][1]["steps"] >= 2
```

- [ ] **Step 2: 跑红→实现→跑绿**：循环骨架同 Task 1 裸循环，加三件事——每轮 token 累计对 `react_token_budget` 熔断、超限走「已有信息收尾+建议可选项 [transfer_human]」路径、每步 logger.info（步数日志=验收 5 后端面）。**Review Focus 2**：fake model 恒返回带 tool_calls 直到超限 → 断言在 max_iterations 内收流不抛。
- [ ] **Step 3: 评估跑一遍**（真实模型标注样例 ≥5 条：验收 2/5 各题型）替代纯单测之外的那条腿，结果进 dev-notes。
- [ ] **Step 4: 一任务一 commit** `feat(ch05): T5 ReAct流式节点+迭代/token双熔断`。

## Task 6: SSE 接线（Graph 编排替换 + suggestions 帧 + 降级面）

**Files:**
- Modify: `app/api/routes.py`（`chat_stream` 编排调用点）、`app/schemas/chat.py`（`Suggestion`、`SuggestionsEvent`）、`app/workflows/graph.py`（`stream_graph_turn(...)` 适配层：图流 → ToolEvent 元组）
- Create: `tests/api/test_chat_stream_ch05.py`

**Interfaces:**
- Consumes: Task 5 事件元组、既有 persister 钩子（落库语义不动，`DBChatPersister` 原样）
- Produces: `POST /api/chat/stream` 新帧 `event: suggestions`，`data={"items":[{"action":"transfer_human|create_ticket","label":"转人工|建工单"}]}`（`model_dump(exclude_none=True)`）；`stream_chat_with_tools` 本体保留（拍板默认：替换接线、函数留作 ch04 回归基线）

- [ ] **Step 1: 写失败测试**（httpx ASGI + fake model，全链）：四出口帧序断言（**Review Focus 4**：`suggestions` 在末个 `token` 后、`[DONE]` 前）；**Review Focus 3**：`conversation_id=None` 无库降级跑完一轮、不发 `conversation` 帧；`token/done/error` 帧字节面回归（ch04 既有断言复用）。
- [ ] **Step 2: 跑红→实现→跑绿**。适配层要点：thread_id = f"conv-{conversation_id or uuid4()}"（降级路径不落 checkpoint 复用——明示在注释）。
- [ ] **Step 3: 一任务一 commit** `feat(ch05): T6 Graph接线/api/chat/stream+suggestions帧(降级/帧序钉死)`。

## Task 7: create_ticket 副作用窄移除 + 建工单 REST 端点

**Files:**
- Modify: `app/db/crud.py`（`create_ticket`，`app/db/crud.py:80`）、`app/api/routes.py`（`POST /api/tickets`）
- Create: `tests/db/test_create_ticket_decoupled.py`、`tests/api/test_tickets_endpoint.py`

**Interfaces:**
- Consumes: 既有 `next_ticket_no`/撞号重试/事务结构（全保留，D4）
- Produces: `POST /api/tickets {conversation_id, title, content}` → `TicketOut(ticket_no)`；**行为变更：建单后 `Conversation.status` 不变**

- [ ] **Step 1: 写失败测试**：真库 fixture（沿用 ch02 `bk.main`/engine 夹具模式）建单前后 status 快照断言不变；tickets 行/编号格式回归断言不变；端点 201 + 重复会话多单合法。
- [ ] **Step 2: 跑红→删那行副作用→跑绿**；全量回归 `pytest`（工具内 create_ticket 路径同步受益）。
- [ ] **Step 3: 一任务一 commit** `feat(ch05): T7 建单不再置已转人工+POST/api/tickets(前端建工单用)`。

## Task 8: 前端两按钮（转人工纯前端 / 建工单调端点）

**Files:**
- Modify: `static/index.html`（及其内联 JS/CSS；若脚本独立文件则同步）

**Interfaces:**
- Consumes: `suggestions` 帧（Task 6）、`POST /api/tickets`（Task 7）
- Produces: 两独立按钮组件——`transfer_human`：点击即消息流插入「已转接人工客服」+「您好，我是客服小猫，请问有什么可以帮您的」（纯前端，**零 fetch**）；`create_ticket`：点击确认后 fetch 端点、回填工单号气泡；按钮一次性（点击后置灰），不点继续发消息 = 正常对话（无任何隐藏动作）

- [ ] **Step 1: 实现 + 手工演练**（前端不可单测面）：浏览器跑验收 3 全路径三遍（都点/只点转人工/都不点）；截图或文字记录进 dev-notes。**若 index.html 有可抽出的纯逻辑（帧→按钮描述）则抽函数配 vitest 级断言——按仓内现状从简，不为此引前端测试框架**。
- [ ] **Step 2: 一任务一 commit** `feat(ch05-ui): 转人工/建工单两独立按钮+确认交互(验收3演练)`。

## Task 9: 端到端验收 + 完结交付

**Files:**
- Modify: `README.md`（ch05 节）、`dev-notes/ch05.md`（完结段）；Create: `tests/e2e/test_ch05_acceptance.py`（真实模型 integration 标记 + 演练脚本）

- [ ] **Step 1: 验收 1–5 逐条跑**（docker compose 起库 + 真模型）：每条 = 一条集成断言或一份演练记录（日志摘录含检索节点/ReAct 多步证据），任何一条过不了回对应任务修，不降口径。
- [ ] **Step 2: 全量测试** `pytest`（默认 deselect integration 口径与 ch04 一致）贴结果进 dev-notes。
- [ ] **Step 3: 完结交付**：功能演示命令（`uvicorn app.main:app` + 浏览器 `static/index.html` + 三条演示话术）、测试结果、dev-notes 路径；README ch05 节。
- [ ] **Step 4: 终批评审 M3**（fresh reviewer 全分支）→ 修行为/安全 finding → 一任务一 commit 语义下的收尾 commit `docs(ch05): 验收+README+dev-notes完结交付`。

**▶ 里程碑评审 M2（Task 5–7）** 在 Task 7 后、Task 8 前跑，同 M1 口径。

## 拍板清单（与计划同批下发，未答按推荐默认）

| # | 问题 | 推荐默认 |
|---|---|---|
| P1 | 置信度闸形态：纯阈值 vs 阈值+LLM 自评二级判 | 纯阈值（最简版，正式置信检查留可观测章） |
| P2 | 意图 JSON 解析失败兜底出口 | 重试 1 次→仍败归 knowledge（带检索+闸防幻觉） |
| P3 | ReAct 熔断参数 | `react_max_iterations=6` / `react_token_budget=8000`（新配置项） |
| P4 | SSE 新帧名 | `suggestions` |
| P5 | 单轮编排去留 | 接线换 Graph，`stream_chat_with_tools` 函数保留作 ch04 回归基线 |
| P6 | 执行方式 | Native（executing-plans，本会话直实施+里程碑 fresh reviewer），MVP=Task 1–9 一口气 |
