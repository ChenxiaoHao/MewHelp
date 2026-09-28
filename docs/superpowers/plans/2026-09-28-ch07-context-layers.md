# ch07 会话上下文管理(三层分层+后台摘要+多会话侧栏)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「简单裁剪」升级为会话内三层上下文(原文层1/半压层2/分段梗概)+ 锚点降级 + 后台异步摘要 + 预算倒推装配 + model_ctx/history_ctx 可观测 + 前端多会话侧栏。

**Architecture:** 新模块 `app/context/`(budget 估算与预算、layers 边界与装配、summarizer 后台任务)横切在既有 LangGraph 裸图之前:新入口节点 `ctx` 每轮挪锚并排任务,coref/intent 吃 history_view,agent 吃 build_model_context 五段序;DB(conversations 锚点列 + conversation_summaries 段表)为权威,InMemorySaver checkpoint 为加速器(空线程回填)。

**Tech Stack:** uv 锁零新依赖 · langgraph 1.2.11 StateGraph/InMemorySaver/add_messages · langchain_core trim_messages(换自算 token_counter)· FastAPI SSE + 2 只读 GET · SQLAlchemy 2 async + MySQL(用户 SQL 原文)· 原生 JS 单页。

**Spec:** `docs/superpowers/specs/2026-09-28-ch07-context-layers-design.md`(拍板 P1–P8 全在其「拍板记录」节,逐条引用)

## Global Constraints

- 零新依赖:uv.lock 一字不动;`token/done/error` 帧字节红线;cid=None 降级路径行为与 ch06 现状一致。
- ch04 遗留三文件 `app/rag/retriever.py`、`app/prompts/self_check.py`、`app/tools/executor.py` 不碰不提交;提交永远精确列文件。
- 用户两步 SQL **原文照录**(含注释)为 `db/init/07a_ch07_summary_projection.sql`、`db/init/07b_ch07_layers.sql`;dev 活库手工等价 ALTER,dev-notes 记日期。
- 中文/数字标识符 ASCII-only 口径沿用(ch06 记忆);Windows GBK 红线:凡 print Unicode 的脚本一律 stdout TextIOWrapper;node --check 提取 JS 显式 UTF-8 写盘。
- 跑测试裸 `uv run pytest -q`(禁 env 前缀);基线 338 passed / 27 deselected,每里程碑只增不减。
- 一任务一 commit(功能+测试+勾选+dev-notes 追加同笔);纯 Prompt/数据面任务以标注样例+smoke 评估替代 TDD(工作要求 1);前端=P7 Vibe 例外。
- 预算三数锚点(demo env):`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5` → 滑窗 5650 / 层1 3954 / 层2 1695。
- Ruling(spec 偏差,已按需求 4 裁决):估算器**不委托** `count_tokens_approximately`(实测中文按 ~3.45 字/token 折,违反「中文按字数折」),T1 起本章自实现 CJK 感知口径,budget/装配 trim/ReAct 计数三处同源;spec 该段在 T1 commit 内同批对齐(规6)。

## Review Focus

评审只对行为/正确性/安全(规6)。本章「规格隐含但任务测试未必全覆盖」的输入面,逐条已在对应任务钉测:

1. **上游 400 风险**:层2 半压与回填的消息形态绝不能产生孤儿 ToolMessage / 带 tool_calls 却无回应的 AIMessage(降级批次形态唯一合法解=整条工具链折叠成一个 marker AIMessage)。→ Task 3/4/7。
2. **后台任务生命周期**:摘要 task 必须自开 session——请求级 session 关流即还;task 内异常绝不冒穿 SSE;重启/in-flight 竞态只 skip 不双写。→ Task 6/7。
3. **空线程语义**:重启/切换后 thread 空但 DB 有史 → 回填生效,coref 不误判首轮;cid=None 面 ctx 节点全 passthrough。→ Task 7。
4. **预算退化**:窗口调小/装不下 → 自检告警 + 保底 trim「绝不丢当前句」,聊天不断线;422 拒超长当前句而非 500。→ Task 1/4。
5. **不压为常态**:默认窗口 20 轮零降级零摘要(验收 3),触发判定全部按 token 折、条数不参与。→ Task 3/11。

---

### Task 1: 配置扩面 + 预算估算器 + 自检 + 输入 422

**Files:**
- Create: `app/context/__init__.py`(空文件)、`app/context/budget.py`
- Modify: `app/core/config.py`(ch07 设置块)、`app/main.py`(lifespan 自检)、`app/schemas/chat.py`(ChatRequest validator)
- Test: `tests/test_budget_ch07.py`、`tests/test_input_limit_ch07.py`
- 同笔:spec「预算模型」节估算器段向实现对齐一句(规6);dev-notes 阶段追加

**Interfaces:**
- Produces: `estimate_text(text)->int`、`estimate_msg(msg)->int`、`estimate_items(seq)->int`(str|BaseMessage 混收,每条 +4 开销);`Budgets(sliding, layer1, layer2, parts:dict)`;`compute_budgets(settings)->Budgets`;`selfcheck_budget(settings)->str|None`(None=正常,否则告警文案)
- Consumes: `settings.*`(本任务新增键)、`CUSTOMER_SERVICE_PROMPT`

- [x] **Step 1: 失败测试——三数锚点与默认口径**

```python
# tests/test_budget_ch07.py
from types import SimpleNamespace
from app.context.budget import compute_budgets, estimate_text, selfcheck_budget

def _demo_settings():
    return SimpleNamespace(model_context_window=18000, max_output_tokens=2000,
        max_user_input_tokens=2000, max_agent_steps=3, tool_result_max_tokens=1200,
        rerank_top_k=5, turns_to_keep=20, steady_tokens_per_turn=500,
        summary_inject_tokens=707, safety_margin_tokens=1000, chunk_size=500)

def test_demo_env_hits_triple():
    b = compute_budgets(_demo_settings())
    assert (b.sliding, b.layer1, b.layer2) == (5650, 3954, 1695)

def test_default_window_budget():
    s = _demo_settings()
    s.model_context_window, s.max_agent_steps = 32000, 6
    b = compute_budgets(s)
    assert (b.sliding, b.layer1, b.layer2) == (10000, 6999, 3000)  # want 面成小值

def test_cjk_one_char_one_token():
    assert estimate_text("汉" * 500) == 500          # 需求4:中文按字折
    assert estimate_text("a" * 100) == 25

def test_selfcheck_warns_when_tiny():
    s = _demo_settings(); s.model_context_window = 6000
    assert selfcheck_budget(s) is not None
```

`compute_budgets` 内 S 现测:`estimate_text(渲染人设 System 文本)`(实测 543;常数 707/1000 即以此为锚凑 4750,prompt 若被改此测试红=校准门,docstring 写明)。

- [x] **Step 2: 跑红 → 实现 budget.py + config 新键 → 跑绿**

config 新增(全带默认):`model_context_window: int = 32000`、`max_output_tokens: int = 2000`、`max_user_input_tokens: int = 2000`、`max_agent_steps: int = 6`、`tool_result_max_tokens: int = 1200`、`rerank_top_k: int = 5`(仅预算面,P3)、`turns_to_keep: int = 20`、`steady_tokens_per_turn: int = 500`、`assistant_head_chars: int = 60`、`summary_inject_tokens: int = 707`、`safety_margin_tokens: int = 1000`、`history_view_messages: int = 6`。

```python
# app/context/budget.py 核心式
def compute_budgets(settings) -> Budgets:
    peak = settings.max_agent_steps * settings.tool_result_max_tokens
    s = estimate_text(_system_text())                      # 543=人设渲染实测
    evidence = settings.rerank_top_k * settings.chunk_size # 单条上限=chunk_size 字
    fixed = s + evidence + settings.summary_inject_tokens + settings.safety_margin_tokens
    window_side = (settings.model_context_window - settings.max_output_tokens
                   - settings.max_user_input_tokens - peak - fixed)
    sliding = max(0, min(settings.turns_to_keep * settings.steady_tokens_per_turn, window_side))
    layer1 = math.floor(sliding * 0.7) - 1 if sliding >= 2 else 0
    layer2 = math.ceil(sliding * 0.3) if sliding else 0
    return Budgets(sliding, layer1, layer2, parts={...})
```

- [x] **Step 3: ChatRequest 当前句超 `max_user_input_tokens` → 422(P5)**

`tests/test_input_limit_ch07.py`:TestClient 对 `/api/chat/stream`(引擎未初始化面亦先过 schema)POST 2001+ 汉字当前句 → 422;正常句 → 非 422。validator 放 `ChatRequest`(model_validator,读 `get_settings()`,估 last message content)。main.py lifespan:`msg = selfcheck_budget(settings)` → `logger.error("上下文预算不足: %s", msg)`(不阻断)。

- [x] **Step 4: 全量绿 + 提交**

`uv run pytest -q` 全绿;spec 估算器段对齐;dev-notes 阶段1 追加(Ruling 记此);commit `feat(ch07-t1): 预算倒推估算器与三数锚点+自检+输入422`。

---

### Task 2: 数据面(用户 SQL 原文 + ORM + crud + 接缝扩展)

**Files:**
- Create: `db/init/07a_ch07_summary_projection.sql`、`db/init/07b_ch07_layers.sql`(**用户原文逐字**)
- Modify: `app/db/models.py`(Conversation+3 列、`ConversationSummary` 新模型)
- Modify: `app/db/crud.py`(锚点/段表/列表/回载查询)
- Test: `tests/test_db_init_seam_ch07.py`、`tests/test_crud_context_ch07.py`
- 运行侧:dev 活库执行两段 ALTER/CREATE(等价 SQL),`SHOW COLUMNS`/`SHOW TABLES` 取证进 dev-notes

**Interfaces:**
- Produces(crud):`get_conv_ctx(session, cid) -> (summary: str|None, upto: int, layer1_from: int)`(NULL→0);`set_layer1_from(session, cid, value)`;`append_summary_segment(session, cid, *, from_msg_id, upto_msg_id, content) -> int`(seq=MAX+1,返回 seq;撞 uk 重算一次);`set_summary_projection(session, cid, *, summary, upto_msg_id)`;`list_messages_after(session, cid, after_id, limit=500)`;`list_user_conversations(session, user_id, limit=50) -> list[(id, created_at, preview, summarized)]`(preview=首条 user 截 40 字;summarized=`summary_upto_msg_id IS NOT NULL`;id 降序)
- Consumes:07a/07b 列名口径(`summary`,`summary_upto_msg_id`,`layer1_from_msg_id`,`conversation_summaries`)

- [x] **Step 1: 失败接缝测试**——解析 `db/init/*.sql` 的 `CREATE TABLE conversations|conversation_summaries` 列 + `ALTER TABLE conversations ADD COLUMN ...`(跨行正则,ch06 接缝测试同款「移走 07 文件演示真 RED、放回演示 GREEN」双证);断言 `Conversation`/`ConversationSummary` 模型列 ⊆ DDL 并集。
- [x] **Step 2: 落 07a/07b 原文 + 模型三列 + `ConversationSummary`**

```python
class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"
    id: Mapped[int] = _pk()
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True),
        ForeignKey("conversations.id"), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(nullable=False)
    from_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
```

- [x] **Step 3: crud 六函数 + test_crud_context_ch07**(复用既有 `db`/MySQL 测试夹具,integration 标记的沿用现配置;单元面 FakeSession 断不到就整组挂 integration 文件跑真库)。锚点 UPDATE 用 `update(Conversation)` 语句直写。
- [x] **Step 4: dev 活库 apply + 取证;全量绿;提交** `feat(ch07-t2): 07a/07b 用户SQL原文入库+锚点/段表 ORM 与 crud+接缝扩展`。

---

### Task 3: 层渲染与降级(半压三规则 + 批次对齐 + ContextStore)

**Files:**
- Create: `app/context/layers.py`(本任务只放渲染/切割纯函数 + `ContextStore` 读侧)
- Test: `tests/test_layers_ch07.py`

**Interfaces:**
- Produces:
  - `render_layer2(rows, settings) -> list[BaseMessage]`:扫描 messages 行(role/content/tool_calls)——`user`→HumanMessage 原样;`assistant` 无 tool_calls→AIMessage 前 `assistant_head_chars` 字+「…」;`assistant` 带 tool_calls 及其后连续 `tool` 行→**整链折叠为一个 AIMessage**,content=`[工具结果·{name}·≈{estimate_tokens(结果全文)} token 已折叠]`(链首 assistant 若有文本则保留其头部再附折叠行);**绝不产出 ToolMessage/带 tool_calls 的 AIMessage**(Review Focus 1)
  - `pick_degrade_cut(layer1_rows, budget, est) -> int|None`:从最新往老累计估算,首个超预算处**向新取整到 human 边界**(切割点必须是保留段首条 human,保留段必 ≤ 预算;向老对齐会死循环——M1 评审 Important 裁决,T4 实测翻转,规4 同笔改措辞);连一整轮都装不下→整批降级交层2 摘要
  - `ContextStore(session_factory, conversation_id, settings)`:`async load_ctx()->(summary, upto, layer1_from)`、`async fetch_all_rows()`、`async fetch_layer1()`、`async set_layer1_from(v)`
- Consumes:crud(Task 2 同名)、estimate_items(Task 1)

- [x] **Step 1: 失败测试**——三规则逐条;工具链折叠形态断言 `not any(isinstance(m, ToolMessage) ...)` 且无 tool_calls;cut 对齐 human 边界(批次不跨轮、保留段首必 human);全绿前不写实现。
- [x] **Step 2: 实现 + 绿**。渲染函数纯,ContextStore 薄包 crud。
- [x] **Step 3: 全量绿;提交** `feat(ch07-t3): 层2半压三规则(工具链整体折叠合法形态)+降级批次对齐human轮界+ContextStore读侧`。

---

### Task 4: 五段装配 + model_ctx/history_ctx 日志 + 文件 handler

**Files:**
- Modify: `app/context/layers.py`(build_model_context / build_history_view / 降级动作)
- Modify: `app/main.py`(FileHandler)、`.gitignore`(+`log/`)
- Test: `tests/test_context_assembly_ch07.py`

**Interfaces:**
- Produces:
  - `build_model_context(store, *, evidence, order_data, current_human, settings) -> list[BaseMessage]`:段1 System(人设渲染)→ 段2 `render_layer2(层2行集)` → 段3 层1 原文(行集→messages,user/assistant 类映射;层1 内 tool 行按现 react 输入语义还原 ToolMessage 合法链)→ 段4 当前 HumanMessage → 段5 合并注入(梗概投影[按 summary_inject_tokens 截尾保新]+「知识库证据:…」+「订单数据:…」拼**一条 HumanMessage**,全空整段省略);末过一次 `trim_messages(strategy="last", token_counter=estimate_items, max_tokens=sliding+s+注入上界预留, include_system=True)` 且复用 `trim_history`「绝不丢当前句」保底
  - `async degrade_if_needed(store, settings) -> tuple[int,int]|None`(挪锚 UPDATE + `logger.info("层1 降级 %d→%d", ...)`);`async build_history_view(store, settings) -> str`(摘要行+层1末 `history_view_messages` 条,`logger.info("history_ctx cid=%s\n%s", ...)`,每轮必打);`log_model_ctx(cid, msgs, tokens)`(「model_ctx cid=… segs=… msgs=… tokens≈…」+逐条正文)
- Consumes:layers T3 函数、crud 锚点、budget

- [x] **Step 1: 失败测试**(caplog):五段顺序逐类断言(段2/3 间无 System;段5 在段4 **之后**且唯一);段5 三子项省并;`层1 降级` 日志与锚 UPDATE 同现;FileHandler 存在且 `encoding="utf-8"`(main.py 导入后 root.handlers 查)。
- [x] **Step 2: 实现 + 全量绿**(ch01 legacy `build_messages` 路径不动)。提交 `feat(ch07-t4): 五段装配序+保底trim+model_ctx/history_ctx每轮留痕+log/app.log(UTF-8)`。

---

### Task 5: 摘要提示词 + 标注样例 smoke(纯 Prompt,eval 代 TDD)

**Files:**
- Create: `app/prompts/summary.py`(`SUMMARIZE_PROMPT`: system 四铁律+「旧梗概仅作背景不重写、本批内容独立成段」; human=`{background}` + `{batch}`)
- Create: `tests/samples/ch07_summary_qa.csv`(≥6 行:`background`, `batch`, `expected_contains`(any-of,`|` 分隔:订单号/手机号/诉求词), `max_chars=200`)
- Create: `evals/smoke_ch07.py`(载 CSV→真模型直调→逐行打印+包含/长度判定,恒 exit 0;stdout TextIOWrapper UTF-8,Windows 红线)
- Modify: `tests/test_prompts_contract_ch07.py`(新文件):prompt 形状契约在场(四铁律关键词、占位符齐)——质量断言不在此,在 smoke

**验收**:smoke 全行通过才算过;miss → 只动 few-shot/铁律措辞,不动代码(ch06 纪律);结论记 dev-notes。

- [x] Step 1 契约测试 RED→GREEN;Step 2 样例 CSV(自造 6 段多轮批次,含「最开始那个订单 1001 退不了要转人工」伏笔——T11 C1 要考);Step 3 真模型 smoke,记录准确率与修 prompt 轮次;Step 4 提交 `feat(ch07-t5): 摘要四铁律prompt+标注样例smoke评估`。

---

### Task 6: 后台异步摘要服务(非阻塞/防重入/失败只 WARN)

**Files:**
- Create: `app/context/summarizer.py`
- Test: `tests/test_summarizer_ch07.py`

**Interfaces:**
- Produces:`schedule_summary(session_factory, cid, settings, model) -> bool`(in-flight 集合防重入→False+`summary skip`;True=已 `asyncio.create_task`);`async run_summary(...)`(自开 session:load→渲染批(层2半压)→SUMMARIZE_PROMPT(旧梗概背景)→ainvoke→清洗(去空行;>400 字截)→`append_summary_segment`(from=upto+1 边界推导, upto=旧 layer1_from)→`set_summary_projection`(全段按 seq 重拼, upto 追边界)→`logger.info("summary done 第%d段 (%d,%d] 耗时%.2fs", ...)`;except:`logger.warning("summary failed cid=%s", exc)`,finally 释放 in-flight)
- Consumes:crud、render_layer2、SUMMARIZE_PROMPT

- [x] **Step 1: 失败测试**:fake model 返回固定梗概→断 append/projection 参数与 done 日志;model 抛→仅 WARN、锚不变、in-flight 空;任务未落时二次 schedule→False+skip 日志;`schedule_summary` **同步返回不等任务**(`await asyncio.sleep(0)` 后任务仍未完成即证不阻塞)。
- [x] Step 2 实现+绿;Step 3 提交 `feat(ch07-t6): 后台摘要任务(自开session/防重入/段追加不回炉/边界追至层1起点/失败WARN不重试)`。

---

### Task 7: 图接线(ctx 入口 + 回填 + coref/agent 换供 + 拓扑测试更新)

**Files:**
- Modify: `app/workflows/nodes.py`(`make_ctx_node`;coref/agent 换用 store 供史/装配)、`app/workflows/graph.py`(入口 `ctx`→coref;`stream_graph_turn` 空线程回填)、`app/api/routes.py`(有 session 时构 `ContextStore` 入 configurable)
- Test: `tests/test_ctx_node_ch07.py`、`tests/test_refill_ch07.py`;Modify `tests/test_graph_topology_ch05.py`(入口断言含 ctx——规4 本任务内直改)

**Interfaces:**
- Produces:节点 `ctx`(passthrough:无 cid/无 store;有 store:`degrade_if_needed`→层2估算超预算→`schedule_summary`);回填 `_refill_input(graph, cfg, store, settings)`:thread 空(`aget_state` 无 messages)且 DB 有 user/assistant 行 → 取层1原文近 `history_view_messages*2` 条前置入图 input(只 Human/AI 两类,无 tool 链;Review Focus 3)
- Consumes:T3/T4/T6 全部公开名

- [x] Step 1 失败测试:ctx 在无 store 轮零副作用;有 store 轮降级+触发被调(mock store);coref 的 history 来自 view(含摘要行)且 `history_ctx` 每轮出(含 chitchat 轮——graph 级);agent 收到五段形 msgs(mock react 断注入点);回填后 coref 不误判首轮。
- [x] Step 2 实现;拓扑测试更新;**双章 e2e 面回归**(A1–A5b/B1–B3 真跑,任何行为变化=本任务 bug);全量绿;提交 `feat(ch07-t7): ctx入口节点+空线程回填+coref/agent切换供+ch05拓扑随迁`。

---

### Task 8: react 注入改形 + 轮数上限接管 + 计数同源

**Files:**
- Modify: `app/agents/react.py`(删 evidence/order_data System 前置两块=装配段5 已代;`while steps < settings.max_agent_steps`;`tokens_used += estimate_msg`;日志行键名),`app/core/config.py`(删 `react_max_iterations`——P3 接管,不留别名)
- Test: `tests/test_react_node_ch05.py` 断言翻转改形 + 构造参数字典名替换;`tests/test_confidence_gate_ch05.py`、`tests/test_graph_topology_ch05.py`、`tests/test_orders_sse_ch06.py`、`tests/test_refund_flow_ch06.py` 的 Settings(...) 键改名(规4 随本任务 commit)

- [x] Step 1 失败测试:传入含段5 的 msgs → react 不再产任何 evidence System;`max_agent_steps=1` 超限收流行为与旧 5 轮版同构(落地为 RF2 测换名 st(max_agent_steps=5) 同构形——轮数参数化后 1/5 行为同一,见 ledger Ruling);估算器换源后 token 熔断阈值断言重校。
- [x] Step 2 实现+全量绿(338+新增,键改名波及逐处核对);提交 `feat(ch07-t8): ReAct去System前置(段5装配代)+max_agent_steps接管轮数上限+计数同源估算器`。

---

### Task 9: 只读会话 API×2

**Files:**
- Create: `app/schemas/conversation.py`(`ConversationItem{id, created_at, preview, summarized}`、`ConversationList{items}`、`MessageItem{id, role, content, created_at}`)
- Modify: `app/api/routes.py`(GET /api/conversations、GET /api/conversations/{id}/messages;引擎 None→503;未知/非属主 cid→404,P8)
- Test: `tests/test_api_conversations_ch07.py`(dependency_overrides FakeSession 模式,ch06 退款 API 同构)

- [x] Step 1 RED→GREEN;Step 2 全量绿;提交 `feat(ch07-t9): GET conversations 列表/消息回载只读API(预览40字/已摘要标记/404/503)`。

---

### Task 10: 前端多会话侧栏(Vibe Coding 例外,P7)

**Files:**
- Modify: `static/index.html`(侧栏 CSS/渲染/切换回载/新对话;失败静默降级一行提示)

**行为口径**(spec 前端节):列表新在前+预览+「已摘要」+高亮当前;点选→GET messages→user/assistant 气泡、tool 行与带 tool_calls 的 assistant 行渲染一行浅色「🔧」标识→conversationId 与本地数组整体换轨续聊;「新对话」置 null 清屏,旧会话留栏;回载失败顶栏提示不挡聊天。

- [x] Step 1 写实现;Step 2 冒烟=提取 <script> 显式 UTF-8 落盘后 `node --check`(Windows 假 OK 红线);Step 3 手测清单并入 T11 浏览器验收;提交 `feat(ch07-t10-ui): 会话侧栏(列表/切换回载/新对话/静默降级)`。

---

### Task 11: 集成 e2e C1–C4 + README/dev-notes 完结交付

**Files:**
- Create: `tests/e2e/test_ch07_acceptance.py`(头部逐字沿用 ch05/06 共引擎 fixture;integration 标记)
- Modify: `README.md`(ch07 节:预算模型/演示 env/侧栏/多 worker 一句挂账)、`dev-notes/ch07.md`(终审批+完结)

**C1 demo env 级联**:Settings(18000 组)真模型真库,话术≈18–22 轮(含一次工具轮)→断言日志/库面出现 `层1 降级`、`summary trigger`、`conversation_summaries` ≥1 段、`summary_upto` 追至旧 layer1_from;末轮问「最开始那个订单后来怎么说」→回答含该 oid(从首问话术取)且不 error 帧。
**C2 默认窗 20 轮**:无降级无摘要(锚不动、段表 0 行)——验收 3。
**C3 非阻塞**:trigger 当轮 done 帧正常关流(任务未完成不影响帧序)。
**C4 回载续聊**:API 面新 store 重启语义(清 checkpointer)→回填轮 coref 日志非 passthrough。

- [ ] Step 1–2 e2e 双章全绿(9+4);Step 3 README+dev-notes 完结+演示命令;Step 4 提交 `test(ch07-t11): 级联/零降级/非阻塞/回填续聊 e2e + README/dev-notes 完结交付`;Step 5 终审包 → 终审批 → finishing 菜单。

---

## 里程碑与评审

M1=T1–T3(预算/数据/层),M2=T4–T6(装配/摘要 prompt/后台),M3=T7–T9(接线/react/API),T10–T11+终审(全分支 fresh reviewer,opus)。评审仅行为/正确性/安全(规6);Minor deferred。
