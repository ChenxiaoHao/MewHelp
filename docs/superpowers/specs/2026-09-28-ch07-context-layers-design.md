# ch07 · 会话上下文管理(三层分层 + 后台摘要 + 多会话侧栏)设计 spec

日期 2026-09-28;分支 ch07(自 master `9789bb1` 切出);需求原文 = 用户当日提示词(功能需求 1–7 + 验收 1–5)+ 两步 SQL 原文。

## 目标与非目标

**目标**:把第一版「简单裁剪」(coref/intent 末 6 条裸切、agent 全史 4000 预算 trim)升级成正经的会话内上下文管理:三层结构(原文/半压/梗概)+ 锚点边界 + 后台异步摘要 + 预算倒推 + 装配顺序固定 + 全程可观测 + 前端多会话侧栏。

**非目标(本章不做)**:语义检索捞历史、按主题重要度留关键事实、跨会话长期记忆、用户画像。

**范围红线**:只管当前会话;`token/done/error` 帧字节级红线不动;零新依赖(uv.lock 不动);ch04 遗留三文件(`app/rag/retriever.py`、`app/prompts/self_check.py`、`app/tools/executor.py`)不碰;cid=None 降级路径(无 DB)行为与现状一致。

## 拍板记录(2026-09-28,用户「全按推荐默认」)

- **P1 checkpoint 语义**:维持 `InMemorySaver`(零新依赖);「落盘」解读为 checkpoint 跨轮保留。切换/重启后线程为空时,从 messages 表**回填**近史(按层1预算取最近后缀)进 thread,锚点照常生效——DB 为权威,checkpoint 为加速器。
- **P2 工具结果落库**:保持现状落 messages 表(ch04 基线不动);「数条数看不见它涨」解读为条数≠体积,**触发判定一律按 token 折算**,层2 渲染时 tool 行=一行标识。
- **P3 MAX_AGENT_STEPS**:`max_agent_steps`(默认 6)**接管** ReAct 轮数上限(替代 `react_max_iterations` 的接线,旧键删除或别名,plan 定);`react_token_budget` 单轮熔断保留。**RERANK_TOP_K**:新设置默认 5,**只作预算面「证据注入条数」估算参数**,不改检索/注入行为(注入证据数仍由 `rerank_top_n` 决定)。
- **P4 MAX_OUTPUT_TOKENS**:只作预算预留(软口径),不下发 `ChatOpenAI(max_tokens=)`。
- **P5 MAX_USER_INPUT_TOKENS**:当前句估算超限 → schema 校验 **422**(新增行为)。
- **P6 X/Y**:`turns_to_keep` 默认 **15**、`steady_tokens_per_turn` 默认 **400**;demo 组下 X×Y=6000>5650,窗口面为 min 小值。默认值 plan 单测校准,以命中验收锚点三数为准。
- **P7 前端**:沿用 ch06 先例 = **Vibe Coding 例外**,不套 TDD(node --check 冒烟 + 用户浏览器验收),后端两接口照常 TDD。
- **P8 回载口径**:`GET /api/conversations/{id}/messages` 返回全部行含 tool 行(前端 user/assistant 渲染气泡、tool 行一行浅色标识);列表 = id 降序 + 首问预览(首条 user 截 40 字)+ 已摘要标记(`summary_upto_msg_id IS NOT NULL`),demo 规模不分页。

## 三层结构与边界语义

**按消息 id 划界,不搬数据**(语义即用户 SQL 注释):

| 区间 | 层 | 渲染形态 |
|---|---|---|
| `id ≤ summary_upto_msg_id` | 已进摘要 | 不再逐条进上下文,只以其梗概段呈现 |
| `summary_upto < id ≤ layer1_from_msg_id` | 层 2 | 半压:user 原话不动;assistant 只留开头 `assistant_head_chars`(默认 60)字;tool 行换一行标识 `[工具结果·{name}·≈{N} token 已折叠]` |
| `id > layer1_from_msg_id` | 层 1 | 原文,一个字不压(含近期 tool 结果全文) |

- 两锚点存 `conversations` 行;NULL 一律按 `0` 解读(新会话全史在层1)。
- **降级只挪 id**:层1 估算 > 层1预算 → `layer1_from_msg_id` 上移(一批最老的层1消息落入层2),日志 `层1 降级 {X}→{Y}`;批次边界对齐到 human 轮边界(不把一轮拆成半压+原文两态)。
- 层2 估算 > 层2预算 → 后台摘要把整批 `(summary_upto, layer1_from]` 压成新第 N 段,完成后 `summary_upto ← 旧 layer1_from`(「边界追到层1起点」),层2 清空重攒。
- 梗概**单表分段、一段一行、只追加不回炉**:旧梗概只作背景给模型看、不参与合并——一个事实一生只经历一次有损压缩。
- DDL:用户两步 SQL **原文照录**为 `db/init/07a_ch07_summary_projection.sql`(conversations +summary +summary_upto_msg_id)与 `db/init/07b_ch07_layers.sql`(+layer1_from_msg_id + `conversation_summaries` 表);docker init 按文件名升序执行,01→…→06→07a→07b 顺序合法(conversations 先于 ALTER)。dev 活库手工执行等价 ALTER,dev-notes 记日期。

## 预算模型(app/context/budget.py)

- **唯一估算器 `estimate_tokens()`**:委托 `count_tokens_approximately`(与 trim_messages、react 累计计数同源——「口径和预算一起校准」的落点)。CJK 折算因子不写死进 spec:T1 单测用固定锚点句实测其行为并断言,预算公式全部经该函数表达。
- `compute_budgets(settings)` 倒推:

  ```
  峰值   = max_agent_steps × tool_result_max_tokens
  固定   = S(人设+红线 system 渲染估算) + RERANK_TOP_K × 单条证据上限估算
           + 梗概注入预留 + 安全余量
  历史   = min( turns_to_keep × steady_tokens_per_turn,
                model_context_window − max_output_tokens − max_user_input_tokens
                − 峰值 − 固定 )
  层1   = floor(历史 × 0.7) − 1        层2 = ceil(历史 × 0.3)
  ```

- **验收锚点(单测锁死)**:demo 组 `MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5` → 滑窗 **5650、层1 3954、层2 1695**。各「固定/预留」分项的默认常数值以命中此三数为校准目标,plan 阶段定死并逐项注释来源。
- **启动自检**:应用启动(lifespan)算一遍;历史预算 ≤ 0 或小到连最短一轮往返都装不下 → `logger.error` 「上下文预算不足」告警,不阻断启动。
- 新 settings(全带默认值、不破坏既有构造):`model_context_window / max_output_tokens / max_user_input_tokens / max_agent_steps / tool_result_max_tokens / rerank_top_k / turns_to_keep / steady_tokens_per_turn / assistant_head_chars / summary_inject_tokens / safety_margin_tokens / history_view_messages(默认6)` + 分项校准常量。`history_token_budget` 保留给回填条数上限等旧语义或退役,plan 定。

## 上下文装配顺序(app/context/layers.py)

**model_ctx(主力 Agent)五段固定序**(需求 3 逐条对应):

1. `[System]` 人设 + 红线 + 工具定义(CUSTOMER_SERVICE_PROMPT 渲染;tools 走 `bind_tools` 参数位)——每轮逐字符恒定,上游前缀缓存可命中;
2. 层 2 半压消息(渲染后首条非 system 为 human,`start_on="human"` 语义保持);
3. 层 1 原文消息;
4. 当前用户句 HumanMessage;
5. `[HumanMessage]` 合并注入:「早前对话梗概 + 知识库证据 + 订单数据」合成**一条**挂当前句**之后**——旧形态(evidence/order_data 各作 SystemMessage 前置,react.py:60-67)作废;子项为空即省略,三项全空整条不发;梗概注入超预留按估算器截断**保尾部**(新段优先)。

- **保底 trim**:最终序列仍过一次 `trim_messages`(strategy=last、本估算器、include_system、绝不丢当前句——复用 `trim_history` 保底分支),只防极端超限,日常不触发动作。
- **history_ctx(指代消解/意图共用)**:摘要行(投影截断)+ 层1 末 `history_view_messages`(默认 6)条原文;每轮必构(含闲聊兜底轮),coref prompt 的 `{history}` 输入形态兼容现状。

## 后台异步摘要(app/context/summarizer.py + app/prompts/summary.py)

- **决策点 = 图新入口节点 `ctx`**(ch05 拓扑测试断言随之改,规4 允许):每轮 coref 之前,有 cid 时读锚 → 算层1降级(UPDATE id)→ 判层2 超预算 → 排任务。降级在本轮内同步落库,摘要在后台。
- 任务 = `asyncio.create_task`,**自开 DB session**(请求级 session 关流即还);每 cid 进程内 in-flight 集合防重入;单 worker 语义,多 worker 失效面同 ch06 pending 挂账,README 记一句。
- 流程:渲染批 → SUMMARIZE_PROMPT(旧梗概作背景段 + 本批半压渲染)→ 模型 → 清洗(去首尾空行、硬上限截断)→ **append** `conversation_summaries`(seq=当前+1,uk_conv_seq 兜并发)→ `summary` 投影列重拼 + `summary_upto_msg_id` 追边界 → 释放 in-flight。
- **失败 = WARN 一条 `summary failed` 即完**——不重试不抛穿,下轮超预算自然重触发(压缩是成本,丢了不致命,阻塞才致命)。
- **摘要提示词四铁律**(纯 Prompt 任务,eval 面验证,见测试策略):①只提炼事实与诉求(问过哪款/报过的订单号手机号/明确诉求/未解决问题);②对话里没有的一个字不许编;③寒暄闲聊不留;④几十~一两百字。旧梗概只作背景不重写。

## 持久化与数据访问

- ORM:`Conversation` +`summary(Text)` +`summary_upto_msg_id(BIGINT UNSIGNED)` +`layer1_from_msg_id(BIGINT UNSIGNED)`;新模型 `ConversationSummary`(与 07b DDL 逐列对齐)。
- crud 增:锚点读写、`append_summary_segment`、会话列表(首问预览 join)、消息回载查询。
- **接缝测试扩展**(ch06 模式):models 列 ⊆ db/init 并集——conversations 面(01 CREATE ∪ 05/06/07a/07b ALTER ADD COLUMN)+ conversation_summaries 面(07b CREATE);把「新环境重建后列缺失静默炸」变成单元 RED。

## 可观测性

- `app/main.py` 增 **FileHandler → log/app.log**(UTF-8 显式编码,Windows GBK 控制台红线;目录自建;`.gitignore` +`log/`)。
- grep 锚(验收 4 直接打勾):`model_ctx cid=…`(摘要全文+滑窗逐条+窗口条数+tokens≈)、`history_ctx cid=…`(摘要行+滑窗,**每轮必打**,不进 Agent 的轮也有)、`summary trigger 层2 约N token > 预算M`、`summary done 第N段 (x,y] 耗时…`、`summary skip` / `summary failed`、`层1 降级 X→Y`。生命周期五态全带 cid/边界。

## HTTP API(P8 口径)

- `GET /api/conversations`:`{items: [{id, created_at, preview, summarized}]}` id 降序,demo_user_id 面;引擎未初始化 503(同构现状)。
- `GET /api/conversations/{id}/messages`:全行升序(role/content/tool_calls 摘要面/created_at);会话不存在或不属于 demo 用户 → 404;引擎未初始化 503。
- 两接口**只读**;`MAX_USER_INPUT_TOKENS` 校验落在既有 `ChatRequest`(超限 422,估算器同源)。

## 前端多会话侧栏(Vibe Coding 例外,P7)

- 左侧栏像素风:「新对话」钮 + 会话列表(新在前、首问预览截 40 字、「已摘要」标记、当前项高亮);
- 点会话 → GET messages 回载渲染(user/assistant 气泡、tool 行一行浅色「🔧」标识)→ `conversationId`/本地消息数组整体换轨,续聊复用现有 send();
- 「新对话」= 置 `conversationId=null` + 清屏,旧会话留侧栏可切回;**未收到 conversation 帧前点旧会话/新对话的竞态与 ch06 同语义处理**;
- 侧栏加载失败:顶部一行错误提示,静默降级,聊天不受影响。

## 测试与验收策略

- **TDD 面**:budget 公式(含 5650/3954/1695 锚点 + 启动自检)、估算器校准断言、layers 三规则渲染 + 边界移动/降级批次对齐、ctx 节点行为、summarizer(fake 模型:成功/失败/防重入/不阻塞)、models/crud/接缝扩展、两 API、422、react 注入改形回归(旧 System 前置断言翻转)、图拓扑。
- **eval 面(纯 Prompt,P7 工作规约)**:`SUMMARIZE_PROMPT` → `tests/samples/ch07_summary_qa.csv` 标注样例(多轮批次 → expected_contains 订单号/诉求关键项 + 禁编造 spot-check)+ `evals/smoke_ch07.py` 真模型逐行打印(UTF-8 stdout 包装,Windows 红线)。
- **集成 e2e**(`tests/e2e/test_ch07_acceptance.py`,头部逐字沿用 ch05/06 共引擎 fixture):C1 demo env 全级联(够轮次 → `层1 降级` → `summary trigger` → `summary done 第N段` → 问「最开始那个订单后来怎么说」靠梗概答对 oid+诉求);C2 默认窗口 20 轮 **零降级零摘要**(装得下就不压);C3 摘要任务不阻塞当轮关流;C4 回载+续聊。验收 1 的「二十轮不爆不崩」由 C2+用户浏览器双保险。
- **基线**:338 单元 + 9 双章 e2e 全绿是每里程碑红线;ch05/06 受影响断言(拓扑/注入)在该任务 commit 内直改(规4)。

## API 核对结论(Context7,2026-09-28,每库一次)

- **LangChain**(`/websites/langchain_oss_python_langchain`):官方高阶 Summarization/ContextEditing Middleware 绑定 `create_agent` 架构,与本章裸 StateGraph+手写 ReAct 环不匹配,**不采用**;继续用 `langchain_core.messages.utils.trim_messages/count_tokens_approximately`(签名 ch01 已按 1.6.3 核对,版本锁定不动)。
- **LangGraph**(`/websites/reference_langchain_python_langgraph`):`add_messages` 用法、`compile(checkpointer=)`+`configurable.thread_id` 通道与现状一致,无 API 变更;回填方案不依赖新 checkpoint 后端。

## 完结交付(需求「工作要求 5」)

演示命令(`uv run uvicorn app.main:app --port 8000` + demo env 话术)、测试结果计数、dev-notes/ch07.md 路径;README ch07 节;多 worker 摘要失效一句挂账说明。
