# ch05 Workflow + Agent 设计 Spec

> **状态**：brainstorm 定稿（重建版）。晨间会话声称的原 spec 从未落盘（幻影产物，取证见 dev-notes/ch05.md「提速规则固化与幻影产物处置」节），本稿依据 dev-notes 逐字决策记录 + 用户正式需求提示词重建，决策内容与晨间定稿一致。
> **需求原文**：用户提示词全文见 dev-notes 与转录留痕；本 spec 是其设计定稿。
> **实施计划**：`docs/superpowers/plans/2026-09-26-ch05-workflow-agent.md`
> **引用规约**：本文与 plan 互相引用一律用节名，不用编号（CLAUDE.md 规约 3）。

## 目标与非目标

**目标**：把 ch01–ch04 的对话/检索/工具能力升级为生产级架构——LangGraph StateGraph 确定性编排做骨架，主力 Agent（ReAct）是骨架中的核心节点；先手写裸循环做祛魅教学对照，再用 LangGraph 重构。

**非目标（本章明示不做）**：指代消解与意图识别的正式版（本章最简实现）；正式置信度检查（留给可观测章）；上下文管理策略升级；MCP 接入；数据飞轮入库；服务重启后 LangGraph 状态恢复；新业务工具；升级任何依赖版本。

## 已定稿决策日志（用户原话逐字，dev-notes 晨间记录）

| # | 决策 | 用户口径 |
|---|---|---|
| D1 | 架构路径 | 「Workflow 确定性编排做骨架，主力 Agent 作为骨架里的核心节点」；三路径比较后选「裸循环教学对照 + 显式 StateGraph 节点/边」 |
| D2 | 分流与按钮 | 以老师的四出口分流与按钮行为为准；「投诉、转人工、建工单三者不能混在一起」；Agent 自动建单方案作废 |
| D3 | Checkpointer | 「ch05 本章不要求服务重启后恢复 LangGraph 会话状态」「请使用 InMemorySaver 作为本章 checkpointer，只保证进程内跨轮状态保存」 |
| D4 | 建单副作用 | 「允许仅移除建单时改会话状态的副作用：tickets 落库机制不变，两动作在数据库语义上也独立」——唯一业务语义改动 |
| D5 | 闲聊口径 | 「闲聊回复零额外调用：分类仍可调用模型，明显寒暄走规则快路」——固定回复阶段零模型调用 |
| D6 | API 核对 | Context7 故障后授权切官方文档；不升级依赖；文档与版本冲突时停止确认 |
| D7 | 保留改动 | `app/prompts/self_check.py`、`app/rag/retriever.py`、`app/tools/executor.py` 三处未提交改动保留原样，只复用接口不修改 |
| D8 | 流程 | 「一口气到实现」；提速规则采纳并固化（CLAUDE.md 八条） |

## 总体架构与图拓扑

```
入口(routes.py /api/chat/stream)
  └─ StateGraph (thread_id = conversation_id, checkpointer = InMemorySaver)
       指代消解(透传) → 意图识别(JSON七类+寒暄快路) → 分流(纯代码四出口)
          ├─ 知识类  → 知识检索(retriever.retrieve) → 置信度闸 → 主力Agent → 日志记录 → END
          │                                        └(证据弱)→ 兜底话术+落低置信池 → 日志记录 → END
          ├─ 业务数据类 ─────────────────────────→ 主力Agent(自调工具) → 日志记录 → END
          ├─ 投诉    → 安抚话术+建议可选项帧 → 日志记录 → END   (不进Agent)
          └─ 闲聊    → 固定话术(零模型调用) → 日志记录 → END
```

- **State 贯穿**：LangGraph State 携带本轮输入、消息历史（checkpointer 按 thread 累积）、意图、路由、检索证据、闸结论、建议可选项、日志字段。
- **持久化边界**：LangGraph State 仅进程内（InMemorySaver，重启即丢，spec 明示）；MySQL 消息/工单落库继续沿用现有 SQLAlchemy 机制，与 checkpoint 无关。
- **裸循环对照**：`app/workflows/naive_agent_loop.py`——无框架手写「调 LLM→有 tool_calls 执行并喂回→无则收敛」循环，教学与测试对照用，不接聊天入口；生产路径是 Graph。

## 七意图四出口映射（写死代码，纯函数）

| 意图 | 出口 | 出口行为 |
|---|---|---|
| 商品咨询 | 知识类 | 强制检索→置信度闸→Agent（携证据） |
| 退款退货 | 知识类 | 同上，检索政策类证据；Agent 仍可自调订单工具 |
| 物流 | 业务数据类 | 不预检索，直接进 Agent |
| 订单 | 业务数据类 | 同上 |
| 售后 | 业务数据类 | 同上 |
| 投诉 | 投诉 | 不进 Agent：安抚话术 + 建议可选项（转人工/建工单分开给） |
| 闲聊 | 闲聊 | 固定话术直接返回 |

意图识别（本章最简版）：单个简单 prompt 输出 JSON 七类之一；明显寒暄（规则快路，问候语词表级）在意图节点直判闲聊，保证固定回复零额外模型调用（D5）。解析失败的兜底出口 → **拍板项**。

## 置信度闸（最简版）

- **位置**：知识类检索之后、进 Agent 之前（前置硬约束：Agent 答复流式吐出，答完再判就晚了）。
- **判据**：复用 ch04 检索得分/证据覆盖度做阈值判断（`RetrieveResult` 的 ScoredRow 分数；阈值取 Settings 配置）。LLM 自评是否作为二级判 → **拍板项**；正式版置信度检查留给可观测章。
- **证据弱**：不进 Agent，返回固定兜底话术；同时用现有 `crud.add_low_confidence_question`（`app/db/crud.py:297`）记录问题留给后续数据飞轮，本章不建新表。**落池唯一写方原则**：知识类预检索走 `retrieve()` 不经 query_faq 工具，故闸节点是该路径唯一写方，与工具内闸 1（query_faq 命中低置信落池）互不重叠。
- **旁路**：业务数据类无检索证据，不走闸。

## 主力 Agent（ReAct）

- 工具 = ch02 Function Calling 注册表 `app/tools/registry.get_tools()`（查订单/查物流/FAQ/工单等五件），本章不新写业务工具；执行统一走 `app/tools/executor.execute_tool`（含超时/重试/摘要）。
- 循环：思考→工具调用→结果喂回→再思考，直到收敛出答案；简单问题一次调用即收敛，复杂问题多步（验收 5）。
- **停止条件与 token 控制**：最大迭代轮数 + 本轮 token 预算上限，超限输出已有信息 + 建议可选项；具体参数值 → **拍板项**。
- **缺信息追问**：Agent 直接输出追问文本并结束本轮；用户下一条消息经 checkpointer 恢复上下文继续——不设专用追问接口。
- **建议可选项**：投诉出口必给（转人工+建工单两个）；Agent 判断合适时也可随回复附带（可单个可双个）；后端只发建议，绝不代执行。

## SSE 契约与前端

- 保留 ch01/ch02/ch04 既有帧不动：`conversation` / `token` / `tool_call` / `tool_result`（含 citations、exclude_none 兼容旧帧）/ `done` / `error`——token/done/error 逐字符红线延续。
- 新增「建议可选项」下发帧（帧名 → **拍板项**）：payload 含 `[{action: transfer_human|create_ticket, label}]`。
- 前端 `static/index.html`：收到可选项帧渲染**两个互相独立**的按钮；点「转人工」= 纯前端展示「已转接人工客服」+ 小猫问候「您好，我是客服小猫，请问有什么可以帮您的」，不发后端请求；点「建工单」= 调用轻量 REST 端点（走 create_ticket 工具本体，不经过模型对话轮）写 `tickets` 表并给确认反馈；两个都不点、继续发消息 = 普通对话，无任何动作。

## 数据库语义改动（唯一一处）

`app/db/crud.py:create_ticket()`：移除 `Conversation.status = '已转人工'` 副作用（D4）。保留建票编号、事务、撞号重试、`tickets` 行、工具调用路径。回归断言：建工单后 conversation.status 不变；转人工不再有后端写库路径（前端模拟）。

## 模块复用清单（已核对签名）

| 复用件 | 位置 | 用途 |
|---|---|---|
| 聊天入口/SSE 帧 | `app/api/routes.py:51 chat_stream` | 编排调用点从单轮换成 Graph，帧协议延续 |
| 单轮编排 | `app/services/tool_chat_service.py:73 stream_chat_with_tools` | ch04 回归基线与件件来源（persist 钩子/事件元组）；是否保留共存 → **拍板项** |
| 检索 | `app/rag/retriever.py:90 retrieve(strategy="hybrid_rerank")` | 知识类预检索（只读接口，D7） |
| LLM 自评 | `app/services/self_check.py:33 evaluate_evidence` | 闸的可选二级判（若拍板启用） |
| 低置信池 | `app/db/crud.py:297 add_low_confidence_question` | 闸拒绝记录，零新表 |
| 工具注册/执行 | `app/tools/registry.get_tools` / `executor.execute_tool` | Agent 工具面（executor 超时改动保留，D7） |
| 落库 | `app/db/crud.py` 会话/消息/工单全套 | 边界不变 |

新增包：`app/workflows/`、`app/agents/`（各含 `__init__.py`）。

## API 与版本核对记录

- Context7 事故：晨间会话 `query_docs` 对 LangGraph/LangChain/SQLAlchemy/FastAPI 持续 `TypeError: fetch failed`，经用户授权降级为**官方文档**核对（dev-notes 晨间节；今后按 CLAUDE.md 规约 8 执行每库一次预算）。
- 已只读核对（官方文档 × 实装版本兼容）：LangGraph 1.2.11 `StateGraph` / `add_conditional_edges` / `compile(checkpointer=...)` / `astream(version='v2')`；LangChain `bind_tools` / `ToolMessage`；FastAPI 0.141.1 SSE；SQLAlchemy 2.0.54 async_sessionmaker。
- `uv.lock` 锁定（本会话实测）：langgraph 1.2.11、langchain 1.4.2、langchain-core 1.6.3、langgraph-checkpoint 4.2.0、fastapi 0.141.1、sqlalchemy 2.0.54、pymilvus 3.0.2。**本章零新增依赖、零版本升级**。
- 开放核对项：真实上游模型对 tool_calls 与 JSON 输出的支持 → 实施首任务冒烟验证（D6 遗留）。

## 验收标准 → 责任节点映射

| # | 验收（需求逐字） | 责任 |
|---|---|---|
| 1 | 政策类问题，日志可见强制检索节点被走到 | 知识检索节点日志（日志记录节）+ Graph 单测 |
| 2 | 「订单 1001 的物流到哪了」Agent 自调工具 | 分流（业务数据类）+ Agent 节点 |
| 3 | 「我要投诉」出两个独立按钮；转人工纯前端；建工单才写 tickets；不点无动作 | 投诉出口 + SSE 可选项帧 + 前端 + create_ticket 改动 |
| 4 | 闲聊拿到固定话术 | 寒暄快路 + 闲聊出口 |
| 5 | 先查订单再查物流的复杂问题 ReAct 多步 | Agent 循环 + 日志可见步数 |

完结交付：功能演示命令、测试结果、dev-notes 路径（需求工作要求 5）。

## 全局约束（plan 继承）

1. 一任务一 commit（代码+勾选+dev-notes 追加同笔）；产物声明先验证落盘（CLAUDE.md）。
2. 三处 ch04 保留改动不触碰；不升级依赖；token/done/error 帧逐字符红线。
3. 可测行为 TDD；纯 Prompt/数据类任务以标注样例或评估集跑一遍替代。
4. 评审里程碑制（3–4 任务一批），口径限行为/正确性/安全。
5. 拍板清单未答项按推荐默认执行并在本文回记一笔。
