# ch06 正式版分流器设计（指代消解 / Query 改写扩写 / 意图四件套 / 退款确定性子流程 / 槽位选择器）

**日期** 2026-09-27 ｜ **上游** ch05（merge `7a9d364`）｜ **状态** brainstorm 定稿，待用户评审
**基线** master `7a9d364`，`PYTHONPATH=. uv run pytest -q` → 280 passed, 24 deselected

ch05 把分流器做成了占位版（coref 原样透传、意图单 prompt 无置信度、七意图粗归四出口）。
本章把它换成正式版——这是 Workflow 架构里最关键的节点：进模型之前，问题先被修好、
判准、按需扩开；退款/售后这类高风险意图不再赌模型自选工具，改走确定性子流程。

## 目标与非目标

**目标**（需求点名）
1. 指代消解：LLM+历史把「它能退吗」补全为自包含问题；已完整的问题原样透传，不强行改写。
2. Query 改写：口语→标准问法，与指代消解同一节点一次 LLM 调用完成。
3. Query 扩写：一个问题→几条侧重不同的检索问法（强制 JSON，字段就 `queries` 一个数组），
   多路检索+去重合并；只对高频低容错场景（退款退货、售后）开启，简单 FAQ 不扩写；
   扩写只发生在检索侧，库里知识只留一份。
4. 意图识别走 Prompt 路线四件套：枚举选项（七意图+「其他」）、强制 JSON（恰好
   `intent`+`confidence` 两字段）、边界 few-shot、「其他」兜底类（拿不准归它，别硬塞业务意图）。
5. 分流精化：退款退货+售后走确定性子流程——先取订单数据，再 Query 扩写+政策条款强制检索，
   最后只有「这一单能不能退」进主 Agent 判定。
6. 槽位：缺订单号模型不许猜；执行阶段弹订单选择器点选回填；退款原因不问，
   提表单时用户从固定类别里选；澄清追问留在主 Agent，意图只决定「要干什么」。
7. 前端配套：聊天流内可点订单卡片（点选自动回填、流程继续）+ 退款小表单（原因固定下拉）。

**非目标**（本章不做）
- 微调小模型 / BERT 意图分类器（Prompt 路线）。
- 跨会话记忆（重启丢 pending 槽位是明示语义，降级见「失败与降级」）。
- 建订单表/接真实订单系统（沿用确定性 mock，见「订单数据源」）。
- 修改三处保留改动（`app/prompts/self_check.py`、`app/rag/retriever.py`、
  `app/tools/executor.py` 的工作区状态）；新增或升级任何依赖。
- 入库侧拆存知识（扩写在检索侧，拍板 P8）。

## 总体架构与图拓扑

```
coref(消解+改写+续跑识别) → intent(快路/降级路/四件套) → 分流
  ├ knowledge  商品咨询、其他: retrieve → gate → agent → logging
  ├ data       物流、订单:               agent → logging          （ch05 不动）
  ├ refund     退款退货、售后: slot检 → ┬ 缺单号: selector 出口 ──┐
  │                                    └ 有单号: fetch_order →    │
  │                                       expand → policy_retrieve│
  │                                       → gate → agent ─────────┤
  ├ complaint  投诉:   complaint 节点（ch05 不动）                 ├→ logging → END
  └ chitchat   闲聊:   chitchat 节点（ch05 不动）                  ┘
```

- knowledge/data/complaint/chitchat 四路与 ch05 逐节点等价（ch05 端到端验收 A1–A5 必须照常绿）。
- 新出口 `refund` 是唯一新路径；`其他` 是新增意图值，复用 knowledge 路径。
- `build_graph` 新节点：`make_coref_node(model)`（替换占位）、`refund_slot_node`、
  `refund_selector_node`、`refund_fetch_node`、`refund_expand_node`、
  `refund_policy_retrieve_node`；gate 节点工厂加 `source` 参数（默认值保 ch05 行为不变）。

## State 扩展与逐轮复位

`ChatState`（`app/workflows/state.py`）新增键（全部进程内、跨轮靠 checkpointer）：

| 键 | 类型 | 写入方 | 语义 |
|----|------|--------|------|
| `intent_confidence` | float | intent | 本次判类置信度（0–1） |
| `pending_flow` | str | selector 出口写 `"refund"`；coref 读后即清 | 跨轮槽位待续标志 |
| `slot_order_id` | str | coref（续跑）/refund_slot（本轮提取） | 已确定的订单号 |
| `order_data` | dict | refund_fetch | 订单详情（供 agent 注入） |
| `expanded_queries` | list[str] | refund_expand | 实际参与检索的问法（≤4，含原问法） |
| `orders_payload` | list[dict] | refund_selector | `orders` 帧数据（临时 UI 数据，不落消息历史） |
| `resolved_query` | str | coref | 语义升级：现在是真消解输出（键沿用） |

coref 仍是逐轮复位点（ch05 M1-I1 语义保留）：清 `evidence/suggestions/answer_text/gate_pass`
并开新 log；**唯一例外**是先读 `pending_flow`/`order_data` 供续跑判断，读完把
`pending_flow` 置空（selector 会在本轮末端重新置位）。

## 指代消解与 Query 改写（需求 1/2）

- 新模块 `app/prompts/coref.py`：`COREF_PROMPT`，输入 = 近 6 条历史渲染成
  「用户：…/客服：…」纯文本 + 本轮原话；输出 = 一行自包含标准问法（**纯文本，不强制 JSON**——
  本节点无结构可失配，散文包裹只取首行非空文本）。
- Prompt 内写死三条规则：①指代必须用历史中的真实实体补全，不许编造历史里没有的实体；
  ②问题已完整时**原样输出**，不强行改写；③只输出问法本身。
- 零调用条件：本轮是该线程首条用户消息（`messages` 里无更早 human）→ 直接透传
  `resolved_query=user_query`，log 记 `coref:"passthrough"`。闲聊快路输入改用 `user_query`
  原话（快路在 intent，顺序不变）。
- 失败降级：LLM 异常/空输出 → 透传原话 + log `coref:"degraded"`（ch04 §8 模式：
  理解层不许拖垮主流程）。
- 续跑识别（与「槽位」节共用）：coref 顶端先查 `pending_flow=="refund"` 且
  `user_query` 全匹配 `^我选择订单\s*(\d{3,})$` → 置 `slot_order_id`、
  `resolved_query` 沿用该单上下文补全式（见「退款子流程」），本轮 log 记 `resume:True`。

## 意图识别四件套（需求 4）

改写 `app/prompts/intent.py` 为 V2（ch05 旧版不留双轨，ch05 单测按先例重定向）：

1. **枚举选项**：八类单选——物流｜订单｜商品咨询｜退款退货｜售后｜投诉｜闲聊｜其他。
2. **强制 JSON**：输出恰好 `{"intent": "...", "confidence": 0.0–1.0}`，无其他字段。
3. **边界 few-shot**（≥4 对，钉住验收 2 的「拿不准」带）：
   - 「退货政策是什么」=商品咨询（问规则内容）≠ 退款退货（要处理具体单子的退款）；
   - 「订单 1001 到哪了」=物流 ≠ 订单（问状态/金额本身）；
   - 「收到的猫粮发霉了要换一袋」=售后；「发霉了我要退款」=退款退货；「你们怎么回事！」=投诉；
   - 拿不准示例 → 「其他」。
4. **「其他」兜底类**：判类口径写明「不确定归其他，别硬塞业务意图」。

`app/workflows/routing.py`：
- `INTENTS` 追加 `"其他"`；`INTENT_ROUTES` v2：商品咨询/其他→knowledge，物流/订单→data，
  **退款退货/售后→refund**（ch05 的 knowledge/data 归属作废），投诉/闲聊不变；
  未知值→按「其他」处理（替掉 ch05「归商品咨询」兜底，拍板 P1）。
- `parse_intent_json` 升级为返回 `(intent, confidence) | None`：intent 必须在八类内；
  confidence 必须可转 float 且 ∈[0,1]；缺任一字段=不合 schema=None。
  （ch05 钉 `str|None` 的旧单测随之重定向。）

intent 节点流程：快路命中→闲聊（零调用，ch05 不动）→ 续跑命中→直接 `route="refund"`（零调用）→
否则判类。降级路（拍板 P2，默认关）：

- `Settings` 新键 `intent_small_model: str = ""`、`intent_confidence_threshold: float = 0.75`。
- `get_model(settings, *, model_name=None)` 加可选参（默认 None=现行为逐字符不变）。
- 配置了 `intent_small_model`：小模型先判；解析成功且 `confidence ≥ 阈值` 直接采用；
  否则大模型复判一次；大模型结果再走同一解析，仍失败→`("其他", 0.0)`。
- 未配置（默认）：即需求点名「用能力强的大模型」，失败重试一次后归「其他」。

## Query 扩写与政策检索（需求 3）

- 新模块 `app/prompts/query_expand.py`：`EXPAND_PROMPT`，强制 JSON 且**字段只有
  `queries` 一个数组**；口径：同一退款诉求拆 ≤3 条不同侧重问法（资格/时限/流程运费），
  加原问法共 ≤4、去重。解析 `parse_queries_json`（纯函数，围栏/散文包裹容错，
  同 ch04 `_FENCE` 模式）。
- 触发是确定性的：**仅 refund 路径**调用扩写（意图白名单=场景判定，不经模型裁决，
  对齐需求「只有高频低容错场景扩写、简单 FAQ 不扩写」）；knowledge/data 路永不扩写。
- 多路检索：`asyncio.gather` 并发 `retriever.retrieve(q, strategy="hybrid_rerank")`；
  **不改 `retriever.py`**（保留文件）——ch04 的 `understand_query` 改写/同义层在
  retrieve 内部照常运转，与本章 graph 层消解职责不同、并存（拍板 P8）。
- 去重合并（纯函数 `merge_evidence`）：按 `chunk_id` 去重、同条取最高 score、降序、
  截断 `settings.rerank_top_n`。

## 退款确定性子流程（需求 5/6）

route=refund 后的节点序（全部确定性代码，模型只在末端出现一次）：

1. **refund_slot**：`extract_order_id(text)` 纯函数正则（`订单\s*[#＃:：]?\s*(\d{3,})`
   与续跑格式 `我选择订单\s*([0-9]{3,})`,ASCII-only——终审 M2/M3 拍板口径,全角单号在槽位/API 两面都 422/None)，输入=resolved_query+本轮原话；**模型绝不猜单号**。
   - 提不到 → **selector 出口**：`refund_selector` 节点填 `orders_payload`（见下节）、
     固定话术 ANSWER=实现常量 `SELECT_ORDER_ASK`「好的，请从下方卡片选择要办理退款的订单：」(2026-09-28 终审规6 对齐,无层按字面匹配)入 `answer_text`+
     messages、置 `pending_flow="refund"` → logging → END。原因不问（表单固定类别承载）。
   - 提到 → 继续。
2. **refund_fetch**：直调 `_make_order(order_id)`（见「订单数据源」）填 `order_data`。
3. **refund_expand**：`EXPAND_PROMPT` → `expanded_queries`；失败/不合 schema →
   退回 `[resolved_query]` 单路（降级不阻断）。
4. **refund_policy_retrieve**：多路并发检索+`merge_evidence` → `evidence`；
   log 记 `retrieve_hits`（强制检索发生与否在日志可见，验收 3 断言点）。
5. **gate**（复用工厂，`source="ch06_refund_gate"`）：不过 → ch05 同款拒答话术+
   `transfer_human` 建议（政策都查不到就不给退款结论）。
6. **agent**：主 Agent 拿「订单数据 + 政策证据」只回答「这一单能不能退、为什么、下一步怎么办」。
   `make_agent_node` 增量：`state["order_data"]` 存在时，在证据 SystemMessage 前追加一条
   「订单数据: <json>」SystemMessage（不动人设 Prompt，不影响其他路径）。
   本路收尾固定挂 `suggestions=[{"action":"refund_apply","label":"发起退款申请"}]`
   （gate 通过即挂——判定文本说「不能退」时按钮语义=仍要提交由人工复核；记录为设计裁决）。
   `done.suggestions` 已非空（如预算熔断出 transfer_human）则不覆盖。

## 订单数据源（拍板 P4）

`app/tools/definitions.py` 重构不改行为：把 `query_order` 体内播种生成抽成模块级
`_make_order(order_id) -> dict`（`random.Random(f"order-{order_id}")` 逐字段等价，
ch05 前向兼容由既有 9 个测试文件守住），`query_order` 改为薄壳调用它。
新增普通函数（**不注册为模型工具**，registry/executor 不动）：
`list_user_orders(user_id) -> list[dict]`：固定演示单号 `["1001","1002","1003"]` 逐个
`_make_order` → 同一种子必然同果，**选择器卡片与点选后详情逐字一致**。

## 槽位选择器与续跑协议（需求 6/7，拍板 P5）

- 新 SSE 帧 `orders`：`{"items":[{order_id,status,amount,created_at,items:[str]}]}`
  （`items` 为「商品名 ×qty」字符串数组）。发帧点=`stream_graph_turn` 适配层：
  `orders_payload` 非空时于末 token 后、done 前、suggestions 前发出；
  落位经 `values` 终态而非 custom writer（可在非流式单测断言，无需活流）。
- 前端点卡片 → 自动发送用户消息 `我选择订单 {order_id}`（可见、入 history；
  与 ch05「建工单按钮=显式动作」同族模式）→ 下一轮 coref 续跑识别直通 refund 路径。
- pending 丢失（重启/匿名线程换 thread）：该消息当普通轮走意图——正则句式大概率仍判退款退货；且 `订单 X` 子串会被内嵌提取正则命中,
  refund_slot 直接拿到单号继续办事(不重弹)——实现优于原记载,2026-09-28 终审订正。

## SSE 契约与端点

- 红线不动：token/done/error 逐字符零变更；既有 tool_call/tool_result/suggestions 帧不变。
- `app/schemas/chat.py`：新 `OrdersEvent{items:list[OrderCard]}`；
  `Suggestion.action` 扩为 `Literal["transfer_human","create_ticket","refund_apply"]`。
- 新端点 `POST /api/refunds`（`app/api/routes.py` + 新 `app/schemas/refund.py`）：
  入参 `{conversation_id, order_id, reason}`，reason ∈
  「七天无理由 / 商品质量问题 / 拍错多拍 / 其他」四类（固定下拉的服务端镜像，非法值 422）；
  写 tickets 表：`ticket_type="售后"`，description 服务端拼
  「【退款申请】订单 {order_id}｜原因：{reason}」(会话 id 已存 tickets 列,不重复进描述——2026-09-28 经 M3 评审按规6 向实现对齐);返回复用 `TicketOut`；
  无会话 503——与 `/api/tickets` 同款降级面。**不建新表**（拍板 P6）。

## 前端配套（需求 7；Vibe Coding 例外直做）

1. `orders` 帧 → 聊天流内订单卡片组（像素风小卡：单号/状态/金额/商品行+「选这个」钮）；
   点击=禁用该组卡片 + 自动发送「我选择订单 {id}」+ 正常走 send 流程。
2. `refund_apply` 建议钮 → 弹轻表单：订单号回显（本轮上下文已知项）、原因**固定类别下拉**、
   提交 → `POST /api/refunds` → 回执「🎫 退款申请 {ticket_no} 已提交」；失败可重试。
3. 无新增后端请求面（除 /api/refunds）。

## 失败与降级汇总

| 故障 | 行为 |
|------|------|
| coref LLM 异常/空输出 | 透传原话（degraded 记日志），流程继续 |
| 意图大模型解析失败 ×2 | 归「其他」→knowledge（带闸安全出口） |
| 降级路小模型调用失败 | 直接升大模型（等价未配置） |
| expand 解析失败 | 原问法单路检索 |
| 政策检索全空/弱 | 闸兜底拒答+转人工+落池(source=ch06_refund_gate) |
| 无 pending 的「我选择订单 X」 | 当普通轮走意图，最坏 selector 再弹 |
| /api/refunds 无 DB | 503，前端提示可重试（同 tickets） |

## API 核对结论（规 8）

本章无新增第三方 API 面：`asyncio.gather`（stdlib）、TypedDict 自定义键跨轮
（ch05 已证）、`ServerSentEvent` 新 event 名（repo 既有模式）、ChatOpenAI 换 model_name
实例化（`get_model` 既有构造改一个参）——全部是锁定版本上**仓库内已跑绿**的用法
（280 套件即版本一致性最强证据），无需 Context7 查询。若 writing-plans/实现阶段冒出
计划外新 API，按规 8 先 Context7 一次再动手。

## 测试策略与验收标准 → 责任节点映射

**单元（TDD，FakeModel 注入）**：coref 节点（透传/补全/降级/零调用）、
`parse_intent_json` v2、route 表 v2、intent 节点（快路/兜底其他/降级路升判）、
`extract_order_id`、`parse_queries_json`、`merge_evidence`、refund 子流程全链
（假检索+假模型：selector 出口帧数据/pending 置位、续跑直通、fetch→expand→retrieve→gate→agent
节点序、order_data 注入、refund_apply 建议）、`list_user_orders` 与 `_make_order` 一致性、
`/api/refunds`（422/503/落表）、适配层 orders 帧序。
**Prompt 质量（非单元可测→标注样例+评估）**：
`tests/samples/ch06_intent_qa.csv`（含「其他」带）、`ch06_coref_qa.csv`（多轮补全+
原样透传对）、扩写走 `evals/smoke_ch06.py`（沿用 `smoke_query_rewrite.py` 范式，真模型冒烟、
打印对照、非 CI 断言）。
**端到端（integration，ch05 模块共环+共引擎 fixture 范式）**：

- B1 多轮 物流→退款→聊回物流（同 conversation_id 线程）：逐轮日志行
  intent 正确 + `resolved` 字段完成指代（日志行扩 `resolved`/`confidence` 两键）。
- B2 怪问题（乱码/情绪句）→ 日志 intent=其他，且 JSON 解析全程未抛穿。
- B3 「这个能退吗」（前轮先聊某商品）→ resolved 被补全 + 日志 nodes 含
  refund 链（fetch/expand/retrieve）+ 回答基于政策。
- B4 浏览器手测：无单号退款问题弹卡片→点选→子流程走完出结论→表单提交拿 ticket_no。

**回归红线**：ch05 单测中钉旧口径者按先例重定向（parse 返回型、route 表）；
ch05 e2e A1–A5 与 280 套件其余全绿；token/done/error 契约测试零改动通过。

## 配置与文件变更清单

新键（`app/core/config.py`，带默认值不破坏构造）：
`intent_small_model=""`、`intent_confidence_threshold=0.75`。

| 动作 | 文件 |
|------|------|
| 新建 | `app/prompts/coref.py`、`app/prompts/query_expand.py`、`app/schemas/refund.py`、`evals/smoke_ch06.py`、`tests/samples/ch06_intent_qa.csv`、`tests/samples/ch06_coref_qa.csv`、新单测 5–6 个 + `tests/e2e/test_ch06_acceptance.py` |
| 修改 | `app/workflows/{state,routing,nodes,graph}.py`、`app/prompts/intent.py`（V2 原地）、`app/core/config.py`、`app/services/chat_service.py`（get_model 加参）、`app/api/routes.py`、`app/schemas/chat.py`、`app/tools/definitions.py`（抽壳+list）、`static/index.html`（Vibe）、`README.md`、受钉旧口径单测 |
| 禁触 | `app/rag/retriever.py`、`app/prompts/self_check.py`、`app/tools/executor.py`（三处保留改动） |

## 拍板记录（2026-09-27，用户「写吧」= P1–P8 推荐默认全收）

P1 其他→knowledge 出口，未知兜底改归其他；P2 降级路实现但默认关
（`intent_small_model` 空=只走大模型）；P3 扩写 ≤4 条含原问法、chunk_id 去重取最高分、
截断 rerank_top_n；P4 不动 query_order 行为、抽 `_make_order` 同播种供
`list_user_orders`，卡片与详情逐字一致；P5 点卡片=发「我选择订单 {id}」消息+pending_flow
确定续跑，pending 丢失最坏再弹；P6 新 POST /api/refunds 落 tickets 表（售后），不建新表；
P7 orders 新帧+refund_apply 建议 action，token/done/error 红线不动；P8 ch04 检索内改写层
与 ch06 graph 消解层职责不同、并存。另记裁决：refund 路 gate 通过即固定挂 refund_apply
（判定为「不能退」时按钮=仍要提交人工复核），见「退款确定性子流程」。

## 全局约束（进计划 Global Constraints）

python 3.12 + uv 锁版本、零新增依赖；一任务一 commit（含勾选+dev-notes 追加）；
token/done/error 逐字符红线、新帧一律「末 token 后 done 前」；前端走 Vibe Coding 例外
（不套 TDD/评审，其余步骤照常）；过程留痕 `dev-notes/ch06.md` 按阶段追加、禁止收尾补记；
点名的技术选型定死，矛盾即停即问；ch05 e2e 与单测回归绿是每任务收尾标准。
