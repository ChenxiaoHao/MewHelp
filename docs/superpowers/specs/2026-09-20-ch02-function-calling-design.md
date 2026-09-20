# ch02 设计稿：Function Calling 工具链（单轮收敛）

- 日期：2026-09-20
- 状态：定稿（用户分节评审通过，"整体方案 OK，按这个方案继续实施"）
- 上游：ch01 纯对话（master @ 6853b0f）
- 技术选型（用户定死）：FastAPI + SQLAlchemy + MySQL（Docker）+ LangChain `@tool`

## §0 决策记录

澄清决策（AskUserQuestion，全选推荐项）：

| # | 决策点 | 结论 |
|---|---|---|
| 1 | 会话历史契约 | **扩展现有契约**：前端照旧携带全量 messages，body 新增 `conversation_id`；服务端建/复用会话并落库；模型上下文仍来自客户端历史 + ch01 裁剪 |
| 2 | SSE 工具状态帧 | **tool_call + tool_result 双帧**（徽章三态：调用中/成功/失败） |
| 3 | 第一轮模型调用 | **第一轮也走流式**：bind_tools + astream，纯闲聊文本逐 token 透传，流末检出 tool_calls |
| 4 | 工具超时重试 | **5s 超时 + 重试 1 次**，.env 可配 |

实现架构：方案 A（分层扩展 + 自研单轮编排）。方案 B（LangChain 预置 AgentExecutor）因是多轮循环架构、与"只做单轮"相悖被否；方案 C（同步 DB）因异步流内混同步调用被否。

用户补充约束（评审通过时追加）：

1. §12 四个硬性二次核对点必须执行（Context7 + 本地实际版本验证，不许凭记忆）
2. **ch01 兼容红线**：SSE `token/done/error` 事件语义不变，不为 ch02 大改 ch01 已稳定代码（见 §13）

## §1 目标与范围

给现有客服装上 Function Calling 工具链：模型自主决定调什么工具 → 执行 → 结果回灌 → 单轮收敛流式作答。工具链长在 SSE 聊天入口上，用户在聊天页问一句就能触发。

**本章不做**：多轮自动循环 Agent Loop、向量检索、RAG、鉴权、部署。

## §2 架构与模块布局

```
app/
├── db/                       # 新增：数据层
│   ├── engine.py             #   create_async_engine + async_sessionmaker + get_session 依赖
│   ├── models.py             #   四张表 ORM（严格对齐附录 A DDL，中文 ENUM）
│   └── crud.py               #   会话创建/复用、消息落库、faq LIKE 查询、工单写入
├── tools/                    # 新增：工具层
│   ├── definitions.py        #   五个 @tool
│   ├── registry.py           #   注册管理（name → tool，供 bind_tools）
│   └── executor.py           #   参数校验 + 超时 + 重试 + 异常→错误结果包装
├── services/
│   ├── chat_service.py       #   ch01 原样不动（build_messages/trim_history/stream_chat/get_model）
│   └── tool_chat_service.py  #   新增：stream_chat_with_tools 单轮编排
├── schemas/chat.py           #   ChatRequest 扩 conversation_id；新 SSE 事件 data 模型
├── api/routes.py             #   扩展：conversation/tool_call/tool_result 帧 + 落库挂点
docker-compose.yml            # 新增：MySQL 8 容器
db/init/01_schema.sql         # 附录 A DDL 原样落盘（容器首启自动建表）
db/init/02_seed.sql           # 附录 B 种子数据
evals/tool_routing_samples.json      # 附录 C 工具路由标注集
evals/run_tool_routing_eval.py       # 路由评估脚本（真实模型，不执行工具）
dev-notes/ch02.md             # 过程留痕
```

数据流：

```
请求 → 建/复用会话（推 conversation 帧）→ user 消息落库
     → 第一轮 bind_tools + astream
         ├─ 无 tool_calls：文本已逐 token 透传 → 落 assistant 行 → done
         └─ 有 tool_calls：落 assistant(tool_calls) 行
              → 逐个执行：推 tool_call 帧 → executor（超时/重试）→ 推 tool_result 帧 → 落 tool 行
              → ToolMessage 回灌 → 第二轮裸 model astream 收敛 → 落 assistant 行 → done
```

## §3 数据层

- **docker-compose.yml**：`mysql:8.0`，端口 3306，库 `mewhelp`，root 密码从 `.env` 读（compose 变量插值）；挂载 `./db/init:/docker-entrypoint-initdb.d:ro`（首启自动执行 DDL + 种子）；healthcheck `mysqladmin ping`；数据落 named volume
- **建表**：附录 A DDL 一字不改。应用不 create_all，DDL 脚本是唯一事实源
- **种子数据**：附录 B——faq 10 条（必含"退货政策"命中条目；**全表不出现"邮费""运费"字样**，保证验收 3 的预期漏召回）；conversations/messages/tickets 各 1 条演示数据验证外键链路
- **ORM**：SQLAlchemy 2.0 async（`create_async_engine` + `async_sessionmaker` + `AsyncSession`）；models 用 `sqlalchemy.dialects.mysql.ENUM` 对齐中文枚举、`JSON` 列存 tool_calls
- **连接串**：`mysql+aiomysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4`；驱动实施实测（§12-③），不通换 asyncmy
- **新增依赖**：`sqlalchemy[asyncio]>=2.0`、`aiomysql`、`cryptography`（MySQL 8 caching_sha2_password 认证需要）
- **启动校验**：lifespan 内 `SELECT 1`，连不上快速失败，日志明确提示"先启动 Docker Desktop"

## §4 工具层

五个 LangChain `@tool`（类型注解 + docstring 自动生成参数 Schema）：

| 工具 | 模型可见参数 | 行为 |
|---|---|---|
| `query_order` | `order_id: str` | mock：金额/状态/下单时间/1-3 件商品，random 生成，回显 order_id 自洽；不建表不接真实接口 |
| `query_product` | `keyword: str` | mock：1-3 个商品（名称/价格/库存/规格） |
| `query_logistics` | `order_id: str` | mock：承运商 + 3-5 条随机轨迹（时间/地点/状态） |
| `query_faq` | `keyword: str` | 真实 SQL：`WHERE question LIKE %kw% OR answer LIKE %kw%`，top 3；无命中返回空列表 |
| `create_ticket` | `description: str`、`ticket_type: 售后\|投诉\|咨询` | 真实写 tickets 表；**conversation_id 由执行器运行时注入，不进模型 schema**（注入 API 按 §12-② 核对）；`ticket_no = T{YYYYMMDD}{当日已有工单数+1:03d}`，IntegrityError 撞号重算重试一次；成功后把 conversations.status 置"已转人工" |

**executor**（`execute_tool(name, args, context)`）：

- `asyncio.wait_for(tool.ainvoke(args), timeout=TOOL_TIMEOUT_SECONDS)`，超时/异常重试 `TOOL_MAX_RETRIES` 次
- 参数校验由 `@tool` 生成的 pydantic schema 承担；非法参数抛出的校验错误与超时/异常同路径处理
- 最终失败不炸请求：返回 `{"error": "工具执行失败: <原因>"}` 作为工具结果回灌，模型据此礼貌收敛
- `tool_result` 帧的 `summary` 由 executor 按工具生成中文简述（物流取最新轨迹、faq 取命中条数、失败取原因），超 80 字符截断

**registry**：`TOOL_REGISTRY: dict[str, BaseTool]` + `get_tools()`，是唯一注册点；executor 只接受注册表内的工具名。

## §5 单轮编排（tool_chat_service.stream_chat_with_tools）

1. `build_messages`（ch01 原函数复用，裁剪与保底规则不动）
2. 第一轮 `model.bind_tools(get_tools()).astream(messages)`：文本 chunk **实时透传**（纯闲聊体验与 ch01 一致），同时累加 chunk；流末检查聚合 `tool_calls`（聚合 API 按 §12-① 核对）
3. 无 tool_calls → 纯闲聊路径结束（行为与 ch01 等价）
4. 有 tool_calls → 对当轮**全部**申请（通常 1 个）逐个：yield `("tool_call", {id,name,args})` → executor 执行 → yield `("tool_result", {id,name,ok,summary})` → 构造 `ToolMessage(content=结果JSON, tool_call_id=id)`
5. 第二轮：`[裁剪后历史 + 第一轮 AIMessage(含 tool_calls) + ToolMessages]` 用**裸 model**（不 bind_tools，物理保证单轮收敛）流式产出最终回答，yield `("token", ...)`
6. 编排函数 yield 带标签事件元组，路由层翻译为 SSE；`done`/`error` 帧语义与 ch01 完全一致

单轮定义：**至多一轮工具决策**；第二轮不带工具，模型只能出文本。

## §6 SSE 协议与请求契约扩展

`ChatRequest` 新增 `conversation_id: int | None = None`。会话身份**由服务端拥有**（DDL 主键 BIGINT 自增，前端不造 uuid）：

- 首轮 null → 服务端建 conversations 行（user_id=DEMO_USER_ID）→ 推 `event: conversation` 帧告知 id
- 前端存 id，后续请求携带；服务端校验存在性，不存在按新建处理
- "＋新对话"按钮：前端置 null → 下轮开新会话行
- **每轮都推 conversation 帧**（幂等，前端直接存）
- 不带该字段的 ch01 curl 验收命令依然兼容（自动建一次性会话）

完整帧序示例：

```
event: conversation  data: {"conversation_id": 3}
event: tool_call     data: {"id":"call_x","name":"query_logistics","args":{"order_id":"1001"}}
event: tool_result   data: {"id":"call_x","name":"query_logistics","ok":true,"summary":"运输中·杭州中转中心"}
event: token         data: "您的包裹..."      （逐 token，同 ch01）
event: done          data: [DONE]             （同 ch01）
event: error         data: {"detail":"..."}   （同 ch01，出错时）
```

## §7 落库策略（ChatPersister）

落库点通过 `ChatPersister` 协议注入编排/路由层（单测用 fake，无需 MySQL）：

| 挂点 | 写入 |
|---|---|
| `on_turn_start` | 建/复用 conversations；user 消息行 |
| `on_tool_calls` | assistant 行（content=第一轮文本可空，tool_calls=申请单 JSON） |
| `on_tool_result` | tool 行（tool_call_id 对号入座，content=结果 JSON） |
| `on_final_answer` | assistant 行（完整回答文本） |

- 纯闲聊只有 on_turn_start + on_final_answer 两笔
- create_ticket 成功 → conversations.status='已转人工'
- **运行期落库失败只记 warning 日志，绝不打断 SSE 聊天流**（演示可用性优先）
- 模型上下文仍来自客户端 messages（决策 1）；DB 用于审计、工单倒查、ch03 铺路

## §8 前端改造（Vibe Coding 例外区）

复古像素风与现有布局**不动**，只加四点：

1. `conversationId` 变量：`conversation` 帧存值；"＋新对话"置 null；请求体携带（null 则省略字段）
2. SSE 解析加 `tool_call` / `tool_result` 两个分支（`conversation` 帧共三个新分支）
3. 工具徽章：`tool_call` 帧在当前 AI 气泡上方挂徽章「🔧 query_logistics 调用中…」，`tool_result` 帧更新为「✓ query_logistics」/「✗ query_logistics」（title 提示 summary）；样式沿用像素风（2px 墨线边框 + 橙底小方块）；同轮多工具多徽章
4. `history` 数组与多轮上下文逻辑**零改动**

## §9 错误处理矩阵

| 故障 | 行为 |
|---|---|
| 工具超时/异常（重试后仍失败） | tool_result 帧 ok=false + `{"error":...}` 回灌 → 模型礼貌说明、建议转人工工单 |
| 模型给非法工具参数 | schema 校验失败 → 同上错误回灌路径 |
| 第一轮/第二轮模型流失败 | `event: error` 帧（同 ch01）；已推出的 conversation/tool 帧保留 |
| 启动时 MySQL 连不上 | lifespan 快速失败，提示启动 Docker Desktop |
| 运行期落库失败 | warning 日志，聊天流不中断 |
| conversation_id 不存在 | 按新建会话处理（不报 4xx，演示容错） |

## §10 测试与评估

**单元测试**（mock 模型 + fake 工具，零 token 零 MySQL，ch01 的 34 个测试不动且必须继续通过）：

- executor：fake 工具（成功/超时/抛异常/非法参数）验证 5s 超时、重试 1 次、错误包装、summary 生成
- 编排：FakeToolModel（编程 chunk 序列：纯文本流 / 含 tool_calls 流）验证事件序列、第二轮裸模型收敛、单轮不再触发工具
- 路由 SSE：conftest Fake + `dependency_overrides`（ch01 模式），断言 conversation/tool_call/tool_result/token/done 帧序与 ch01 兼容性（不带 conversation_id 的旧请求体仍 200）
- persister：fake recorder 验证四个挂点调用时机与参数
- models：表结构断言（中文 ENUM 值、JSON 列、FK、ticket_no 主键）
- crud：faq LIKE 查询与工单号生成的纯逻辑部分（撞号重试）单测；真实 SQL 行为由演示命令冒烟

**评估集**（Prompt 类产出替代 TDD，真实模型）：

- `evals/tool_routing_samples.json`：附录 C 的 10 条标注
- `run_tool_routing_eval.py`：system prompt + 样例 → bind_tools → 第一轮 `ainvoke`（不执行工具）→ 比对工具名集合（`no_tool` = 空集合）→ 输出逐条通过/失败 + 通过率
- System Prompt 改造：在 ch01 客服 Prompt 上追加"工具使用指引"段——涉及订单/物流/商品/FAQ/工单的具体数据必须调工具获取；工具无结果时如实告知并建议创建人工工单；**无工具结果时严禁编造数据**（衔接 ch01 约束 4 的"能力声明+替代行为"写法）。质量由评估集验证
- 验收 3 的漏召回（"邮费是多少" → query_faq 查不到）是**预期结果**，评估集 remark 记录，dev-notes 留痕，留给 ch03 向量检索升级

**验收标准**（用户原文）：

1. 浏览器问「订单 1001 的物流到哪了」→ 气泡带工具徽章 + 按 mock 轨迹作答
2. 问「退货政策是什么」→ query_faq 命中并作答
3. 问「邮费是多少」→ query_faq 漏召回（预期），记录在案

## §11 配置与依赖新增

`.env.example` 追加（沿用 ch01 dotenv 优先级：init > .env > 系统 env）：

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=<compose 与此共用>
MYSQL_DB=mewhelp
DEMO_USER_ID=demo_user
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=1
```

Settings 增对应字段 + `database_url` 组装 property。`/api/health` 响应体不变（ch01 契约）。

## §12 实施时硬性二次核对点（用户点名，写入计划为硬性步骤）

| # | 核对点 | 方法 |
|---|---|---|
| ① | `bind_tools + astream` 的 chunk 累加与 `tool_calls` 聚合行为 | Context7 + 本地安装版本实测（qwen 上游兼容性） |
| ② | `create_ticket` 的 conversation_id 运行时注入机制 | 按当前 LangChain 版本核对（ToolRuntime / InjectedToolArg / RunnableConfig），Context7 + `inspect.signature` |
| ③ | aiomysql Windows 安装 | `uv add` 实测；不通换 asyncmy（驱动属实现细节，不算选型变更；两者都不通才停工问用户） |
| ④ | MySQL Docker 启动与认证兼容 | 容器实测（mysql:8.0 + caching_sha2_password + cryptography） |

## §13 ch01 兼容红线（用户约束）

- SSE `token` / `done` / `error` 三种事件的语义、data 格式**不变**；新事件只增不改
- `chat_service.py`（build_messages / trim_history / stream_chat / get_model）**不改**，编排在新文件
- `/api/extract`、`/api/health` 契约不变
- ch01 的 34 个单测不改且必须全绿；ch01 README 三条 curl 验收命令必须继续通过
- 前端只做 §8 列出的四点增量，复古像素风 UI 不动

## 附录 A：建表 DDL（用户提供，原样落盘 db/init/01_schema.sql）

```sql
-- =============================================================
-- ch02 · Function Calling 工具链 · 建表 DDL
-- 本章新建:faq / conversations / messages / tickets 四张表
-- 商品、订单、物流走工具内 mock,不建表
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- 建表顺序:先 conversations,再依赖它的 messages / tickets
-- =============================================================

-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```

## 附录 B：faq 种子数据（db/init/02_seed.sql，10 条）

| # | question | answer（摘要） | category |
|---|---|---|---|
| 1 | 退货政策是什么？ | 签收 7 天内无理由退换，商品需未洗涤未使用、吊牌完整；15 天内质量问题免费修换 | 退换货 |
| 2 | 如何申请退换货？ | 「我的订单」→「申请退换」，填原因提交，审核后快递上门取件 | 退换货 |
| 3 | 退款多久到账？ | 收到退货并验收后 1-3 个工作日原路退回 | 退换货 |
| 4 | 下单后多久发货？ | 现货 48 小时内发出，一般 2-4 天送达；预售以商品页为准 | 物流配送 |
| 5 | 支持哪些快递公司？ | 默认顺丰或京东物流，部分偏远地区支持指定其他快递 | 物流配送 |
| 6 | 怎么联系人工客服？ | 聊天中说明转人工即可创建工单，服务时间 9:00-21:00 | 售后服务 |
| 7 | 商品有质量问题怎么办？ | 15 天内免费修换，请提供订单号与问题描述 | 售后服务 |
| 8 | 如何修改收货地址？ | 未发货在订单详情修改；已发货联系客服拦截 | 订单服务 |
| 9 | 登录密码忘了怎么重置？ | 登录页「忘记密码」，手机验证后重置 | 账户 |
| 10 | 如何开具发票？ | 提交订单时勾选「开具发票」填税号，电子发票 48 小时内发送 | 发票 |

**红线**：全表（question + answer）不得出现「邮费」「运费」字样——保证验收 3 漏召回。

演示数据：conversations 1 行（demo_user，已结束）、messages 2 行（user/assistant）、tickets 1 行（T20260901001，售后，已处理）。

## 附录 C：工具路由评估集标注（evals/tool_routing_samples.json，10 条）

| # | 用户问题 | 期望工具 | 备注 |
|---|---|---|---|
| 1 | 订单 DD20260901001 的物流到哪了 | query_logistics | 验收 1 同款 |
| 2 | 帮我查一下订单 1001 买了什么 | query_order | |
| 3 | 你们有蓝牙耳机卖吗，多少钱 | query_product | |
| 4 | 退货政策是什么 | query_faq | 验收 2，应命中种子 #1 |
| 5 | 邮费是多少 | query_faq | **预期漏召回**：LIKE 查不到，remark 记录，留 ch03 升级 |
| 6 | 你们发货用什么快递 | query_faq | 应命中种子 #5 |
| 7 | 你们一直没人处理我的问题，帮我登记个投诉 | create_ticket | ticket_type=投诉 |
| 8 | 你好 | no_tool | 闲聊 |
| 9 | 谢谢，再见 | no_tool | 闲聊 |
| 10 | 怎么申请退款 | query_faq | 应命中种子 #2/#3 |
