# ch08 即插即用工具系统 设计规格

- 日期:2026-09-29
- 状态:设计已于 chat 获批(用户「可以,你继续吧」,2026-09-29);本文档为待用户复核的书面 spec
- 前置:ch07 已合并 master@29970e9(单元 403 passed / 集成 golden 13 passed)
- 需求原文与 DDL 原文:见文末「附录一/附录二」,逐字收录,权威以此为准

## 拍板记录(2026-09-29,规7 批审:用户对 P1–P9 未逐条表态,全按推荐默认执行,此处记死)

| 编号 | 决定 | 内容 |
|---|---|---|
| P1 | 三恒文件正名 | `app/tools/executor.py` 恒M 惯例解除:ch08 重构对象,工作树那行未提交的 `timeout_seconds 5.0→30.0` **弃用**;超时基准按「配置与依赖」节定 10s。`app/prompts/self_check.py` 用户本地重写的判据 prompt **保留并随 ch08 首个波及任务正名一笔提交**。`app/rag/retriever.py` 调试 print **恢复 HEAD(弃)**。 |
| P2 | 新依赖破例 | `uv add mcp langchain-mcp-adapters jsonschema`(jsonschema 实证不在 uv.lock 传递依赖)。历章「零新依赖」惯例首例破例;版本=uv 解析为准,不手挑、不自行升级。 |
| P3 | Server 拓扑 | 物流 8101、售后 8102,路由均 `/mcp`,本机无 TLS。 |
| P4 | 发现时机 | 每对话轮 `get_tools()` 现拿,**无缓存**(adapters 实证每次调用新建会话→验收3 天然过)。 |
| P5 | MCP 工具权限 | 外部 MCP 工具一律按**只读**呈现给模型;`write` 仅内置 create_ticket。外部 server 若实现写语义,本章不支持(权限表只认我方登记)。 |
| P6 | interrupt 落点 | **独立 ticket_confirm 节点**,节点第一行即 interrupt(重放安全);否决 react 环内 interrupt(resume=节点从头重放→重烧 LLM 且不确定)。 |
| P7 | query_logistics 下线 | 删内置版,MCP 版沿用同名;内置移除后**无共存窗口**。server 未启动=当轮无物流工具(注册中心只呈现可用面),验收2 演示须先起 server。 |
| P8 | 审计写法 | 轮内 await 落库(单机 demo 时序最可靠),写失败只 WARN 不拦执行;`result_summary` 截 500 字符,`error_message` 截 512(与 DDL 列宽一致)。 |
| P9 | 验收1 演示形态 | 往 logistics_server.py 加一个新 `@mcp.tool`(查仓库类)→ 只重启 8101 → 客服侧零改动零重启,下轮即问即用。 |

## 现状形状(ch08 起点,侦察实证)

- `app/tools/definitions.py`:5 个 `@tool` async 内置(query_order / query_product / **query_logistics(本章下线)** / query_faq(读 RunnableConfig) / create_ticket(径连 `app/db/crud.py:81 create_ticket`)),mock random 数据照 ch02 做法。
- `app/tools/registry.py`:静态 `TOOL_REGISTRY: dict[str, BaseTool]` + `get_tools()`(固定顺序)/ `get_tool(name)`。
- `app/tools/executor.py`:**单漏斗**——react、`app/services/tool_chat_service.py`、`app/services/naive_agent_loop.py` 三个调用面全走 `execute_tool(name, args, tool_call_id, context) -> ToolOutcome`;`ToolContext{conversation_id, timeout_seconds, max_retries}`;兜底原则「绝不向上抛,失败变 ok=False 回灌模型」(ch05 spec §9 同源,ch08 保留并扩强)。
- `app/agents/react.py:70-71`:bind 集合现排除 create_ticket;:109-117 是 ch05 D2 红线的「幻调执行闸」(SimpleNamespace 硬拒)——**本章被权限闸取代**(见「建工单确认流」节的红线变更)。
- `app/core/config.py:23-24`:`tool_timeout_seconds: float = 5.0`、`tool_max_retries: int = 1`(env 可覆)。
- graph(`app/workflows/graph.py`):InMemorySaver 线程 `conv-{cid}`;`stream_graph_turn` 帧面 token/tool_call/tool_result/orders/suggestions,done 帧属 routes SSE 适配件;ch06 选择器=「orders 帧+关流+表单 POST」,**无 interrupt 先例**——本章确认流为图内全新机制。
- 依赖缺口:pyproject 无 mcp / langchain-mcp-adapters;uv.lock 无 jsonschema。
- 引用面 ~20 文件(registry/executor/definitions 三层):react、tool_chat_service、naive_agent_loop、evals/run_tool_routing_eval、tests(test_tools / test_executor / test_executor_citations / test_routes_ch02 / test_schemas_ch02 / test_tool_chat_service / test_react_node_ch05 / test_chat_stream_ch05 / test_naive_loop_ch05 / test_model_smoke_ch05_integration / e2e test_ch05_acceptance)。波及清单在 plan 逐列。

## 总体架构

```
模型/bind_tools ← 注册中心快照(内置登记 ∪ MCP 每轮现拿,ToolSpec 三件套)
        │ tool_call
        ▼
执行引擎单点 execute_tool(spec, args, tool_call_id, ctx)
  校验闸 → 权限闸 → 超时/重试(只暂时性) → 三类分诊 → 结果格式化 → 审计落行
        │                                    (write 无凭证 → 拒 → 确认流)
        ▼                                    ▼
   ToolOutcome 回灌 react 环          ticket_confirm 节点(interrupt ⇄ 前端卡片 resume)
```

文件布局(新增/重构):

- 重构 `app/tools/registry.py`(ToolSpec + 内置登记 + MCP 发现合并 + 降级)
- 重构 `app/tools/executor.py`(五道闸单点)
- 新增 `app/tools/audit.py`(审计写入 sink,可注入)
- 新增 `mcp_servers/logistics_server.py`、`mcp_servers/aftersale_server.py`(独立进程)
- 修改 `app/workflows/state.py`、`app/workflows/graph.py`、`app/workflows/nodes.py`(确认节点)、`app/agents/react.py`(create_ticket 入 bind + preview 捕获)、`app/api/routes.py`(confirm/resume 端点)、`app/db/models.py`+`app/db/crud.py`(ToolAuditLog)
- 新增 `db/init/08_ch08_tool_audit.sql`(附录二逐字)
- 前端(静态页):ticket 预览卡片渲染——**Vibe Coding 例外**,帧契约由本 spec「确认流」节钉死,实现随用户描述迭代,不套流程

## 注册中心(app/tools/registry.py 重构)

三件套登记形状:

```python
@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool          # 可执行件(langchain @tool / adapters StructuredTool)
    description: str        # 用途描述(三件套之二;模型可见面)
    permission: str         # "readonly" | "write" —— 只认我方登记(P5)
    source: str             # "builtin" | "mcp"
    mcp_server: str | None  # MCP server 名("logistics"/"aftersale"),内置为 None
```

- **内置登记**:模块导入即登记(服务启动时),create_ticket=`write`,其余=`readonly`;query_logistics 内置件删除(P7)。
- **MCP 发现**:每轮 `await snapshot_tools(settings) -> list[ToolSpec]`——`MultiServerMCPClient` 按 settings 两 URL 构连,`get_tools()` 现拿(P4 无缓存);返回工具包成 ToolSpec(permission 恒 readonly)。
- **合并规则**:MCP 工具名撞内置名 → **丢弃 MCP 件 + WARN**(外部声明不可信,内置面我方定义;本章唯一预期撞名=无,query_logistics 内置已撤,MCP 补位)。
- **降级**:单个 server 连不上(拒连/超时/工具枚举失败)→ try/except 只 WARN,快照=内置 ∪ 其余可用 server——聊天不断线(T7「store 面异常全吞」原则延伸到工具发现面)。
- **同步 legacy 面**:`tool_chat_service` / `naive_agent_loop` 保留同步内置视图(`BUILTIN_SPECS: dict[str, ToolSpec]`),**不接 MCP**(它们是 ch04/05 兼容面,主力图链才是本章验收面);executor 五道闸对全部调用面生效(单漏斗)。
- **模型可见面(变更)**:react bind 集合 = 当轮快照全部工具,**含 create_ticket**——模型要能发起建单,引擎才有东西可拦。ch05 D2「建单唯一入口=前端按钮」红线由本章需求7 取代为「Agent 发起 + 前端确认 + resume 执行」;ch05 投诉流按钮路(`/api/tickets` POST)原样不动。

## 执行引擎五道闸(app/tools/executor.py 重构)

`execute_tool(spec: ToolSpec, args: dict, tool_call_id: str, ctx: ToolContext) -> ToolOutcome`——查找从 execute_tool 内部上移到调用方(react 用当轮快照;查不到名=幻觉调用,沿用 react 现护栏回 ok=False,并按「权限拒绝」落审计——未登记=未授权)。

### 权限闸

- **闸序=校验→权限**(spec 自查订正,2026-09-29):写调用若先过权限闸被拒,「必填缺失先回校验错误让模型追问」(验收4)永远触发不了;必填齐了才谈得上转确认流。
- 判定唯一依据=`ToolSpec.permission`;MCP 自带任何声明一概不采信、模型不临场判断(需求3 原文语义)。
- `write` 且 `ctx.ticket_confirmed=False` → 拒:回 `ToolOutcome(ok=False, result={"error": "写操作需客户明确确认后执行"})`。凭证是 **ToolContext 服务端字段**,不是工具参数——模型 args 受 Schema 校验,伪造不进来(需求「不给模型绕过去的机会」的机制保证)。
- create_ticket 被拒且参数合法 → react 把 `{ticket_type, description}` 捕获进 `state.ticket_preview`,图路由转 ticket_confirm 节点(「建工单确认流」节);此时**不写审计**(中间态,终局按确认流结论写,见审计节);非确认流程的 write 拒(理论面,本章无)→ 审计「权限拒绝」。

### 校验闸(需求2)

- 执行前 `jsonschema` 按 `spec.tool.args`(JSON Schema)统一校验:类型不符/必填缺失/取值越界 → 拦下**不抛异常**,`result={"error": "参数不合法: <校验说明中文>"}`,回灌模型让它追问用户或重组调用。
- 审计状态「校验拦下」。
- create_ticket 走此闸实现验收4「缺必填先追问」:描述缺失 → 校验错误回灌 → 模型向用户追问(提示词效果用评估样例验证,见「测试策略」)。

### 超时与重试(需求4,验收6)

- `asyncio.wait_for`,`ctx.timeout_seconds` 默认=settings.tool_timeout_seconds(settings 默认 **5.0→10.0**,P1;MCP 冷会话 5s 偏紧,demo 可 env 调小演示超时)。
- **只重试暂时性故障**:白名单 `(TimeoutError, ConnectionError, OSError)`(含 MCP transport 断连);重试次数=ctx.max_retries(settings 键不动,默认 1)。
- **业务空结果不重试**:工具正常返回零记录 = `成功`,格式化给人话空态。
- **write 默认不自动重试**:permission=write 时 attempts=1(超时未必没执行,重复执行比失败更糟——需求4 原文);审计 `retry_count=0`。
- 终局超时 → 状态「超时」;重试后成功 → 「成功」且 retry_count>0。

### 错误三类分诊(如实回,需求4)

| 类别 | 触发 | 回灌 result | 审计状态 |
|---|---|---|---|
| 参数不合法 | 校验闸拦下 | `{"error": "参数不合法: …"}` | 校验拦下 |
| 查询落空 | 工具正常返回零结果 | 原结果+人话空态说明(不是 error) | 成功 |
| 真故障 | 不可重试异常/超时耗尽 | `{"error": "工具执行失败: …"}` | 失败 / 超时 |

### 结果格式化

- 挑有用字段:MCP 工具结果过字段白名单投影(表在 server 契约节钉);内置工具维持现有返回形状。
- 枚举码翻人话:翻译表集中在 executor(`{"TRANSPORT": "运输中", "SIGNED": "已签收", …}`);Server 侧回结构化原码,格式化单点=客服侧(保证内置/MCP 一视同仁)。
- 序列化 `ensure_ascii=False`(现状保留)。
- `make_summary` 扩展覆盖新工具(≤80 字符徽章语义不变)。

## 审计 tool_audit_logs(需求5)

- DDL 附录二**逐字**入 `db/init/08_ch08_tool_audit.sql`;dev 活库手工等价建表 + 接缝测(照 ch07 惯例)。
- ORM:`app/db/models.py` 新增 `ToolAuditLog`(无 FK 无关系,`created_at` 用 server_default;`status`/`tool_source` 用 Enum 列,值取 DDL 中文枚举原样)。
- 每次进入执行引擎的调用恰落一条(成功/失败/超时/校验拦下/权限拒绝),`conversation_id` 取 ctx、`arguments` 存原始 dict、`duration_ms` 实测、`retry_count` 实录、`mcp_server` 取 spec。
- **create_ticket 例外(P8 细则)**:被拒转确认流的调用按**终局**落一条——resume「确认提交」执行成功→「成功」(result_summary 带工单号);resume「取消」→「权限拒绝」(error_message=「客户在预览卡片取消」)。避免预览流留下假「权限拒绝」。
- 写入:`app/tools/audit.py` 提供 async sink(ctx 注入,默认 DB writer;单测注入记录器;**sink=None 降级跳过**——无 DB 面的单测/评估不破)。轮内 await,失败只 WARN 不拦执行(需求5 红线)。
- 幻觉未登记调用的审计行:`tool_source` NOT NULL(DDL)——调用发生在我方面,记 `'builtin'` + `mcp_server=NULL` + 状态「权限拒绝」(未登记=未授权);`arguments` 存模型原始 dict。

## MCP Servers(需求6,P3)

| | logistics_server | aftersale_server |
|---|---|---|
| 端口/路由 | 8101 `/mcp` | 8102 `/mcp` |
| 工具 | `query_logistics(order_id)`(接管同名) | `query_warranty(order_id)`、`query_return_progress(order_id)` |
| 数据 | mock 轨迹(承运商池+节点状态码),照 ch02 random 做法 | mock 在保判定+退货进度节点列表 |

- 独立进程、不接真系统、不建表;契约(客服侧格式化/白名单依据):
  - query_logistics → `{carrier, tracking_no, current_status: <码>, nodes: [{time, status: <码>, desc}]}`;查无→`{found: false}`。
  - query_warranty → `{order_id, in_warranty: bool, expire_date, basis}`。
  - query_return_progress → `{found, return_no, current_status: <码>, nodes: [...]}`。
- 技术栈定死:MCP 官方 Python SDK + Streamable HTTP。**Context7 核对结论(规8,一处记死)**:v2 形制 `MCPServer` + `@mcp.tool()` + `mcp.run(transport="streamable-http")` / `streamable_http_app()`+lifespan `session_manager.run()`;文档版本列 v1.12.4 仍载 `FastMCP` 旧名——**实现以 uv 实际解析版本的导出名为准**,T1 首验(若 v1:import 路径 `mcp.server.fastmcp` 一行之差,不动架构)。
- 启动命令(demo 交付用):`.venv/Scripts/python.exe mcp_servers/logistics_server.py`(脚本内 `settings.port` 钉 8101;uvicorn 内嵌)。

## 建工单确认流(需求7,P6)

### 图拓扑变更

- State 新增:`ticket_preview: dict`(轮级:react 捕获 `{tool_call_id, ticket_type, description}`)、`ticket_decision: str`。
- agent 节点出口条件边:`ticket_preview` 非空 → `ticket_confirm`,否则 → `logging`。
- `ticket_confirm` 节点(**interrupt 前零副作用,重放安全**):

```python
async def ticket_confirm_node(state, config):
    decision = interrupt(dict(state["ticket_preview"]))   # 节点第一行即暂停
    ...                                                    # resume 后从头重放,到此取回 decision
    # decision == "confirm" → execute_tool(create_ticket, preview.args,
    #                            ctx(ticket_confirmed=True)) → 落 tickets 表
    #                            answer_text = f"已为您创建工单 {ticket_no}…"
    # decision == "cancel"  → 不执行 + 审计「权限拒绝」 + answer_text = "好的,已取消本次建单。"
```

- 终局走 `logging` → END;确认/取消话术为固定文本(不再烧 LLM,resume 后确定性出答)。

### 流与端点

- `stream_graph_turn` 增 stream_mode `updates`,检测 `__interrupt__` 载荷 → 发新帧 `("ticket_preview", {"ticket_type", "description", "conversation_id"})` 后正常收尾(routes 照发 done)——前端体验与 ch06 orders 卡同形,机制为 LangGraph 原生暂停(与需求7「照第6章订单选择器的做法」的类比一致,差异已在拍板 P6 记死)。
- 新端点 `POST /api/tickets/confirm`,body `{conversation_id, decision: "confirm" | "cancel"}` → `graph.astream(Command(resume=decision), config{thread_id: conv-{cid}})` → 返回**新 SSE 流**(复用 routes 现 SSE 适配件,工单号以 token 帧续播)。
- 线程态丢失(InMemorySaver 进程内,ch07 D3 同源边界):resume 命中无待确认 interrupt 的 thread → **409** `{"detail": "确认已过期,请重新发起建单"}`。
- **implicit-cancel 语义**(预览卡片弹出后用户不点按钮、直接发新消息):普通聊天轮进入前检测线程有 pending interrupt → 先以 `"cancel"` 自动 resume(建单不执行+审计「权限拒绝」,语义准确:未经确认即放弃),再正常跑新输入。绝不让 pending interrupt 把新轮炸掉或吃掉。
- ch05 投诉流按钮路 `/api/tickets` POST 原样不动(需求7 原文红线)。

### react.py 变更收口

- bind 集合改为当轮快照全集(含 create_ticket);:109-117 ch05 幻调硬闸**删除**,拒绝职责并入 executor 权限闸;create_ticket 的 tool_result 帧照现有面(ok=False+summary「等待客户确认」)。

## 配置与依赖

- `uv add mcp langchain-mcp-adapters jsonschema`(P2)。
- Settings 新键:`mcp_logistics_url` 默认 `http://127.0.0.1:8101/mcp`、`mcp_aftersale_url` 默认 `http://127.0.0.1:8102/mcp`;`tool_timeout_seconds` 默认 5.0→10.0。
- 演示 env:无必填新键(URL 默认即 demo 值;演示超时调小 env `TOOL_TIMEOUT_SECONDS=2` 这类)。

## 测试策略

- **单元(TDD,全量覆盖新逻辑)**:ToolSpec 合并/撞名丢弃/server 降级;校验闸三类拦截;权限闸(write 拒/凭证放行/模型面不可伪造);重试分类白名单+write 恒 1 次;三类分诊回灌文本;格式化投影+枚举翻话+中文不转义;审计行形状(含 sink=None 降级、写败 WARN、create_ticket 终局规则);ticket_confirm 重放安全(interrupt 前无副作用钉测);stream_graph_turn interrupt→ticket_preview 帧;confirm 端点 confirm/cancel 双路+409。
- **集成(@integration 标记)**:起双 server 真链路——发现合并、经 MCP 查物流(验收2 骨架)、server 侧临时加工具后 client 新快照即见(验收3 自动化近似;正式验收 P9 手工演);MCP 超时/拒连降级面。
- **e2e**:验收4/5 全流(聊天建单→追问→preview 帧→confirm→落单带工单号 / cancel→不建单+审计「权限拒绝」);验收6 超时演示自动化钉。
- **非单测产物(工作要求1)**:「校验拦下后模型自纠」「缺必填先追问不瞎编」「分诊话术如实回」= 标注样例+评估集(evals/ 扩展),跑一遍验证,不硬套 TDD。
- **前端预览卡片**:Vibe Coding 例外,不套流程;帧契约=`ticket_preview` 帧字段(本节钉死,前端只消费)。

## 验收映射

| # | 验收条目 | 落点 |
|---|---|---|
| 1 | 只注册即用不动核心代码 | P9:server 加 `@mcp.tool` → 重启该 server → 下轮即用(零客服侧代码);e2e/集成钉 |
| 2 | MCP 查物流 | 集成+演示;query_logistics 唯一来源=MCP(P7) |
| 3 | Server 加工具客服侧不动 | P4 每轮现拿;集成测试模拟 |
| 4 | 追问+预览+确认落单带工单号 | 校验闸+确认流+e2e |
| 5 | 取消=不建单+审计「权限拒绝」 | confirm 端点 cancel 路+审计终局规则 |
| 6 | 超时有重试+retry_count+状态+耗时齐;写超时不自动重试 | 重试白名单+write attempts=1+审计列;e2e 钉 |

## 已知边界与挂账(明示不装)

- InMemorySaver 进程内:多 worker/重启 → 待确认 interrupt 丢,confirm 得 409(ch07 D3 同源挂账,本章把它变成用户可见语义)。
- 审计表只增不清(保留策略后续章)。
- 每轮两次 MCP 会话的本地开销(demo 规模可接受;生产化=缓存/TTL 挂账,P4 备选)。
- 外部 server 写语义不支持(P5:一律 readonly)。
- Skill 机制、更多外部系统:本章不做(需求文「本章不做」节原文)。
- naive_agent_loop / tool_chat_service 两个 legacy 面不吃 MCP 新工具(单漏斗的闸与审计照吃)。

## 附录一 · 用户原话全文(2026-09-29 需求文)

> 我要用 Superpowers 模式把客服系统的工具层从几个写死的内置工具,升级成即插即用的工具系统。
>
> **功能需求**
> 1. 工具注册中心:不管内置还是 MCP 来的工具,一律带齐工具名、用途描述、JSON Schema 参数定义三样,登记进同一份工具表;内置工具服务启动时登记,MCP 工具连上 Server 动态发现、现问现拿;新工具注册进来就能被主力 Agent 用上,不改核心代码、不重启服务
> 2. 参数校验:工具执行前统一按 JSON Schema 校验一遍,类型不对、必填缺失、取值越界的拦下;拦下不抛异常了事,把校验错误说明包成一条工具结果回灌给模型,让它追问用户或重新组织调用
> 3. 权限控制:工具分只读、写两类,写操作由执行引擎把门;本系统的写操作就 create_ticket 一个,规矩是——只有客户明确要求建工单时才发起,真正执行还要等第 7 点的前端确认回来,没确认的调用引擎直接拒绝,不给模型绕过去的机会;外部 MCP 工具的用途声明是对方自己写的、不可信,能不能调(尤其写操作)只认我们这侧的权限规则,不看 Server 声明、不让模型临场判断
> 4. 执行引擎:所有工具调用走同一处,统一管超时、重试、错误处理、结果格式化;重试只给网络抖动这类暂时性故障,业务空结果不重试,写操作默认不自动重试(超时未必没执行,重复执行比失败更糟);错误按参数不合法、查询落空、真故障三类分诊,坏消息如实回给模型;结果只挑回答用得上的字段、内部枚举码翻成人话、序列化 JSON 中文不转义
> 5. 审计留痕:建 tool_audit_logs 表,每次工具调用落一条——所属会话、tool_call_id、工具名、来源是内置还是哪个 MCP Server、调用参数、结果摘要、状态(成功/失败/超时/校验拦下/权限拒绝)、错误说明、重试次数、耗时、时间;被权限拒、被校验拦的调用同样要落;审计表不挂外键,写审计失败不许反过来拦工具执行
> 6. MCP 接入:自建两个业务 MCP Server——物流(查物流轨迹)、售后(查在保、查退货进度),各自独立进程,Server 内部照第 2 章工具的做法随机生成 mock 数据返回,不接真实系统、不建表;客服系统作为 MCP Client 接入,拿回的工具跟内置清单合成一份,主力 Agent 一视同仁地调;物流查询由物流 MCP Server 接管,第 2 章内置的 query_logistics 从清单里下线,别留重名工具
> 7. 建工单确认流(带前端配套):客户明确要求建工单才走这条流,Agent 先核对信息够不够,缺问题描述这类必填项就主动追问用户,不许瞎编;凑齐后发起 create_ticket,执行引擎不直接执行,照第 6 章订单选择器的做法用 LangGraph interrupt 把工单预览(工单类型、问题描述)推给前端;前端渲染成预览卡片,带「确认提交」「取消」两个按钮:点确认,回传后端 resume 放行,落 tickets 表,回复里把工单号带给用户;点取消,不执行,这次调用按「权限拒绝」落审计;第 5 章投诉流程里前端按钮建单那条路保持不动,点按钮本身就是用户确认
>
> **技术栈**:MCP Server 用官方 Python SDK 自建、Streamable HTTP;MCP Client 用 langchain-mcp-adapters 的 MultiServerMCPClient 多 Server 接入。
> **本章不做**:Skill 机制(生态概念不落代码);仓储这类更多外部系统。
> **验收标准**:见「验收映射」节六条(原文同)。
> **工作要求**:全程 Superpowers;非单测代码任务 TDD 换标注样例/评估集;前端卡片 Vibe Coding 例外;dev-notes/ch08.md 每阶段实时追记四样(关键原话/关键产出/拒绝纠偏/翻车返工);库 API 一律 Context7 先查;选型矛盾即停问用户;完结交付=演示命令+测试结果+dev-notes 路径。

## 附录二 · tool_audit_logs DDL 原文(逐字入 db/init/08_ch08_tool_audit.sql)

```sql
-- =============================================================
-- ch08 · 工具系统 · 建表 DDL
-- 本章新建:tool_audit_logs(工具调用审计留痕,统一执行引擎每次调用落一条)
-- 不挂外键:审计写入不能被引用约束拦住,conversation_id 只建普通索引
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 工具调用审计:内置和 MCP 工具都记,被权限拒、被校验拦的调用同样落一条
CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT '工具名',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT '工具来源:内置 / MCP 接入',
  mcp_server      VARCHAR(64)     NULL                    COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments       JSON            NULL                    COMMENT '调用参数',
  result_summary  TEXT            NULL                    COMMENT '返回结果,过长截断存摘要',
  status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message   VARCHAR(512)    NULL                    COMMENT '失败 / 拦下时的原因说明',
  retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0     COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms     INT UNSIGNED    NULL                    COMMENT '耗时毫秒',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';
```

## 附录三 · LangGraph interrupt 核对结论(规8,Context7 三核之一)

- `from langgraph.types import interrupt`;节点内 `interrupt(payload)` 抛 GraphInterrupt 暂停,值经 checkpoint 浮现客户端;**必须配 checkpointer**(现 InMemorySaver 满足)。
- 恢复=`Command(resume=值)` + 同 `thread_id`;**恢复=该节点从头重放、全部逻辑重新执行**——P6 独立节点、interrupt 前置零副作用的设计依据。
- 响应按 interrupt 序号匹配(单 interrupt 场景无歧义)。
