# MewHelp 电商智能客服系统

「喵帮」电商智能客服项目，按章递进（每章一节，见下文；含开发留痕 dev-notes/）：

ch01 SSE 流式对话 → ch02 Function Calling 工具链+MySQL → ch03 RAG 语义检索 → ch04 混合检索+重排+评估体系
→ ch05 LangGraph 工作流+ReAct → ch06 分流器 → ch07 上下文三层管理 → ch08 工具注册中心+MCP+三道闸+确认流
→ ch09 Langfuse 可观测性+数据飞轮 → ch10 自训 17 类主题分类器（RoBERTa-wwm-ext 全参微调）+旁路归类+主题分布后台

## 技术栈

Python 3.12 · FastAPI（内置 `fastapi.sse`）· LangChain 1.x · SQLAlchemy 2（async）+ MySQL 8.0（Docker）· OpenAI 协议直连任意上游（百炼 qwen / GPT / Claude / DeepSeek / Ollama，只改 `.env`）

## 快速开始

```bash
# 1. 安装依赖（需先安装 uv: https://docs.astral.sh/uv/）
uv sync

# 2. 配置模型接入（三行搞定换模型）
cp .env.example .env
#    编辑 .env 填入: OPENAI_BASE_URL / OPENAI_API_KEY / MODEL_NAME
#    百炼密钥获取: https://bailian.console.aliyun.com/?apiKey=1
#    注意: .env 优先于系统环境变量（防环境变量污染）

# 3. 启动服务
uv run uvicorn app.main:app --port 8000
```

- 聊天页面: http://127.0.0.1:8000/ （左侧流式对话 + 工具调用徽章，右侧售后提取面板）
- API 文档: http://127.0.0.1:8000/docs
- ch02 需先起 MySQL：见下文「ch02 · 前置」

## API

| 端点 | 说明 |
|---|---|
| `POST /api/chat/stream` | SSE 流式对话。事件：`token`（增量文本，data 为 JSON 字符串）/ `done`（`[DONE]`）/ `error`（`{"detail":...}`）。请求体 `{"messages":[{role,content}...]}`，末条必须 user |
| `POST /api/extract` | 售后描述 → `{"order_id", "issue_type", "expected_solution"}` 结构化 JSON |
| `GET /api/health` | `{"status","model","history_token_budget"}`（不含敏感信息） |

## 验收命令（对应 ch01 三条验收标准）

> ⚠️ Windows Git Bash 下 curl 内联中文 JSON 会被 argv 编码损坏（报 "There was an error parsing the body"）。
> 解决：把请求体存为 UTF-8 文件后 `curl -d @body.json`。PowerShell / Linux / macOS 可直接内联。

```bash
# 验收1: SSE 流式回复（-N 关闭缓冲，观察逐 token 输出，结尾 [DONE]）
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好，你们支持7天无理由退货吗？"}]}'

# 验收2: 多轮上下文（第二轮携带第一轮历史，回复应接住"那运费呢"的指代）
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [
    {"role": "user", "content": "我买的鞋子想退货"},
    {"role": "assistant", "content": "好的喵，请提供订单号，并说明退货原因～"},
    {"role": "user", "content": "订单号是 DD20260905077，尺码不合适。那运费呢？"}
  ]}'

# 验收3: 结构化提取
curl -X POST http://127.0.0.1:8000/api/extract \
  -H "Content-Type: application/json" \
  -d '{"description": "订单 DD20260901001 的耳机右声道没声音，才买一周，我要退货退款。"}'
# 预期: {"order_id":"DD20260901001","issue_type":"退货","expected_solution":"..."}
```

## ch02：Function Calling 工具链

### 前置：启动 MySQL

1. 启动 Docker Desktop
2. `docker compose up -d`（首次初始化约 30-60s，等 `docker compose ps` 显示 healthy）

容器宿主侧端口 **3307**（本机原生 MySQL 占用 3306）。`.env` 需包含 `MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DB/DEMO_USER_ID/TOOL_TIMEOUT_SECONDS/TOOL_MAX_RETRIES`（见 `.env.example`，全有默认值）。MySQL 连不上时服务启动即退出并提示先起容器。

### 五个工具

| 工具 | 数据源 | 说明 |
|---|---|---|
| query_order | mock | 按订单号查状态/金额/明细（同订单号结果稳定） |
| query_product | mock | 按商品编号查名称/价格/库存 |
| query_logistics | mock | 按订单号查承运商/轨迹 |
| query_faq | MySQL faq 表 | 关键词 LIKE 检索平台政策 |
| create_ticket | MySQL tickets 表 | 创建人工工单，会话置「已转人工」 |

模型第一轮决定是否调工具（可同轮并行多个），工具结果回灌后第二轮收敛出最终回答——**单轮收敛，无多轮 Agent 循环**（后续章节）。

### SSE 事件（在 ch01 的 token/done/error 之上新增）

| 事件 | data | 时机 |
|---|---|---|
| conversation | `{"conversation_id": 3}` | 每轮第一帧，服务端会话 id |
| tool_call | `{"id","name","args"}` | 模型决定调用工具 |
| tool_result | `{"id","name","ok","summary"}` | 工具执行结束（ok:false 含超时/异常） |

每轮完整落库 messages 四行链：user → assistant(带 tool_calls) → tool(结果) → assistant(最终回答)。

### 验收命令（对应 ch02 三条验收标准）

中文 body 先写 UTF-8 文件再 `-d @file`（注意事项同 ch01）：

```bash
# 验收1: 物流查询 → tool_call(query_logistics) → tool_result(ok:true) → 按结果作答
printf '{"messages":[{"role":"user","content":"订单 1001 的物流到哪了"}]}' > a1.json
curl -N -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @a1.json

# 验收2: 退货政策 → query_faq 命中（summary=命中 1 条），回答含「7 天」等种子要点
printf '{"messages":[{"role":"user","content":"退货政策是什么"}]}' > a2.json
curl -N -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @a2.json

# 验收3: 邮费 → query_faq 漏召回（LIKE 检索的预期结果）→ 如实告知查不到并建议转人工（**ch03 起此问已能语义命中，见下文 ch03 节**）
printf '{"messages":[{"role":"user","content":"邮费是多少"}]}' > a3.json
curl -N -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @a3.json
# 漏召回原文留痕 dev-notes/ch02.md「Task 10」段，ch03 向量检索升级的输入

rm a1.json a2.json a3.json
```

## 测试与评估

```bash
uv run pytest                               # 单元测试（mock 模型，零 token 消耗）
uv run python evals/run_extraction_eval.py  # 提取 Prompt 标注评估集（真实模型，5 条样例）
uv run python evals/run_tool_routing_eval.py # 工具路由评估集（真实模型，10 条样例，阈值 8/10，固定 temperature=0 可复现）
```

## 项目结构

```
app/
├── main.py            # FastAPI 入口（lifespan：配置校验、MySQL 探活、静态托管）
├── core/config.py     # pydantic-settings（.env 优先于系统环境变量）
├── prompts/           # 客服 System Prompt（含工具使用指引）/ 提取 Prompt（few-shot）
├── schemas/           # Pydantic 数据类（chat 含三种 SSE 事件 / extraction）
├── db/                # engine（async 引擎+探活）/ models（四表）/ crud（FAQ 检索、建单等）
├── tools/             # definitions（五工具）/ registry / executor（超时+重试+错误包装）
├── services/          # chat_service / extract_service / tool_chat_service（单轮编排）/ persistence（落库）
└── api/routes.py      # SSE 对话（工具链+落库）/ 提取 / health
db/init/               # MySQL DDL（四表）+ 种子数据（faq 10 条 + 演示数据）
docker-compose.yml     # 本地 MySQL 8.0（3307，--skip-character-set-client-handshake 锁 utf8mb4）
static/index.html      # 聊天单页（SSE 解析 + 工具徽章 + 会话 id 闭环）
evals/                 # 标注评估集（提取 5 样例 / 工具路由 10 样例）
docs/superpowers/      # spec 与实施计划
dev-notes/             # 开发过程留痕（ch01 / ch02）
```

## ch02 不做（后续章节）

多轮 Agent 循环、RAG/向量检索（ch03：修复 FAQ LIKE 漏召回）、鉴权、部署。

## ch03:RAG 知识库(query_faq 语义检索版)

### 准备(老库升级一次性)
```bash
docker compose up -d
docker compose exec -i mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" mewhelp' < db/init/03_ch03_schema.sql
docker compose exec -i mysql sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" mewhelp' < db/init/04_ch03_seed.sql
```

### 建库与挖知识
```bash
uv run python -m app.jobs.build_knowledge          # 全量重建 knowledge/*.md → MySQL pending → Milvus → done
uv run python evals/run_rag_eval.py                # 检索评估 hit-rate@3 ≥10/12(纯文档库上跑)
uv run python -m app.jobs.mine_qa                  # 历史对话 → LLM 抽 QA → 三道闸 → 入库+向量化
uv run python evals/run_qa_mining_eval.py          # 抽取质量评估(真 LLM)
```

### 验收演示
- 验收 1:`uv run uvicorn app.main:app` 后问「邮费是多少」→ query_faq 徽章 + 99 包邮/8 元答案
- 验收 2:`--fault-after 3`(exit 42)→ 查 pending → `--skip-existing` 补齐 → `--check` 差集 ∅;
  全量重建会清 qa_mined(设计内),`mine_qa --reprocess-kept --dedup-only` 找回
- 对账:`uv run python -m app.jobs.build_knowledge --check`

参数在 `.env`(RAG_TOP_K / RAG_SCORE_THRESHOLD / QA_DEDUP_THRESHOLD…,默认值见 app/core/config.py)。
开发过程留痕:`dev-notes/ch03.md`;设计:`docs/superpowers/specs/2026-09-21-…-design.md`。

## ch04 RAG 进阶:混合检索 + 重排 + 评估体系

- **混合检索**:Milvus 原生 BM25(text 字段挂 BM25 Function、内置 chinese analyzer)+ dense 向量各 Top-50 → `hybrid_search` + RRF(k=60)融合;品类元数据先过滤再检索。
- **重排**:SiliconFlow `bge-reranker-v2-m3` 精排 Top-10;组装 prompt 用首尾排布(奇数位升序+偶数位降序),最相关在两端。
- **Query 理解**:口语→标准问法改写 + 检索侧同义词扩展(不入库);评估走 `evals/cache/rewrite_cache.json` 保四臂同输入。
- **前置双闸拒答**:检索侧 top1 置信 < `retrieval_low_conf_threshold`(实测校准终值)→ 拒答+进池;生成前证据自评不足 → 拒答+进池(`low_confidence_questions`,source 标写方)。
- **引用与反馈**:回答 `[n]` ↔ tool_result 帧 `citations` → 聊天页点角标弹窗看原文+章节路径(`GET /api/chunks/{id}`);每条回答 👍/👎 一次性锁定(纯前端 localStorage,数据飞轮入口)。
- **忠实度**:LLM-as-judge(`evals/run_faith_eval.py`)判 fabricated → `faith_cases` 台账(`static/faith.html`,GET/PATCH `/api/faith_cases`,复发自动退回)。
- **评估**:`evals/run_strategy_eval.py` 四策略×分桶 Hit/Recall@3、@10 + MRR@10 + D 桶阈值校准表 → `evals/reports/`。

```bash
uv run python -m app.jobs.build_knowledge   # 老师 6 文档重建(knowledge_chunks + Milvus 双写)
uv run python evals/run_strategy_eval.py    # 验收①:四策略对比报告
uv run python evals/run_faith_eval.py       # 忠实度评估 + 台账落盘
uv run python evals/demo_ch04.py            # 验收②③④数据面演练
uv run pytest -q                            # 全量单测
uv run pytest -m integration -q             # 集成(需 docker mysql/milvus + key)
```

注意:重跑 `run_strategy_eval`/`run_faith_eval` 会**覆盖** `evals/reports/` 两份已提交报告(忠实度报告尾部手工「补充解读」策展节需重附);全量跑为 live 云调用、有成本(成本闸见各脚本 `--limit`/`--sample`);`run_faith_eval --sample 0 --d-limit 0` 会把报告洗成空表。

## ch05 LangGraph 工作流编排 + ReAct Agent

- **Graph 编排**:`StateGraph` 指代消解→意图识别→四路分流(knowledge→retrieve→gate/agent、data→agent、complaint、chitchat)→日志→END;`InMemorySaver` 仅进程内跨轮(D3),线程键 `conv-{cid}`;降级(cid=None)走 anon 线程+整包客户端历史(与 ch01 无状态语义对齐)。
- **两层防幻觉**:意图 JSON 解析失败重试 1 次仍败归 knowledge(P2);知识置信闸(纯阈值,P1)证据弱直接兜底拒答、不进 Agent,拒答同时落 `low_confidence_questions` 池。
- **ReAct 流式 Agent**:思考→工具→结果喂回→再思考直到收敛;`max_agent_steps=6`(ch07 T8 接管,原 `react_max_iterations` 键删除) / `react_token_budget=8000` 双熔断(P3),超限用已有信息收尾并建议转人工;每步 `logger.info` 步数可见;ch07 T8 起证据/订单注入并入段5 装配,react 层不再做 System 前置。
- **suggestions 帧 + 两独立按钮**:投诉出口末 token 后、done 前发 `suggestions` 帧(P4);前端「转人工」纯前端话术零 fetch,「建工单」确认后才 `POST /api/tickets` 写 tickets——两按钮互不绑定,建单不再置会话「已转人工」(crud 副作用已移除,D4);不点=无任何动作。
- **接线与基线**:`/api/chat/stream` 换 Graph 编排;ch04 `stream_chat_with_tools` 函数保留作回归基线(P5)。token/done/error 帧逐字符红线不动。

```bash
uv run uvicorn app.main:app --port 8000        # 起服务(docker mysql/milvus 需在线)
# 浏览器开 http://127.0.0.1:8000/ 演示话术:
#   验收1 退货政策是什么        → 日志行 ch05 graph turn 含 retrieve/gate
#   验收2 订单 1001 的物流到哪了 → Agent 自调工具(tool_call 帧)
#   验收3 我要投诉              → 两个独立按钮;只点转人工=零后端动作;点建工单才出工单号
#   验收4 你好呀                → 固定话术
#   验收5 先查下订单 1001 买的是什么商品，再按这个商品名在 FAQ 里查下使用说明 → ReAct 两步(agent_steps≥2 日志可见)
uv run pytest -q                                              # 全量单测(默认 deselect integration)
uv run pytest tests/e2e/test_ch05_acceptance.py -m integration  # 端到端验收1-5(真模型+真库,活环境)
```

## ch06 Workflow 最关键的一个节点:正式版分流器

- **指代消解+Query 改写合一**:`coref` 节点带历史(近 6 条)把「它/这个/换货吧」补全成自包含标准问法;已完整的问题原样通过不强行改写;短确认/选项作答必须带上历史里的主体事由。
- **意图识别 Prompt 四件套**:八类枚举(含「其他」)/强制 JSON 恰好 `{intent, confidence}`/边界 few-shot(「退货政策是什么」→商品咨询 vs「这单猫粮要退掉」→退款退货)/拿不准归其他不硬塞——纯 Prompt 方案,不训分类模型;解析失败重试 1 次仍败归「其他」。
- **Query 扩写(仅退款退货/售后)**:一次提问拆≤4 条检索问法(原问法居首),强制 `{"queries": [...]}`;政策条款多路并行检索→chunk_id 去重取最高分→截 rerank_top_n;知识库存法零改动(只查询侧拆,不入库侧拆存)。
- **退款确定性子流程**:intent=退款退货/售后 → 槽位检(正则提取单号,模型不许瞎猜)→ 缺号弹 `orders` 帧卡片(P5 点卡片发「我选择订单 {id}」协议句续跑,选择句+槽位直通零模型消耗)→ 取单数据 → 扩写+政策强制检索 → 置信闸 → 只有「这一单能不能退」进主 Agent;gate 通过固定挂 `refund_apply` 建议。
- **退款申请表单(P6)**:`refund_apply` 建议 → 前端轻表单,原因只收固定四类(七天无理由/商品质量问题/拍错多拍/其他),提交 `POST /api/refunds` 落 tickets 表 `ticket_type=「售后」`,描述服务端拼装、自由文案进不来。
- **降本降级路(默认关)**:`intent_small_model` 配小模型先判,置信 < `intent_confidence_threshold`(0.75)再升主模型;关时行为=ch05 直判。
- **SSE 帧序红线**:token…→(persist)→`orders`→`suggestions`→done;token/done/error 逐字符不动(ch05 契约)。

```bash
uv run uvicorn app.main:app --port 8000        # 起服务(docker mysql/milvus 需在线)
# 浏览器开 http://127.0.0.1:8000/ 演示话术:
#   验收1 订单1001的物流到哪了 → 这个订单我想退掉 → 它的物流呢(三轮意图+消解逐轮对)
#   验收2 asdfgh 我不知道我想问啥 你们软件好奇怪 → 归「其他」走知识路,全程无异常
#   验收3 我买了订单1001的冻干猫粮 → 这个能退吗 → refund 链+政策检索命中+Agent 结论
#   验收4 我要退款(不说单号) → 弹订单卡片 → 点一张 → 结论 → 末条挂「发起退款申请」→ 表单拿票号
uv run pytest -m integration -q -p no:cacheprovider tests/e2e/   # 双章 e2e(A1–A5,B1–B3)
uv run python evals/smoke_ch06.py                                # prompt 质量真模型冒烟(不进 CI)
```

## ch07 会话上下文管理:三层分层 + 后台异步摘要 + 多会话侧栏

- **三层模型**:层1=近史原文(DB messages 按 `layer1_from_msg_id` 锚切)/层2=降级批(只挪锚不动行)/梗概=`conversation_summaries` 分段表+单列投影。装配=五段:System→层2半压(用户全留、客服留头 `assistant_head_chars`)→层1 原文→当前句→段5 合并注入(梗概投影+知识库证据+订单数据,**一条 Human**;react 不再 System 前置)。
- **预算公式同源**:`compute_budgets` = 窗-输出-输入-工具峰值(步数×工具上限)-固定项(System 543/证据面 2500/梗概注入 707/安全垫 1000),滑窗取 `min(窗口面, turns_to_keep×steady)`;层1/层2 七三开。demo 组六键(`MODEL_CONTEXT_WINDOW=18000 MAX_AGENT_STEPS=3` 等;六键中仅这两键异于默认)钉死锚点 **5650/3954/1695**;token 计数=CJK 估算器(汉字 1:1、其余 4:1),react 熔断同源。
- **后台异步摘要不阻塞**:ctx 入口节点每轮先降级再判层2 超预算→`asyncio.create_task` 排任务(防重入 in-flight);当轮帧序零变化,summary done 落库+投影追平边界。**多 worker 挂账**:in-flight 是进程内集合,spec 明言单 worker 语义;多进程部署需外置队列(未做)。
- **只读 API×2(P8)**:`GET /api/conversations`(id 降序+首问预览 40 字+已摘要标记)、`GET /api/conversations/{id}/messages`(全行升序含 tool 行;不存在/非属主同答 404;引擎未起 503)。前端侧栏:点选换轨续聊(清 checkpointer 场景=回填供史)、tool 行浅色 🔧、「＋新对话」旧会话留栏;列表失败=一行提示不挡聊天。
- **日志留痕(验收4 grep 锚)**:`model_ctx cid=`(逐条+tokens≈)/`history_ctx cid=`(**每轮必打**,闲聊轮也有)/`summary trigger|done|skip|failed`/`层1 降级 X→Y`;落盘 `log/app.log`(UTF-8)。**已知边界**:FileHandler 无轮转(≈10-30KB/轮,RotatingFileHandler 零依赖可解未做);梗概投影 `summary` 列 TEXT 上限≈64KB(约 160 段后触顶);>400 字梗概硬截可断数字串尾(罕见)。
- **输入闸**:`MAX_USER_INPUT_TOKENS`(默认 2000)超限 422(估算器同源,`ChatRequest` 校验)。

```bash
# demo env(spec 验收锚,写进 .env 或命令行):
MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000 \
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5 \
uv run uvicorn app.main:app --port 8000
# 浏览器开 http://127.0.0.1:8000/ 演示话术:
#   验收3 默认窗连聊 20 轮 → log grep -c "层1 降级" = 0(装得下就不压)
#   验收4 demo env 连聊 ~20 轮(用户句用 200 字级长文;层2 触发按半压后估算,
#         轻话术轮均≈80 攒不满 1695 预算)→ grep "summary trigger"/"层1 降级"/"model_ctx" 齐现,当轮关流不卡
#   验收5 侧栏:点旧会话回载气泡、tool 行浅色🔧、「＋新对话」清屏旧会话留栏
uv run pytest -m integration -q -p no:cacheprovider tests/e2e/   # 三章 e2e(A1–A5,B1–B3,C1–C4)
uv run python evals/smoke_ch07.py                                # 摘要 prompt 真模型冒烟(不进 CI)
```

## ch08 工具层升级:注册中心 + MCP 即插即用 + 三道闸 + 确认流

- **注册中心三件套**:`ToolSpec(tool, permission, source, mcp_server)` / `BUILTIN_SPECS`(内置视图)/ `snapshot_tools(settings)` 每对话轮现拿「内置 ∪ MCP」快照,**无缓存**(P4:adapters 每次调用新建会话,Server 侧加/改工具下轮即见,客服侧零改动)。`query_logistics` 唯一来源=MCP(P7 内置版已下线,无共存窗口)。
- **三道闸(单漏斗 executor)**:闸序=校验→权限(spec 自查订正)。JSON Schema 校验闸(必填缺失/枚举外→「参数不合法: 」回灌,模型向用户追问,不瞎编);权限闸(read/write 两级,唯一 write=`create_ticket`——未确认时停在 `awaiting_confirmation` 中间态进确认流,**中间态不落审计**);执行闸=超时+重试白名单(只对暂时性故障 ConnectionError/Timeout 类重试,业务硬错不重;`write` 恒 attempts=1——重复建单比失败更糟,写超时给「人工核实」提示)。三类失败分诊回灌文案(校验拦下/权限拒绝/执行失败+超时)。
- **审计 `tool_audit_logs`**(db/init/08):每次真实执行一行——工具名/来源/参数摘要/结果摘要/状态(成功/失败/超时/校验拦下/权限拒绝)/`retry_count`/`duration_ms` 全列;审计写败只 WARN 不拦执行(需求5 红线)。
- **双自建 MCP Server(Streamable HTTP)**:`mcp_servers/logistics_server.py`(8101:`query_logistics`)、`mcp_servers/aftersale_server.py`(8102:`query_warranty`/`query_return_progress`)。env:`MCP_LOGISTICS_URL`/`MCP_AFTERSALE_URL`(默认 `http://127.0.0.1:8101/mcp`、`.../8102/mcp`);连不上=当轮静默降级只剩内置面(WARN)。
- **LangGraph interrupt 建工单确认流**:create_ticket 必填齐→权限闸给 `ticket_request` 内部事件→条件边进 `ticket_confirm` 节点 `interrupt` 暂停,末帧 `ticket_preview {tool_call_id, ticket_type, description, conversation_id}`(暂停轮无 done/suggestions,前端卡片照挂)。`POST /api/tickets/confirm {conversation_id, decision: confirm|cancel}` 以 `Command(resume)` 续跑:confirm→真建单+审计「成功」+零 LLM 模板答复回工单号;cancel→不建单+审计「权限拒绝」;新消息隐式 cancel(drain 旧 interrupt 落账);resume 重放安全(interrupt 前零副作用);挂起态再 confirm=409,缺参=422(依赖层预检,标准 JSON 非 SSE)。
- **前端**:preview 帧→工单预览卡片(「✅ 确认建单」「✖ 取消」,点击后双钮置灰);确认后独立 SSE 简读续播工单号。

```bash
# 演示(三条命令按序;8101/8102 不起=当轮无 MCP 工具,不报错)
.venv/Scripts/python.exe mcp_servers/logistics_server.py    # 终端1:物流 MCP 8101
.venv/Scripts/python.exe mcp_servers/aftersale_server.py    # 终端2:售后 MCP 8102
uv run uvicorn app.main:app --port 8000                     # 终端3:主服务,浏览器开 http://127.0.0.1:8000/
# 验收演示脚本(P9 手工面):
#   验收1 只注册即用:往 logistics_server.py 加一个新 @mcp.tool(如查仓库)→只重启 8101
#         →客服侧零改动零重启,下轮即问即用
#   验收2 MCP 查物流:起两 Server 后问「订单 1001 的物流到哪了」→🔧 query_logistics(mcp 源)
#   验收4 建单确认:「帮我查下订单1002的物流，查完后务必建个工单记录猫粮包装破损的情况，
#         方便后续补发」(话术敏感性:纯「建个工单+缺货」会被判退款流,约 1/3 命中率,
#         可换顺承句重试)→出预览卡→点确认→回工单号 T…
#   验收5 取消:预览卡点「✖ 取消」→不建单,tool_audit_logs 落「权限拒绝」
#   验收6 超时:log/grep「超时」+ SELECT retry_count/duration_ms FROM tool_audit_logs
uv run pytest -q                                              # 单元全绿(含 e2e C4–C6 单元级钉)
uv run pytest -m integration -q -p no:cacheprovider          # 含 ch08 双 Server 真链路(T6)+三章 e2e
uv run python evals/run_ch08_eval.py                          # 三桶话术评估(先起两 Server;不进 CI)
```
- **已知边界(spec 挂账原文)**:InMemorySaver 进程内——多 worker/重启丢待确认 interrupt,confirm 得 409(=用户可见「确认已过期」语义);审计表只增不清;每轮两次 MCP 会话的本地开销(demo 规模可接受,缓存/TTL 挂账);外部 server 写语义不支持(P5:MCP 工具一律 readonly);naive_agent_loop / tool_chat_service 两个 legacy 面不吃 MCP 新工具(单漏斗闸与审计照吃)。

## ch09 可观测性与数据飞轮:Langfuse 追踪 + 缺口流水线 + 审核回写 + 评估趋势

- **Langfuse 自托管观测链**(profile `langfuse`:postgres/redis/clickhouse/minio + web/worker):LangGraph 编译挂 `CallbackHandler`,每轮一条 trace——意图/置信度/工具名/拒答 reason 进 metadata,retriever 召回段进 span;意图以 `intent:<名>` tag 落 trace 供报表聚合。密钥三键(`LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY`)只进 gitignored `.env`,compose 里全占位;未配置=回调静默不挂,主流程零拖累。
- **置信度双闸(定位如实记,终审口径订正)**:排序闸 `rag_score_threshold`(ch04 遗留,进 LLM 前的候选过滤)与池化闸是两个不同的闸,升级只动后者:池化线=ch09 T4 的 `evidence_conf` 综合分(形 `sum`,权重 0.9/0.05/0.05=top1/有效条数/间隙,θ=`evidence_conf_threshold` 0.168,300 题校准择优),在闸节点判「答不了」即低置信落池;`retrieval_low_conf_threshold`=0.161 为 ch04 闸1(rerank 后 top 口径)兼 rule 形缺键回落(ch05 老行为零漂移),本章未改其语义。
- **三入口落池 + 当轮召回快照**:`low_confidence_questions` 加 `retrieved_chunks` 快照列(source 扩 `user_feedback`)。漏斗全收口:`query_faq` 低置信拒答、`refund_gate`/gate 闸拒答、`self_check` 兜圈拒答、前端 👎(`POST /api/feedback`)——池写成功即 `spawn_process` 自触发流水线(拍板 2A 三入口平权,fire-and-forget 强引用防 GC)。重复 👎 允许重复落池(归并兜)。
- **数据飞轮流水线** `app/services/flywheel.py`:`process_lcq_row` 标准化→语义查重(候选=待审 ≤50,窗满 WARN 如实)→命中累加/未命中新行+建议答案,`matched_review_id` 写回同事务封口(幂等边界=非 NULL 即出集;并发输家条件更新整笔作废)。幻觉候选 id 按未命中处理+WARN。CLI 补扫:`uv run python -m app.jobs.flywheel [--limit N|--dry-run]`;LLM 提示词评估=标注样例集 9/10 过阈(dev-notes 阶段十二含唯一 FAIL 样本人核记录)。
- **审核 API + 后台页**:`GET /api/review_queue`、`GET /api/review_queue/{id}/detail`(归并原话+快照)、`PATCH /api/review_queue/{id}`(仅 待审→通过|驳回;通过必带核准答案否则 422;并发/迟到写手条件更新 422「非法流转」)。通过=双落:chunk 三元指纹复用(同问同答不重做,done 命中置态照走=「复用也是通过」)→ embedding+upsert+flush → 置态——Milvus/向量任一步失败 502 状态留待审可重试(半成功不许存在)。页面 `static/review.html`(复古像素形制,聊天页顶栏入口),👎 已真接上报。
- **评估趋势与成本报表** `app.jobs.eval_cycle`:一轮 `eval_runs` 行(recall@3/recall@10/mrr@10 + faithfulness 抽样,检索臂与阈值口径同 ch04 策略评估);`--trend` 出环比表、`--report` 按 intent tag 聚合 Langfuse trace 成本/延迟(不可达明确报错不拖主流程)。控制台 ASCII,中文表 UTF-8 落 `evals/reports/`。

```bash
# 部署(一次性;Milvus 栈已在跑的前提下)
docker compose --profile langfuse up -d        # lf 四件套+web(3001)+worker
docker exec milvus-minio sh -c 'mc alias set local http://127.0.0.1:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" && mc mb -p local/langfuse'   # 建观测桶(口令取容器 env;已存在=幂等)
# .env 补 LANGFUSE_HOST=http://127.0.0.1:3001 + 两把 key(web UI 建 project 后拿)

# ⚠⚠ 旧 MySQL 卷必读 ⚠⚠
# db/init/09、10 两个 DDL 只对「新卷」自动执行。已有 mysql-data 卷(从未 down -v)
# 必须手工应用,否则新列缺失——落池/快照整行写失败被吞:
#   docker compose exec -T mysql mysql -u root -p<pwd> mewhelp < db/init/09_*.sql
#   docker compose exec -T mysql mysql -u root -p<pwd> mewhelp < db/init/10_*.sql
# (本仓开发即此状态,09/10 已手工应用并双证接缝;down -v 会连 mysql/milvus 卷一起毁,禁止)

# 演示(按序)
uv run uvicorn app.main:app --port 8000        # 主服务
uv run python -m app.jobs.flywheel --dry-run   # 看未处理池行
uv run python -m app.jobs.flywheel --limit 50  # 补扫(幂等,已处理出集)
uv run python -m app.jobs.eval_cycle --limit 10 --trigger 手动
uv run python -m app.jobs.eval_cycle --limit 10 --skip-faith --trigger 定时
uv run python -m app.jobs.eval_cycle --trend   # → evals/reports/ch09_eval_trend.md
uv run python -m app.jobs.eval_cycle --report  # → evals/reports/ch09_langfuse_report.md
uv run pytest -q                                # 单元全绿(含 e2e 验收 2–6 单元级钉)
uv run pytest -m integration -q -p no:cacheprovider   # 活库/活 Milvus 面(含 ch04–09 全 integration)
```
- **验收演示脚本(P 手工面)**:
  - 验收1 观测链:问一句→Langfuse UI(3001)出现 trace,metadata 含 intent/置信度;拒答轮 reason 可见。
  - 验收2 缺口成案:问知识库没有的问题(如「我家猫不吃冻干能退吗」)→兜底话术→`static/review.html` 待审列表出现,详情含原话+当轮快照。
  - 验收3 审核闭环:详情弹层填核准答案→「通过」→同问再问即答对(硬基线:真模型面;T10 手测取证 chunk53 仍在库,演示资产)。
  - 验收4 👎 链:历史回答点 👎→队列数 +1(source=user_feedback),流水线自触发。
  - 验收5 趋势:`--trend` 两轮以上环比表(真库 run id 5/6 已落)。
  - 验收6 报表:`--report` 意图组表(注:qwen 兼容端未配单价,`totalCost` 恒 0 如实呈现;条数/延迟可用,配价后自动出成本)。
- **已知边界(spec 挂账)**:单用户 demo 无鉴权(admin 页/审核 API 裸奔,公网部署前必须加);查重候选窗=待审 ≤50,窗满理论可产生 twin 行;👎 上报失败前端只标注不重试;eval faith 面单样本超时降级跳过不进分母;Langfuse trace 采样全量(demo 规模)。

## ch10 自训主题分类器:RoBERTa-wwm-ext 全参微调 + 旁路批量归类 + 主题分布后台

- **17 类多标签主题分类器**(`hfl/chinese-roberta-wwm-ext` 全参微调,非 LoRA):类目契约唯一权威=术语表 `finetune/glossary.json`(四大类 退换货/物流/尺码/发票 领头,近邻边界钉:修归保修维修退归退换货、运费管钱物流管货、价保补差价优惠活动券满减),词表漂移红线=运行时双 assert(TopicClassifier init / evaluate 加载各一处)+ DDL 注释、前端图例两处派生面,同源自 glossary。
- **数据管道**:清洗(脱敏手机号/长单号/地址+全角归一)→ LLM 照术语表造数补缺(每类 100,「其他」减半,多诉求 directive)→ LLM 预标(JSON `{"labels":[..],"text":"错别字修正"}` 一调用双职责)+ 抽审面 `finetune/audit/annotation_review.csv`(全部多标签+每类 10%;按分层抽审口径收口——边界混淆类重点抽 42 条,判对 41/42,唯一金标错已改判,不做全量逐条审)→ 标签组合分层 80/10/10(终审 I-1:空白归一 keep-first 去重后重分发,同题跨集泄漏封死;test 241 行封存为 FINAL)→ 增强(同义词替换+中性前缀)只扩 train。定稿种子行 2220(去重 labeled 1957 + 短口语双诉求种子 263),多标签占 58.2%;分布 train 3450(含增强)/valid 241/test 241。
- **训练/评测**:`finetune/train.py`(lr2e-5/batch16/max_len128/epochs≤10/早停3/fp16,RTX 4060 全程 ~4 分钟)——训练毕 valid 扫 [0.30,0.70] 步 0.05 取 macro-F1 最优阈值写 `topic_config.json`(推理只认 config,不硬编码 0.5)。封存 test(n=241,去泄漏后口径)macro-F1 **0.8388** · micro-F1 **0.8866**(阈值 0.55;此前 0.8771 系同题跨集泄漏虚高,终审 I-1 去重重分布后以 r4 为准;边界抽审改判 1 条金标后终版),逐类 P/R/F1+每类二值混淆矩阵落 `finetune/reports/ch10_eval_report.md`,错例台账 misclassified.csv 人工抽判 20(真错~9 集中近邻边界与多标签次标签漏召,金标错~4——见 dev-notes 阶段七;错例两轮回灌均只动 train 面:第一轮口语锚进同义词表+定向造数 210 条,第二轮短口语双诉求种子 263 条——r3 暴露字面学习:裸「买大了想退」尺码 0.02,补概念样本后 r4 双命中恢复)。
- **旁路部署(实时主链路零 import 零调用)**:`app/jobs/topic_classify.py` 攒低置信度池未归类行 → 进程内批量推理 → `topic_classifications`(uk_question_id 一人一行幂等,`--rerun` 刷新,`--dry-run` 选池数钉「已归类不重选」),Langfuse 落一条 trace;ONNX 导出为可选件未做(拍板 4)。演示资产=池 qid 1-3 三行真归类。
- **只读 API + 后台页**:`GET /api/topics/distribution?days=N`(labels JSON Python 层展开计数,多标签句每类各计一次,pct 按标签出现总次数归一,count 降序;days<1→422,无库→503);`static/topic.html` 复古像素条形图(时间窗下拉,飞轮审核页入口),验收三钉 `tests/e2e/test_ch10_acceptance.py`。
- **后续优化方向(范围已冻结,不再开新章)**:近邻边界类数据回灌(保修维修/运费/价保/优惠活动,错例台账集中于这四对);终审 4 个 deferred minors(min_count 实现/days 溢出防御/日志可读性/双标注路径);ONNX 导出属性能优化,不影响功能完整度。

```bash
uv run python -m finetune.train --epochs 10                  # 全参微调+扫阈值(~4 分钟)
uv run python -m finetune.evaluate                           # 封存 test → 报告+错例台账
uv run python -m app.jobs.topic_classify --limit 50          # 旁路批量归类活库真写
uv run pytest tests/e2e/test_ch10_acceptance.py -m integration -v   # 验收三钉
