# MewHelp 电商智能客服系统

「喵帮」电商智能客服项目，按章递进：

- **ch01 纯对话**：SSE 流式多轮对话 + Prompt 模板管理 + 售后描述结构化提取 + token 预算历史裁剪
- **ch02 Function Calling 工具链**：五个 LangChain 工具 + MySQL 四表落库 + 单轮调用收敛 + 前端工具徽章

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
docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp < db/init/03_ch03_schema.sql
docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp < db/init/04_ch03_seed.sql
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
