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

# 验收3: 邮费 → query_faq 漏召回（LIKE 检索的预期结果）→ 如实告知查不到并建议转人工
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
