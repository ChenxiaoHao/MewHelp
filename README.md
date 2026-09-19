# MewHelp 电商智能客服系统 · ch01 纯对话

「喵帮」电商智能客服的第一章：跑通纯对话。SSE 流式多轮对话 + Prompt 模板管理 + 售后描述结构化提取 + token 预算历史裁剪。

## 技术栈

Python 3.12 · FastAPI（内置 `fastapi.sse`）· LangChain 1.x · OpenAI 协议直连任意上游（百炼 qwen / GPT / Claude / DeepSeek / Ollama，只改 `.env`）

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

- 聊天页面: http://127.0.0.1:8000/ （左侧流式对话，右侧售后提取面板）
- API 文档: http://127.0.0.1:8000/docs

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

## 测试与评估

```bash
uv run pytest                              # 单元测试（mock 模型，零 token 消耗）
uv run python evals/run_extraction_eval.py # 提取 Prompt 标注评估集（真实模型，5 条样例）
```

## 项目结构

```
app/
├── main.py            # FastAPI 入口（lifespan 配置校验、静态托管）
├── core/config.py     # pydantic-settings（.env 优先于系统环境变量）
├── prompts/           # 客服 System Prompt / 提取 Prompt（few-shot）
├── schemas/           # Pydantic 数据类（chat / extraction）
├── services/          # chat_service（裁剪+流式）/ extract_service（结构化）
└── api/routes.py      # SSE 对话 / 提取 / health
static/index.html      # 聊天单页（fetch+ReadableStream 解析 SSE）
evals/                 # 标注评估集
docs/superpowers/      # spec 与实施计划
dev-notes/ch01.md      # 开发过程留痕
```

## 本章不做（后续章节）

工具调用、Agent 循环、服务端会话存储、RAG、鉴权、部署。
