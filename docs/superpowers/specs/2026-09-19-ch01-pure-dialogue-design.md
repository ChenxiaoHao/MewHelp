# ch01 电商智能客服系统 · 纯对话 — 设计稿

- 日期：2026-09-19
- 状态：已定稿（用户批准，含 3 处调整）
- 章节范围：ch01 仅纯对话，不做工具调用 / Agent 循环

## 1. 目标与验收标准

做一个电商智能客服系统的第一章：跑通纯对话。

验收标准（全部满足才算完成）：

1. curl 调对话接口能看到 SSE 流式回复（逐 token 推送）
2. 连续问两轮，第二轮能接住第一轮的上下文（客户端携带历史 + 服务端裁剪）
3. 发一段售后描述，`/api/extract` 能返回结构化 JSON（订单号 / 诉求类型 / 期望方案）

## 2. 技术栈（用户定死，不可自行更换）

- Python 3.12 + uv（依赖管理与虚拟环境）
- FastAPI（SSE 使用新版内置 `fastapi.sse.EventSourceResponse` / `ServerSentEvent`；安装后验证导入可用）
- LangChain（`langchain` + `langchain-openai`）
- 模型接入：OpenAI 协议直连上游，`ChatOpenAI(base_url=..., api_key=..., model=...)`；地址 / 模型名 / 密钥全部来自 `.env`，可切换 GPT / Claude / DeepSeek / Ollama / 百炼 qwen
- pydantic-settings（配置）、pytest + httpx（测试）

**硬性工作约束**：涉及 FastAPI / LangChain 等库的具体 API 用法，实现前必须先用 Context7 MCP 按当前安装版本核对最新官方文档，禁止凭记忆写。发现选型矛盾或走不通，停下来问用户，不得自行换方案。

## 3. 架构与请求流

```
浏览器聊天页(static)          curl / 客户端
      │ POST /api/chat/stream      │ POST /api/extract
      ▼                            ▼
  FastAPI (app/api/routes.py)
      │                            │
      ▼                            ▼
  chat_service                 extract_service
  ┌─ prompts/客服SystemPrompt   ┌─ prompts/提取Prompt(含few-shot)
  ├─ trim_messages 历史裁剪      └─ with_structured_output(AfterSaleExtraction)
  └─ ChatOpenAI.astream()
      │
      ▼
  OpenAI 协议上游（.env 决定）
```

**多轮上下文最简版：服务端无状态。** 历史消息由客户端（聊天页 JS 内存 / curl 请求体）每次随请求携带；服务端只做 System Prompt 拼接 + token 预算裁剪。会话存储留待后续章节。

## 4. 项目结构

```
MewHelp_Project/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 入口：挂载路由、静态页面、启动配置校验
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py     # pydantic-settings 读 .env
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── customer_service.py   # 客服 System Prompt（ChatPromptTemplate）
│   │   └── extraction.py         # 售后信息提取 Prompt（含 2 条 few-shot 标注样例）
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── extraction.py         # AfterSaleExtraction、IssueType、请求/响应模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_service.py       # 模型工厂、历史裁剪、astream 流式对话
│   │   └── extract_service.py    # with_structured_output 提取
│   └── api/
│       ├── __init__.py
│       └── routes.py             # /api/chat/stream、/api/extract、/api/health
├── static/
│   └── index.html        # 简易聊天单页（Vibe Coding 区，不走评审流程）
├── tests/
│   ├── conftest.py               # mock 模型 fixture
│   ├── test_trim.py              # 历史裁剪单测
│   ├── test_schemas.py           # schema 校验单测
│   ├── test_prompts.py           # prompt 渲染单测
│   └── test_routes.py            # 路由 + SSE 组帧测试（mock 模型）
├── evals/
│   └── extraction_samples.json   # 5 条售后描述 + 人工标注期望字段（评估集）
├── dev-notes/
│   └── ch01.md                   # 开发过程留痕（按阶段追记，不许收尾补记）
├── docs/superpowers/specs/       # 本设计稿与后续 spec
├── .env.example                  # 占位配置模板（.env 进 .gitignore）
├── .gitignore
├── pyproject.toml                # uv 管理
└── README.md                     # 启动方式 + 3 条 curl 验收命令
```

## 5. API 契约

### 5.1 POST /api/chat/stream

请求体：

```json
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              {"role": "user", "content": "最后一条必须是 user"}]}
```

响应：SSE 流（`fastapi.sse.EventSourceResponse`）

- `event: token`，`data` 为增量文本（逐 token / chunk 推送）
- `event: done`，`raw_data` 为 `[DONE]`（流正常结束标记）
- `event: error`，`data` 为 JSON `{"detail": "..."}`（上游失败时发送后正常关流，HTTP 仍为 200，SSE 惯例）

校验：`messages` 非空、最后一条 role 必须为 `user`、role 仅允许 `user`/`assistant`，违反返回 422。

### 5.2 POST /api/extract

请求体：`{"description": "售后描述文本"}`（非空校验）

响应 200：`AfterSaleExtraction` JSON。上游失败返回 502 + 错误详情。

### 5.3 GET /api/health

响应（只返回必要的非敏感运行信息，**不返回 base_url、不返回密钥**）：

```json
{"status": "ok", "model": "模型名", "history_token_budget": 4000}
```

## 6. Prompt 管理

- `prompts/customer_service.py`：`ChatPromptTemplate`。System Prompt 内容要点：
  - 角色：「喵帮」电商智能客服
  - 约束：只回答电商 / 订单 / 售后相关问题；不越权承诺赔偿或退款结果；主动引导用户提供订单号；不确定时如实说明并建议转人工；语气友好简洁
- `prompts/extraction.py`：提取 Prompt 模板，含 2 条 few-shot 标注样例，指导模型按固定字段输出
- 全部模板化集中管理，调 Prompt 不改业务代码

## 7. 结构化输出 Schema（调整后定稿）

```python
class IssueType(str, Enum):
    refund = "退款"
    return_goods = "退货"   # return 是 Python 关键字，用 return_goods
    exchange = "换货"
    logistics = "物流"
    quality = "质量问题"
    other = "其他"

class AfterSaleExtraction(BaseModel):
    order_id: str | None = Field(None, description="订单号，用户未提供时为 None")
    issue_type: IssueType = Field(description="诉求类型")
    expected_solution: str = Field(description="用户期望的处理方案")
```

（按用户要求已删除 confidence 字段。）经 `model.with_structured_output(AfterSaleExtraction)` 调用。

## 8. 历史裁剪与 token 预算

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

trimmed = trim_messages(
    history,
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=settings.history_token_budget,
    start_on="human",
    include_system=True,
)
```

- System Prompt 永远保留，超出预算从最旧历史裁起
- 保底规则（实施中发现并补入，2026-09-19）：若预算小到连最新一条消息都装不下（trim_messages 实测会将其裁掉、只剩 system），强制返回 [system] + [最后一条]，绝不丢用户当前问题
- 预算 `.env` 可配：`HISTORY_TOKEN_BUDGET=4000`
- 用近似 token 计数，不绑定具体上游 tokenizer
- ⚠️ **实施约束（用户明确要求）**：`trim_messages` 的具体参数与行为（含 `include_system` / `start_on` / `end_on` 语义）在编码前必须再次用 Context7 按**当时实际安装的版本**核对，不得凭记忆实现；上面代码仅为设计意图示意。

## 9. 配置

`app/core/config.py` 用 pydantic-settings 读取 `.env`：

| 变量 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `OPENAI_BASE_URL` | 是 | — | OpenAI 协议上游地址 |
| `OPENAI_API_KEY` | 是 | — | 密钥（Ollama 可填占位值） |
| `MODEL_NAME` | 是 | — | 模型名 |
| `HISTORY_TOKEN_BUDGET` | 否 | 4000 | 历史裁剪 token 预算 |
| `TEMPERATURE` | 否 | 0.7 | 采样温度 |

启动时校验必填项，缺失即报清晰错误，不带病启动。`.env` 进 `.gitignore`，仓库只提交 `.env.example`。

## 10. 聊天页（Vibe Coding 例外区）

`static/index.html` 单文件 HTML+JS，由 FastAPI 托管：

- 消息气泡界面；`fetch` POST + ReadableStream 解析 SSE（`EventSource` 不支持 POST）
- 历史保存在 JS 内存，随每次请求携带
- 侧边「售后提取」面板：贴入描述 → 调 `/api/extract` → 展示 JSON
- 此页面按用户描述效果直接迭代，不套 brainstorm / TDD / code review 流程

## 11. 错误处理

- 流式接口：上游异常 → `event: error` 后正常关流（HTTP 200）
- 非流式接口：上游异常 → 502 + `{"detail": ...}`
- 入参校验失败：422 + 明确错误消息
- 配置缺失：启动时报错退出

## 12. 测试策略（按用户工作要求分类）

| 对象 | 验证方式 |
|---|---|
| 裁剪逻辑、schema、prompt 渲染、SSE 组帧、路由 | pytest 单测（mock 模型，零 token 消耗） |
| System Prompt / 提取 Prompt 质量 | 标注样例评估集：`evals/extraction_samples.json` 5 条售后描述 + 人工标注期望字段，跑真实模型对比（替代 TDD 步骤） |
| 端到端 | README 中 3 条 curl 验收命令，对应验收标准 1/2/3 |

## 13. 范围外（本章不做）

- 工具调用、Agent 循环、会话服务端存储、RAG、鉴权、部署
- 聊天页面为例外区：做简易版即可，效果迭代不走评审流程

## 14. 过程留痕要求

`dev-notes/ch01.md` 按阶段追记（brainstorm 定稿 / 计划评审通过 / 每个任务完成 / code review 结论 / finish），每段记录四样：用户关键原话、关键产出（spec / plan 路径、评审结论）、用户拒绝或纠偏内容、翻车与返工。**禁止收尾时一次性补记。**

## 15. 设计评审记录

- 2026-09-19 用户批准方案 A（分层结构 + LangChain 全链路），并提出 3 处调整：
  1. §7 schema 删除 confidence 字段 → 已采纳
  2. §5.3 /api/health 不返回 base_url，只返回 status / model / history_token_budget → 已采纳
  3. §8 trim_messages 参数实施前按实际安装版本再次用 Context7 核对 → 已写入实施约束
