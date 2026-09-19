# ch01 电商智能客服·纯对话 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑通纯对话：SSE 流式多轮对话接口 + Prompt 模板管理 + with_structured_output 售后信息提取 + token 预算历史裁剪，附简易聊天页与真实模型评估集。

**Architecture:** 分层结构（api / services / prompts / schemas / core），服务端无状态，历史由客户端携带，服务端做 System Prompt 拼接 + trim_messages 预算裁剪。模型经 langchain-openai ChatOpenAI 以 OpenAI 协议直连 .env 指定的任意上游。

**Tech Stack:** Python 3.12 + uv、FastAPI（新版内置 `fastapi.sse`）、LangChain（langchain / langchain-openai）、pydantic-settings、pytest + pytest-asyncio + httpx。

**Spec:** `docs/superpowers/specs/2026-09-19-ch01-pure-dialogue-design.md`

## Global Constraints

- 技术栈定死：Python + FastAPI + LangChain + OpenAI 协议直连；发现矛盾或走不通 **停下来问用户，不得自行换方案**（含：`fastapi.sse` 导入失败、依赖装不上、API 签名与文档不符）。
- 涉及 FastAPI / LangChain 具体 API 的代码，**写之前先用 Context7 MCP 按当前安装版本核对**（spec §8 对 `trim_messages` 有硬性二次核对要求）。本计划中的库调用代码均以 2026-09-19 Context7 查询结果为据，实施时仍须复核。
- 纯 Prompt / 数据类任务（Task 3 的 Prompt 质量、Task 9）不用 TDD，用 **标注样例评估集** 验证；其余代码任务照常 TDD。
- `static/index.html` 聊天页是 **Vibe Coding 例外区**：Task 8 只给能用的最简版，之后按用户描述直接改，不套评审流程。
- `.env` 永不提交；仓库只有 `.env.example`。
- 每个 Task 完成后由 **orchestrator**（主会话，掌握用户对话上下文）追记 `dev-notes/ch01.md` 一段（用户关键原话 / 关键产出 / 纠偏 / 翻车返工），随该 Task 的 commit 或单独 commit 提交。**禁止收尾一次性补记。**
- 每个 git commit 消息末尾附：`Co-Authored-By: Claude Code <noreply@anthropic.com>`
- 运行目录：`D:/CODE/python/MewHelp_Project`（Windows + Git Bash）。测试一律 `uv run pytest ...`。

---

### Task 1: 项目脚手架 + 配置层

**Files:**
- Create: `pyproject.toml`、`.gitignore`、`.env.example`、`.python-version`、`app/__init__.py`、`app/core/__init__.py`、`app/core/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces:
  - `class Settings(BaseSettings)`：字段 `openai_base_url: str`、`openai_api_key: str`、`model_name: str`、`history_token_budget: int = 4000`、`temperature: float = 0.7`
  - `get_settings() -> Settings`（`lru_cache`，读 `.env`）

- [ ] **Step 1: 确认 uv 可用**

Run: `uv --version`
Expected: 打印版本号。若命令不存在：**停止，问用户**如何安装（winget / pip install uv），不得自行更换依赖管理工具。

- [ ] **Step 2: 初始化项目**

```bash
cd /d/CODE/python/MewHelp_Project
uv init --python 3.12 --name mewhelp --bare
uv add fastapi "uvicorn[standard]" langchain langchain-openai pydantic-settings httpx
uv add --dev pytest pytest-asyncio
```

Expected: 生成 `pyproject.toml`、`.python-version`、`uv.lock`，依赖安装成功。

- [ ] **Step 3: 验证 fastapi.sse 可用（版本红线）**

Run: `uv run python -c "from fastapi.sse import EventSourceResponse, ServerSentEvent; print('sse ok')"`
Expected: 输出 `sse ok`。若 ImportError：**停止，问用户**（可能需要升级 fastapi 版本或改用 StreamingResponse 手拼 SSE——属于方案变更，需用户拍板）。

- [ ] **Step 4: 在 pyproject.toml 追加 pytest 配置**

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 5: 写 .gitignore 和 .env.example**

`.gitignore`：

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

`.env.example`：

```
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=qwen-plus
HISTORY_TOKEN_BUDGET=4000
TEMPERATURE=0.7
```

- [ ] **Step 6: 写失败测试 tests/test_config.py**

```python
import pytest
from pydantic import ValidationError


def _set_required_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")


def test_settings_reads_env_with_defaults(monkeypatch):
    _set_required_env(monkeypatch)
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.openai_base_url == "http://localhost:11434/v1"
    assert s.model_name == "test-model"
    assert s.history_token_budget == 4000
    assert s.temperature == 0.7


def test_settings_budget_overridable(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "1200")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.history_token_budget == 1200


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

- [ ] **Step 7: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError: app.core.config` 或 ImportError）

- [ ] **Step 8: 实现 app/core/config.py**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_base_url: str
    openai_api_key: str
    model_name: str
    history_token_budget: int = 4000
    temperature: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

同时创建空文件：`app/__init__.py`、`app/core/__init__.py`、`tests/__init__.py`。

- [ ] **Step 9: 运行测试确认通过**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat(ch01): 项目脚手架 + pydantic-settings 配置层

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: Pydantic Schemas（数据类）

**Files:**
- Create: `app/schemas/__init__.py`、`app/schemas/extraction.py`、`app/schemas/chat.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `IssueType(str, Enum)`：`refund/return_goods/exchange/logistics/quality/other`，值为中文
  - `AfterSaleExtraction(BaseModel)`：`order_id: str | None`、`issue_type: IssueType`、`expected_solution: str`（**无 confidence 字段**，用户已裁掉）
  - `ExtractRequest(BaseModel)`：`description: str`（min_length=1）
  - `ChatMessage(BaseModel)`：`role: Literal["user", "assistant"]`、`content: str`（min_length=1）
  - `ChatRequest(BaseModel)`：`messages: list[ChatMessage]`（min_length=1，最后一条必须 role=user，违反抛 ValueError→422）
  - `HealthResponse(BaseModel)`：`status: str`、`model: str`、`history_token_budget: int`（**无 base_url**，用户已裁掉）

- [ ] **Step 1: 写失败测试 tests/test_schemas.py**

```python
import pytest
from pydantic import ValidationError


def test_issue_type_values():
    from app.schemas.extraction import IssueType

    assert IssueType.refund.value == "退款"
    assert IssueType.return_goods.value == "退货"
    assert IssueType.exchange.value == "换货"
    assert IssueType.logistics.value == "物流"
    assert IssueType.quality.value == "质量问题"
    assert IssueType.other.value == "其他"


def test_after_sale_extraction_fields():
    from app.schemas.extraction import AfterSaleExtraction

    m = AfterSaleExtraction(
        order_id="DD123", issue_type="refund", expected_solution="全额退款"
    )
    assert m.order_id == "DD123"
    assert set(m.model_fields) == {"order_id", "issue_type", "expected_solution"}


def test_after_sale_extraction_order_id_optional():
    from app.schemas.extraction import AfterSaleExtraction

    m = AfterSaleExtraction(issue_type="quality", expected_solution="补发")
    assert m.order_id is None


def test_extract_request_rejects_empty():
    from app.schemas.extraction import ExtractRequest

    with pytest.raises(ValidationError):
        ExtractRequest(description="")


def test_chat_request_rejects_last_not_user():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(
            messages=[
                {"role": "user", "content": "在吗"},
                {"role": "assistant", "content": "在的"},
            ]
        )


def test_chat_request_rejects_bad_role():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "system", "content": "hack"}])


def test_chat_request_ok():
    from app.schemas.chat import ChatRequest

    req = ChatRequest(
        messages=[
            {"role": "user", "content": "在吗"},
            {"role": "assistant", "content": "在的"},
            {"role": "user", "content": "退货流程是什么"},
        ]
    )
    assert len(req.messages) == 3


def test_health_response_no_base_url():
    from app.schemas.chat import HealthResponse

    h = HealthResponse(status="ok", model="qwen-plus", history_token_budget=4000)
    assert "base_url" not in h.model_fields
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 app/schemas/extraction.py**

```python
from enum import Enum

from pydantic import BaseModel, Field


class IssueType(str, Enum):
    refund = "退款"
    return_goods = "退货"  # return 是 Python 关键字，用 return_goods
    exchange = "换货"
    logistics = "物流"
    quality = "质量问题"
    other = "其他"


class AfterSaleExtraction(BaseModel):
    """从用户售后描述中提取的结构化字段。"""

    order_id: str | None = Field(None, description="订单号，用户未提供时为 None")
    issue_type: IssueType = Field(description="诉求类型")
    expected_solution: str = Field(description="用户期望的处理方案")


class ExtractRequest(BaseModel):
    description: str = Field(min_length=1, description="用户的售后描述原文")
```

- [ ] **Step 4: 实现 app/schemas/chat.py**

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    @model_validator(mode="after")
    def last_message_must_be_user(self) -> "ChatRequest":
        if self.messages[-1].role != "user":
            raise ValueError("messages 最后一条的 role 必须是 user")
        return self


class HealthResponse(BaseModel):
    status: str
    model: str
    history_token_budget: int
```

同时创建空文件 `app/schemas/__init__.py`。

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ch01): Pydantic schemas(提取字段/对话请求/健康检查)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Prompt 模板管理

**Files:**
- Create: `app/prompts/__init__.py`、`app/prompts/customer_service.py`、`app/prompts/extraction.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `customer_service.CUSTOMER_SERVICE_PROMPT: ChatPromptTemplate`（输入变量 `messages`，用 MessagesPlaceholder；`.invoke({"messages": [...]})` 返回含 system 的消息列表）
  - `extraction.EXTRACTION_PROMPT: ChatPromptTemplate`（输入变量 `description`）

**验证方式说明**：Prompt 是纯文本资产，单测只验证「渲染行为正确」（模板能渲染、约束关键词在、变量占位正确）；**Prompt 质量**在 Task 9 用标注评估集对真实模型验证（用户要求：非可单测代码用评估集替代 TDD）。

- [ ] **Step 1: 用 Context7 核对 ChatPromptTemplate / MessagesPlaceholder 当前 API**

查询：`/langchain-ai/docs` → "ChatPromptTemplate from_messages MessagesPlaceholder system prompt current API"。确认 `("placeholder", "{messages}")` 元组写法或 `MessagesPlaceholder("messages")` 哪个是当前推荐，按核对结果写代码。

- [ ] **Step 2: 写失败测试 tests/test_prompts.py**

```python
from langchain_core.messages import AIMessage, HumanMessage


def test_customer_service_prompt_renders_with_system():
    from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT

    rendered = CUSTOMER_SERVICE_PROMPT.invoke(
        {"messages": [HumanMessage(content="你好")]}
    ).to_messages()
    assert rendered[0].type == "system"
    assert rendered[-1].content == "你好"


def test_customer_service_prompt_contains_constraints():
    from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT

    rendered = CUSTOMER_SERVICE_PROMPT.invoke(
        {
            "messages": [
                HumanMessage(content="在吗"),
                AIMessage(content="在的"),
                HumanMessage(content="怎么退货"),
            ]
        }
    ).to_messages()
    system_text = rendered[0].content
    # 角色设定与行为约束关键词
    for keyword in ["喵帮", "客服", "订单号", "转人工"]:
        assert keyword in system_text
    # 历史消息保序传入
    assert [m.type for m in rendered[1:]] == ["human", "ai", "human"]


def test_extraction_prompt_renders_description():
    from app.prompts.extraction import EXTRACTION_PROMPT

    rendered = EXTRACTION_PROMPT.invoke(
        {"description": "订单DD1的鞋子开胶了，想退货"}
    ).to_messages()
    assert rendered[-1].content == "订单DD1的鞋子开胶了，想退货"
    # few-shot 样例存在于 system 部分
    system_text = rendered[0].content
    assert "DD20260901001" in system_text
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 4: 实现 app/prompts/customer_service.py**

System Prompt 全文（可按 Task 9 评估结果迭代，改动只在此文件）：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """你是「喵帮」电商平台的智能客服喵喵，负责解答购物、订单、物流、售后相关问题。

行为约束：
1. 只回答与电商购物、订单、物流、售后相关的问题；无关问题（如写作业、闲聊八卦）礼貌拒绝并引回主题。
2. 不越权承诺：不得替平台承诺赔偿金额、退款一定通过、具体到账时间等结果性内容；涉及此类诉求时说明流程并建议转人工客服。
3. 处理售后问题时，主动引导用户提供订单号，以便查询。
4. 不确定或查询不到的信息如实说明，不编造订单状态、物流信息或平台政策。
5. 语气友好、简洁，使用中文回答，适当使用「喵」保持品牌风格但不堆砌。"""

CUSTOMER_SERVICE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```

（若 Step 1 Context7 核对结果显示当前版本推荐 `("placeholder", "{messages}")` 写法，则改用该写法，行为等价。）

- [ ] **Step 5: 实现 app/prompts/extraction.py**

```python
from langchain_core.prompts import ChatPromptTemplate

EXTRACTION_SYSTEM = """你是电商售后工单信息提取器。从用户描述中提取以下字段：
- order_id: 订单号（形如 DD 开头加数字的编号）；用户未提供时输出 null，不要编造。
- issue_type: 诉求类型，只能从 [退款, 退货, 换货, 物流, 质量问题, 其他] 中选一个。
- expected_solution: 用户期望的处理方案，用一句简短中文概括。

样例1：
输入：订单 DD20260901001 的耳机右声道没声音，才买一周，我要退货退款。
输出：order_id=DD20260901001, issue_type=退货, expected_solution=退货并退款

样例2：
输入：买的杯子收到就是碎的，订单号找不到了，希望补发一个。
输出：order_id=null, issue_type=质量问题, expected_solution=补发商品"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM),
        ("human", "{description}"),
    ]
)
```

同时创建空文件 `app/prompts/__init__.py`。

- [ ] **Step 6: 运行测试确认通过**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(ch01): Prompt 模板(客服角色设定+售后提取 few-shot)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: 历史裁剪（token 预算）

**Files:**
- Create: `app/services/__init__.py`、`app/services/chat_service.py`（本任务只写裁剪部分）
- Test: `tests/test_trim.py`

**Interfaces:**
- Consumes: `Settings.history_token_budget`（Task 1）
- Produces:
  - `trim_history(history: list[BaseMessage], budget: int) -> list[BaseMessage]`
  - `to_langchain_messages(chat_messages: list[ChatMessage]) -> list[BaseMessage]`（role→HumanMessage/AIMessage 转换）

- [ ] **Step 1: 【用户硬性要求】用 Context7 按当前安装版本核对 trim_messages**

先 Run: `uv run python -c "import langchain_core; print(langchain_core.__version__)"` 记录版本。
再用 Context7 查 `/langchain-ai/docs`："trim_messages strategy last token_counter count_tokens_approximately include_system start_on parameters"，逐项核对以下参数的存在与语义：`strategy="last"`、`token_counter=count_tokens_approximately`、`max_tokens`、`start_on="human"`、`include_system=True`。
**任何一项与计划代码不符：停止，问用户。**

- [ ] **Step 2: 写失败测试 tests/test_trim.py**

```python
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def _conv(n_turns: int, filler: str = "这是一段用于填充token的中文文本内容。"):
    msgs = [SystemMessage(content="system prompt")]
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"{filler}第{i}轮问题"))
        msgs.append(AIMessage(content=f"{filler}第{i}轮回答"))
    return msgs


def test_trim_keeps_system_and_latest():
    from app.services.chat_service import trim_history

    msgs = _conv(10)
    trimmed = trim_history(msgs, budget=60)
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[-1].content == msgs[-1].content
    assert len(trimmed) < len(msgs)


def test_trim_first_non_system_is_human():
    from app.services.chat_service import trim_history

    trimmed = trim_history(_conv(10), budget=60)
    assert isinstance(trimmed[1], HumanMessage)


def test_trim_noop_under_budget():
    from app.services.chat_service import trim_history

    msgs = _conv(2)
    trimmed = trim_history(msgs, budget=100_000)
    assert len(trimmed) == len(msgs)


def test_to_langchain_messages_maps_roles():
    from app.schemas.chat import ChatMessage
    from app.services.chat_service import to_langchain_messages

    out = to_langchain_messages(
        [
            ChatMessage(role="user", content="你好"),
            ChatMessage(role="assistant", content="在的"),
        ]
    )
    assert isinstance(out[0], HumanMessage)
    assert isinstance(out[1], AIMessage)
```

（预算 60 是近似值：`count_tokens_approximately` 默认按字符折算，中文长句一轮远超 60，必然触发裁剪；若实测未裁剪，调小 budget 而不是改断言逻辑。）

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_trim.py -v`
Expected: FAIL（`app.services.chat_service` 不存在）

- [ ] **Step 4: 实现 app/services/chat_service.py（裁剪部分）**

```python
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    trim_messages,
)

from app.schemas.chat import ChatMessage


def to_langchain_messages(chat_messages: list[ChatMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(content=m.content)
        if m.role == "user"
        else AIMessage(content=m.content)
        for m in chat_messages
    ]


def trim_history(history: list[BaseMessage], budget: int) -> list[BaseMessage]:
    """按 token 预算从最旧消息裁起；System 永远保留，裁剪后首条非 system 消息为 human。"""
    return trim_messages(
        history,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=budget,
        start_on="human",
        include_system=True,
    )
```

同时创建空文件 `app/services/__init__.py`。
（参数以 Step 1 Context7 核对结果为准；如有出入按核对结果改，并在 dev-notes 记录差异。）

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_trim.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ch01): trim_messages 历史裁剪 + token 预算控制

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: 模型工厂 + 流式对话服务

**Files:**
- Modify: `app/services/chat_service.py`（追加）
- Test: `tests/test_stream_chat.py`

**Interfaces:**
- Consumes: `Settings`（Task 1）、`CUSTOMER_SERVICE_PROMPT`（Task 3）、`trim_history` / `to_langchain_messages`（Task 4）
- Produces:
  - `get_model(settings: Settings) -> ChatOpenAI`
  - `build_messages(chat_messages: list[ChatMessage], settings: Settings) -> list[BaseMessage]`（转换→拼 system→裁剪，返回可直接喂给模型的消息列表）
  - `stream_chat(messages: list[BaseMessage], model) -> AsyncIterator[str]`（yield 增量文本）

- [ ] **Step 1: 用 Context7 核对 astream chunk 的文本属性**

查 `/langchain-ai/docs`："ChatOpenAI astream AIMessageChunk text content attribute current API"。确认增量文本取 `chunk.text` 还是 `chunk.content`（新旧版本有差异）。**以核对结果为准写代码，计划中按 `chunk.text` 预写。**

- [ ] **Step 2: 写失败测试 tests/test_stream_chat.py**

```python
import pytest
from langchain_core.messages import HumanMessage


class FakeChunk:
    def __init__(self, text):
        self.text = text
        self.content = text  # 兼容两种属性取法


class FakeModel:
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or ["你", "好", "喵"]
        self.error = error

    async def astream(self, messages, **kwargs):
        if self.error:
            raise self.error
        for t in self.chunks:
            yield FakeChunk(t)


async def test_stream_chat_yields_text_chunks():
    from app.services.chat_service import stream_chat

    out = [t async for t in stream_chat([HumanMessage(content="hi")], FakeModel())]
    assert out == ["你", "好", "喵"]


async def test_stream_chat_propagates_upstream_error():
    from app.services.chat_service import stream_chat

    model = FakeModel(error=RuntimeError("upstream down"))
    with pytest.raises(RuntimeError):
        _ = [t async for t in stream_chat([HumanMessage(content="hi")], model)]


def test_build_messages_prepends_system_and_trims(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "60")
    from app.core.config import Settings, get_settings
    from app.schemas.chat import ChatMessage
    from app.services.chat_service import build_messages

    get_settings.cache_clear()
    settings = Settings(_env_file=None)
    msgs = [
        ChatMessage(role="user", content=f"很长的问题{i}" * 30) for i in range(6)
    ]
    out = build_messages(msgs, settings)
    assert out[0].type == "system"
    assert out[-1].content == msgs[-1].content
    assert len(out) < len(msgs) + 1


def test_get_model_uses_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")
    monkeypatch.setenv("MODEL_NAME", "qwen2.5:7b")
    from app.core.config import Settings
    from app.services.chat_service import get_model

    m = get_model(Settings(_env_file=None))
    assert m.model_name == "qwen2.5:7b"
    assert m.temperature == 0.7
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_stream_chat.py -v`
Expected: FAIL（`stream_chat` / `build_messages` / `get_model` 不存在）

- [ ] **Step 4: 追加实现到 app/services/chat_service.py**

```python
from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.prompts.customer_service import CUSTOMER_SERVICE_PROMPT


def get_model(settings: Settings) -> ChatOpenAI:
    """OpenAI 协议直连上游；换 GPT/Claude/DeepSeek/Ollama 只改 .env。"""
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.model_name,
        temperature=settings.temperature,
    )


def build_messages(
    chat_messages: list[ChatMessage], settings: Settings
) -> list[BaseMessage]:
    history = to_langchain_messages(chat_messages)
    rendered = CUSTOMER_SERVICE_PROMPT.invoke({"messages": history}).to_messages()
    return trim_history(rendered, settings.history_token_budget)


async def stream_chat(
    messages: list[BaseMessage], model
) -> AsyncIterator[str]:
    async for chunk in model.astream(messages):
        text = getattr(chunk, "text", None) or chunk.content
        if text:
            yield text
```

（`chunk.text` vs `chunk.content` 以 Step 1 核对结果为准；`getattr` 兜底写法仅在两者语义确认等价时保留，否则按文档写唯一正确取法。）

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_stream_chat.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ch01): ChatOpenAI 模型工厂 + build_messages + stream_chat 流式服务

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: 结构化提取服务

**Files:**
- Create: `app/services/extract_service.py`
- Test: `tests/test_extract_service.py`

**Interfaces:**
- Consumes: `AfterSaleExtraction`（Task 2）、`EXTRACTION_PROMPT`（Task 3）、`get_model`（Task 5）
- Produces: `async extract_after_sale(description: str, model) -> AfterSaleExtraction`

- [ ] **Step 1: 用 Context7 核对 with_structured_output 链式用法**

查 `/langchain-ai/docs`："with_structured_output Pydantic prompt pipe chain ainvoke current API"。确认 `EXTRACTION_PROMPT | model.with_structured_output(Schema)` 的管道组合与 `ainvoke` 入参形态。

- [ ] **Step 2: 写失败测试 tests/test_extract_service.py**

```python
class FakeStructuredModel:
    """模拟 with_structured_output 返回的 runnable。"""

    def __init__(self, schema, result):
        self.schema = schema
        self.result = result

    async def ainvoke(self, value):
        assert isinstance(value, dict) and "description" in value
        return self.result


class FakeModelWithStructured:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredModel(schema, self.result)


async def test_extract_returns_schema_instance():
    from app.schemas.extraction import AfterSaleExtraction, IssueType
    from app.services.extract_service import extract_after_sale

    expected = AfterSaleExtraction(
        order_id="DD20260901001",
        issue_type=IssueType.return_goods,
        expected_solution="退货并退款",
    )
    model = FakeModelWithStructured(expected)
    out = await extract_after_sale("耳机坏了要退货，订单DD20260901001", model)
    assert out == expected
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_extract_service.py -v`
Expected: FAIL（模块不存在）

- [ ] **Step 4: 实现 app/services/extract_service.py**

```python
from app.prompts.extraction import EXTRACTION_PROMPT
from app.schemas.extraction import AfterSaleExtraction


async def extract_after_sale(description: str, model) -> AfterSaleExtraction:
    structured_model = model.with_structured_output(AfterSaleExtraction)
    chain = EXTRACTION_PROMPT | structured_model
    return await chain.ainvoke({"description": description})
```

- [ ] **Step 5: 运行测试确认通过**

Run: `uv run pytest tests/test_extract_service.py -v`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(ch01): with_structured_output 售后信息提取服务

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: API 路由 + 应用组装（SSE / extract / health）

**Files:**
- Create: `app/api/__init__.py`、`app/api/routes.py`、`app/main.py`
- Test: `tests/test_routes.py`、`tests/conftest.py`

**Interfaces:**
- Consumes: Task 1-6 全部产物
- Produces:
  - `dep_chat_model() -> ChatOpenAI`（FastAPI 依赖，测试用 `app.dependency_overrides` 替换）
  - `POST /api/chat/stream`（SSE：`event: token` data 为 JSON 字符串的增量文本 / `event: done` data `[DONE]` / `event: error` data `{"detail": ...}`）
  - `POST /api/extract` → 200 `AfterSaleExtraction` JSON；上游异常 → 502 `{"detail": ...}`
  - `GET /api/health` → `{"status": "ok", "model": ..., "history_token_budget": ...}`
  - `GET /` → 聊天页（static/index.html）

- [ ] **Step 1: 用 Context7 核对 EventSourceResponse / ServerSentEvent 用法**

查 `/websites/fastapi_tiangolo`："EventSourceResponse ServerSentEvent POST yield async generator response_class"。核对：POST + `response_class=EventSourceResponse` + async generator yield `ServerSentEvent` 的写法、`data`（JSON 序列化）与 `raw_data`（原样字符串）语义、异步生成器中 try/except 的可用形态。

- [ ] **Step 2: 写 tests/conftest.py（共享 Fake 模型与客户端 fixture）**

```python
import pytest
from httpx import ASGITransport, AsyncClient


class FakeChunk:
    def __init__(self, text):
        self.text = text
        self.content = text


class FakeChatModel:
    """路由测试用假模型：流式吐固定文本，结构化提取返回固定结果。"""

    structured_result = None  # 由各测试按需覆盖

    async def astream(self, messages, **kwargs):
        for t in ["你好", "呀", "喵"]:
            yield FakeChunk(t)

    def with_structured_output(self, schema, **kwargs):
        class _Runnable:
            async def ainvoke(self, value):
                return FakeChatModel.structured_result

        return _Runnable()


class BrokenChatModel:
    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover （使其成为 async generator）

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("upstream down")


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("MODEL_NAME", "fake-model")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def client(fake_env):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 3: 写失败测试 tests/test_routes.py**

```python
import json


def _override_model(app, model):
    from app.api.routes import dep_chat_model

    app.dependency_overrides[dep_chat_model] = lambda: model
    return app


async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "ok",
        "model": "fake-model",
        "history_token_budget": 4000,
    }


async def test_chat_stream_sse_frames(client):
    from app.main import app
    from tests.conftest import FakeChatModel

    _override_model(app, FakeChatModel())
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: token" in body
    # data 为 JSON 字符串化的增量文本
    token_lines = [
        line.removeprefix("data:").strip()
        for line in body.splitlines()
        if line.startswith("data:") and line.removeprefix("data:").strip() not in ("[DONE]",)
    ]
    joined = "".join(json.loads(t) for t in token_lines if t.startswith('"'))
    assert joined == "你好呀喵"
    assert "[DONE]" in body
    app.dependency_overrides.clear()


async def test_chat_stream_validation_last_not_user(client):
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "assistant", "content": "嗨"}]},
    )
    assert r.status_code == 422


async def test_chat_stream_error_event_on_upstream_failure(client):
    from app.main import app
    from tests.conftest import BrokenChatModel

    _override_model(app, BrokenChatModel())
    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}]},
    )
    assert r.status_code == 200  # SSE 惯例：错误走事件流
    assert "event: error" in r.text
    app.dependency_overrides.clear()


async def test_extract_returns_json(client):
    from app.main import app
    from app.schemas.extraction import AfterSaleExtraction, IssueType
    from tests.conftest import FakeChatModel

    FakeChatModel.structured_result = AfterSaleExtraction(
        order_id="DD123",
        issue_type=IssueType.refund,
        expected_solution="全额退款",
    )
    _override_model(app, FakeChatModel())
    r = await client.post(
        "/api/extract", json={"description": "订单DD123不想要了，退款"}
    )
    assert r.status_code == 200
    assert r.json() == {
        "order_id": "DD123",
        "issue_type": "退款",
        "expected_solution": "全额退款",
    }
    app.dependency_overrides.clear()


async def test_extract_502_on_upstream_failure(client):
    from app.main import app
    from tests.conftest import BrokenChatModel

    _override_model(app, BrokenChatModel())
    r = await client.post("/api/extract", json={"description": "任意描述"})
    assert r.status_code == 502
    assert "detail" in r.json()
    app.dependency_overrides.clear()


async def test_extract_rejects_empty_description(client):
    r = await client.post("/api/extract", json={"description": ""})
    assert r.status_code == 422
```

- [ ] **Step 4: 运行测试确认失败**

Run: `uv run pytest tests/test_routes.py -v`
Expected: FAIL（`app.main` 不存在）

- [ ] **Step 5: 实现 app/api/routes.py**

```python
import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.schemas.chat import ChatRequest, HealthResponse
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.services.chat_service import build_messages, get_model, stream_chat
from app.services.extract_service import extract_after_sale

logger = logging.getLogger(__name__)
router = APIRouter()


def dep_chat_model():
    return get_model(get_settings())


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    req: ChatRequest, model=Depends(dep_chat_model)
) -> AsyncIterable[ServerSentEvent]:
    settings = get_settings()
    messages = build_messages(req.messages, settings)
    try:
        async for text in stream_chat(messages, model):
            yield ServerSentEvent(data=text, event="token")
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例：错误进事件流后正常关流
        logger.exception("chat stream failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")


@router.post("/api/extract", response_model=AfterSaleExtraction)
async def extract(req: ExtractRequest, model=Depends(dep_chat_model)):
    try:
        return await extract_after_sale(req.description, model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract failed")
        from fastapi import HTTPException

        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/api/health", response_model=HealthResponse)
async def health():
    s = get_settings()
    return HealthResponse(
        status="ok", model=s.model_name, history_token_budget=s.history_token_budget
    )
```

（`HTTPException` 移到文件顶部 import；此处内联仅为示意——实现时写在顶部。以 Step 1 Context7 核对结果为准调整 SSE 细节。）

- [ ] **Step 6: 实现 app/main.py**

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.routes import router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_settings()  # 启动即校验 .env 必填项，不带病启动
    except ValidationError as exc:
        raise SystemExit(
            f"[启动失败] .env 配置缺失或非法，请参考 .env.example 补全：\n{exc}"
        ) from exc
    yield


app = FastAPI(title="MewHelp 电商智能客服", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse("static/index.html")
```

注意：`StaticFiles(directory="static")` 要求目录存在，Task 8 才建 index.html——本任务先创建 `static/` 目录并放一个占位 `static/index.html`（内容 `<html><body><p>chat ui coming</p></body></html>`，Task 8 覆盖），保证路由测试可跑。

- [ ] **Step 7: 运行测试确认通过**

Run: `uv run pytest -v`
Expected: 全部测试通过（含之前任务）

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(ch01): SSE 对话路由 + 提取路由 + health + FastAPI 组装

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 聊天页（Vibe Coding 例外区，不走评审流程）

**Files:**
- Modify: `static/index.html`（覆盖 Task 7 占位）

**Interfaces:**
- Consumes: `POST /api/chat/stream`、`POST /api/extract`（Task 7）
- Produces: 浏览器可用的单页聊天界面

- [ ] **Step 1: 实现最简可用版 static/index.html**

要点（完整代码实现时直接写，这是 Vibe 区，之后按用户描述效果迭代）：
- 左侧聊天区：消息气泡（user 右对齐 / assistant 左对齐）、输入框 + 发送按钮
- `fetch("/api/chat/stream", {method:"POST", ...})` 后用 `resp.body.getReader()` + `TextDecoder` 读流，按空行 `\n\n` 切 SSE 帧，解析 `event:` 与 `data:` 行；`event: token` 时 `JSON.parse(data)` 得增量文本追加到当前气泡；`event: done` 结束；`event: error` 显示错误气泡
- 历史存 JS 数组 `messages`，每轮把完整数组（含刚收到的 assistant 回复）随请求发送
- 右侧「售后提取」面板：textarea + 按钮 → `POST /api/extract` → `<pre>` 展示 JSON
- 无框架、无构建，单文件内联 CSS/JS

- [ ] **Step 2: 手工冒烟（需 .env 配好真实上游）**

Run: `uv run uvicorn app.main:app --port 8000`，浏览器开 `http://127.0.0.1:8000/`
Expected: 发消息看到逐 token 流式回复；连续两轮第二轮接住上下文；提取面板返回 JSON。
若无可用 .env：跳过真实冒烟，仅验证页面加载与 422/错误分支，真实验证并入 Task 10。

- [ ] **Step 3: Commit**

```bash
git add static/index.html
git commit -m "feat(ch01): 简易聊天单页(SSE 流式渲染+售后提取面板) [vibe]

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: 提取 Prompt 标注评估集（替代 TDD 的验证步骤）

**Files:**
- Create: `evals/extraction_samples.json`、`evals/run_extraction_eval.py`

**Interfaces:**
- Consumes: `extract_after_sale`（Task 6）、`get_model` / `get_settings`（Task 1/5）、真实 `.env`
- Produces: 可重复运行的评估脚本，输出逐条对比表 + 通过率

- [ ] **Step 1: 写标注样例 evals/extraction_samples.json（5 条，人工标注）**

```json
[
  {
    "description": "订单 DD20260901001 的耳机右声道没声音，才买一周，我要退货退款。",
    "expected": {"order_id": "DD20260901001", "issue_type": "退货", "expected_solution_keywords": ["退货", "退款"]}
  },
  {
    "description": "我上周买的鞋子尺码偏小，订单号是 DD20260905077，想换一双大一码的。",
    "expected": {"order_id": "DD20260905077", "issue_type": "换货", "expected_solution_keywords": ["换"]}
  },
  {
    "description": "DD20260910123 这个订单显示已签收但我根本没收到货，快递到底去哪了？",
    "expected": {"order_id": "DD20260910123", "issue_type": "物流", "expected_solution_keywords": ["物流", "查", "签收"]}
  },
  {
    "description": "买的杯子收到就是碎的，订单号在手边找不到，希望你们补发一个。",
    "expected": {"order_id": null, "issue_type": "质量问题", "expected_solution_keywords": ["补发"]}
  },
  {
    "description": "订单 DD20260912888 我拍错颜色了，还没发货的话直接给我退了吧。",
    "expected": {"order_id": "DD20260912888", "issue_type": "退款", "expected_solution_keywords": ["退"]}
  }
]
```

判定规则：`order_id`、`issue_type` 精确匹配（枚举取 `.value` 中文比较）；`expected_solution` 用关键词包含判定（自由文本不做精确匹配），并打印实际值供人工复核。

- [ ] **Step 2: 实现 evals/run_extraction_eval.py**

```python
"""提取 Prompt 评估集：跑真实模型对比人工标注。用法: uv run python evals/run_extraction_eval.py"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale


async def main() -> int:
    samples = json.loads(
        (Path(__file__).parent / "extraction_samples.json").read_text(encoding="utf-8")
    )
    model = get_model(get_settings())
    passed = 0
    for i, s in enumerate(samples, 1):
        got = await extract_after_sale(s["description"], model)
        exp = s["expected"]
        ok_id = got.order_id == exp["order_id"]
        ok_type = got.issue_type.value == exp["issue_type"]
        ok_sol = any(k in got.expected_solution for k in exp["expected_solution_keywords"])
        ok = ok_id and ok_type and ok_sol
        passed += ok
        print(
            f"[{'PASS' if ok else 'FAIL'}] #{i} "
            f"order_id: {got.order_id!r}(期望 {exp['order_id']!r}) | "
            f"issue_type: {got.issue_type.value}(期望 {exp['issue_type']}) | "
            f"expected_solution: {got.expected_solution!r}(关键词 {exp['expected_solution_keywords']})"
        )
    print(f"\n通过率: {passed}/{len(samples)}")
    return 0 if passed == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 3: 配置 .env 并运行评估**

前置：用户提供真实上游的 `.env`（复制 `.env.example` 填入密钥）。
Run: `uv run python evals/run_extraction_eval.py`
Expected: 5/5 或 ≥4/5；未达标则迭代 `app/prompts/extraction.py`（只改 Prompt，不改判定），重跑至达标或两轮无改善时**停下来问用户**。
**System Prompt 对话质量**同场验收：`uv run uvicorn app.main:app --port 8000` 后用 Task 10 的 curl 命令 1/2 人工确认角色设定与约束生效。

- [ ] **Step 4: Commit**

```bash
git add evals
git commit -m "test(ch01): 售后提取标注评估集(5样例)+评估脚本

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: README + 端到端验收

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: 全部前序任务
- Produces: 项目说明 + 3 条验收命令；验收通过记录进 dev-notes

- [ ] **Step 1: 写 README.md**

内容必须包含：项目简介、快速开始（`uv sync` → 复制 `.env.example` 为 `.env` 填密钥 → `uv run uvicorn app.main:app --port 8000`）、聊天页地址、评估集运行方式，以及 3 条验收命令：

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
```

Expected 结果说明写入 README（验收 3 应返回 `{"order_id":"DD20260901001","issue_type":"退货","expected_solution":"..."}`）。

- [ ] **Step 2: 起服务跑全部验收**

```bash
uv run pytest -v                              # 单测全绿
uv run python evals/run_extraction_eval.py    # 评估集达标
uv run uvicorn app.main:app --port 8000       # 另开终端跑 3 条 curl
```

Expected: 验收标准 1/2/3 全部满足。任何一条失败：修复后重跑，翻车与修复过程记入 dev-notes。

- [ ] **Step 3: Commit + dev-notes 收尾段**

```bash
git add README.md
git commit -m "docs(ch01): README 快速开始 + 3 条验收命令

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

orchestrator 在 dev-notes/ch01.md 补 finish 段（演示命令、测试结果、翻车记录），随后走 finishing-a-development-branch。
