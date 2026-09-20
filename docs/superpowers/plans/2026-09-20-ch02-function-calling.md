# ch02 Function Calling 工具链 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 MewHelp 客服装上 Function Calling 工具链——模型自主选工具（5 个 @tool）、执行（超时/重试/错误回灌）、单轮收敛流式作答，聊天记录落 MySQL 四张表，聊天页显示工具徽章。

**Architecture:** 分层扩展（spec 方案 A）：`app/db`（SQLAlchemy 2.0 async + aiomysql）与 `app/tools`（definitions/registry/executor）两个新包；编排在 `app/services/tool_chat_service.py` 新文件（ch01 的 chat_service 一字不动）；路由扩三种 SSE 事件（conversation/tool_call/tool_result），token/done/error 语义不变；落库经 ChatPersister 协议注入，失败只记日志不断流。

**Tech Stack:** Python 3.12 · uv · FastAPI 0.141 · LangChain 1.x（@tool/bind_tools/ToolMessage）· SQLAlchemy 2.0 async · aiomysql · MySQL 8.0（Docker Compose）· qwen-plus（百炼，OpenAI 协议）

**Spec:** `docs/superpowers/specs/2026-09-20-ch02-function-calling-design.md`（计划与 spec 同行，执行者两份都读）

## Global Constraints

- **ch01 兼容红线（spec §13，用户点名）**：SSE `token`/`done`/`error` 语义与 data 格式不变；`app/services/chat_service.py` 不改；`/api/extract`、`/api/health` 契约不变；ch01 的 7 个测试文件不改且全绿；README 三条 ch01 curl 验收命令继续通过。唯一允许动的共享文件是 `tests/conftest.py`（只**增强** Fake 能力，见 Task 8）
- **4 个硬性核对点（用户点名，spec §12）**：① bind_tools+astream 的 chunk 聚合 → Task 7 Step 1（本地）+ Task 10（qwen 真实上游）；② create_ticket 的 conversation_id 注入 → Task 5 Step 1；③ aiomysql Windows 安装实测 → Task 1 Step 2（不通换 asyncmy，两个都不通才停工问用户）；④ MySQL Docker 启动与认证兼容 → Task 3（caching_sha2_password 需 cryptography）。核对未过或文档与实测矛盾 → **停工问用户**
- **双轨验证**：可单测代码一律 TDD（先红后绿）；Prompt/工具路由质量用评估集（Task 10，替代 TDD）；`static/index.html` 聊天页是 **Vibe 例外区**（Task 9：直接改，用户浏览器验证迭代，不套 brainstorm/TDD/review）
- 每个任务完成：`dev-notes/ch02.md` 追记一段（①用户关键原话 ②关键产出 ③用户拒绝或纠偏 ④翻车与返工；追记时 Edit 的 old_string 必须锚定该段落特有文字，ch01 翻车 3 教训），然后单独 commit
- commit message 末尾带一行：`Co-Authored-By: Claude Code <noreply@anthropic.com>`
- Windows Git Bash：curl 中文 body 一律先写 UTF-8 文件再 `-d @file`（ch01 翻车 10）；python 内联脚本用 `uv run python - <<'PY' ... PY` heredoc；无 /tmp，临时文件放项目根 `_tmp_*` 并随手删
- 测试命令统一 `uv run pytest`（pyproject 已配 asyncio_mode=auto，async 测试函数直接可跑）；除 Task 3/4/10 的真库冒烟外，单测零 token、零 MySQL 依赖
- faq 种子红线：全表不得出现「邮费」「运费」字样（保证验收 3 漏召回，spec 附录 B）
- 单轮定义（spec §5）：至多一轮工具决策；第二轮用**裸 model**（不 bind_tools）物理保证收敛；当轮全部 tool_calls 都执行（通常 1 个）

---

### Task 1: 实施分支 + 依赖安装（硬性核对点③）+ 配置层扩展

**Files:**
- Modify: `pyproject.toml`（uv add 自动改）
- Modify: `app/core/config.py`（Settings 新增 8 字段 + database_url property）
- Modify: `.env.example`（追加 ch02 段）
- Create: `tests/test_config_ch02.py`

**Interfaces:**
- Consumes: ch01 的 `Settings`（pydantic-settings，`model_config = SettingsConfigDict(env_file=".env", extra="ignore")`，`settings_customise_sources` 使 .env 优先）
- Produces: `Settings.mysql_host/mysql_port/mysql_user/mysql_password/mysql_db/demo_user_id/tool_timeout_seconds/tool_max_retries`（全带默认值，ch01 各处 `Settings(_env_file=None)` 构造不受影响）；`Settings.database_url` → `"mysql+aiomysql://{user}:{password}@{host}:{port}/{db}?charset=utf8mb4"`（Task 3 引擎用）

- [x] **Step 1: 切实施分支**

```bash
git checkout -b ch02-function-calling
```

（ch01 先例：spec/plan 提交在 master，实施分支从那里切。）

- [x] **Step 2: 安装依赖并实测 aiomysql（硬性核对点③）**

```bash
uv add "sqlalchemy[asyncio]" aiomysql cryptography
uv run python -c "import aiomysql, sqlalchemy; print('aiomysql', aiomysql.__version__, '| sqlalchemy', sqlalchemy.__version__)"
```

Expected: 打印两个版本号，无 ImportError。
**若 aiomysql 安装/导入失败** → `uv add asyncmy` 改试；连接串方言换 `mysql+asyncmy://`（Task 3 同步改），并追记 dev-notes。**两个驱动都不通 → 停工问用户。**

- [x] **Step 3: 写失败测试 `tests/test_config_ch02.py`**

```python
import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")


def test_database_url_assembly(env, monkeypatch):
    monkeypatch.setenv("MYSQL_PASSWORD", "pw123")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.database_url == (
        "mysql+aiomysql://root:pw123@127.0.0.1:3306/mewhelp?charset=utf8mb4"
    )


def test_ch02_defaults(env):
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.mysql_host == "127.0.0.1"
    assert s.mysql_port == 3306
    assert s.mysql_user == "root"
    assert s.mysql_password == "mewhelp_dev"
    assert s.mysql_db == "mewhelp"
    assert s.demo_user_id == "demo_user"
    assert s.tool_timeout_seconds == 5.0
    assert s.tool_max_retries == 1
```

- [x] **Step 4: 跑测试确认失败**

Run: `uv run pytest tests/test_config_ch02.py -v`
Expected: FAIL（AttributeError: 'Settings' object has no attribute 'database_url' / mysql_host）

- [x] **Step 5: 实现 `app/core/config.py` 扩展**

在 Settings 类现有字段之后、`settings_customise_sources` 之前插入（现有字段与 classmethod 一字不动）：

```python
    # --- ch02: MySQL（docker compose 起本地容器，全部带默认值，不破坏 ch01 构造） ---
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "mewhelp_dev"
    mysql_db: str = "mewhelp"
    # --- ch02: 会话与工具执行 ---
    demo_user_id: str = "demo_user"
    tool_timeout_seconds: float = 5.0
    tool_max_retries: int = 1

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_db}?charset=utf8mb4"
        )
```

- [x] **Step 6: 跑测试确认通过 + ch01 回归**

Run: `uv run pytest tests/test_config_ch02.py tests/test_config.py -v`
Expected: 全 PASS（ch01 的 test_config 不受影响，因新字段全有默认值）

- [x] **Step 7: `.env.example` 追加 ch02 段（文件末尾）**

```dotenv

# --- ch02: MySQL（docker compose 起本地容器）---
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=mewhelp_dev
MYSQL_DB=mewhelp
# --- ch02: 会话与工具执行 ---
DEMO_USER_ID=demo_user
TOOL_TIMEOUT_SECONDS=5
TOOL_MAX_RETRIES=1
```

同时把同一段追加到本地 `.env` 末尾（不动已有 OPENAI_* 行；.env 在 .gitignore 中，不进 git）。

- [x] **Step 8: dev-notes 追记 + commit**

在 `dev-notes/ch02.md` 末尾追记「Task 1」段（四要素；重点记核对点③实测结果：aiomysql 版本、是否换 asyncmy）。

```bash
git add pyproject.toml uv.lock app/core/config.py .env.example tests/test_config_ch02.py dev-notes/ch02.md
git commit -m "feat(ch02): 配置层扩展(MySQL/工具执行参数) + aiomysql 依赖实测

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

（计划文档已按 ch01 先例在 writing-plans 阶段提交至 master，本分支切出时自带。）

---

### Task 2: ORM 四表模型（纯 metadata 单测，不碰数据库）

**Files:**
- Create: `app/db/__init__.py`（空文件）
- Create: `app/db/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: SQLAlchemy 2.0 `DeclarativeBase` + `Mapped`/`mapped_column`
- Produces: `Base`（Task 3 create_all 不用——建表走 SQL 脚本，Base 仅供 metadata 断言）；`Conversation`/`Message`/`Faq`/`Ticket` 四个 ORM 类，表名/列名/类型与 spec 附录 A DDL 逐列对齐（Task 4 crud、Task 8 persister 用）

- [x] **Step 1: 写失败测试 `tests/test_models.py`**

DDL 用户原话（spec 附录 A 原样收录）是列对齐的唯一依据，测试断言关键结构：

```python
from sqlalchemy import JSON


def test_conversation_table():
    from app.db.models import Conversation

    t = Conversation.__table__
    assert t.name == "conversations"
    assert t.c.id.primary_key
    assert t.c.user_id.type.length == 64
    assert list(t.c.status.type.enums) == ["进行中", "已转人工", "已结束"]
    assert {"created_at", "updated_at"} <= set(t.c.keys())


def test_message_table():
    from app.db.models import Message

    t = Message.__table__
    assert t.name == "messages"
    assert list(t.c.role.type.enums) == ["user", "assistant", "tool"]
    assert isinstance(t.c.tool_calls.type, JSON)
    assert t.c.content.nullable
    fks = {fk.target_fullname for fk in t.c.conversation_id.foreign_keys}
    assert "conversations.id" in fks


def test_faq_table():
    from app.db.models import Faq

    t = Faq.__table__
    assert t.name == "faq"
    assert t.c.question.type.length == 512
    assert t.c.answer.nullable is False
    assert t.c.category.index


def test_ticket_table():
    from app.db.models import Ticket

    t = Ticket.__table__
    assert t.name == "tickets"
    assert t.c.ticket_no.primary_key
    assert t.c.ticket_no.type.length == 32
    assert list(t.c.ticket_type.type.enums) == ["售后", "投诉", "咨询"]
    assert list(t.c.status.type.enums) == ["待处理", "已处理"]
    fks = {fk.target_fullname for fk in t.c.conversation_id.foreign_keys}
    assert "conversations.id" in fks
```

- [x] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL（ModuleNotFoundError: No module named 'app.db'）

- [x] **Step 3: 实现 `app/db/models.py`**

```python
"""ORM 模型：四张表与 spec 附录 A 的 DDL 逐列对齐。

建表不走 metadata.create_all——以 db/init/01_schema.sql（用户 DDL 原样）为准，
容器首启自动执行；本文件仅供查询/写入映射与结构测试。
"""

from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, Enum, ForeignKey, String, Text, func
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[int]:
    return mapped_column(
        BIGINT(unsigned=True), primary_key=True, autoincrement=True
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = _pk()
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"),
        nullable=False,
        server_default="进行中",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = _pk()
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        Enum("user", "assistant", "tool", name="message_role"), nullable=False
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_calls: Mapped[list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


class Faq(Base):
    __tablename__ = "faq"

    id: Mapped[int] = _pk()
    question: Mapped[str] = mapped_column(String(512), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(
        Enum("售后", "投诉", "咨询", name="ticket_type"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        nullable=False,
        server_default="待处理",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
```

注意：`from datetime import date` 若 lint 报未使用则删掉（Ticket 无 Date 列，导入以实际为准；上面代码保留最小集合即可）。

- [x] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS（ch01 34 个 + Task 1 的 2 个配置测试 + 新增 4 个模型测试）

- [x] **Step 5: dev-notes 追记 + commit**

```bash
git add app/db tests/test_models.py dev-notes/ch02.md
git commit -m "feat(ch02): ORM 四表模型(与用户 DDL 逐列对齐)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: Docker MySQL + DDL/种子脚本 + 引擎层（硬性核对点④）

**⚠️ STOP — 本任务开始前，先提醒用户启动 Docker Desktop（用户在执行约束里点名）。等 `docker info` 能通再继续。**

**Files:**
- Create: `docker-compose.yml`
- Create: `db/init/01_schema.sql`（spec 附录 A 的 DDL 原样复制，一字不改）
- Create: `db/init/02_seed.sql`（faq 种子 10 条，spec 附录 B；红线：无「邮费」「运费」字样）
- Create: `app/db/engine.py`
- Create: `tests/test_engine.py`（纯单测：URL 装配与惰性，不连库）

**Interfaces:**
- Consumes: `Settings.database_url`（Task 1）
- Produces: `init_engine(settings: Settings) -> None`（幂等，main.py lifespan 调）；`get_engine() -> AsyncEngine`（未 init 抛 RuntimeError）；`get_session_factory() -> async_sessionmaker[AsyncSession]`；`async check_db() -> None`（`SELECT 1`，失败抛异常）；`dispose_engine() -> None`（Task 8 路由依赖、Task 10 冒烟用）

- [x] **Step 1: 确认 Docker daemon 可用**

```bash
docker info --format '{{.ServerVersion}}'
```

Expected: 打印版本号。**不通 → 停下来等用户启动 Docker Desktop，不要自己反复重试。**

- [x] **Step 2: 写 `docker-compose.yml`**

```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: mewhelp-mysql
    environment:
      MYSQL_ROOT_PASSWORD: mewhelp_dev
      MYSQL_DATABASE: mewhelp
      TZ: Asia/Shanghai
    ports:
      - "3306:3306"
    command:
      - --character-set-server=utf8mb4
      - --collation-server=utf8mb4_unicode_ci
    volumes:
      - mewhelp_mysql_data:/var/lib/mysql
      - ./db/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "127.0.0.1", "-pmewhelp_dev"]
      interval: 5s
      timeout: 3s
      retries: 20

volumes:
  mewhelp_mysql_data:
```

（`db/init/*.sql` 按文件名序在**首次初始化空数据卷**时自动执行；改了 SQL 需 `docker compose down -v` 重建卷。）

- [x] **Step 3: 写 `db/init/01_schema.sql`**

打开 spec 附录 A，把用户提供的四张表 DDL **原样复制**进来（含 `DROP TABLE IF EXISTS` 顺序、`ENGINE=InnoDB DEFAULT CHARSET=utf8mb4`、注释、索引）。复制后核对：

```bash
grep -c "CREATE TABLE" db/init/01_schema.sql
```

Expected: `4`

- [x] **Step 4: 写 `db/init/02_seed.sql`（faq 种子 10 条，spec 附录 B 表格逐条转 INSERT）**

spec 附录 B 的 10 条 question/category **逐字对齐**，answer 按附录 B 摘要扩写成完整句；外加附录 B 要求的**演示数据**（conversations 1 行 + messages 2 行 + tickets 1 行）：

```sql
-- faq 种子数据（spec 附录 B 的 10 条，answer 按摘要扩写）。
-- 红线：全表（question + answer）不得出现「邮费」「运费」字样（保证验收 3 漏召回）。
-- 演示数据：conversations 1 行（demo_user，已结束）+ messages 2 行 + tickets 1 行（T20260901001，售后，已处理）。
USE mewhelp;

INSERT INTO faq (question, answer, category) VALUES
('退货政策是什么？', '签收后 7 天内可无理由退换，商品需保持未洗涤、未使用且吊牌完整；15 天内出现质量问题可享受免费修换。', '退换货'),
('如何申请退换货？', '进入「我的订单」找到对应订单，点击「申请退换」，填写原因后提交。审核通过后会有快递上门取件，请保持手机畅通。', '退换货'),
('退款多久到账？', '我们收到退回商品并验收通过后，1-3 个工作日内按原支付路径退回，到账时间以支付渠道为准，请留意收款明细。', '退换货'),
('下单后多久发货？', '现货商品 48 小时内发出，一般 2-4 天送达；预售商品以商品页标注的发货时间为准，请以下单页面说明为准。', '物流配送'),
('支持哪些快递公司？', '默认发顺丰或京东物流，部分偏远地区支持指定其他快递。如需指定请在下单前联系客服确认是否可达。', '物流配送'),
('怎么联系人工客服？', '在聊天中说明要转人工即可为您创建工单，人工客服服务时间为每天 9:00-21:00，我们会尽快跟进处理。', '售后服务'),
('商品有质量问题怎么办？', '签收 15 天内可享受免费修换。请提供订单号与问题描述（附照片更佳），我们会优先为您处理。', '售后服务'),
('如何修改收货地址？', '订单未发货时可在订单详情页自行修改收货地址；已发货订单请尽快联系客服尝试快递拦截，拦截不保证成功。', '订单服务'),
('登录密码忘了怎么重置？', '在登录页点击「忘记密码」，使用绑定手机号完成短信验证后即可重置密码。', '账户'),
('如何开具发票？', '提交订单时勾选「开具发票」并填写发票抬头与税号，电子发票将在 48 小时内发送到您预留的邮箱。', '发票');

-- 演示数据（工单倒查/演示展示用）
INSERT INTO conversations (user_id, status) VALUES ('demo_user', '已结束');
SET @demo_conv_id = LAST_INSERT_ID();
INSERT INTO messages (conversation_id, role, content) VALUES
(@demo_conv_id, 'user', '收到的杯子有裂纹，想退换。'),
(@demo_conv_id, 'assistant', '好的，已为您创建售后工单，人工客服会尽快与您联系处理退换事宜。');
INSERT INTO tickets (ticket_no, conversation_id, description, ticket_type, status) VALUES
('T20260901001', @demo_conv_id, '杯子到货破损，用户要求退换', '售后', '已处理');
```

写完后自查红线：

```bash
grep -n "邮费\|运费" db/init/02_seed.sql
```

Expected: **零输出**（红线：全表不得出现该字样）。若有命中，改写该行措辞后复查。

- [x] **Step 5: 起容器并等 healthy**

```bash
docker compose up -d
docker compose ps --format "table {{.Name}}\t{{.Status}}"
```

Expected: `mewhelp-mysql` 状态含 `(healthy)`（首启初始化约 30–60s，可循环等待）。
**若容器起不来（端口占用/认证报错）** → 贴 `docker compose logs mysql` 关键行给用户，停工等指示（硬性核对点④）。

- [x] **Step 6: 验证建表灌数 + 认证兼容（硬性核对点④实测）**

caching_sha2_password 认证需要 cryptography（Task 1 已装）。用 aiomysql 直连实测：

```bash
uv run python - <<'PY'
import asyncio
import aiomysql

async def main():
    conn = await aiomysql.connect(
        host="127.0.0.1", port=3306, user="root",
        password="mewhelp_dev", db="mewhelp", charset="utf8mb4",
    )
    cur = await conn.cursor()
    await cur.execute("SHOW TABLES")
    tables = sorted(r[0] for r in await cur.fetchall())
    print("tables:", tables)
    assert tables == ["conversations", "faq", "messages", "tickets"], tables
    await cur.execute("SELECT COUNT(*) FROM faq")
    (n,) = await cur.fetchone()
    print("faq rows:", n)
    assert n == 10
    await cur.execute("SELECT COUNT(*) FROM faq WHERE question LIKE '%邮费%' OR answer LIKE '%邮费%' OR question LIKE '%运费%' OR answer LIKE '%运费%'")
    (m,) = await cur.fetchone()
    assert m == 0, "faq 种子出现邮费/运费字样，违反红线"
    await cur.execute("SELECT COUNT(*) FROM conversations")
    assert (await cur.fetchone())[0] == 1, "演示会话缺失"
    await cur.execute("SELECT COUNT(*) FROM messages")
    assert (await cur.fetchone())[0] == 2, "演示消息缺失"
    await cur.execute("SELECT ticket_no, ticket_type, status FROM tickets")
    assert await cur.fetchall() == [("T20260901001", "售后", "已处理")], "演示工单缺失或不对"
    print("OK: 4 tables, 10 faq rows, no 邮费/运费, demo data seeded")
    conn.close()

asyncio.run(main())
PY
```

Expected: `OK: 4 tables, 10 faq rows, no 邮费/运费, demo data seeded`。**若报认证错误**（`cryptography is required` / `Authentication plugin 'caching_sha2_password'`）→ 确认 `uv run python -c "import cryptography"`；仍不通则在 compose command 追加 `--default-authentication-plugin=mysql_native_password` 后 `docker compose down -v && docker compose up -d` 重来，并追记 dev-notes。

- [x] **Step 7: 写失败测试 `tests/test_engine.py`（纯单测，不连库）**

```python
import pytest


def test_get_engine_before_init_raises():
    from app.db import engine as eng

    # 单测环境从未 init_engine（除非其他测试先跑过，故先强制置空）
    eng._engine = None
    with pytest.raises(RuntimeError):
        eng.get_engine()


def test_init_engine_uses_settings_url(monkeypatch):
    from app.core.config import Settings
    from app.db import engine as eng

    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    s = Settings(_env_file=None)
    e = eng.init_engine(s)
    assert "aiomysql" in str(e.url) or "asyncmy" in str(e.url)
    assert e is eng.get_engine()
    assert eng.get_session_factory() is not None
    eng._engine = None  # 复位，避免污染其他测试


def test_init_engine_idempotent(monkeypatch):
    from app.core.config import Settings
    from app.db import engine as eng

    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    s = Settings(_env_file=None)
    e1 = eng.init_engine(s)
    e2 = eng.init_engine(s)
    assert e1 is e2
    eng._engine = None
```

- [x] **Step 8: 跑测试确认失败**

Run: `uv run pytest tests/test_engine.py -v`
Expected: FAIL（ModuleNotFoundError: app.db.engine / AttributeError: _engine）

- [x] **Step 9: 实现 `app/db/engine.py`**

```python
"""异步引擎与会话工厂。main.py lifespan 负责 init/dispose，业务代码只取 session。"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings) -> AsyncEngine:
    """幂等初始化（pool_pre_ping 防容器重启后的陈旧连接）。"""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
        _session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )
        logger.info("async engine initialized: %s", _engine.url.render_as_string(hide_password=True))
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("engine not initialized; call init_engine(settings) first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("engine not initialized; call init_engine(settings) first")
    return _session_factory


async def check_db() -> None:
    """启动探活：SELECT 1。失败抛异常由 main.py 决定退出提示。"""
    async with get_session_factory()() as session:
        await session.execute(text("SELECT 1"))


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("async engine disposed")
    _engine = None
    _session_factory = None
```

- [x] **Step 10: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS

- [x] **Step 11: dev-notes 追记 + commit**

dev-notes「Task 3」段重点记核对点④实测结果（认证方式、是否加 mysql_native_password、healthy 等待时长）与种子红线 grep 自查结果。

```bash
git add docker-compose.yml db/init app/db/engine.py tests/test_engine.py dev-notes/ch02.md
git commit -m "feat(ch02): Docker MySQL + DDL/faq种子 + 异步引擎层

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: crud 层（会话/消息/FAQ 检索/工单号生成）+ 真库冒烟

**Files:**
- Create: `app/db/crud.py`
- Create: `tests/test_crud.py`（FakeSession 纯单测 + SQL 构造断言）

**Interfaces:**
- Consumes: Task 2 的 `Conversation/Message/Faq/Ticket`；`AsyncSession`
- Produces（Task 5 create_ticket、Task 8 persister/路由用，签名以此为准）:
  - `async create_or_get_conversation(session: AsyncSession, conversation_id: int | None, user_id: str) -> Conversation`（id 为 None 或查无此会话 → 新建）
  - `async add_message(session: AsyncSession, conversation_id: int, role: str, content: str | None = None, tool_calls: list | None = None, tool_call_id: str | None = None) -> Message`
  - `build_faq_query(keyword: str, limit: int = 3) -> Select`（question LIKE OR answer LIKE）
  - `async search_faq(session: AsyncSession, keyword: str, limit: int = 3) -> list[Faq]`
  - `next_ticket_no(today_count: int, today: date) -> str`（纯函数：`T{YYYYMMDD}{today_count+1:03d}`）
  - `async create_ticket(session: AsyncSession, *, conversation_id: int | None, description: str, ticket_type: str) -> Ticket`（撞号 IntegrityError 重算重试一次；成功后把会话 status 置「已转人工」）

- [x] **Step 1: 写失败测试 `tests/test_crud.py`**

```python
from datetime import date


class FakeSession:
    """最小 AsyncSession 替身：只覆盖 crud 用到的方法。"""

    def __init__(self, existing=None):
        self.existing = existing
        self.added = []
        self.commits = 0
        self.rollbacks = 0
        self.refreshed = []

    async def get(self, model, pk):
        return self.existing

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        for o in self.added:
            if hasattr(o, "id") and getattr(o, "id") is None:
                o.id = 42  # 模拟自增主键回填

    async def rollback(self):
        self.rollbacks += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)


def test_next_ticket_no_format():
    from app.db.crud import next_ticket_no

    d = date(2026, 9, 20)
    assert next_ticket_no(0, d) == "T20260920001"
    assert next_ticket_no(7, d) == "T20260920008"
    assert next_ticket_no(99, d) == "T20260920100"


def test_build_faq_query_is_or_like():
    from app.db.crud import build_faq_query

    sql = str(
        build_faq_query("退货", limit=3).compile(compile_kwargs={"literal_binds": True})
    )
    assert "question LIKE" in sql
    assert "answer LIKE" in sql
    assert "OR" in sql
    assert "%退货%" in sql
    assert "LIMIT" in sql


async def test_create_conversation_when_id_none():
    from app.db import crud

    s = FakeSession(existing=None)
    conv = await crud.create_or_get_conversation(s, None, "demo_user")
    assert conv.user_id == "demo_user"
    assert s.commits == 1
    assert conv in s.refreshed


async def test_reuse_existing_conversation():
    from app.db import crud
    from app.db.models import Conversation

    existing = Conversation(id=7, user_id="demo_user")
    s = FakeSession(existing=existing)
    conv = await crud.create_or_get_conversation(s, 7, "demo_user")
    assert conv is existing
    assert s.commits == 0


async def test_fallback_new_when_stale_id():
    """前端带了一个库里不存在的 conversation_id → 新建而不是报错。"""
    from app.db import crud

    s = FakeSession(existing=None)
    conv = await crud.create_or_get_conversation(s, 999, "demo_user")
    assert conv.id is not None or conv in s.refreshed
    assert s.commits == 1


async def test_add_message_passes_through_fields():
    from app.db import crud

    s = FakeSession()
    msg = await crud.add_message(
        s, 7, "assistant", content=None, tool_calls=[{"id": "c1", "name": "query_order"}]
    )
    assert msg.role == "assistant"
    assert msg.conversation_id == 7
    assert msg.tool_calls[0]["name"] == "query_order"
    assert s.commits == 1
```

- [x] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_crud.py -v`
Expected: FAIL（No module named 'app.db.crud'）

- [x] **Step 3: 实现 `app/db/crud.py`**

```python
"""数据访问层：会话/消息/FAQ 检索/工单。运行期落库失败的降级策略在调用方（persister）。"""

import logging
from datetime import date

from sqlalchemy import Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket

logger = logging.getLogger(__name__)


async def create_or_get_conversation(
    session: AsyncSession, conversation_id: int | None, user_id: str
) -> Conversation:
    """conversation_id 由服务端拥有：None 或查无 → 新建；查到 → 复用。"""
    if conversation_id is not None:
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            return conv
    conv = Conversation(user_id=user_id)
    session.add(conv)
    await session.commit()
    await session.refresh(conv)
    return conv


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(msg)
    await session.commit()
    return msg


def build_faq_query(keyword: str, limit: int = 3) -> Select:
    like = f"%{keyword}%"
    return (
        select(Faq)
        .where(or_(Faq.question.like(like), Faq.answer.like(like)))
        .limit(limit)
    )


async def search_faq(
    session: AsyncSession, keyword: str, limit: int = 3
) -> list[Faq]:
    rows = (await session.execute(build_faq_query(keyword, limit))).scalars().all()
    return list(rows)


def next_ticket_no(today_count: int, today: date) -> str:
    """T{YYYYMMDD}{当日已有工单数+1:03d}，如 T20260920001。"""
    return f"T{today:%Y%m%d}{today_count + 1:03d}"


async def create_ticket(
    session: AsyncSession,
    *,
    conversation_id: int | None,
    description: str,
    ticket_type: str,
) -> Ticket:
    """写工单 + 会话置「已转人工」。并发撞号（IntegrityError）重算重试一次。"""
    if conversation_id is None:
        raise ValueError("缺少会话上下文，无法创建工单")
    for attempt in (0, 1):
        today = date.today()
        count = (
            await session.execute(
                select(func.count())
                .select_from(Ticket)
                .where(func.date(Ticket.created_at) == today)
            )
        ).scalar_one()
        ticket = Ticket(
            ticket_no=next_ticket_no(count, today),
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        session.add(ticket)
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            if attempt == 1:
                raise
            logger.warning("ticket_no collision, recomputing (attempt %d)", attempt + 1)
            continue
        await session.refresh(ticket)
        return ticket
    raise RuntimeError("unreachable")
```

- [x] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS

- [x] **Step 5: 真库冒烟（需 Task 3 容器仍在跑；crud 全链路 + 验收 3 的漏召回预检）**

```bash
uv run python - <<'PY'
import asyncio
from app.core.config import get_settings
from app.db import crud, engine


async def main():
    engine.init_engine(get_settings())
    await engine.check_db()
    async with engine.get_session_factory()() as s:
        hits = await crud.search_faq(s, "退货政策")
        assert hits, "「退货政策」应命中（验收 2 前提）"
        print("faq hit:", hits[0].question)
        miss = await crud.search_faq(s, "邮费")
        assert not miss, "「邮费」应漏召回（验收 3 前提）"
        print("faq miss on 邮费: OK")
        conv = await crud.create_or_get_conversation(s, None, "demo_user")
        print("conversation id:", conv.id)
        t = await crud.create_ticket(
            s, conversation_id=conv.id, description="冒烟测试工单", ticket_type="咨询"
        )
        await s.refresh(conv)
        print("ticket_no:", t.ticket_no, "| conv status:", conv.status)
        assert t.ticket_no.startswith("T") and conv.status == "已转人工"
        await crud.add_message(s, conv.id, "user", content="冒烟")
        print("SMOKE OK")

    await engine.dispose_engine()

asyncio.run(main())
PY
```

Expected: `SMOKE OK`（若 ticket_no 非 001 说明库里已有当天冒烟残留，无碍）。冒烟产生的测试行留在库里可接受（演示数据）。

- [x] **Step 6: dev-notes 追记 + commit**

```bash
git add app/db/crud.py tests/test_crud.py dev-notes/ch02.md
git commit -m "feat(ch02): crud 层(会话/消息/FAQ LIKE/工单号生成与撞号重试) + 真库冒烟

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: 五个 @tool 业务工具 + 注册管理（硬性核对点②）

**Files:**
- Create: `app/tools/__init__.py`（空文件）
- Create: `app/tools/definitions.py`
- Create: `app/tools/registry.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Consumes: `get_session_factory()`（Task 3，query_faq/create_ticket 开自己的会话——工具执行发生在编排层持有的请求 session 之外，避免跨协程共享 session）；`crud.search_faq/create_ticket`（Task 4）；`Settings.demo_user_id`（Task 1）
- Produces:
  - `definitions.py`: `query_order(order_id: str) -> dict`、`query_product(product_id: str) -> dict`、`query_logistics(order_id: str) -> dict`（三个 mock，`random` 造数据）、`query_faq(keyword: str) -> dict`（返回 `{"keyword":..., "hits":[{"id","question","answer","category"}]}`）、`create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"], config: RunnableConfig) -> dict`（返回 `{"ticket_no":..., "status":"待处理"}`；conversation_id 从 `config["configurable"]["conversation_id"]` 取，**模型 schema 不含此参数**）
  - `registry.py`: `TOOL_REGISTRY: dict[str, BaseTool]`、`get_tools() -> list[BaseTool]`（bind_tools 用，固定顺序）、`get_tool(name: str) -> BaseTool | None`

- [x] **Step 1: 硬性核对点② —— create_ticket 的 conversation_id 注入机制（Context7 + 本地实测，写实现前做）**

1. Context7 查（libraryId 用 resolve-library-id 对 "LangChain" 的结果，选 Python 主库）：query 形如 `python @tool inject RunnableConfig configurable hide argument from model schema InjectedToolArg`。确认当前版本推荐的注入方式。
2. 本地写探针文件 `_tmp_probe_tool.py` 实测 **RunnableConfig 注入**（首选，spec §4）：

```python
import asyncio
from typing import Literal

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool
def probe(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    config: RunnableConfig,
) -> dict:
    """探针工具。description: 描述; ticket_type: 类型"""
    cid = (config.get("configurable") or {}).get("conversation_id")
    return {"cid": cid}


print("model-visible args:", sorted(probe.args.keys()))
out = asyncio.run(
    probe.ainvoke(
        {"description": "x", "ticket_type": "咨询"},
        config={"configurable": {"conversation_id": 99}},
    )
)
print("invoke result:", out)
assert sorted(probe.args.keys()) == ["description", "ticket_type"], "config 泄漏进模型 schema！"
assert out == {"cid": 99}, out
print("PROBE OK: RunnableConfig 注入可用且不进模型 schema")
```

Run: `uv run python _tmp_probe_tool.py`，然后 `rm _tmp_probe_tool.py`。

**判定**：PROBE OK → 用 RunnableConfig 方案继续。失败 → 备选 **InjectedToolArg**：

```python
from typing import Annotated
from langchain_core.tools import InjectedToolArg

@tool
def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict: ...
```

（备选方案下 Task 6 executor 的调用要改为 `tool.ainvoke({**args, "conversation_id": ctx.conversation_id})`——两处必须同改，dev-notes 记一笔。）**两个机制都不可用 → 停工问用户。**

3. 顺手实测 **dict 返回值透传行为**（决定 Step 2 测试断言写法与 Task 6 executor 的 str 兜底是否会被触发）：

```bash
uv run python -c "
import asyncio
from langchain_core.tools import tool

@tool
def echo(x: str) -> dict:
    '''回声。x: 输入'''
    return {'got': x}

out = asyncio.run(echo.ainvoke({'x': 'hi'}))
print(type(out).__name__, out)
"
```

Expected: `dict {'got': 'hi'}`（当前 langchain_core 对非 str 返回值原样透传）。**若打印 `str`**：Step 2 各测试的 `out[...]` 断言前需先做同款解析（executor 的 str→json 兜底逻辑见 Task 6，届时其 `test_string_result_coerced` 就是主路径而非兜底），dev-notes 记一笔实测结论。

- [x] **Step 2: 写失败测试 `tests/test_tools.py`**

mock 三工具用确定性 monkeypatch（random 播种/打桩），DB 两工具用 FakeSession/monkeypatch crud：

```python
import asyncio
import random

import pytest


def test_registry_contains_five_tools_in_order():
    from app.tools.registry import TOOL_REGISTRY, get_tool, get_tools

    assert list(TOOL_REGISTRY) == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]
    assert [t.name for t in get_tools()] == list(TOOL_REGISTRY)
    assert get_tool("query_order") is TOOL_REGISTRY["query_order"]
    assert get_tool("nope") is None


def test_create_ticket_schema_hides_conversation_id():
    """硬性核对点②的回归断言：模型可见参数只有 description/ticket_type。"""
    from app.tools.registry import get_tool

    assert sorted(get_tool("create_ticket").args.keys()) == [
        "description",
        "ticket_type",
    ]


def test_mock_tools_have_docstrings_and_schemas():
    from app.tools.registry import get_tool

    for name in ("query_order", "query_product", "query_logistics"):
        t = get_tool(name)
        assert t.description.strip(), f"{name} 缺 docstring（模型选工具靠它）"
        assert "order_id" in t.args or "product_id" in t.args


async def test_query_logistics_shape(monkeypatch):
    from app.tools import definitions as d

    monkeypatch.setattr(random, "randint", lambda a, b: a)  # 确定性
    monkeypatch.setattr(random, "choice", lambda seq: seq[0])
    out = await d.query_logistics.ainvoke({"order_id": "1001"})
    assert out["order_id"] == "1001"
    assert out["carrier"] and out["current_status"]
    assert isinstance(out["traces"], list) and len(out["traces"]) >= 1
    assert {"time", "location", "detail"} <= set(out["traces"][0].keys())


async def test_query_order_and_product_shape():
    from app.tools import definitions as d

    o = await d.query_order.ainvoke({"order_id": "1001"})
    assert o["order_id"] == "1001"
    assert {"status", "amount", "items", "created_at"} <= set(o.keys())
    p = await d.query_product.ainvoke({"product_id": "2001"})
    assert p["product_id"] == "2001"
    assert {"name", "price", "stock", "category"} <= set(p.keys())


async def test_query_faq_hits(monkeypatch):
    from app.db.models import Faq
    from app.tools import definitions as d

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def fake_search_faq(session, keyword, limit=3):
        return [Faq(id=1, question=f"{keyword}是什么", answer="答案A", category="售后政策")]

    monkeypatch.setattr(d, "search_faq", fake_search_faq)
    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    out = await d.query_faq.ainvoke({"keyword": "退货政策"})
    assert out["keyword"] == "退货政策"
    assert out["hits"][0]["answer"] == "答案A"


async def test_query_faq_miss_shape(monkeypatch):
    """漏召回也要返回结构化空结果（模型据此如实告知）。"""
    from app.tools import definitions as d

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    async def fake_search_faq(session, keyword, limit=3):
        return []

    monkeypatch.setattr(d, "search_faq", fake_search_faq)
    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    out = await d.query_faq.ainvoke({"keyword": "邮费"})
    assert out == {"keyword": "邮费", "hits": []}


async def test_create_ticket_uses_config_conversation_id(monkeypatch):
    from app.tools import definitions as d

    seen = {}

    async def fake_create_ticket(session, *, conversation_id, description, ticket_type):
        seen.update(conversation_id=conversation_id, description=description, ticket_type=ticket_type)
        from app.db.models import Ticket

        return Ticket(ticket_no="T20260920001", status="待处理")

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    # 注意：definitions.py 里 `from app.db.crud import create_ticket as crud_create_ticket`，
    # 打桩目标是模块内别名 crud_create_ticket，不是 @tool 对象本身
    monkeypatch.setattr(d, "crud_create_ticket", fake_create_ticket)
    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    out = await d.create_ticket.ainvoke(
        {"description": "商品破损要退货", "ticket_type": "售后"},
        config={"configurable": {"conversation_id": 7}},
    )
    assert out == {"ticket_no": "T20260920001", "status": "待处理"}
    assert seen == {
        "conversation_id": 7,
        "description": "商品破损要退货",
        "ticket_type": "售后",
    }


async def test_create_ticket_without_conversation_raises(monkeypatch):
    """没有会话上下文（如评估脚本直调）→ ValueError，由 executor 包成错误结果。"""
    from app.tools import definitions as d

    class Sess:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(d, "get_session_factory", lambda: (lambda: Sess()))
    with pytest.raises(ValueError):
        await d.create_ticket.ainvoke(
            {"description": "x", "ticket_type": "咨询"},
            config={"configurable": {}},
        )
```

- [x] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL（No module named 'app.tools'）

- [x] **Step 4: 实现 `app/tools/definitions.py`**

mock 数据随机但字段形状固定（spec §4）；docstring 即工具描述（模型选工具的唯一依据，写清楚「什么时候用我」）：

```python
"""五个业务工具。query_order/query_product/query_logistics 为 mock（不接真实接口、不建表）；
query_faq 查 faq 表（SQL LIKE）；create_ticket 写 tickets 表。

工具函数一律返回 dict（结构化结果），异常向上抛由 executor 统一包装——
definitions 里不写错误处理（职责分离）。
"""

import random
from datetime import datetime, timedelta
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.db.crud import create_ticket as crud_create_ticket
from app.db.crud import search_faq
from app.db.engine import get_session_factory

_ORDER_STATUSES = ["待付款", "待发货", "运输中", "已签收", "已取消"]
_CARRIERS = ["中通快递", "圆通速递", "韵达快递", "顺丰速运"]
_CITIES = ["杭州转运中心", "苏州分拨中心", "南京集散中心", "上海虹桥网点", "北京大兴网点"]
_PRODUCT_NAMES = ["喵帮定制猫爬架", "冻干鸡肉猫粮 2kg", "宠物自动饮水机", "猫砂盆除臭剂", "磨爪逗猫棒套装"]
_CATEGORIES = ["猫粮", "用品", "零食", "清洁"]


@tool
async def query_order(order_id: str) -> dict:
    """按订单号查询订单状态、金额与商品明细。用户问「我的订单」「订单到哪一步了」「订单状态」时使用。order_id: 订单号，如 1001。"""
    rnd = random.Random(f"order-{order_id}")  # 同订单号结果稳定，演示可复现
    n_items = rnd.randint(1, 3)
    return {
        "order_id": order_id,
        "status": rnd.choice(_ORDER_STATUSES),
        "amount": round(rnd.uniform(29, 599), 2),
        "created_at": (datetime.now() - timedelta(days=rnd.randint(0, 30))).strftime("%Y-%m-%d %H:%M"),
        "items": [
            {"name": rnd.choice(_PRODUCT_NAMES), "qty": rnd.randint(1, 2)}
            for _ in range(n_items)
        ],
    }


@tool
async def query_product(product_id: str) -> dict:
    """按商品编号查询商品名称、价格、库存与分类。用户问某个商品「多少钱」「有没有货」「商品信息」时使用。product_id: 商品编号，如 2001。"""
    rnd = random.Random(f"product-{product_id}")
    return {
        "product_id": product_id,
        "name": rnd.choice(_PRODUCT_NAMES),
        "price": round(rnd.uniform(9.9, 399.0), 2),
        "stock": rnd.randint(0, 200),
        "category": rnd.choice(_CATEGORIES),
    }


@tool
async def query_logistics(order_id: str) -> dict:
    """按订单号查询物流轨迹：承运商、当前状态与最近几条轨迹。用户问「物流到哪了」「快递走到哪了」「什么时候到」时使用。order_id: 订单号，如 1001。"""
    rnd = random.Random(f"logistics-{order_id}")
    n = rnd.randint(2, 4)
    now = datetime.now()
    cities = rnd.sample(_CITIES, n)
    traces = [
        {
            "time": (now - timedelta(hours=8 * (n - i))).strftime("%Y-%m-%d %H:%M"),
            "location": city,
            "detail": rnd.choice(["快件已到达", "快件已发出，下一站", "运输中", "已揽收"]),
        }
        for i, city in enumerate(cities)
    ]
    return {
        "order_id": order_id,
        "carrier": rnd.choice(_CARRIERS),
        "current_status": rnd.choice(["运输中", "派送中", "已签收", "已揽收"]),
        "traces": traces,
    }


@tool
async def query_faq(keyword: str) -> dict:
    """按关键词检索平台常见问题（退货政策、发货时间、发票、会员积分等）。用户咨询平台规则/政策类问题时优先使用。keyword: 检索关键词，如「退货政策」。"""
    async with get_session_factory()() as session:
        rows = await search_faq(session, keyword)
    return {
        "keyword": keyword,
        "hits": [
            {"id": f.id, "question": f.question, "answer": f.answer, "category": f.category}
            for f in rows
        ],
    }


@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    config: RunnableConfig,
) -> dict:
    """创建人工客服工单（转人工）。仅当用户明确要求转人工，或问题超出工具与 FAQ 能力、需要人工跟进时使用。description: 用一句话概括用户的问题与诉求; ticket_type: 工单类型，售后/投诉/咨询三选一。"""
    conversation_id = (config.get("configurable") or {}).get("conversation_id")
    async with get_session_factory()() as session:
        ticket = await crud_create_ticket(
            session,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        return {"ticket_no": ticket.ticket_no, "status": ticket.status}
```

（若核对点②判定改用 InjectedToolArg，此处 create_ticket 按 Step 1 备选代码改写，测试 `test_create_ticket_uses_config_conversation_id` 的 invoke 参数同步改为在输入 dict 里带 conversation_id 的形式，并在 dev-notes 记录切换原因。）

- [x] **Step 5: 实现 `app/tools/registry.py`**

```python
"""工具注册管理：唯一权威清单。bind_tools、executor 查找、评估脚本都从这里取，防止三处清单漂移。"""

from langchain_core.tools import BaseTool

from app.tools.definitions import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
)

TOOL_REGISTRY: dict[str, BaseTool] = {
    "query_order": query_order,
    "query_product": query_product,
    "query_logistics": query_logistics,
    "query_faq": query_faq,
    "create_ticket": create_ticket,
}


def get_tools() -> list[BaseTool]:
    """bind_tools 用的工具列表（固定顺序，便于测试与排查）。"""
    return list(TOOL_REGISTRY.values())


def get_tool(name: str) -> BaseTool | None:
    return TOOL_REGISTRY.get(name)
```

- [x] **Step 6: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS（test_tools 8 个新测试 + 既有全部）

- [x] **Step 7: dev-notes 追记 + commit**

dev-notes「Task 5」段重点记核对点②结论（RunnableConfig 还是 InjectedToolArg、探针输出、Context7 查到的说法）。

```bash
git add app/tools tests/test_tools.py dev-notes/ch02.md
git commit -m "feat(ch02): 五个 @tool 业务工具(3 mock + faq 检索 + 工单) + 注册管理

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: 工具执行器（超时/重试/错误包装/摘要）

**Files:**
- Create: `app/tools/executor.py`
- Create: `tests/test_executor.py`

**Interfaces:**
- Consumes: `get_tool(name)`（Task 5）；`ToolContext.timeout_seconds/max_retries` 的取值来自 `Settings.tool_timeout_seconds/tool_max_retries`（Task 1，Task 7 编排层负责装配）
- Produces（Task 7 编排、Task 8 落库用，签名以此为准）:
  - `@dataclass ToolContext`: `conversation_id: int | None = None`、`timeout_seconds: float = 5.0`、`max_retries: int = 1`
  - `@dataclass ToolOutcome`: `name: str`、`tool_call_id: str`、`ok: bool`、`result: Any`（dict；失败时 `{"error": "工具执行失败: ..."}`）、`summary: str`（≤80 字符，前端徽章文案/落库）
  - `make_summary(name: str, result: Any) -> str`
  - `async execute_tool(name: str, args: dict, tool_call_id: str, context: ToolContext) -> ToolOutcome`——**绝不向上抛异常**：未注册工具/超时/任何 Exception 都包成 `ok=False` 的 ToolOutcome（错误信息回灌模型收敛，spec §9 错误矩阵）

- [ ] **Step 1: 写失败测试 `tests/test_executor.py`**

```python
import asyncio

import pytest

from app.tools import executor as ex
from app.tools.executor import ToolContext, execute_tool


class FakeTool:
    """替身：行为由注入的 async 函数决定，记录调用次数与 config。"""

    def __init__(self, behavior):
        self.behavior = behavior
        self.calls = 0
        self.seen_config = None
        self.seen_args = None

    async def ainvoke(self, args, config=None, **kwargs):
        self.calls += 1
        self.seen_args = args
        self.seen_config = config
        return await self.behavior(self.calls, args)


@pytest.fixture
def ctx():
    return ToolContext(conversation_id=42, timeout_seconds=0.05, max_retries=1)


async def test_success_passes_configurable(monkeypatch, ctx):
    async def ok(call, args):
        return {"order_id": args["order_id"], "status": "运输中", "amount": 99.0}

    fake = FakeTool(ok)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_order", {"order_id": "1001"}, "call_1", ctx)
    assert out.ok is True
    assert out.name == "query_order" and out.tool_call_id == "call_1"
    assert out.result["status"] == "运输中"
    assert fake.seen_config["configurable"]["conversation_id"] == 42
    assert fake.seen_args == {"order_id": "1001"}
    assert "运输中" in out.summary


async def test_timeout_retries_then_error(monkeypatch, ctx):
    async def slow(call, args):
        await asyncio.sleep(1)
        return {}

    fake = FakeTool(slow)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_order", {}, "call_1", ctx)
    assert out.ok is False
    assert fake.calls == 2  # 首次 + 重试 1 次
    assert "超时" in out.result["error"]


async def test_transient_error_then_success(monkeypatch, ctx):
    async def flaky(call, args):
        if call == 1:
            raise RuntimeError("boom")
        return {"keyword": "x", "hits": []}

    fake = FakeTool(flaky)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {"keyword": "x"}, "call_1", ctx)
    assert out.ok is True and fake.calls == 2
    assert out.summary == "未命中"


async def test_permanent_error_exhausts_retries(monkeypatch, ctx):
    async def always_fail(call, args):
        raise ValueError("bad args")

    fake = FakeTool(always_fail)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {}, "call_1", ctx)
    assert out.ok is False and fake.calls == 2
    assert "bad args" in out.result["error"]


async def test_unknown_tool_returns_error_outcome(ctx):
    out = await execute_tool("no_such_tool", {}, "call_1", ctx)
    assert out.ok is False
    assert "未注册" in out.result["error"]


async def test_string_result_coerced(monkeypatch, ctx):
    """工具若返回字符串（部分 LangChain 版本对 dict 会转 str），executor 兜底还原。"""
    async def returns_str(call, args):
        return '{"hits": [], "keyword": "y"}'

    fake = FakeTool(returns_str)
    monkeypatch.setattr(ex, "get_tool", lambda name: fake)
    out = await execute_tool("query_faq", {}, "call_1", ctx)
    assert out.ok and out.result == {"hits": [], "keyword": "y"}


def test_make_summary_shapes():
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": []}) == "未命中"
    assert ex.make_summary("query_faq", {"keyword": "k", "hits": [{"question": "q"}]}) == "命中 1 条"
    s = ex.make_summary("query_logistics", {"order_id": "1", "carrier": "中通快递", "current_status": "派送中", "traces": []})
    assert "派送中" in s
    t = ex.make_summary("create_ticket", {"ticket_no": "T20260920001", "status": "待处理"})
    assert t == "工单 T20260920001 已创建"
    e = ex.make_summary("query_order", {"error": "工具执行失败: boom"})
    assert e.startswith("失败")
    long = ex.make_summary("query_order", {"blob": "很" * 200})
    assert len(long) <= 80
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_executor.py -v`
Expected: FAIL（No module named 'app.tools.executor'）

- [ ] **Step 3: 实现 `app/tools/executor.py`**

```python
"""工具执行基础设施：查找注册表、Schema 由 LangChain @tool 声明式校验、
asyncio.wait_for 超时、有限重试、异常→错误结果包装、摘要生成。

失败兜底原则（spec §9）：execute_tool 绝不向上抛——任何失败都变成
ok=False 的 ToolOutcome，其 result["error"] 作为 ToolMessage 回灌给模型，
让模型向用户礼貌收敛，而不是打断 SSE 流。
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from app.tools.registry import get_tool

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    conversation_id: int | None = None
    timeout_seconds: float = 5.0
    max_retries: int = 1


@dataclass
class ToolOutcome:
    name: str
    tool_call_id: str
    ok: bool
    result: Any  # dict：成功=工具返回；失败={"error": "..."}
    summary: str  # ≤80 字符：前端徽章文案 / tool 消息落库 content


def make_summary(name: str, result: Any) -> str:
    """把工具结果压成一句 ≤80 字符的中文摘要（前端徽章/落库用）。"""
    if not isinstance(result, dict):
        text = str(result)
    elif "error" in result:
        text = f"失败: {result['error']}"
    elif name == "query_faq":
        hits = result.get("hits", [])
        text = "未命中" if not hits else f"命中 {len(hits)} 条"
    elif name == "create_ticket":
        text = f"工单 {result.get('ticket_no', '?')} 已创建"
    elif name in ("query_order", "query_product", "query_logistics"):
        status = result.get("current_status") or result.get("status")
        label = result.get("name") or result.get("carrier") or ""
        text = " ".join(x for x in (status, label) if x) or json.dumps(result, ensure_ascii=False)
    else:
        text = json.dumps(result, ensure_ascii=False)
    return text[:80]


async def execute_tool(
    name: str, args: dict, tool_call_id: str, context: ToolContext
) -> ToolOutcome:
    tool = get_tool(name)
    if tool is None:
        err = f"未注册的工具: {name}"
        logger.warning("tool lookup failed: %s", name)
        return ToolOutcome(name, tool_call_id, False, {"error": err}, make_summary(name, {"error": err}))

    config = {"configurable": {"conversation_id": context.conversation_id}}
    attempts = max(1, context.max_retries + 1)
    last_err = "未知错误"
    for i in range(attempts):
        try:
            result = await asyncio.wait_for(
                tool.ainvoke(args, config=config), timeout=context.timeout_seconds
            )
            if isinstance(result, str):
                # 部分 LangChain 版本会把 dict 返回值转成 str，兜底还原
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    result = {"text": result}
            return ToolOutcome(name, tool_call_id, True, result, make_summary(name, result))
        except TimeoutError:
            last_err = f"执行超时(>{context.timeout_seconds}s)"
            logger.warning("tool %s timeout (attempt %d/%d)", name, i + 1, attempts)
        except Exception as exc:  # noqa: BLE001 —— 兜底包装是设计目标
            last_err = f"{type(exc).__name__}: {exc}"
            logger.warning("tool %s failed (attempt %d/%d): %s", name, i + 1, attempts, exc)
    err_result = {"error": f"工具执行失败: {last_err}"}
    return ToolOutcome(name, tool_call_id, False, err_result, make_summary(name, err_result))
```

str 兜底分支（`isinstance(result, str)` 段）是否为主路径，以 **Task 5 Step 1-3** 的 dict 透传实测结论为准（已在那里跑过探针，无需重复）：实测为 dict 透传 → 该分支纯属防御；实测被 stringify → 该分支即主路径，`test_string_result_coerced` 覆盖它。两种情况上面代码都成立，把实测结论抄进 dev-notes 即可。

- [ ] **Step 4: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS

- [ ] **Step 5: dev-notes 追记 + commit**

```bash
git add app/tools/executor.py tests/test_executor.py dev-notes/ch02.md
git commit -m "feat(ch02): 工具执行器(5s超时+重试1次+错误包装回灌+摘要)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: 单轮编排 tool_chat_service（硬性核对点①）

**Files:**
- Create: `app/services/tool_chat_service.py`
- Create: `tests/test_tool_chat_service.py`

**Interfaces:**
- Consumes: `build_messages`/`stream_chat`（ch01 chat_service，**只 import 不修改**）；`get_tools()`（Task 5）；`ToolContext/ToolOutcome/execute_tool`（Task 6）；`Settings`（Task 1）；`ChatMessage`（ch01 schemas）
- Produces（Task 8 路由用，签名以此为准）:
  - `ToolEvent = tuple[str, Any]`——`("token", str)` | `("tool_call", {"id","name","args"})` | `("tool_result", {"id","name","ok","summary"})`（spec §5/§6）
  - `ChatPersister`（Protocol）：`async on_tool_calls(conversation_id: int, content: str, tool_calls: list)`、`async on_tool_result(conversation_id: int, outcome: ToolOutcome)`、`async on_final_answer(conversation_id: int, content: str)`（spec §7 的 on_turn_start 挂点在路由层直接走 crud，不经此协议——见 Task 8）
  - `async stream_chat_with_tools(chat_messages: list[ChatMessage], settings: Settings, model, *, conversation_id: int | None = None, persister: ChatPersister | None = None) -> AsyncIterator[ToolEvent]`

- [ ] **Step 1: 硬性核对点① —— chunk 累加与 tool_calls 聚合（Context7 + 本地实测，写实现前做）**

1. Context7 查（LangChain Python 主库）：query 形如 `bind_tools astream streaming tool call chunks AIMessageChunk aggregation add chunks together`。记录文档说法。
2. 本地实测（零 token，纯 langchain_core）：

```bash
uv run python - <<'PY'
from langchain_core.messages import AIMessageChunk

# 模拟上游把一次 tool call 的 args 拆成两个 chunk 推送
c1 = AIMessageChunk(content="")
c2 = AIMessageChunk(content="", tool_call_chunks=[
    {"name": "query_logistics", "args": '{"order_id": ', "id": "call_1", "index": 0, "type": "tool_call_chunk"},
])
c3 = AIMessageChunk(content="", tool_call_chunks=[
    {"name": None, "args": '"1001"}', "id": None, "index": 0, "type": "tool_call_chunk"},
])
full = c1 + c2 + c3
print("aggregated tool_calls:", full.tool_calls)
assert len(full.tool_calls) == 1, full.tool_calls
tc = full.tool_calls[0]
assert tc["name"] == "query_logistics", tc
assert tc["args"] == {"order_id": "1001"}, tc
assert tc["id"] == "call_1", tc
# 文本 chunk 的 .text 透传行为（ch01 stream_chat 同款取法）
assert AIMessageChunk(content="你好").text == "你好"
print("AGGREGATION OK: + 累加自动聚合分片 args")
PY
```

Expected: `AGGREGATION OK`。**若断言失败**（聚合行为与预期不符，如 args 未拼接/字段名不同）→ 把实际输出贴进 dev-notes，停工问用户（spec §12-① 约定）。核对通过后，Step 3 实现中的 `full = chunk if full is None else full + chunk` 与 `full.tool_calls` 即为经实测的写法。

- [ ] **Step 2: 写失败测试 `tests/test_tool_chat_service.py`**

```python
import pytest
from langchain_core.messages import AIMessageChunk, ToolMessage

from app.schemas.chat import ChatMessage
from app.services import tool_chat_service as svc
from app.tools.executor import ToolOutcome


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    from app.core.config import Settings

    return Settings(_env_file=None)


class FakeToolModel:
    """两轮各给一段流脚本；第二轮记录收到的完整 messages（验证回灌）。"""

    def __init__(self, round1, round2):
        self.scripts = [round1, round2]
        self.calls = 0
        self.round2_messages = None
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx == 1:
            self.round2_messages = list(messages)
        for c in self.scripts[idx]:
            yield c


class FakePersister:
    def __init__(self):
        self.calls = []

    async def on_tool_calls(self, conversation_id, content, tool_calls):
        self.calls.append(("tool_calls", conversation_id, content, tool_calls))

    async def on_tool_result(self, conversation_id, outcome):
        self.calls.append(("tool_result", conversation_id, outcome))

    async def on_final_answer(self, conversation_id, content):
        self.calls.append(("final", conversation_id, content))


def user_msg(text="你好"):
    return [ChatMessage(role="user", content=text)]


async def collect(agen):
    return [e async for e in agen]


def tc_chunk(args_fragment, first=False):
    """构造 tool_call 分片 chunk（first=True 带 name/id，模拟真实上游拆包）。"""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "query_logistics" if first else None,
                "args": args_fragment,
                "id": "call_1" if first else None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )


async def test_pure_chat_passthrough(settings):
    """无工具决策：token 实时透传，第二轮根本不发生，行为与 ch01 等价。"""
    model = FakeToolModel(
        round1=[AIMessageChunk(content="你好"), AIMessageChunk(content="呀")],
        round2=[],
    )
    p = FakePersister()
    events = await collect(
        svc.stream_chat_with_tools(
            user_msg(), settings, model, conversation_id=1, persister=p
        )
    )
    assert events == [("token", "你好"), ("token", "呀")]
    assert model.calls == 1
    assert model.round2_messages is None
    assert len(model.bound_tools) == 5  # 第一轮确实 bind 了 5 个工具
    assert ("final", 1, "你好呀") in p.calls
    assert [c[0] for c in p.calls] == ["final"]


async def test_tool_flow_events_and_round2_feedback(settings, monkeypatch):
    model = FakeToolModel(
        round1=[tc_chunk('{"order_id": ', first=True), tc_chunk('"1001"}')],
        round2=[AIMessageChunk(content="您的包裹"), AIMessageChunk(content="在杭州")],
    )

    async def fake_execute(name, args, tcid, ctx):
        assert name == "query_logistics"
        assert args == {"order_id": "1001"}  # 分片 args 已聚合解析
        assert tcid == "call_1"
        assert ctx.conversation_id == 3
        assert ctx.timeout_seconds == settings.tool_timeout_seconds
        assert ctx.max_retries == settings.tool_max_retries
        return ToolOutcome(
            name=name, tool_call_id=tcid, ok=True,
            result={"current_status": "运输中"}, summary="运输中",
        )

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    p = FakePersister()
    events = await collect(
        svc.stream_chat_with_tools(
            user_msg("订单1001物流到哪了"), settings, model,
            conversation_id=3, persister=p,
        )
    )
    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "token", "token"]
    assert events[0][1] == {"id": "call_1", "name": "query_logistics", "args": {"order_id": "1001"}}
    assert events[1][1] == {"id": "call_1", "name": "query_logistics", "ok": True, "summary": "运输中"}

    # 第二轮回灌：裸 model 收到 [...历史, AIMessage(含tool_calls), ToolMessage(结果JSON)]
    assert model.calls == 2
    r2 = model.round2_messages
    tm = [m for m in r2 if isinstance(m, ToolMessage)]
    assert len(tm) == 1
    assert tm[0].tool_call_id == "call_1"
    assert "运输中" in tm[0].content
    ai = [m for m in r2 if getattr(m, "tool_calls", None)]
    assert ai and ai[0].tool_calls[0]["name"] == "query_logistics"

    # 落库三挂点按序各一次
    assert [c[0] for c in p.calls] == ["tool_calls", "tool_result", "final"]
    assert p.calls[0][2] == ""  # 第一轮无文本
    assert p.calls[0][3][0]["name"] == "query_logistics"
    assert p.calls[2][2] == "您的包裹在杭州"


async def test_tool_failure_still_converges(settings, monkeypatch):
    """工具失败：ok=False 帧照发，错误 JSON 照样回灌，第二轮礼貌收敛（spec §9）。"""
    model = FakeToolModel(
        round1=[tc_chunk('{"order_id": "1001"}', first=True)],
        round2=[AIMessageChunk(content="抱歉，暂时查询不到")],
    )

    async def fake_execute(name, args, tcid, ctx):
        return ToolOutcome(
            name=name, tool_call_id=tcid, ok=False,
            result={"error": "工具执行失败: 执行超时(>5.0s)"}, summary="失败: 执行超时",
        )

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    events = await collect(
        svc.stream_chat_with_tools(user_msg("查订单"), settings, model, conversation_id=5)
    )
    kinds = [k for k, _ in events]
    assert kinds == ["tool_call", "tool_result", "token"]
    assert events[1][1]["ok"] is False
    r2 = model.round2_messages
    tm = [m for m in r2 if isinstance(m, ToolMessage)][0]
    assert "error" in tm.content


async def test_persister_failure_never_breaks_stream(settings):
    model = FakeToolModel(round1=[AIMessageChunk(content="好")], round2=[])

    class BrokenPersister:
        async def on_tool_calls(self, *a):
            raise RuntimeError("db down")

        async def on_tool_result(self, *a):
            raise RuntimeError("db down")

        async def on_final_answer(self, *a):
            raise RuntimeError("db down")

    events = await collect(
        svc.stream_chat_with_tools(
            user_msg(), settings, model, conversation_id=1, persister=BrokenPersister()
        )
    )
    assert events == [("token", "好")]  # 落库炸了流也要完整


async def test_no_persister_is_noop(settings):
    model = FakeToolModel(round1=[AIMessageChunk(content="喵")], round2=[])
    events = await collect(
        svc.stream_chat_with_tools(user_msg(), settings, model)
    )
    assert events == [("token", "喵")]
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_tool_chat_service.py -v`
Expected: FAIL（No module named 'app.services.tool_chat_service'）

- [ ] **Step 4: 实现 `app/services/tool_chat_service.py`**

```python
"""单轮 Function Calling 编排（spec §5）。

流程：build_messages（ch01 复用）→ 第一轮 bind_tools+astream（文本实时透传，
chunk 用 + 累加聚合 tool_calls，写法经硬性核对点①实测）→ 有工具申请则逐个
执行并 yield 状态帧 → 结果以 ToolMessage 回灌 → 第二轮**裸 model**（不
bind_tools，物理保证单轮收敛）流式产出最终回答。

ch01 兼容：纯闲聊路径的事件流与 chat_service.stream_chat 完全等价；
本文件不修改 chat_service 的任何代码。
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Protocol

from langchain_core.messages import AIMessageChunk, ToolMessage

from app.core.config import Settings
from app.schemas.chat import ChatMessage
from app.services.chat_service import build_messages, stream_chat
from app.tools.executor import ToolContext, ToolOutcome, execute_tool
from app.tools.registry import get_tools

logger = logging.getLogger(__name__)

# ("token", str) | ("tool_call", {id,name,args}) | ("tool_result", {id,name,ok,summary})
ToolEvent = tuple[str, Any]


class ChatPersister(Protocol):
    """编排层落库挂点（spec §7）。on_turn_start 在路由层直接走 crud。"""

    async def on_tool_calls(self, conversation_id: int, content: str, tool_calls: list) -> None: ...

    async def on_tool_result(self, conversation_id: int, outcome: ToolOutcome) -> None: ...

    async def on_final_answer(self, conversation_id: int, content: str) -> None: ...


def _text_of(msg: Any) -> str:
    """AIMessage(Chunk).content 可能是 str 或分片 list，统一压成 str。"""
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in c
        )
    return str(c)


def _serialize_tool_calls(tool_calls: list) -> list:
    """tool_calls → 可 JSON 化 list（落库 messages.tool_calls 列）。"""
    return [
        {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args") or {}}
        for tc in tool_calls
    ]


async def _persist(persister: ChatPersister | None, hook: str, *args: Any) -> None:
    """落库失败只 warning，绝不打断 SSE（spec §7/§9）。"""
    if persister is None:
        return
    try:
        await getattr(persister, hook)(*args)
    except Exception:  # noqa: BLE001
        logger.warning("persister.%s failed; stream continues", hook, exc_info=True)


async def stream_chat_with_tools(
    chat_messages: list[ChatMessage],
    settings: Settings,
    model: Any,
    *,
    conversation_id: int | None = None,
    persister: ChatPersister | None = None,
) -> AsyncIterator[ToolEvent]:
    messages = build_messages(chat_messages, settings)
    bound = model.bind_tools(get_tools())
    ctx = ToolContext(
        conversation_id=conversation_id,
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
    )

    # ---- 第一轮：bind_tools + astream，文本实时透传，chunk 累加聚合 ----
    full: AIMessageChunk | None = None
    async for chunk in bound.astream(messages):
        full = chunk if full is None else full + chunk
        text = getattr(chunk, "text", "") or ""
        if text:
            yield ("token", text)

    tool_calls = list(getattr(full, "tool_calls", None) or []) if full is not None else []
    if not tool_calls:
        # ---- 纯闲聊：第一轮即最终回答（与 ch01 行为等价）----
        await _persist(persister, "on_final_answer", conversation_id, _text_of(full) if full else "")
        return

    # ---- 有工具决策：先落 assistant(tool_calls) 行，再逐个执行 ----
    await _persist(
        persister, "on_tool_calls", conversation_id,
        _text_of(full), _serialize_tool_calls(tool_calls),
    )

    round2: list = [*messages, full]  # 第一轮完整 AIMessage（含 tool_calls）回传上游
    for tc in tool_calls:
        yield ("tool_call", {"id": tc["id"], "name": tc["name"], "args": tc.get("args") or {}})
        outcome = await execute_tool(tc["name"], tc.get("args") or {}, tc["id"], ctx)
        yield (
            "tool_result",
            {
                "id": outcome.tool_call_id,
                "name": outcome.name,
                "ok": outcome.ok,
                "summary": outcome.summary,
            },
        )
        round2.append(
            ToolMessage(
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=outcome.tool_call_id,
            )
        )
        await _persist(persister, "on_tool_result", conversation_id, outcome)

    # ---- 第二轮：裸 model（不 bind_tools）+ ch01 stream_chat，物理保证单轮收敛 ----
    parts: list[str] = []
    async for t in stream_chat(round2, model):
        parts.append(t)
        yield ("token", t)
    await _persist(persister, "on_final_answer", conversation_id, "".join(parts))
```

- [ ] **Step 5: 跑测试确认通过 + 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS（新增 5 个编排测试）

- [ ] **Step 6: dev-notes 追记 + commit**

dev-notes「Task 7」段重点记核对点①结论：Context7 说法、本地探针输出（聚合后 tool_calls 的实际字段）、qwen 真实上游留待 Task 10 实测。

```bash
git add app/services/tool_chat_service.py tests/test_tool_chat_service.py dev-notes/ch02.md
git commit -m "feat(ch02): 单轮工具编排(第一轮bind_tools流式透传+聚合, 第二轮裸模型收敛)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 落库实现 + 契约扩展 + 路由/生命周期接入 + conftest 加法升级

本任务把 ch02 接进 HTTP 层。ch01 红线在本任务的落点：`token`/`done`/`error` 三种帧的序列化写法**逐字符保留现状**（token=`data=text`、done=`raw_data="[DONE]"`、error=`data={"detail":...}`）；conftest 只做加法；ch01 测试文件零改动、必须全绿。

**Files:**
- Create: `app/services/persistence.py`
- Modify: `app/schemas/chat.py`（ChatRequest 加 conversation_id；文件尾加 3 个事件模型）
- Modify: `app/api/routes.py`（imports + dep_db_session + chat_stream 重写；extract/health 不动）
- Modify: `app/main.py`（lifespan 扩展）
- Modify: `tests/conftest.py`（FakeChatModel/BrokenChatModel 加法升级）
- Create: `tests/test_persistence.py`、`tests/test_schemas_ch02.py`、`tests/test_routes_ch02.py`

**Interfaces:**
- Consumes: `crud.*`（Task 4）、`get_session_factory`（Task 3）、`stream_chat_with_tools/ChatPersister`（Task 7）、`ToolOutcome`（Task 6）、`Settings.demo_user_id`（Task 1）
- Produces: `DBChatPersister(session, conversation_id)`；`dep_db_session()`（async generator 依赖：引擎未初始化 → yield None）；SSE 新事件 `conversation`/`tool_call`/`tool_result`；`ChatRequest.conversation_id`；lifespan 内 `init_engine/check_db/dispose_engine`（Task 9/10 前端与验收用）

- [ ] **Step 1: conftest 加法升级（先做，证明其本身无害）**

`tests/conftest.py` 三处改动，**其余内容一字不动**：

1）顶部 import 增加一行：

```python
from langchain_core.messages import AIMessageChunk
```

2）`FakeChatModel` 整类替换为（原 FakeChunk 类保留在原位，加一行注释说明 ch02 起 conftest 不再使用它；test_stream_chat.py 用的是自己的本地副本，不受影响）：

```python
class FakeChatModel:
    """路由测试用假模型：流式吐固定文本（真实 AIMessageChunk，支持 ch02 编排的
    chunk 累加/tool_calls 聚合），结构化提取返回 structured_result。"""

    structured_result = None  # 由各测试按需覆盖

    def __init__(self):
        self.bound_tools = None

    def bind_tools(self, tools, **kwargs):
        """ch02 编排会调用；记录工具清单并返回自身（流脚本已确定）。"""
        self.bound_tools = tools
        return self

    async def astream(self, messages, **kwargs):
        for t in ["你好", "呀", "喵"]:
            yield AIMessageChunk(content=t)

    def with_structured_output(self, schema, **kwargs):
        return FakeStructuredRunnable()
```

3）`BrokenChatModel` 整类替换为（只加 bind_tools，行为不变）：

```python
class BrokenChatModel:
    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        raise RuntimeError("upstream down")
        yield  # pragma: no cover （使其成为 async generator）

    def with_structured_output(self, schema, **kwargs):
        raise RuntimeError("upstream down")
```

Run: `uv run pytest -v`
Expected: **ch01 全部测试仍绿**（此时路由还没改，升级本身必须无害；若红说明升级写错了，先修）

- [ ] **Step 2: 写失败测试（三个新测试文件）**

`tests/test_schemas_ch02.py`：

```python
import pytest
from pydantic import ValidationError


def test_conversation_id_optional_and_defaults_none():
    from app.schemas.chat import ChatRequest

    r = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert r.conversation_id is None


def test_conversation_id_accepts_positive_int():
    from app.schemas.chat import ChatRequest

    r = ChatRequest(messages=[{"role": "user", "content": "hi"}], conversation_id=7)
    assert r.conversation_id == 7


def test_conversation_id_rejects_zero():
    from app.schemas.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(messages=[{"role": "user", "content": "hi"}], conversation_id=0)


def test_event_models_match_spec_frame_shape():
    """spec §6 帧 data 形状契约。"""
    from app.schemas.chat import ConversationEvent, ToolCallEvent, ToolResultEvent

    assert ConversationEvent(conversation_id=3).model_dump() == {"conversation_id": 3}
    assert ToolCallEvent(
        id="call_x", name="query_logistics", args={"order_id": "1001"}
    ).model_dump() == {"id": "call_x", "name": "query_logistics", "args": {"order_id": "1001"}}
    assert ToolResultEvent(
        id="call_x", name="query_logistics", ok=True, summary="运输中"
    ).model_dump() == {"id": "call_x", "name": "query_logistics", "ok": True, "summary": "运输中"}
```

`tests/test_persistence.py`：

```python
import json


async def test_persister_maps_hooks_to_rows(monkeypatch):
    """spec §7：on_tool_calls→assistant行(tool_calls JSON)；on_tool_result→tool行(结果JSON+tool_call_id)；on_final_answer→assistant行(全文)。"""
    from app.services import persistence as pers
    from app.tools.executor import ToolOutcome

    rows = []

    async def fake_add_message(
        session, conversation_id, role, content=None, tool_calls=None, tool_call_id=None
    ):
        rows.append((conversation_id, role, content, tool_calls, tool_call_id))

    monkeypatch.setattr(pers.crud, "add_message", fake_add_message)
    p = pers.DBChatPersister(session=object(), conversation_id=7)

    await p.on_tool_calls(7, "", [{"id": "c1", "name": "query_faq", "args": {"keyword": "退货"}}])
    outcome = ToolOutcome("query_faq", "c1", True, {"keyword": "退货", "hits": []}, "未命中")
    await p.on_tool_result(7, outcome)
    await p.on_final_answer(7, "最终回答")

    assert rows == [
        (7, "assistant", None, [{"id": "c1", "name": "query_faq", "args": {"keyword": "退货"}}], None),
        (7, "tool", json.dumps({"keyword": "退货", "hits": []}, ensure_ascii=False), None, "c1"),
        (7, "assistant", "最终回答", None, None),
    ]


async def test_persister_swallows_db_errors(monkeypatch):
    """运行期落库失败只 warning，绝不上抛（spec §9）。"""
    from app.services import persistence as pers
    from app.tools.executor import ToolOutcome

    async def boom(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(pers.crud, "add_message", boom)
    p = pers.DBChatPersister(session=object(), conversation_id=7)
    await p.on_tool_calls(7, "", [{"id": "c1", "name": "x", "args": {}}])
    await p.on_tool_result(7, ToolOutcome("x", "c1", False, {"error": "e"}, "失败"))
    await p.on_final_answer(7, "文本")  # 三个都不抛 = 通过
```

`tests/test_routes_ch02.py`：

```python
import json

from langchain_core.messages import AIMessageChunk


class FakeConv:
    id = 7


class FakeCrud:
    """routes 与 persistence 共用的 crud 替身：记录所有写入。"""

    def __init__(self):
        self.messages = []
        self.conv_calls = []

    async def create_or_get_conversation(self, session, conversation_id, user_id):
        self.conv_calls.append((conversation_id, user_id))
        return FakeConv()

    async def add_message(
        self, session, conversation_id, role, content=None, tool_calls=None, tool_call_id=None
    ):
        self.messages.append((conversation_id, role, content, tool_calls, tool_call_id))


class ScriptedToolModel:
    """第一轮吐 tool_call 分片，第二轮（裸模型）吐文本。"""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "query_faq",
                        "args": '{"keyword": "退货政策"}',
                        "id": "c1",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            )
        else:
            yield AIMessageChunk(content="7天无理由退换")


def _override(app, dep, value):
    app.dependency_overrides[dep] = lambda: value


async def test_no_db_degrades_to_ch01_frames(client):
    """引擎未初始化 → dep_db_session yield None：无 conversation 帧、无落库，token/done 与 ch01 完全一致。"""
    from app.api.routes import dep_chat_model
    from app.main import app
    from tests.conftest import FakeChatModel

    _override(app, dep_chat_model, FakeChatModel())
    r = await client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "你好"}]}
    )
    assert r.status_code == 200
    assert "event: conversation" not in r.text
    assert "event: tool_call" not in r.text
    data = [
        line[len("data:"):].strip()
        for line in r.text.splitlines()
        if line.startswith("data:") and line[len("data:"):].strip().startswith('"')
    ]
    assert "".join(json.loads(d) for d in data) == "你好呀喵"
    assert "[DONE]" in r.text


async def test_full_frames_with_db(client, monkeypatch):
    """帧序 conversation→token→done；on_turn_start 落 user 行；on_final_answer 落 assistant 行。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod
    from tests.conftest import FakeChatModel

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, FakeChatModel())

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "你好"}], "conversation_id": 7},
    )
    body = r.text
    assert body.index("event: conversation") < body.index("event: token")
    conv_lines = [
        line for line in body.splitlines()
        if line.startswith("data:") and "conversation_id" in line
    ]
    assert json.loads(conv_lines[0][len("data:"):].strip()) == {"conversation_id": 7}
    assert fake_crud.conv_calls == [(7, "demo_user")]
    assert (7, "user", "你好", None, None) in fake_crud.messages
    assert any(m[1] == "assistant" and m[2] == "你好呀喵" for m in fake_crud.messages)
    assert "[DONE]" in body


async def test_tool_frames_order_and_shape(client, monkeypatch):
    """帧序 conversation→tool_call→tool_result→token→done（spec §6）。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import tool_chat_service as svc
    from app.tools.executor import ToolOutcome

    monkeypatch.setattr(routes_mod, "crud", FakeCrud())
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, ScriptedToolModel())

    async def fake_execute(name, args, tcid, ctx):
        assert ctx.conversation_id == 7  # conversation 帧的 id 已注入工具上下文
        return ToolOutcome(name, tcid, True, {"keyword": "退货政策", "hits": [{"question": "退货政策是什么？"}]}, "命中 1 条")

    monkeypatch.setattr(svc, "execute_tool", fake_execute)

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "退货政策是什么"}]},
    )
    body = r.text
    idx = {ev: body.index(f"event: {ev}") for ev in ("conversation", "tool_call", "tool_result", "token", "done")}
    assert idx["conversation"] < idx["tool_call"] < idx["tool_result"] < idx["token"] < idx["done"]
    tc_line = [l for l in body.splitlines() if l.startswith("data:") and '"query_faq"' in l and '"args"' in l][0]
    assert json.loads(tc_line[len("data:"):].strip()) == {
        "id": "c1", "name": "query_faq", "args": {"keyword": "退货政策"}
    }
    tr_line = [l for l in body.splitlines() if l.startswith("data:") and '"ok"' in l][0]
    assert json.loads(tr_line[len("data:"):].strip()) == {
        "id": "c1", "name": "query_faq", "ok": True, "summary": "命中 1 条"
    }


async def test_conversation_id_zero_rejected_422(client):
    from app.main import app  # noqa: F401 —— 确保 app 已构建

    r = await client.post(
        "/api/chat/stream",
        json={"messages": [{"role": "user", "content": "hi"}], "conversation_id": 0},
    )
    assert r.status_code == 422


async def test_ch01_style_request_without_conversation_id_still_200(client, monkeypatch):
    """不带 conversation_id 的 ch01 请求体依然兼容（spec §6）。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import persistence as pers_mod
    from tests.conftest import FakeChatModel

    fake_crud = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake_crud)
    monkeypatch.setattr(pers_mod, "crud", fake_crud)
    _override(app, routes_mod.dep_db_session, object())
    _override(app, routes_mod.dep_chat_model, FakeChatModel())

    r = await client.post(
        "/api/chat/stream", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert r.text.count("event: token") == 3
    assert fake_crud.conv_calls == [(None, "demo_user")]  # 自动建一次性会话
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_schemas_ch02.py tests/test_persistence.py tests/test_routes_ch02.py -v`
Expected: FAIL（conversation_id 字段不存在 / No module named app.services.persistence / dep_db_session 不存在）

- [ ] **Step 4: 实现 `app/schemas/chat.py` 扩展**

1）`ChatRequest` 的 `messages` 字段之后加一行（validator 等其余内容不动）：

```python
    conversation_id: int | None = Field(default=None, ge=1)
```

2）文件末尾追加：

```python
class ConversationEvent(BaseModel):
    """SSE `conversation` 帧 data：服务端告知本轮会话 id（每轮都推，幂等）。"""

    conversation_id: int


class ToolCallEvent(BaseModel):
    """SSE `tool_call` 帧 data：模型决定调用工具（前端渲染「调用中」徽章）。"""

    id: str
    name: str
    args: dict


class ToolResultEvent(BaseModel):
    """SSE `tool_result` 帧 data：工具执行结束（前端更新徽章 ✓/✗ + summary 提示）。"""

    id: str
    name: str
    ok: bool
    summary: str
```

（`Field`/`BaseModel` 已在文件头 import；若 `Field` 未 import 则补上。）

- [ ] **Step 5: 实现 `app/services/persistence.py`**

```python
"""DBChatPersister：编排层 ChatPersister 协议（spec §7）的实现。

挂点对应关系：on_turn_start 在路由层直接走 crud（建/复用会话 + user 行，
发生在流开始前，需要拿到 conversation_id 才能装配后续一切）；其余三个挂点
由编排层在流中途回调。每个挂点内部自捕获异常只 warning——运行期落库失败
一律降级，绝不打断 SSE 聊天流（spec §9）。
"""

import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.tools.executor import ToolOutcome

logger = logging.getLogger(__name__)


class DBChatPersister:
    def __init__(self, session: AsyncSession, conversation_id: int):
        self._session = session
        self._conversation_id = conversation_id

    async def on_tool_calls(self, conversation_id: int, content: str, tool_calls: list) -> None:
        try:
            await crud.add_message(
                self._session,
                self._conversation_id,
                "assistant",
                content=content or None,
                tool_calls=tool_calls,
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist assistant(tool_calls) failed", exc_info=True)

    async def on_tool_result(self, conversation_id: int, outcome: ToolOutcome) -> None:
        try:
            await crud.add_message(
                self._session,
                self._conversation_id,
                "tool",
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=outcome.tool_call_id,
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist tool result failed", exc_info=True)

    async def on_final_answer(self, conversation_id: int, content: str) -> None:
        try:
            await crud.add_message(
                self._session, self._conversation_id, "assistant", content=content
            )
        except Exception:  # noqa: BLE001
            logger.warning("persist final answer failed", exc_info=True)
```

- [ ] **Step 6: 实现 `app/api/routes.py` 修改**

1）imports 区改为（新增 3 行 import，删去不再直接使用的 `build_messages/stream_chat`；`get_model` 保留）：

```python
import logging
from collections.abc import AsyncIterable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.core.config import get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.schemas.chat import (
    ChatRequest,
    ConversationEvent,
    HealthResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from app.schemas.extraction import AfterSaleExtraction, ExtractRequest
from app.services.chat_service import get_model
from app.services.extract_service import extract_after_sale
from app.services.persistence import DBChatPersister
from app.services.tool_chat_service import stream_chat_with_tools
```

2）`dep_chat_model` 之后新增：

```python
async def dep_db_session():
    """DB 会话依赖注入点：测试用 dependency_overrides 替换。
    引擎未初始化（ch01 单测环境/未配 MySQL）→ yield None，路由降级：
    不推 conversation 帧、不落库，token/done/error 行为与 ch01 完全一致。"""
    try:
        factory = get_session_factory()
    except RuntimeError:
        yield None
        return
    async with factory() as session:
        yield session
```

3）`chat_stream` 整函数替换为（extract/health 两个路由**一字不动**）：

```python
@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    req: ChatRequest,
    settings=Depends(dep_settings),
    model=Depends(dep_chat_model),
    session=Depends(dep_db_session),
) -> AsyncIterable[ServerSentEvent]:
    # ---- on_turn_start 挂点（spec §7）：建/复用会话 + user 行 + conversation 帧 ----
    conversation_id = req.conversation_id
    persister = None
    if session is not None:
        try:
            conv = await crud.create_or_get_conversation(
                session, conversation_id, settings.demo_user_id
            )
            conversation_id = conv.id
            await crud.add_message(
                session, conversation_id, "user", content=req.messages[-1].content
            )
            persister = DBChatPersister(session, conversation_id)
            yield ServerSentEvent(
                data=ConversationEvent(conversation_id=conversation_id).model_dump(),
                event="conversation",
            )
        except Exception:  # noqa: BLE001 —— 运行期落库失败只降级，不挡聊天（spec §9）
            logger.warning("conversation bootstrap failed; continue without persistence", exc_info=True)
            persister = None

    # ---- 单轮工具编排；token/done/error 帧写法与 ch01 逐字符一致（spec §13 红线）----
    try:
        async for kind, payload in stream_chat_with_tools(
            req.messages,
            settings,
            model,
            conversation_id=conversation_id,
            persister=persister,
        ):
            if kind == "token":
                yield ServerSentEvent(data=payload, event="token")
            elif kind == "tool_call":
                yield ServerSentEvent(
                    data=ToolCallEvent(**payload).model_dump(), event="tool_call"
                )
            elif kind == "tool_result":
                yield ServerSentEvent(
                    data=ToolResultEvent(**payload).model_dump(), event="tool_result"
                )
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例：错误进事件流后正常关流
        logger.exception("chat stream failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")
```

- [ ] **Step 7: 实现 `app/main.py` lifespan 扩展**

imports 区加一行：

```python
from app.db.engine import check_db, dispose_engine, init_engine
```

lifespan 整函数替换为（其余内容一字不动）：

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()  # 启动即校验 .env 必填项，不带病启动
    except ValidationError as exc:
        raise SystemExit(
            f"[启动失败] .env 配置缺失或非法，请参考 .env.example 补全：\n{exc}"
        ) from exc
    init_engine(settings)
    try:
        await check_db()  # 硬性核对点④的运行期形态：连不上快速失败
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[启动失败] 无法连接 MySQL。请先启动 Docker Desktop，然后执行 "
            f"`docker compose up -d`，等容器 healthy 后重试。详情: {exc}"
        ) from exc
    logger.info(
        "MySQL 已连接 %s:%s/%s", settings.mysql_host, settings.mysql_port, settings.mysql_db
    )
    yield
    await dispose_engine()
```

（httpx ASGITransport 不触发 lifespan，单测环境不需要 MySQL——这是 ch01 测试能在无 DB 环境跑通的前提，保持不变。）

- [ ] **Step 8: 全量回归（ch01 红线的判决点）**

Run: `uv run pytest -v`
Expected: **全绿**——ch01 34 个（零改动）+ ch02 新增全部。任何 ch01 测试变红都是红线事故：先修复实现，**禁止改 ch01 测试**。

- [ ] **Step 9: 起服务人工冒烟（需 Task 3 容器在跑）**

```bash
uv run uvicorn app.main:app --port 8000 &
sleep 3
curl -s http://127.0.0.1:8000/api/health
printf '{"messages":[{"role":"user","content":"你好"}]}' > _tmp_body.json
curl -sN -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @_tmp_body.json | head -20
rm _tmp_body.json
kill %1
```

Expected: health 200；SSE 首帧 `event: conversation`，随后 token 帧与 `[DONE]`；启动日志有「MySQL 已连接」。（此步用真实 qwen，消耗少量 token；若模型不可用仅验证 conversation 帧与错误帧即可，完整验证在 Task 10。）

- [ ] **Step 10: dev-notes 追记 + commit**

dev-notes「Task 8」段记：conftest 加法升级内容、ch01 回归判决结果、冒烟输出摘要。

```bash
git add app/services/persistence.py app/schemas/chat.py app/api/routes.py app/main.py tests/conftest.py tests/test_persistence.py tests/test_schemas_ch02.py tests/test_routes_ch02.py dev-notes/ch02.md
git commit -m "feat(ch02): SSE 接入工具编排(conversation/tool_call/tool_result 帧) + 落库 + 启动探活

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: System Prompt 工具指引 + 前端工具徽章（Vibe 例外区）

Prompt 是"非可单测产出"：结构性护栏用轻量测试（关键词存在），**质量由 Task 10 评估集验证**。前端 `static/index.html` 是 Vibe Coding 例外区（spec §8）：直接改、浏览器验证、用户反馈迭代，不套 TDD/review；但 spec §8 的"只加四点"边界必须守住，复古像素风与现有布局不动。

**Files:**
- Modify: `app/prompts/customer_service.py`（SYSTEM_PROMPT 约束 3/4 改写）
- Create: `tests/test_prompts_ch02.py`
- Modify: `static/index.html`（四点增量：conversationId / 3 新事件分支 / 工具徽章 / history 零改动）

**Interfaces:**
- Consumes: SSE `conversation`/`tool_call`/`tool_result` 帧（Task 8）
- Produces: 新 SYSTEM_PROMPT（Task 10 评估集与验收用）；前端 conversationId 闭环（验收 1/2/3 的浏览器载体）

- [ ] **Step 1: 写失败测试 `tests/test_prompts_ch02.py`**

```python
def test_system_prompt_keeps_ch01_keywords():
    """ch01 test_prompts 的四个关键词必须保留（红线连带约束）。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    for kw in ["喵帮", "客服", "订单号", "转人工"]:
        assert kw in SYSTEM_PROMPT


def test_system_prompt_has_tool_guidance():
    """spec §10：工具指引三要素——只能来自工具 / 严禁编造 / 无结果时如实告知并建议工单。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    for kw in ["工具", "严禁编造", "人工工单"]:
        assert kw in SYSTEM_PROMPT


def test_ch01_no_system_claim_replaced():
    """ch01 约束 4 的『没有接入任何查询系统』能力声明必须已被工具指引取代（现在真的接了）。"""
    from app.prompts.customer_service import SYSTEM_PROMPT

    assert "没有接入任何订单/物流查询系统" not in SYSTEM_PROMPT
```

Run: `uv run pytest tests/test_prompts_ch02.py -v` → Expected: FAIL（当前 Prompt 无「工具」指引、旧声明还在）

- [ ] **Step 2: 改写 `app/prompts/customer_service.py` 的 SYSTEM_PROMPT**

整段替换为（约束 1/2/5 一字不动；3 微调、4 重写；模板 CUSTOMER_SERVICE_PROMPT 不动）：

```python
SYSTEM_PROMPT = """你是「喵帮」电商平台的智能客服喵喵，负责解答购物、订单、物流、售后相关问题。

行为约束：
1. 只回答与电商购物、订单、物流、售后相关的问题；无关问题（如写作业、闲聊八卦）礼貌拒绝并引回主题。
2. 不越权承诺：不得替平台承诺赔偿金额、退款一定通过、具体到账时间等结果性内容；涉及此类诉求时说明流程并建议转人工客服。
3. 处理售后问题时，主动引导用户提供订单号，以便调用工具查询。
4. 工具使用指引：订单状态、物流轨迹、商品信息、平台政策等具体数据只能来自工具返回结果——涉及这些信息时先调用相应工具再作答，严禁编造工具结果，不得在未调用工具时声称"已查到""已核实"。工具无结果或执行失败时，如实告知用户暂时查询不到，并建议创建人工工单（转人工）跟进。
5. 语气友好、简洁，使用中文回答，适当使用「喵」保持品牌风格但不堆砌。"""
```

- [ ] **Step 3: 跑测试确认通过（含 ch01 prompt 测试）**

Run: `uv run pytest tests/test_prompts_ch02.py tests/test_prompts.py -v`
Expected: 全 PASS（ch01 test_prompts 的关键词断言依赖「喵帮/客服/订单号/转人工」，上面文本已保留）

- [ ] **Step 4: 前端四点增量改造 `static/index.html`（Vibe）**

改动 1 —— `<script>` 顶部 `const history = [];` 之后加：

```js
// ch02: 服务端会话 id（conversation 帧存值；「＋新对话」置 null；null 时请求体省略该字段）
let conversationId = null;
```

改动 2 —— `addMessage` 整函数替换（bubble 外包一层 `.msg-col`，徽章挂气泡上方；返回值多一个 col）：

```js
function addMessage(cls, text) {
  const row = document.createElement('div');
  row.className = 'msg ' + cls;
  if (cls === 'assistant') {
    const av = document.createElement('div');
    av.className = 'avatar';
    av.innerHTML = '<svg class="cat" aria-hidden="true"><use href="#cat-icon"></use></svg>';
    row.appendChild(av);
  }
  const col = document.createElement('div');
  col.className = 'msg-col';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  col.appendChild(bubble);
  row.appendChild(col);
  messagesEl.appendChild(row);
  scrollBottom();
  return { row, bubble, col };
}

// ch02: 工具徽章（像素风小方块，挂当前 AI 气泡上方；同轮多工具多徽章）
function addToolBadge(col, name) {
  const b = document.createElement('div');
  b.className = 'tool-badge pending';
  b.textContent = '🔧 ' + name + ' 调用中…';
  col.insertBefore(b, col.firstChild);
  scrollBottom();
  return b;
}
```

改动 3 —— `resetChat` 首行后加 `conversationId = null;`：

```js
function resetChat() {
  history.length = 0;
  conversationId = null;  // ch02: 新对话 → 服务端开新会话行
  messagesEl.innerHTML = '';
  addMessage('assistant', GREETING);
}
```

改动 4 —— `send()` 内四处：

a. `const { row, bubble } = addMessage('assistant', '');` 改为：

```js
  const { row, bubble, col } = addMessage('assistant', '');
  bubble.classList.add('typing');
  const badges = {};  // ch02: tool_call id → 徽章元素
```

（原 `bubble.classList.add('typing');` 行并入上面，勿重复。）

b. fetch body 行改为：

```js
      body: JSON.stringify(
        conversationId == null
          ? { messages: history }
          : { messages: history, conversation_id: conversationId }
      ),
```

c. 事件分发 `if (event === 'token') {...}` 之后、`else if (event === 'error')` 之前插入三个新分支：

```js
        } else if (event === 'conversation') {
          conversationId = JSON.parse(data).conversation_id;
        } else if (event === 'tool_call') {
          const tc = JSON.parse(data);
          badges[tc.id] = addToolBadge(col, tc.name);
        } else if (event === 'tool_result') {
          const tr = JSON.parse(data);
          const badge = badges[tr.id];
          if (badge) {
            badge.classList.remove('pending');
            badge.classList.add(tr.ok ? 'ok' : 'fail');
            badge.textContent = (tr.ok ? '✓ ' : '✗ ') + tr.name;
            badge.title = tr.summary || '';  // 悬停看摘要
          }
        }
```

d. `history.push({ role: 'assistant', content: answer });` 等 history 逻辑**零改动**（spec §8-4）；错误路径 `row.remove()` 会连带移除徽章（在 col 内），无需额外处理。

改动 5 —— `<style>` 内 `.bubble` 相关样式之后加（配色沿用现有变量：#FF9B50 橙 / #F6D89C 米黄 / #292929 墨线）：

```css
/* ch02: 消息列（徽章+气泡纵向排列）与工具徽章（像素风：2px 墨线边框 + 橙底小方块） */
.msg-col { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.tool-badge {
  align-self: flex-start;
  font-size: 11px;
  line-height: 1.4;
  padding: 2px 6px;
  background: #FF9B50;
  color: #292929;
  border: 2px solid #292929;
  box-shadow: 2px 2px 0 rgba(41, 41, 41, .35);
  white-space: nowrap;
}
.tool-badge.pending { background: #F6D89C; }
.tool-badge.fail { background: #E8DCC5; }
```

- [ ] **Step 5: JS 语法自检（ch01 模式：提取 script → node --check）**

```bash
uv run python - <<'PY'
import re
html = open("static/index.html", encoding="utf-8").read()
scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
open("_tmp_check.js", "w", encoding="utf-8").write("\n".join(scripts))
print("extracted", sum(len(s) for s in scripts), "chars")
PY
node --check _tmp_check.js && echo "JS OK" && rm _tmp_check.js
```

Expected: `JS OK`

- [ ] **Step 6: 全量回归**

Run: `uv run pytest -v`
Expected: 全 PASS

- [ ] **Step 7: 浏览器 Vibe 验证（需 MySQL 容器 + 起服）**

```bash
uv run uvicorn app.main:app --port 8000 &
```

**请用户在浏览器 http://127.0.0.1:8000/ 验证**（Vibe 例外区：按用户描述的效果迭代，改完重复本步）：
1. 问「订单 1001 的物流到哪了」→ 气泡上方出现「🔧 query_logistics 调用中…」→ 变「✓ query_logistics」，悬停有 summary；回答按 mock 轨迹
2. 问「退货政策是什么」→「✓ query_faq」徽章 + 按种子答案作答
3. 「＋新对话」后再问 → conversationId 已重置（服务端日志可见新会话行）
4. 布局回归：像素风无走样、气泡最大宽度正常、提取弹窗仍可用（msg-col 包装后若 max-width 表现变化，与用户一起微调 CSS）

验证通过后 `kill %1` 停服。

- [ ] **Step 8: dev-notes 追记 + commit**

dev-notes「Task 9」段记：Prompt 改写对照（旧约束 4 → 工具指引）、前端四点增量、用户浏览器验证反馈与迭代（Vibe 留痕：用户原话 + 每轮改了什么）。

```bash
git add app/prompts/customer_service.py tests/test_prompts_ch02.py static/index.html dev-notes/ch02.md
git commit -m "feat(ch02): Prompt 工具指引 + 聊天页工具徽章/会话id闭环 [vibe]

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: 工具路由评估集 + 三条验收 + 核对点① qwen 上游实测 + ch01 回归

**Files:**
- Create: `evals/tool_routing_samples.json`（spec 附录 C 的 10 条标注）
- Create: `evals/run_tool_routing_eval.py`

**Interfaces:**
- Consumes: 新 SYSTEM_PROMPT（Task 9）、`get_tools()`（Task 5）、`get_model`（ch01）、全部已接入的 HTTP 层（Task 8/9）
- Produces: 评估通过率记录、三条验收证据（帧序列 + 浏览器表现 + DB 行）、验收 3 漏召回留痕（dev-notes，ch03 输入）

- [ ] **Step 1: 写 `evals/tool_routing_samples.json`（spec 附录 C 逐条转录）**

```json
[
  {"id": 1, "question": "订单 DD20260901001 的物流到哪了", "expected_tools": ["query_logistics"], "remark": "验收 1 同款"},
  {"id": 2, "question": "帮我查一下订单 1001 买了什么", "expected_tools": ["query_order"], "remark": ""},
  {"id": 3, "question": "你们有蓝牙耳机卖吗，多少钱", "expected_tools": ["query_product"], "remark": ""},
  {"id": 4, "question": "退货政策是什么", "expected_tools": ["query_faq"], "remark": "验收 2，应命中种子 #1"},
  {"id": 5, "question": "邮费是多少", "expected_tools": ["query_faq"], "remark": "路由正确即通过；执行期 LIKE 漏召回是预期，留 ch03 向量检索升级"},
  {"id": 6, "question": "你们发货用什么快递", "expected_tools": ["query_faq"], "remark": "应命中种子 #5"},
  {"id": 7, "question": "你们一直没人处理我的问题，帮我登记个投诉", "expected_tools": ["create_ticket"], "remark": "ticket_type=投诉"},
  {"id": 8, "question": "你好", "expected_tools": [], "remark": "闲聊，no_tool"},
  {"id": 9, "question": "谢谢，再见", "expected_tools": [], "remark": "闲聊，no_tool"},
  {"id": 10, "question": "怎么申请退款", "expected_tools": ["query_faq"], "remark": "应命中种子 #2/#3"}
]
```

（`expected_tools: []` 即 no_tool = 空集合。样例 5 考察的是**路由**——模型该调 query_faq；漏召回发生在**执行**，由验收 3 单独验证。）

- [ ] **Step 2: 写 `evals/run_tool_routing_eval.py`**

```python
"""工具路由评估（Prompt 类产出的评估集替代 TDD，spec §10）。

真实模型调用：SYSTEM_PROMPT + 每条样例 → bind_tools → 第一轮 ainvoke
（只看决策，不执行工具，零 DB 依赖）→ 比对工具名集合。
用法：uv run python evals/run_tool_routing_eval.py   （需 .env 有效 OPENAI_* 配置）
判定：≥8/10 过阈值（目标 10/10）；FAIL 样例逐条分析原因并记录 dev-notes。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.prompts.customer_service import SYSTEM_PROMPT  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402
from app.tools.registry import get_tools  # noqa: E402

PASS_THRESHOLD = 8


async def main() -> int:
    samples = json.loads(
        (Path(__file__).parent / "tool_routing_samples.json").read_text(encoding="utf-8")
    )
    model = get_model(get_settings()).bind_tools(get_tools())

    passed = 0
    for s in samples:
        resp = await model.ainvoke(
            [("system", SYSTEM_PROMPT), HumanMessage(content=s["question"])]
        )
        actual = sorted(tc["name"] for tc in (getattr(resp, "tool_calls", None) or []))
        expected = sorted(s["expected_tools"])
        ok = actual == expected
        passed += ok
        mark = "PASS" if ok else "FAIL"
        remark = f" | remark: {s['remark']}" if s.get("remark") else ""
        print(f"[{mark}] #{s['id']:>2} {s['question']} | 期望 {expected} 实际 {actual}{remark}")

    total = len(samples)
    print(f"\n通过率: {passed}/{total}（阈值 {PASS_THRESHOLD}/{total}）")
    if passed < PASS_THRESHOLD:
        print("未达阈值：分析 FAIL 样例（Prompt 指引问题 or 模型能力问题），调整后重跑。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 3: 跑评估（真实模型，约 10 次调用）**

Run: `uv run python evals/run_tool_routing_eval.py`
Expected: `通过率: ≥8/10`，退出码 0。
**未达阈值** → 逐条分析 FAIL 输出：若因 SYSTEM_PROMPT 指引不足 → 回 Task 9 Step 2 微调措辞（保持四关键词）重跑；若模型能力问题（如样例 3 商品名歧义）→ 记 dev-notes 并**停下与用户确认**是否调整样例标注或阈值。

- [ ] **Step 4: 起服 + 三条验收 curl（帧级证据；同时是核对点①的 qwen 真实上游实测）**

前提：MySQL 容器 healthy（`docker compose ps`）、`.env` 有效。

```bash
uv run uvicorn app.main:app --port 8000 &
sleep 3

# 验收 1：问物流 → query_logistics 徽章帧 + 按结果作答（真实 qwen 流式 tool_calls 聚合在此实测——
# 若 tool_call 帧缺失或 args 残缺，即核对点①上游不兼容，停工带帧输出问用户）
printf '{"messages":[{"role":"user","content":"订单 1001 的物流到哪了"}]}' > _tmp_a1.json
curl -sN -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @_tmp_a1.json | grep -E "^(event|data):" | head -30

# 验收 2：退货政策 → query_faq 命中
printf '{"messages":[{"role":"user","content":"退货政策是什么"}]}' > _tmp_a2.json
curl -sN -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @_tmp_a2.json | grep -E "^(event|data):" | head -30

# 验收 3：邮费 → query_faq 漏召回（预期），模型如实告知
printf '{"messages":[{"role":"user","content":"邮费是多少"}]}' > _tmp_a3.json
curl -sN -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d @_tmp_a3.json | grep -E "^(event|data):" | head -30

rm _tmp_a1.json _tmp_a2.json _tmp_a3.json
```

Expected：
- 验收 1：帧序含 `conversation` → `tool_call`(name=query_logistics) → `tool_result`(ok:true) → 若干 `token` → `done`；token 拼接内容引用了轨迹/承运商（非编造：与 tool_result 前的 mock 返回一致）
- 验收 2：`tool_call`(query_faq) → `tool_result`(ok:true, summary=命中 1 条)；回答含「7 天」等种子答案要点
- 验收 3：`tool_call`(query_faq) → `tool_result`(ok:true, summary=**未命中**)；模型如实说查不到并建议工单/转人工——**这是预期结果**，把 token 回答原文粘进 dev-notes「漏召回记录」留给 ch03

浏览器端（Task 9 已验徽章，此处复核验收 1/2 的徽章 ✓ 状态与悬停 summary 即可）。

- [ ] **Step 5: 落库验证（spec §7 全链路证据）**

```bash
docker exec mewhelp-mysql mysql -uroot -pmewhelp_dev mewhelp -e \
"SELECT id, conversation_id, role, LEFT(content, 36) AS content, tool_call_id, tool_calls IS NOT NULL AS has_tc FROM messages ORDER BY id DESC LIMIT 12;"
```

Expected: 最近会话可见完整一轮：`user` 行 → `assistant` 行(has_tc=1, content 可空) → `tool` 行(tool_call_id 对号) → `assistant` 行(最终回答全文)。再看会话状态：

```bash
docker exec mewhelp-mysql mysql -uroot -pmewhelp_dev mewhelp -e \
"SELECT c.id, c.status, t.ticket_no FROM conversations c LEFT JOIN tickets t ON t.conversation_id = c.id ORDER BY c.id DESC LIMIT 5;"
```

（若验收过程让模型创建了工单：status=已转人工 + ticket_no 非空；没有工单调用则仅验证会话行存在。）

- [ ] **Step 6: ch01 全面回归（红线判决）**

```bash
kill %1  # 停服
uv run pytest -v          # 全量单测：ch01 34 个零改动全绿 + ch02 全部
```

再跑 README 的三条 ch01 curl 验收命令（health / chat stream / extract；中文 body 走 UTF-8 文件），行为与 ch01 交付时一致。

- [ ] **Step 7:（可选，建议做）启动探活反向验证**

```bash
docker compose stop mysql
uv run uvicorn app.main:app --port 8000   # 预期：SystemExit 提示启动 Docker Desktop + docker compose up -d
docker compose start mysql                # 恢复
```

- [ ] **Step 8: dev-notes 追记 + commit**

dev-notes「Task 10」段：评估通过率与 FAIL 分析、核对点① qwen 上游实测结论、三条验收证据摘要（帧序 + 回答要点）、**漏召回记录**（验收 3 原文，ch03 输入）、ch01 回归结果。

```bash
git add evals dev-notes/ch02.md
git commit -m "test(ch02): 工具路由评估集(10样例)+评估脚本; 三条验收与漏召回留痕

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: README 更新 + finish（分支收尾 + 完结交付摘要）

**Files:**
- Modify: `README.md`
- Modify: `dev-notes/ch02.md`（finish 段）

- [ ] **Step 1: README 新增 ch02 章节**

在 ch01 验收命令章节之后插入（结构与既有 README 风格对齐，标题层级按现有文件调整）：

```markdown
## ch02：Function Calling 工具链

### 前置：启动 MySQL

1. 启动 Docker Desktop
2. `docker compose up -d`（首次初始化约 30-60s，等 `docker compose ps` 显示 healthy）

`.env` 需包含 `MYSQL_HOST/MYSQL_PORT/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DB/DEMO_USER_ID/TOOL_TIMEOUT_SECONDS/TOOL_MAX_RETRIES`（见 `.env.example`，全有默认值）。

### 五个工具

| 工具 | 数据源 | 说明 |
|---|---|---|
| query_order | mock | 按订单号查状态/金额/明细（同订单号结果稳定） |
| query_product | mock | 按商品编号查名称/价格/库存 |
| query_logistics | mock | 按订单号查承运商/轨迹 |
| query_faq | MySQL faq 表 | 关键词 LIKE 检索平台政策 |
| create_ticket | MySQL tickets 表 | 创建人工工单，会话置「已转人工」 |

### SSE 事件（在 ch01 的 token/done/error 之上新增）

| 事件 | data | 时机 |
|---|---|---|
| conversation | `{"conversation_id": 3}` | 每轮第一帧，服务端会话 id |
| tool_call | `{"id","name","args"}` | 模型决定调用工具 |
| tool_result | `{"id","name","ok","summary"}` | 工具执行结束 |

### 验收命令

（三条 curl：物流徽章链路 / 退货政策命中 / 邮费漏召回——与 dev-notes ch02 Task 10 Step 4 相同，中文 body 先写 UTF-8 文件再 `-d @file`）

### 工具路由评估

`uv run python evals/run_tool_routing_eval.py`（真实模型 10 条样例，阈值 8/10）
```

同时更新 README 的项目结构清单（按现有树的风格补行）：`app/db/`（engine/models/crud）、`app/tools/`（definitions/registry/executor）、`app/services/tool_chat_service.py`、`app/services/persistence.py`、`db/init/`、`docker-compose.yml`、`evals/tool_routing_samples.json`、`evals/run_tool_routing_eval.py`。

- [ ] **Step 2: dev-notes/ch02.md 追记「Finish」段**

四要素齐备：①用户全程关键原话（spec 确认 + 执行约束）②关键产出（11 任务清单、测试总数、评估通过率、三条验收结论、漏召回记录指针）③用户拒绝或纠偏（如实记录执行期发生的）④翻车与返工（如实记录执行期发生的）。外加**完结交付摘要**（用户工作要求）：功能演示命令（docker compose up -d → uvicorn → 浏览器三问）、测试结果、dev-notes 路径。

- [ ] **Step 3: commit docs**

```bash
git add README.md dev-notes/ch02.md
git commit -m "docs(ch02): README 工具链章节 + dev-notes finish 段(交付摘要)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

- [ ] **Step 4: 调用 superpowers:finishing-a-development-branch 技能收尾**

流程（ch01 先例）：分支上全量测试绿 → merge 到 master → master 上复测全绿 → 删除 ch02-function-calling 分支。合并前确认 `git status` 干净（临时文件已清）。

- [ ] **Step 5: 向用户交付完结摘要**

内容：三条验收标准逐条结论（含验收 3 漏召回留痕位置）、评估通过率、测试总数（ch01 34 + ch02 新增）、演示命令、dev-notes/spec/plan 三个文档路径。

---

## 自检记录（writing-plans self-review）

- **Spec 覆盖**：§2 模块布局→Task 1-8 文件路径逐一对应；§3→Task 2/3/8；§4→Task 5/6；§5→Task 7；§6→Task 8/9；§7→Task 8（on_turn_start 在路由、三挂点经 persister，§7 表全覆盖）；§8→Task 9 四点增量；§9 错误矩阵→Task 6（超时/异常包装）、7（失败收敛）、8（落库降级/启动探活/不存在 id 新建）各有测试；§10→各任务 TDD + Task 10 评估与验收；§11→Task 1；§12 四核对点→Task 1③/3④/5②/7①+10①上游；§13 红线→Global Constraints + Task 8 Step 8、Task 10 Step 6 两道判决；附录 A→Task 3 Step 3（原样复制+grep 核对）；附录 B→Task 3 Step 4（逐条转录+演示数据）；附录 C→Task 10 Step 1
- **占位符扫描**：全部步骤含真实代码/命令/预期输出；DDL 一步为「从 spec 附录 A 原样复制」（唯一权威源是用户 DDL 原文，spec 已收录，复制+grep 计数核对，非 TBD）
- **类型一致性**：`ToolContext(conversation_id, timeout_seconds, max_retries)`、`ToolOutcome(name, tool_call_id, ok, result, summary)`、`execute_tool(name, args, tool_call_id, context)`、`stream_chat_with_tools(chat_messages, settings, model, *, conversation_id, persister)`、persister 三挂点签名、`crud.*` 六函数签名、`init_engine/get_engine/get_session_factory/check_db/dispose_engine`、`TOOL_REGISTRY/get_tools/get_tool`、Settings 八字段+database_url——跨任务引用已逐一对齐
