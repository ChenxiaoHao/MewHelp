# ch08 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把工具层从写死内置五件套升级为即插即用系统:注册中心(内置+MCP 每轮现拿)、单漏斗执行引擎(校验→权限→超时/重试→分诊→格式化→审计)、双自建 MCP Server、LangGraph interrupt 建工单确认流。

**Architecture:** 所有工具调用汇于 `execute_tool` 单点五道闸,元数据以 `ToolSpec` 三件套(name/description/JSON Schema + permission/source/mcp_server)统一承载;MCP 经 langchain-mcp-adapters `MultiServerMCPClient.get_tools(server_name=)` 每轮现拿(无缓存,单 server 故障降级不断线);create_ticket 是唯一 write——引擎拒无凭证调用并捕获预览,state 路由到独立 `ticket_confirm` 节点(节点第一行即 `interrupt`,重放零副作用),前端确认经 `POST /api/tickets/confirm` 以 `Command(resume=…)` 恢复同线程。

**Tech Stack:** MCP 官方 Python SDK(Streamable HTTP)+ langchain-mcp-adapters + jsonschema(全部 uv 解析版本,P2 破例新增);FastAPI SSE / LangGraph 1.2.11 InMemorySaver / SQLAlchemy 2 async / MySQL 8(docker 3307)。

**Spec:** `docs/superpowers/specs/2026-09-29-ch08-tool-system-design.md`(附录一=需求原文,附录二=DDL 原文,拍板 P1–P9 已记死)

## Global Constraints

- Python 3.12 + uv;测试默认 `pytest -m "not integration"`;Windows/Git Bash,控制台 GBK——任何含中文的验证输出一律落 UTF-8 文件再 Read。
- 一任务一 commit:功能+测试+计划勾选+dev-notes/ch08.md 阶段追加同笔;**禁 `git add -A`,永远精确列文件**;commit 尾 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- 新依赖只允许 `uv add mcp langchain-mcp-adapters jsonschema`(P2),不自行升级任何既有依赖。
- `db/init/08_ch08_tool_audit.sql` 必须逐字等于 spec 附录二 DDL(含 SET NAMES utf8mb4 与注释)。
- 三恒文件处置(P1):`app/tools/executor.py` 正名收编(T1 先 `git checkout --` 弃工作树未提交行,后续任务基于 HEAD 重构并正常提交);`app/prompts/self_check.py` 用户重写版随 T1 正名提交;`app/rag/retriever.py` T1 `git checkout --` 恢复 HEAD(弃调试 print)。
- 代码注释中文;测试字面量 ASCII-only(全角数字是历史翻车点);SSE 惯例=错误进 error 帧后正常关流。
- `ensure_ascii=False` 序列化红线贯穿所有 JSON 输出。
- dev-notes/ch08.md 每任务完成追加四样:关键原话/关键产出/拒绝纠偏/翻车与返工(无也写「无」)。

## Review Focus

1. **pending-interrupt 时用户发新消息**(不点卡片直接聊):隐式 cancel 必须先跑通(线程 drain 后新轮正常回答),不得 500、不得静默吞轮——T7 钉测。
2. **resume 决策值被前端乱填**(如 `"delete"`):请求模型 Literal 422,线程 interrupt 不被消耗——T8 钉测。
3. **MCP server 返回非 dict(str)**:executor 现有 json.loads 防御分支必须保活,格式化面不破——T5 钉测。
4. **审计 DB 不可用**(引擎未初始化/连接炸):只 WARN 丢行,工具执行照常返回——T1+T5 钉测。
5. **MCP 工具撞内置名**:丢弃+WARN,外部声明永不进权限表——T2 钉测。
6. 幻觉未登记工具名:拒绝回灌+审计「权限拒绝」(tool_source=builtin)——T3 钉测。

---

### Task 1: 依赖 + 审计表(DDL/ORM/sink)+ 三恒文件正名

**Files:**
- Modify: `pyproject.toml` `uv.lock`(uv add 产物)
- Create: `db/init/08_ch08_tool_audit.sql`(逐字=spec 附录二)
- Modify: `app/db/models.py`(尾部新增 ToolAuditLog)
- Modify: `app/db/crud.py`(新增 insert_tool_audit)
- Create: `app/tools/audit.py`(AuditRecord + db_audit_sink)
- Test: `tests/test_audit_ch08.py`
- Modify(正名): `app/prompts/self_check.py`(工作树现状直接提交)
- Restore(弃,不入库): `app/rag/retriever.py`、`app/tools/executor.py` 工作树行

**Interfaces:**
- Consumes: 无(基座任务)
- Produces: `audit.AuditRecord`(字段=DDL 列名 snake_case)、`audit.db_audit_sink(record)`、`crud.insert_tool_audit(session, **record_fields) -> None`、模型 `ToolAuditLog`

- [x] **Step 1: 装依赖 + 恒文件处置**

```bash
uv add mcp langchain-mcp-adapters jsonschema
git checkout -- app/rag/retriever.py app/tools/executor.py   # P1:调试 print 弃;executor 未提交行弃,后续基于 HEAD
```

Expected: `uv.lock` 三包入列(记解析到的版本到 dev-notes);`git status` 只剩 self_check.py 一个 M。

- [x] **Step 2: DDL 落盘 + dev 活库建表**

`db/init/08_ch08_tool_audit.sql` ← spec 附录二逐字。活库(docker mysql 3307)应用:

```bash
docker exec -i $(docker ps -q --filter expose=3306) mysql -uroot -pmewhelp_dev mewhelp < db/init/08_ch08_tool_audit.sql
```

Expected: 无报错;`SHOW COLUMNS FROM tool_audit_logs` 13 列、status ENUM 中文值原样(乱码即 charset 事故,回读 spec DDL 注释)。

- [x] **Step 3: 写失败测试** `tests/test_audit_ch08.py`

```python
"""ch08 T1:审计行形状 + sink 降级语义(引擎未初始化不拦执行)。"""
from dataclasses import asdict

from app.db import models
from app.tools import audit


def _record(**kw):
    base = dict(conversation_id=1, tool_call_id="c1", tool_name="query_order",
                tool_source="builtin", mcp_server=None, arguments={"order_id": "1001"},
                result_summary="运输中", status="成功", error_message=None,
                retry_count=0, duration_ms=12)
    base.update(kw)
    return audit.AuditRecord(**base)


def test_model_columns_match_ddl():
    cols = {c.name: c for c in models.ToolAuditLog.__table__.columns}
    assert set(cols) == {"id", "conversation_id", "tool_call_id", "tool_name",
                         "tool_source", "mcp_server", "arguments", "result_summary",
                         "status", "error_message", "retry_count", "duration_ms",
                         "created_at"}
    assert not models.ToolAuditLog.__table__.foreign_keys            # 无 FK(spec 审计节)
    assert cols["status"].type.enums == ["成功", "失败", "超时", "校验拦下", "权限拒绝"]


async def test_db_sink_engine_unavailable_warns(caplog):
    # 引擎未初始化:get_session_factory() RuntimeError → WARN 丢行,绝不 raise
    await audit.db_audit_sink(_record())
    assert any("audit" in r.getMessage().lower() for r in caplog.records)
```

Run: `pytest tests/test_audit_ch08.py -v` → FAIL(ImportError: audit 模块不存在)。

- [x] **Step 4: 实现**

`app/tools/audit.py`:

```python
"""工具调用审计 sink(spec 审计节):AuditRecord 逐列对 DDL;db_audit_sink
引擎未初始化/写败只 WARN 丢行——审计失败不许反过来拦工具执行(需求5 红线)。"""

import logging
from dataclasses import asdict, dataclass

from app.db import crud
from app.db.engine import get_session_factory

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str          # "builtin" | "mcp"
    mcp_server: str | None
    arguments: dict | None
    result_summary: str
    status: str               # 成功/失败/超时/校验拦下/权限拒绝
    error_message: str | None
    retry_count: int
    duration_ms: int | None


async def db_audit_sink(record: AuditRecord) -> None:
    try:
        factory = get_session_factory()
    except RuntimeError:
        logger.warning("audit dropped (engine unavailable) name=%s", record.tool_name)
        return
    try:
        async with factory() as session:
            await crud.insert_tool_audit(session, **asdict(record))
    except Exception:  # noqa: BLE001 —— 写败 WARN,上抛由 executor 再兜一层
        logger.warning("audit write failed name=%s", record.tool_name, exc_info=True)
```

`app/db/crud.py` 追加:

```python
async def insert_tool_audit(session, **fields) -> None:
    session.add(ToolAuditLog(**fields))
    await session.commit()
```

`app/db/models.py` 追加(import 补 `JSON, String, Text, Integer` 缺项 + `from sqlalchemy.dialects.mysql import TINYINT, INTEGER`):

```python
class ToolAuditLog(Base):
    """ch08 工具调用审计(DDL=db/init/08,逐字对齐;无 FK 红线)。"""
    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(BIGINT(unsigned=True), nullable=True, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp", name="tool_source"), nullable=False)
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝", name="tool_audit_status"),
        nullable=False, index=True)
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(TINYINT(unsigned=True), nullable=False, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(INTEGER(unsigned=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default=func.now())
```

- [x] **Step 5: 集成接缝测**(活库真插真读) `tests/test_audit_ch08.py` 追加:

```python
import pytest

@pytest.mark.integration
async def test_seam_insert_roundtrip_live_db():
    from sqlalchemy import select
    from app.db.engine import get_session_factory
    from app.db.models import ToolAuditLog
    async with get_session_factory()() as session:
        await crud.insert_tool_audit(session, **asdict(_record(
            tool_call_id="seam-t1", status="校验拦下")))
        row = (await session.execute(select(ToolAuditLog).where(
            ToolAuditLog.tool_call_id == "seam-t1"))).scalar_one()
        assert row.status == "校验拦下"          # 中文枚举不乱码
        await session.delete(row)
        await session.commit()
```

Run: `pytest tests/test_audit_ch08.py -m integration -v` → PASS(输出落 UTF-8 文件读,GBK 红线)。

- [x] **Step 6: 全量单测 + 提交**

`pytest -q` 预期:仅既有 403±1(新审计测)。提交:

```bash
git add pyproject.toml uv.lock db/init/08_ch08_tool_audit.sql app/db/models.py app/db/crud.py app/tools/audit.py tests/test_audit_ch08.py app/prompts/self_check.py dev-notes/ch08.md docs/superpowers/plans/2026-09-29-ch08-tool-system.md
git commit -m "feat(ch08-t1): 依赖三件入列+tool_audit_logs DDL/ORM/sink+引擎降级测;三恒正名(self_check 提交/retriever/executor 弃)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 2: 注册中心(ToolSpec/内置登记/MCP 现拿/降级撞名)+ query_logistics 下线

**Files:**
- Modify: `app/tools/registry.py`(整文件重构)
- Modify: `app/tools/definitions.py`(删 query_logistics 函数体与 `_CITIES` 等仅其用常量)
- Modify: `app/core/config.py`(+`mcp_logistics_url`/`mcp_aftersale_url` 两键)
- Test: `tests/test_registry_ch08.py`;波及修:`tests/test_tools.py`、`tests/test_executor.py`、`tests/test_naive_loop_ch05.py`、`tests/test_react_node_ch05.py`、`tests/test_tool_chat_service.py`、`tests/test_schemas_ch02.py`、`tests/e2e/test_ch05_acceptance.py`、`evals/run_tool_routing_eval.py`、`app/tools/executor.py`(docstring 提 logistics 处)

**Interfaces:**
- Consumes: T1 无直接依赖(同库)
- Produces: `registry.ToolSpec(tool, permission, source, mcp_server=None)`(frozen dataclass,`.name`/`.description` 为 tool 派生 property);`registry.BUILTIN_SPECS: dict[str, ToolSpec]`(键序=query_order,query_product,query_faq,create_ticket);`registry.get_tools()/get_tool(name)` 签名不变(内置视图,MCP 不掺);`async registry.snapshot_tools(settings, *, client=None) -> dict[str, ToolSpec]`;`registry.make_mcp_client(settings) -> MultiServerMCPClient`

- [x] **Step 1: 写失败测试** `tests/test_registry_ch08.py`

```python
"""ch08 T2:注册中心合并/撞名丢弃/单 server 降级(spec 注册中心节)。"""
import pytest
from langchain_core.tools import tool

from app.core.config import Settings
from app.tools import registry


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"fake {name}"
        self.args = {}


class _BoomClient:
    async def get_tools(self, *, server_name):
        raise ConnectionError("server gone")


class _OneSidedClient:
    """logistics 给两件(其一撞内置),aftersale 给一件。"""
    async def get_tools(self, *, server_name):
        if server_name == "logistics":
            return [_FakeTool("query_waimai"), _FakeTool("query_order")]
        return [_FakeTool("query_warranty")]


def _settings():
    return Settings(openai_base_url="x", openai_api_key="x", model_name="x")


def test_builtin_specs_shape():
    assert list(registry.BUILTIN_SPECS) == ["query_order", "query_product",
                                            "query_faq", "create_ticket"]   # 无 logistics
    assert registry.BUILTIN_SPECS["create_ticket"].permission == "write"
    assert registry.BUILTIN_SPECS["query_faq"].permission == "readonly"
    assert all(s.source == "builtin" for s in registry.BUILTIN_SPECS.values())


async def test_snapshot_merges_mcp_and_drops_name_clash():
    specs = await registry.snapshot_tools(_settings(), client=_OneSidedClient())
    assert "query_waimai" in specs and "query_warranty" in specs
    assert specs["query_waimai"].source == "mcp"
    assert specs["query_waimai"].mcp_server == "logistics"
    assert specs["query_order"].source == "builtin"      # 撞名:内置存活


async def test_snapshot_single_server_failure_degrades(caplog):
    class Half(_OneSidedClient):
        async def get_tools(self, *, server_name):
            if server_name == "aftersale":
                raise ConnectionError("boom")
            return [_FakeTool("query_waimai")]

    specs = await registry.snapshot_tools(_settings(), client=Half())
    assert "query_waimai" in specs and "query_warranty" not in specs
    assert set(specs) >= set(registry.BUILTIN_SPECS)     # 内置面完整
    assert any("mcp discovery" in r.getMessage() for r in caplog.records)


async def test_snapshot_boom_client_all_servers_builtin_only():
    specs = await registry.snapshot_tools(_settings(), client=_BoomClient())
    assert set(specs) == set(registry.BUILTIN_SPECS)
```

Run → FAIL(ToolSpec 不存在)。

- [x] **Step 2: 实现 registry.py 重构**

```python
"""工具注册中心(spec 注册中心节):三件套统一登记,内置启动登记,MCP 每轮现拿。

ToolSpec.permission 是权限唯一权威——外部 MCP 自带声明一概不采信(需求3);
snapshot_tools 每轮调用(P4 无缓存,adapters 实证每次 get_tools 新建会话),
单 server 连不上只 WARN 降级,聊天不断线(T7 store 面全吞原则延伸)。
"""

import logging
from dataclasses import dataclass
from typing import Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.definitions import create_ticket, query_faq, query_order, query_product

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    permission: Literal["readonly", "write"]   # 只认我方登记(P5)
    source: Literal["builtin", "mcp"]
    mcp_server: str | None = None

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def description(self) -> str:
        return self.tool.description or ""


BUILTIN_SPECS: dict[str, ToolSpec] = {
    "query_order": ToolSpec(query_order, "readonly", "builtin"),
    "query_product": ToolSpec(query_product, "readonly", "builtin"),
    "query_faq": ToolSpec(query_faq, "readonly", "builtin"),
    # 全系统唯一 write(需求3);执行闸在 executor 权限闸,bind 与否在 T7 切换
    "create_ticket": ToolSpec(create_ticket, "write", "builtin"),
}

_MCP_SERVERS = ("logistics", "aftersale")


def get_tools() -> list[BaseTool]:
    """内置视图(bind/legacy 面兼容签名;主力图链请用 snapshot_tools)。"""
    return [s.tool for s in BUILTIN_SPECS.values()]


def get_tool(name: str) -> BaseTool | None:
    spec = BUILTIN_SPECS.get(name)
    return spec.tool if spec else None


def make_mcp_client(settings) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "http", "url": settings.mcp_logistics_url},
        "aftersale": {"transport": "http", "url": settings.mcp_aftersale_url},
    })


async def snapshot_tools(settings, *, client=None) -> dict[str, ToolSpec]:
    """当轮工具面:内置 ∪ 各 MCP server 现拿(server_name 逐连,失败只丢该面)。"""
    specs = dict(BUILTIN_SPECS)
    client = client or make_mcp_client(settings)
    for sname in _MCP_SERVERS:
        try:
            tools = await client.get_tools(server_name=sname)
        except Exception:  # noqa: BLE001 —— 发现面降级不断线
            logger.warning("mcp discovery failed server=%s; builtin view kept",
                           sname, exc_info=True)
            continue
        for t in tools:
            if t.name in specs:          # 撞名:内置权威,外部件丢弃(权限面不可污染)
                logger.warning("mcp tool %s (server=%s) clashes with builtin; dropped",
                               t.name, sname)
                continue
            specs[t.name] = ToolSpec(t, "readonly", "mcp", sname)   # P5:MCP 恒只读
    return specs
```

`app/core/config.py` ch02 段后追加:

```python
    # --- ch08: MCP 接入(P3 定死本机端口;demo 值=默认,零配置可用) ---
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"
    mcp_aftersale_url: str = "http://127.0.0.1:8102/mcp"
```

`definitions.py`:query_logistics @tool 整函数删,`_CARRIERS/_CITIES` 若无他用一并删(现仅它用)。

- [x] **Step 3: 波及修**——跑 `pytest -q`,按失败清单逐处改:断言清单去 logistics、fake 表去 `query_logistics` 项、`e2e test_ch05_acceptance` 与 `test_react_node_ch05.py:90` 用 `query_order` 顶替该调用名(断言语义「工具链跑通」不变,规4 断言随任务改);`test_tools.py` 的 `TOOL_REGISTRY` 断言迁 `BUILTIN_SPECS` 键序;executor docstring 的枚举分支 `elif name in (... "query_logistics")` 删 logistics 项。
- [x] **Step 4: 全绿 + 提交**(文件=上表全部 + dev-notes + 计划勾选)

```bash
git commit -m "feat(ch08-t2): ToolSpec 注册中心(MCP 每轮现拿/撞名丢/降级吞)+内置 query_logistics 下线,MCP url 两键入 settings

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 3: executor 签名迁移 + 校验闸 + 未登记拒绝审计

**Files:**
- Modify: `app/tools/executor.py`(核心重构第一步)
- Modify: `app/agents/react.py`(执行循环改传 spec;`_execute` 幻觉拒绝处发审计)
- Modify: `app/services/tool_chat_service.py`、`app/services/naive_agent_loop.py`(legacy:内置视图 lookup,未命中走既有错误形状)
- Test: `tests/test_executor.py` 适配重写 + 新增 `tests/test_validation_ch08.py`

**Interfaces:**
- Consumes: T2 `BUILTIN_SPECS/ToolSpec`;T1 `AuditRecord/db_audit_sink`
- Produces: `execute_tool(spec: ToolSpec, args: dict, tool_call_id: str, ctx: ToolContext) -> ToolOutcome`(旧 name-lookup 签名作废);`ToolContext(+audit_sink=None 键,None=用 db_audit_sink 默认)`;`executor.validate_args(spec, args) -> str | None`(中文错误文本或 None);`executor.audit_denied(ctx, spec_name, tool_call_id, args, reason)`(幻觉面共用)

- [x] **Step 1: 写失败测试** `tests/test_validation_ch08.py`

```python
"""ch08 T3:JSON Schema 校验闸(spec 校验闸节)——拦下不抛异常,错误回灌+审计。"""
from langchain_core.tools import tool

from app.tools.audit import AuditRecord
from app.tools.executor import ToolContext, execute_tool
from app.tools.registry import ToolSpec

REC = []


async def _sink(rec: AuditRecord):
    REC.append(rec)


@tool
async def probe(order_id: str, qty: int = 1) -> dict:
    """查一件东西。order_id: 订单号。"""
    return {"ok": True, "order_id": order_id, "qty": qty}


def _spec():
    return ToolSpec(probe, "readonly", "builtin")


async def test_missing_required_blocks_with_chinese_reason():
    out = await execute_tool(_spec(), {"qty": 2}, "c1", ToolContext(audit_sink=_sink))
    assert not out.ok and "参数不合法" in out.result["error"]
    assert "order_id" in out.result["error"]           # 指名缺的字段
    assert REC and REC[-1].status == "校验拦下"


async def test_wrong_type_blocked():
    out = await execute_tool(_spec(), {"order_id": 123}, "c2",
                             ToolContext(audit_sink=_sink))
    assert not out.ok and "参数不合法" in out.result["error"]
    assert REC[-1].status == "校验拦下"


async def test_valid_args_pass_through():
    out = await execute_tool(_spec(), {"order_id": "1001"}, "c3",
                             ToolContext(audit_sink=_sink))
    assert out.ok and out.result["order_id"] == "1001"


async def test_literal_enum_out_of_range_blocked():
    out = await execute_tool(
        ToolSpec(_ticket_probe(), "readonly", "builtin"),
        {"description": "x", "ticket_type": "爆炸"}, "c4", ToolContext(audit_sink=_sink))
    assert not out.ok and REC[-1].status == "校验拦下"
```

(`_ticket_probe()` 返回内置 `app.tools.definitions.create_ticket`,Literal 枚举现成。)

Run → FAIL(execute_tool 现签名无 spec)。

- [x] **Step 2: executor 重构(第一步)**

`ToolContext` 加 `audit_sink=None`、`ticket_confirmed=False`(T4 用,一次进齐)。核心改动:

```python
from jsonschema import Draft202012Validator

from app.tools.audit import AuditRecord, db_audit_sink

_SCHEMA_CACHE: dict[int, dict] = {}


def tool_json_schema(tool) -> dict:
    sc = getattr(tool, "tool_call_schema", None)
    if isinstance(sc, type):
        try:
            return sc.model_json_schema()
        except AttributeError:
            pass
    return {"type": "object", "properties": tool.args or {}}   # 兜底拼装


def validate_args(spec, args: dict) -> str | None:
    key = id(spec.tool)
    schema = _SCHEMA_CACHE.setdefault(key, tool_json_schema(spec.tool))
    errs = sorted(Draft202012Validator(schema).iter_errors(args),
                  key=lambda e: list(e.path))
    if not errs:
        return None
    e = errs[0]
    loc = "/".join(str(p) for p in e.path) or "(root)"
    return f"参数不合法: {loc} {e.message}"[:512]


async def _emit(ctx, record: AuditRecord) -> None:
    sink = ctx.audit_sink if ctx.audit_sink is not None else db_audit_sink
    try:
        await sink(record)
    except Exception:  # noqa: BLE001 —— 审计永不反拦执行(需求5)
        logger.warning("audit emit failed name=%s", record.tool_name, exc_info=True)


def _record(ctx, spec, tool_call_id, args, status, summary, error=None,
            retries=0, duration_ms=0) -> AuditRecord:
    return AuditRecord(conversation_id=ctx.conversation_id, tool_call_id=tool_call_id,
                       tool_name=spec.name, tool_source=spec.source,
                       mcp_server=spec.mcp_server, arguments=args,
                       result_summary=summary[:500], status=status,
                       error_message=error[:512] if error else None,
                       retry_count=retries, duration_ms=duration_ms)


async def execute_tool(spec, args, tool_call_id, ctx):
    err = validate_args(spec, args)
    if err:
        await _emit(ctx, _record(ctx, spec, tool_call_id, args, "校验拦下", err, err))
        return ToolOutcome(spec.name, tool_call_id, False, {"error": err},
                           make_summary(spec.name, {"error": err}))
    # ↓ 原函数体:get_tool(name) 行删除,tool 改 spec.tool;其余暂保持(重试分诊 T5)
```

原「未注册的工具」分支整段删除(lookup 职责移到调用方)。

- [x] **Step 3: 调用方适配**

react.py 执行点(T7 才接 MCP 快照,本任务先用内置视图):

```python
from app.tools.audit import AuditRecord
from app.tools.registry import BUILTIN_SPECS
...
spec = BUILTIN_SPECS.get(tc["name"])
if spec is None:
    # 幻觉调用=未授权:拒绝 + 审计「权限拒绝」(Review Focus 6)
    await audit_denied(ctx, tc["name"], tc["id"], tc.get("args") or {})
    outcome = SimpleNamespace(...)   # 沿用现拒绝形状,summary 前缀「未注册的工具:」
else:
    outcome = await execute_tool(spec, tc.get("args") or {}, tc["id"], ctx)
```

`audit_denied(ctx, name, tool_call_id, args)` 落在 executor(构造合成 spec 字段:tool_source="builtin"、status="权限拒绝",直发 `_emit`)。tool_chat_service/naive_agent_loop:先 `spec = BUILTIN_SPECS.get(name)` → None 则原错误形状,否则新签名调用。

- [x] **Step 4: `tests/test_executor.py` 全数迁新签名**(FakeTool 包 `ToolSpec(FakeTool(...), "readonly", "builtin")` 直传;`_install_fake_tools` monkeypatch 模式作废),跑绿全量提交。

```bash
git commit -m "feat(ch08-t3): execute_tool 收 ToolSpec+jsonschema 校验闸(拦下回灌不抛异常)+幻觉未登记拒绝审计

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 4: 权限闸(唯一 write=create_ticket;凭证在 ToolContext,模型不可伪造)

**Files:**
- Modify: `app/tools/executor.py`(校验闸后插权限闸)
- Test: `tests/test_permission_ch08.py`

**Interfaces:**
- Consumes: T3 管线;`ToolContext.ticket_confirmed`(T3 已进形)
- Produces: `ToolOutcome(+awaiting_confirmation: bool = False)`;写拒绝文本 `"写操作需客户在预览卡片确认后才执行"`(T7 react 捕获 preview 的触发信号)

- [x] **Step 1: 写失败测试**

```python
"""ch08 T4:权限闸(spec 权限闸节)。MCP 声明不可信:P5 恒只读由 T2 构造保证,
此处钉 write 无凭证拒/有凭证放行/拒绝不落审计(终局走确认流)。"""
REC = []

@tool
async def fake_write(description: str) -> dict:
    """写。"""
    return {"ticket_no": "T1"}

def _wspec():
    return ToolSpec(fake_write, "write", "builtin")

async def test_write_without_credential_refused():
    out = await execute_tool(_wspec(), {"description": "x"}, "c1",
                             ToolContext(audit_sink=_sink))
    assert not out.ok and out.awaiting_confirmation
    assert "预览卡片确认" in out.result["error"]
    assert not REC                       # 等待确认=中间态,审计按终局(spec 审计节)

async def test_write_with_credential_executes():
    out = await execute_tool(_wspec(), {"description": "x"}, "c2",
                             ToolContext(ticket_confirmed=True, audit_sink=_sink))
    assert out.ok and out.result["ticket_no"] == "T1"
    assert REC[-1].status == "成功"      # T5 补全成功行;T4 先钉「放行」本身

async def test_invalid_write_hits_validation_not_permission():
    # 闸序=校验→权限:必填缺失先回「参数不合法」(验收4 追问路径的机制地基)
    out = await execute_tool(_wspec(), {}, "c3", ToolContext(audit_sink=_sink))
    assert "参数不合法" in out.result["error"] and not out.awaiting_confirmation
```

Run → FAIL。

- [x] **Step 2: 实现**——校验闸通过后:

```python
    if spec.permission == "write" and not ctx.ticket_confirmed:
        err = "写操作需客户在预览卡片确认后才执行"
        return ToolOutcome(spec.name, tool_call_id, False, {"error": err},
                           "等待客户确认", awaiting_confirmation=True)
```

`ToolOutcome` 加 `awaiting_confirmation: bool = False`。T4 测「成功」断言以现状最小放行(审计全量在 T5)。

- [x] **Step 3: 全绿+提交** `git commit -m "feat(ch08-t4): 权限闸——write 无凭证拒/凭证服务端持有模型不可伪造,闸序校验→权限钉测 …"`(尾注同规)

### Task 5: 超时/重试分诊 + 结果格式化 + 全路径审计 + 默认超时 10s

**Files:**
- Modify: `app/tools/executor.py`(执行循环替换)
- Modify: `app/core/config.py`(`tool_timeout_seconds` 默认 5.0→10.0)
- Test: `tests/test_retry_ch08.py`、`tests/test_format_ch08.py`;`tests/test_executor.py` 超时断言校准

**Interfaces:**
- Consumes: T3 `_emit/_record`;T4 `awaiting_confirmation`
- Produces: `TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)`;`executor.STATUS_LABELS: dict[str, str]`;`format_result(spec, result) -> dict`;审计终局全量:成功/失败/超时(retry_count/duration_ms 实录)

- [x] **Step 1: 写失败测试**(节选,三文件合成一个行为一步)

```python
"""ch08 T5:重试只给暂时性故障;write 恒单试;三类分诊;MCP 投影+枚举翻话(spec 执行引擎节)。"""
import asyncio

class SlowTool:
    name = "slow"; description = "慢"; args = {}
    def __init__(self, sleep): self.sleep = sleep; self.calls = 0
    async def ainvoke(self, args, config=None, **kw):
        self.calls += 1
        await asyncio.sleep(self.sleep)
        return {"done": True}

class BoomTool:
    name = "boom"; description = "炸"; args = {}
    def __init__(self, exc): self.exc = exc; self.calls = 0
    async def ainvoke(self, args, config=None, **kw):
        self.calls += 1
        raise self.exc

async def test_transient_retried_then_success():
    t = BoomTool(ConnectionError("net jitter")); t.ainvoke
    specs=[BoomTool(ConnectionError("jitter"))]
    class Flaky:
        name="flaky"; description=""; args={}
        def __init__(self): self.calls=0
        async def ainvoke(self, a, config=None, **k):
            self.calls+=1
            if self.calls==1: raise ConnectionError("jitter")
            return {"ok":1}
    out = await execute_tool(ToolSpec(Flaky(), "readonly", "builtin"), {}, "c1",
                             ToolContext(timeout_seconds=1, max_retries=1, audit_sink=_sink))
    assert out.ok and REC[-1].status == "成功" and REC[-1].retry_count == 1

async def test_timeout_status_audited_with_duration():
    out = await execute_tool(ToolSpec(SlowTool(0.05), "readonly", "builtin"), {}, "c2",
                             ToolContext(timeout_seconds=0.01, max_retries=1, audit_sink=_sink))
    assert not out.ok and REC[-1].status == "超时" and REC[-1].retry_count == 1
    assert REC[-1].duration_ms is not None and REC[-1].duration_ms >= 0

async def test_business_error_not_retried():
    t = BoomTool(ValueError("业务硬错"))
    await execute_tool(ToolSpec(t, "readonly", "mcp", "logistics"), {}, "c3",
                       ToolContext(timeout_seconds=1, max_retries=3, audit_sink=_sink))
    assert t.calls == 1 and REC[-1].status == "失败" and REC[-1].retry_count == 0

async def test_write_never_auto_retried():
    t = BoomTool(ConnectionError("抖动也不试"))
    out = await execute_tool(ToolSpec(t, "write", "builtin"), {"x": 1}, "c4",
                             ToolContext(ticket_confirmed=True, max_retries=3, audit_sink=_sink))
    assert t.calls == 1 and "执行失败" in out.result["error"]   # 需求4:重复执行比失败更糟

def test_format_projection_and_labels():
    spec = ToolSpec(_mcp_fake(), "readonly", "mcp", "logistics")
    raw = {"carrier": "中通", "current_status": "TRANSPORT", "_internal": 1,
           "nodes": [{"n": i} for i in range(8)]}
    out = format_result(spec, raw)
    assert out["current_status"] == "运输中" and "_internal" not in out
    assert len(out["nodes"]) == 5                      # 只留最近 5 条
    builtin = ToolSpec(_builtin_fake(), "readonly", "builtin")
    assert format_result(builtin, raw) is raw          # 内置形状不动(现状兼容红线)
```

(`STATUS_LABELS = {"COLLECTED": "已揽收", "TRANSPORT": "运输中", "DELIVERING": "派送中", "SIGNED": "已签收", "APPROVED": "审核通过", "RECEIVING": "收到退货中", "REFUNDING": "退款处理中", "CLOSED": "已关闭"}` 在测试以 `executor.STATUS_LABELS[...]` 引用,不复制字面。)

- [x] **Step 2: 实现**——执行循环替换为:

```python
TRANSIENT_ERRORS = (TimeoutError, ConnectionError, OSError)

def format_result(spec, result):
    if spec.source != "mcp" or not isinstance(result, dict):
        return result
    out = {k: v for k, v in result.items() if not str(k).startswith("_")}
    if isinstance(out.get("nodes"), list):
        out["nodes"] = out["nodes"][-5:]
    for key in ("current_status", "status"):
        if out.get(key) in STATUS_LABELS:
            out[key] = STATUS_LABELS[out[key]]
    return out

async def execute_tool(spec, args, tool_call_id, ctx):
    err = validate_args(spec, args)
    if err: ...  # T3 校验闸不变
    if spec.permission == "write" and not ctx.ticket_confirmed: ...  # T4 权限闸不变
    t0 = time.monotonic()
    config = {"configurable": {"conversation_id": ctx.conversation_id}}
    attempts = 1 if spec.permission == "write" else max(1, ctx.max_retries + 1)
    status, retries_used, last_err, result = "失败", 0, "未知错误", None
    for i in range(attempts):
        retries_used = i
        try:
            result = await asyncio.wait_for(
                spec.tool.ainvoke(args, config=config), timeout=ctx.timeout_seconds)
            status = "成功"
            break
        except TimeoutError:
            status, last_err = "超时", f"执行超时(>{ctx.timeout_seconds}s)"
        except TRANSIENT_ERRORS as exc:
            status, last_err = "失败", f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 —— 非暂时性:真故障,不重试
            status, last_err = "失败", f"{type(exc).__name__}: {exc}"
            break
    duration_ms = int((time.monotonic() - t0) * 1000)
    if status == "成功":
        if isinstance(result, str):
            try: result = json.loads(result)          # Review Focus 3:防御分支保活
            except json.JSONDecodeError: result = {"text": result}
        result = format_result(spec, result)
        citations = (build_citations(result.get("hits", [])) or None
                     if spec.name == "query_faq" and isinstance(result, dict) else None)
        await _emit(ctx, _record(ctx, spec, tool_call_id, args, "成功",
                                 make_summary(spec.name, result),
                                 retries=retries_used, duration_ms=duration_ms))
        return ToolOutcome(spec.name, tool_call_id, True, result,
                           make_summary(spec.name, result), citations=citations)
    if status == "超时" and spec.permission == "write":
        last_err += "(写操作超时未自动重试,请人工核实是否已执行)"
    err_result = {"error": f"工具执行失败: {last_err}"}
    await _emit(ctx, _record(ctx, spec, tool_call_id, args, status,
                             make_summary(spec.name, err_result), last_err,
                             retries=retries_used, duration_ms=duration_ms))
    return ToolOutcome(spec.name, tool_call_id, False, err_result,
                       make_summary(spec.name, err_result))
```

settings 默认 10.0;`tests/test_executor.py` 原「超时重试」断言按新分诊校准(规4)。

- [x] **Step 3: 全绿+提交** `"feat(ch08-t5): 重试白名单+write 恒单试+三类分诊全量审计+MCP 投影枚举翻话,超时默认 10s …"`

### Task 6: 双 MCP Server + 集成链路

**Files:**
- Create: `mcp_servers/__init__.py`(空,包标记)、`mcp_servers/logistics_server.py`、`mcp_servers/aftersale_server.py`
- Test: `tests/test_mcp_servers_ch08_integration.py`(全部 @integration)

**Interfaces:**
- Consumes: T2 `make_mcp_client/snapshot_tools`;T3/T5 `execute_tool`
- Produces: 工具契约(名字/schema/返回形态)= spec「MCP Servers」节三契约;`_STATUS_LABELS` 对应新码(已入 T5 STATUS_LABELS)

- [x] **Step 0: 定名验版(TDD 前置的事实核对,uv 解析版本为准)**

```bash
uv run python -c "import mcp, mcp.server.fastmcp as f; print(f.__name__, f.FastMCP)"
```

若 ImportError → 试 `from mcp.server.mcpserver import MCPServer`;哪个通哪个名写进两 server 文件(Context7 两形制,差异仅 import 行与构造名),记 ledger/dev-notes。

- [x] **Step 1: 失败测**

```python
"""ch08 T6(integration):真 Streamable HTTP 链路——发现合并/经 MCP 查/拒连降级。
server 以子进程起,P9 手工验收 1/3 用同两条命令(README 收录)。"""
import subprocess, sys, time
import pytest

pytestmark = pytest.mark.integration

def _wait_port(host, port, timeout=15):
    import socket
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False

@pytest.fixture
async def servers():
    procs = [subprocess.Popen([sys.executable, f"mcp_servers/{n}_server.py"])
             for n in ("logistics", "aftersale")]
    try:
        assert _wait_port("127.0.0.1", 8101) and _wait_port("127.0.0.1", 8102)
        yield
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)

async def test_mcp_discovery_call_chain(servers):
    st = Settings(openai_base_url="x", openai_api_key="x", model_name="x")
    specs = await snapshot_tools(st)
    assert specs["query_logistics"].source == "mcp"
    assert {"query_warranty", "query_return_progress"} <= set(specs)
    out = await execute_tool(specs["query_logistics"], {"order_id": "1001"}, "c1",
                             ToolContext(timeout_seconds=5))
    assert out.ok and out.result["current_status"] in executor.STATUS_LABELS.values() or \
        out.result["current_status"] in ("运输中", "派送中", "已签收", "已揽收")

async def test_server_down_snapshot_degrades(monkeypatch):
    st = Settings(..., mcp_logistics_url="http://127.0.0.1:9599/mcp",
                  mcp_aftersale_url="http://127.0.0.1:9598/mcp")
    specs = await snapshot_tools(st)          # 无 client 现构,端口无人听
    assert set(specs) == set(BUILTIN_SPECS)   # 不抛,内置面完整
```

(成功断言里 server 回的是原码、executor 翻人话,二形态都认——格式化面 T5 已钉,这里只验链路。)

- [x] **Step 2: logistics_server.py**

```python
"""ch08 物流 MCP Server(需求6):独立进程,8101 /mcp,Streamable HTTP。
mock 数据照 ch02 做法按订单号播种(演示可复现),不接真实系统、不建表。"""
import random
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP   # T6 Step 0 定名(见 ledger)

_CARRIERS = ["中通快递", "圆通速递", "韵达快递", "顺丰速运"]
_NODES = ["杭州转运中心", "苏州分拨中心", "南京集散中心", "上海虹桥网点", "北京大兴网点"]
_STATUS_FLOW = ["COLLECTED", "TRANSPORT", "TRANSPORT", "DELIVERING", "SIGNED"]

mcp = FastMCP("logistics", host="127.0.0.1", port=8101)


@mcp.tool()
async def query_logistics(order_id: str) -> dict:
    """按订单号查询物流轨迹:承运商、运单号、当前状态码与最近节点。

    order_id: 订单号,如 1001。
    """
    rnd = random.Random(f"logistics-{order_id}")
    n = rnd.randint(2, 4)
    now = datetime.now()
    picked = rnd.sample(_STATUS_FLOW, n)
    nodes = [{"time": (now - timedelta(hours=8 * (n - i))).strftime("%Y-%m-%d %H:%M"),
              "status": picked[i],
              "desc": {"COLLECTED": "快件已揽收", "TRANSPORT": "运输中",
                       "DELIVERING": "派送中", "SIGNED": "已签收"}[picked[i]],
              } for i in range(n)]
    return {"carrier": rnd.choice(_CARRIERS),
            "tracking_no": f"ZTO{rnd.randint(10**11, 10**12 - 1)}",
            "current_status": picked[-1],
            "nodes": nodes}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- [x] **Step 3: aftersale_server.py**(同形制,port=8102,name="aftersale")

```python
@mcp.tool()
async def query_warranty(order_id: str) -> dict:
    """按订单号查询是否在保修期:在保布尔、到期日与判定依据。order_id: 订单号。"""
    rnd = random.Random(f"warranty-{order_id}")
    days = rnd.randint(-60, 300)
    return {"order_id": order_id, "in_warranty": days > 0,
            "expire_date": (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d"),
            "basis": "自签收日起 365 天"}


@mcp.tool()
async def query_return_progress(order_id: str) -> dict:
    """按订单号查询退货单进度:无退货单返回 found=false。order_id: 订单号。"""
    if order_id.endswith("9"):                  # 演示可解释:尾号 9 暂无在途退货单
        return {"found": False, "order_id": order_id}
    rnd = random.Random(f"return-{order_id}")
    flow = ["APPROVED", "RECEIVING", "REFUNDING", "CLOSED"]
    i = rnd.randint(0, 3)
    return {"found": True, "order_id": order_id, "return_no": f"R{order_id}{i}",
            "current_status": flow[i],
            "nodes": [{"status": s, "desc": {"APPROVED": "审核通过",
                     "RECEIVING": "收到退货中", "REFUNDING": "退款处理中",
                     "CLOSED": "已关闭"}[s]} for s in flow[:i + 1]]}
```

- [x] **Step 4: 集成跑** `pytest tests/test_mcp_servers_ch08_integration.py -m integration -v`(输出落文件读)。
- [x] **Step 5: 提交** `"feat(ch08-t6): 物流/售后双 MCP Server(Streamable HTTP 8101/8102,mock 播种)+真链路集成测 …"`

### Task 7: 确认流图侧(preview 捕获 + ticket_confirm 节点 + interrupt 帧)

**Files:**
- Modify: `app/workflows/state.py`(+ticket_preview)、`app/workflows/nodes.py`(agent_node 传快照/捕获、coref reset、新节点)、`app/agents/react.py`(bind 全集含 create_ticket、删 :109-117 硬闸、发 ticket_request 内部事件)、`app/workflows/graph.py`(条件边+resume_value+interrupt 检测+隐式 cancel drain)、`app/tools/definitions.py`(create_ticket docstring 改 ch08 语义)
- Test: `tests/test_confirm_flow_ch08.py`

**Interfaces:**
- Consumes: T5 executor 全链;`ToolContext.ticket_confirmed`
- Produces: state 键 `ticket_preview: dict`;react 事件 `("ticket_request", {"tool_call_id", "args"})`(仅 agent_node 消费,不出 writer 白名单);图节点 `ticket_confirm`;`stream_graph_turn(..., resume_value=None)` 新 kwarg;帧 `("ticket_preview", {tool_call_id, ticket_type, description, conversation_id})`

- [ ] **Step 1: 写失败测试**

```python
"""ch08 T7:建单确认流图侧(spec 确认流节)。"""
# 脚本模型第一轮调 create_ticket(必填齐),第二轮收尾文本
turns = [_turn(text="我帮您准备工单。",
                tool_calls=[("create_ticket",
                             {"description": "猫粮到货破损", "ticket_type": "售后"}, "t1")]),
         _turn(text="预览卡片已发出,请确认。")]
frames = [f async for f in stream_graph_turn(
    [SimpleNamespace(content="帮我建个工单,猫粮到货破损了")], _settings(), model,
    conversation_id=7, persister=None, ctx_store=None)]
assert ("ticket_preview", ...) in kinds          # 见下逐条断言

async def test_preview_frame_and_pause():
    frames = ...
    kinds = [k for k, _ in frames]
    assert "ticket_preview" in kinds
    f = dict(frames[kinds.index("ticket_preview")])[1]  # payload
    assert f["ticket_type"] == "售后" and f["description"] == "猫粮到货破损"
    assert not any(k == "done" for k in kinds[kinds.index("ticket_preview"):])  # 暂停语义:preview 后即末帧

async def test_resume_confirm_lands_ticket(reset_graph_state):
    # 同线程续:stream_graph_turn(..., resume_value="confirm") → token 含工单号
    frames = [f async for f in stream_graph_turn(None, st, model, conversation_id=7,
                                                 resume_value="confirm")]
    text = "".join(p for k, p in frames if k == "token")
    assert "T" in text and "已为您创建工单" in text

async def test_resume_cancel_audits_permission_denied(reset_graph_state, recorder):
    ... resume_value="cancel" ...
    assert "取消" in text and recorder[-1].status == "权限拒绝"
    # tickets 表不增行:count 断言(以 FakeStore/monkeypatch crud 计次)

async def test_implicit_cancel_on_new_message(reset_graph_state, recorder):
    """Review Focus 1:卡片弹出后直接发新消息 → 不 500、旧单自动按取消收账、新轮正常答。"""
    await _reach_preview(...)                    # helper:跑第一轮到 preview
    frames = [f async for f in stream_graph_turn(
        [SimpleNamespace(content="算了,先问下发货地")], st, model2,
        conversation_id=7, persister=None, ctx_store=None)]
    assert ("token", ...) 正常出且无 error 帧
    assert recorder[-1].status == "权限拒绝"      # 隐式 cancel 落账

async def test_interrupt_replay_side_effect_free(reset_graph_state, recorder):
    """resume 重放:ticket_confirm 从头跑,interrupt 前必须零副作用——
    断 confirm 路径审计恰一条、crud 建单恰一次(计数 monkeypatch)。"""
```

- [ ] **Step 2: react.py 改**

bind 行(T7 起接快照,agent_node 传入):

```python
async def react_agent_stream(state, settings, model, *, persister=None, specs=None):
    ...
    specs = specs if specs is not None else BUILTIN_SPECS
    bound = model.bind_tools([s.tool for s in specs.values()])   # 含 create_ticket(需求7)
```

同时改 `app/tools/definitions.py` create_ticket docstring:ch05「仅当用户明确要求转人工…使用」旧口径 → ch08 语义(「客户明确要求建工单时调用;description 必填、须来自客户原话不得编造;调用后由客户在卡片确认才真正执行」——验收4 追问行为的地基,T10 评估集验效果)。

执行循环:`spec = specs.get(tc["name"])` 分支保留 T3 幻觉拒绝;删除 :109-117 create_ticket 硬闸整块(ch05 D2 红线由本章取代,注释留史)。outcome 处理处:

```python
            if outcome.awaiting_confirmation:
                yield ("ticket_request", {"tool_call_id": tc["id"],
                                          "args": tc.get("args") or {}})
```

(`tool_result` 帧照发——ok=False+summary「等待客户确认」,模型见错误自收敛收尾。)

- [ ] **Step 3: nodes.py**:`make_agent_node` 内 `specs = await snapshot_tools(settings)`(per 轮一次)传 react;捕获 `ticket_request`(不 writer 外发)→ `upd["ticket_preview"] = {"tool_call_id":..., "ticket_type": args.get("ticket_type", "咨询"), "description": args.get("description", "")}`;coref reset dict(:131)加 `"ticket_preview": {}`。新节点:

```python
async def ticket_confirm_node(state, config):
    """ch08 确认节点(spec 确认流节):interrupt 前置零副作用——
    langgraph 恢复语义=本节点从头重放,interrupt() 是第一行动才安全(P6)。"""
    preview = state["ticket_preview"]
    decision = interrupt(dict(preview))
    conf = config.get("configurable") or {}
    cid = conf.get("conversation_id")
    log = _note(state, "ticket_confirm")
    if decision == "confirm":
        ctx = ToolContext(conversation_id=cid, timeout_seconds=settings_default_timeout(),
                          ticket_confirmed=True)
        outcome = await execute_tool(BUILTIN_SPECS["create_ticket"],
                                     {"description": preview["description"],
                                      "ticket_type": preview["ticket_type"]},
                                     preview.get("tool_call_id") or "confirm", ctx)
        ans = (f"已为您创建工单 {outcome.result['ticket_no']},处理进度会另行通知。"
               if outcome.ok else
               f"工单创建失败:{outcome.summary},请稍后再试或使用页面下方「建工单」按钮。")
    else:
        await _emit_none_sink_cancel(cid, preview)   # 直发 AuditRecord(status="权限拒绝",
        #   error_message="客户在预览卡片取消", tool_name="create_ticket",
        #   arguments={...preview...}, result_summary="已取消,未执行")
        ans = "好的,已取消本次建单。"
    return {"answer_text": ans, "log": log,
            "messages": [AIMessage(content=ans)], "ticket_preview": {}}
```

(`settings_default_timeout()`:节点拿不到 settings——从 `config.configurable.settings_ref` 取,agent_node 注入 settings 进 cfg;或模块级 get_settings()。择:节点级 `get_settings()`,ledger 记 Ruling。)

- [ ] **Step 4: graph.py**:

```python
g.add_node("ticket_confirm", N.ticket_confirm_node)
g.add_conditional_edges("agent", _after_agent,
                        {"confirm": "ticket_confirm", "done": "logging"})
g.add_edge("ticket_confirm", "logging")
```

`_after_agent(state) = "confirm" if state.get("ticket_preview") else "done"`。

`stream_graph_turn`:新 kwarg `resume_value`;resume 分支跳 refill/user 行,input=`Command(resume=resume_value)`;普通分支 astream 前:

```python
    snap = await graph.aget_state(cfg)
    if getattr(snap, "interrupts", ()):            # 隐式 cancel(Review Focus 1)
        await graph.ainvoke(Command(resume="cancel"), config=cfg)
```

流尽后:

```python
    snap = await graph.aget_state(cfg)
    pend = getattr(snap, "interrupts", ()) or ()
    if pend:
        yield ("ticket_preview", {**dict(pend[0].value),
                                  "conversation_id": conversation_id})
        return                                      # 暂停轮:无 suggestions/fix 语义
```

- [ ] **Step 5: 全绿+提交** `"feat(ch08-t7): 建单确认流图侧——preview 捕获/ticket_confirm(interrupt 重放零副作用)/隐式 cancel drain/ticket_preview 帧;ch05 D2 硬闸退役 …"`

### Task 8: POST /api/tickets/confirm(resume 端点 + 409 + 帧复用)

**Files:**
- Modify: `app/schemas/ticket.py`(+TicketConfirmRequest)、`app/schemas/chat.py`(+TicketPreviewEvent)、`app/api/routes.py`(新端点 + SSE 帧映射抽 helper 两端复用)
- Test: `tests/test_confirm_endpoint_ch08.py`

**Interfaces:**
- Consumes: T7 `stream_graph_turn(resume_value=…)`、`ticket_preview` 帧
- Produces: `POST /api/tickets/confirm` body `{conversation_id:int, decision:"confirm"|"cancel"}` → SSE 流;409 无 pending;SSE event 名 `ticket_preview`,data=TicketPreviewEvent

- [ ] **Step 1: 写失败测试**

```python
async def test_confirm_sse_streams_ticket_no(client_with_reach_preview):
    # fixture:FakeScript 模型跑第一轮至 preview 帧(同 T7 helper,同进程 checkpointer)
    resp = client.post("/api/tickets/confirm",
                      json={"conversation_id": 7, "decision": "confirm"})
    assert resp.status_code == 200 and "text/event-stream" in resp.headers["content-type"]
    body = "".join(resp.iter_lines(decode_unicode=False))...  # SSE 文本含 token 帧工单号

async def test_cancel_endpoint(client_with_reach_preview, recorder):
    ... decision "cancel" ...; assert recorder[-1].status == "权限拒绝"

async def test_no_pending_interrupt_is_409(client):
    resp = client.post("/api/tickets/confirm", json={"conversation_id": 999,
                                                     "decision": "confirm"})
    assert resp.status_code == 409

async def test_unknown_decision_422(client_with_reach_preview):
    resp = client.post(..., json={"conversation_id": 7, "decision": "delete"})
    assert resp.status_code == 422      # Review Focus 2:线程 interrupt 未被消耗
    # 复证:随后 decision=confirm 仍能成
```

- [ ] **Step 2: 实现**

routes.py:帧→SSE 映射抽 `_sse_event(kind, payload)`(现 chat_stream elif 链搬入,chat_stream 改调;新 `elif kind == "ticket_preview": yield ServerSentEvent(data=TicketPreviewEvent(**payload).model_dump(), event="ticket_preview")`)。新端点:

```python
@router.post("/api/tickets/confirm", response_class=EventSourceResponse)
async def ticket_confirm(req: TicketConfirmRequest,
                        settings=Depends(dep_settings), model=Depends(dep_chat_model),
                        session=Depends(dep_db_session)):
    """ch08 需求7:预览卡片回传 → Command(resume) 同线程续跑,SSE 续播答复。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    graph = build_graph(settings, model)
    cfg = {"configurable": {"thread_id": f"conv-{req.conversation_id}",
                            "conversation_id": req.conversation_id, "persister": None,
                            "ctx_store": None}}
    snap = await graph.aget_state(cfg)
    if not (getattr(snap, "interrupts", ()) or ()):
        raise HTTPException(status_code=409, detail="确认已过期,请重新发起建单")
    persister = DBChatPersister(session, req.conversation_id)
    try:
        async for kind, payload in stream_graph_turn(
                None, settings, model, conversation_id=req.conversation_id,
                persister=persister, resume_value=req.decision):
            yield _sse_event(kind, payload)
        yield ServerSentEvent(raw_data="[DONE]", event="done")
    except Exception as exc:  # noqa: BLE001 —— SSE 惯例同 chat_stream
        logger.exception("confirm resume failed")
        yield ServerSentEvent(data={"detail": str(exc)}, event="error")
```

(stream_graph_turn resume 分支需容忍 chat_messages=None:current 取 ""、skip refill、无 user 行——T7 已铺,此处如未铺一并补上并测。)

- [ ] **Step 3: 全绿+提交** `"feat(ch08-t8): /api/tickets/confirm resume 端点——SSE 续播+409 过期+422 乱值 …"`

### Task 9: 前端工单预览卡片(后端契约 TDD;html/js 部分 Vibe 例外)

**Files:**
- Modify: `static/index.html`(SSE 事件分发加 `ticket_preview` 分支 + 卡片渲染 + 两按钮 + POST 续流消费)
- Test: 后端无新文件(T8 已覆盖端点;本任务验证=手工清单)

**Interfaces:** Consumes `ticket_preview` 帧 `{ticket_type, description, conversation_id, tool_call_id}` 与 confirm 端点 SSE(帧面=T8 事件);Produces 无。

- [ ] **Step 1(手工冒烟,TDD 例外——工作要求1 Vibe)**:起服务两 Server(`.venv/Scripts/python.exe mcp_servers/logistics_server.py` ×2 + 主服务),浏览器聊天「帮我建个工单,猫粮缺货」→ 观察卡片渲染/确认建单后工单号续播/取消后「已取消」;复发新消息验证隐式 cancel。翻车点(Git Bash 下 curl 轮询、done 帧关流)照 ch07 经验先轮询再起测。
- [ ] **Step 2: 提交**(index.html + 计划勾选 + dev-notes,注明 Vibe 例外由用户后续描述驱动迭代)

### Task 10: 标注样例评估集(提示词效果类,非单测产物)

**Files:**
- Create: `evals/ch08_samples.jsonl`(标注样例)、`evals/run_ch08_eval.py`(@integration,真模型跑)
- Modify: `dev-notes/ch08.md`(评估结果记录)

**Interfaces:** Consumes 全部执行引擎;Produces:三份桶结论(校验回灌自纠/必填追问不瞎编/三类分诊如实回)。

- [ ] **Step 1: 样例表** `evals/ch08_samples.jsonl` 每行 `{"id", "user", "expect": "ask_missing|self_fix|no_fabricate", "notes"}`,≥10 条(缺描述建单×3、参数类型错×3、查询落空×2、真故障×2)。「真故障」桶以 monkeypatch BoomTool 注入快照跑脚本(评估面脚本允许注入假件——被测的是模型对错误回灌的话术收敛)。
- [ ] **Step 2: 跑** `pytest` 外独立 `uv run python evals/run_ch08_eval.py`(落 UTF-8 结果文件读),人工核对三桶话术:追问不编造、错误如实转述、空结果不说成功。不达标 → 改 create_ticket/工具 docstring 或 react 人设尾部提示(规4 任务内直改),重跑。
- [ ] **Step 3: 提交** `"test(ch08-t10): 校验回灌/必填追问/分诊话术标注样例集+评估跑批结论 …"`

### Task 11: 验收 e2e + README + 完结交付

**Files:**
- Create: `tests/e2e/test_ch08_acceptance.py`(单元级 Fake 面钉六条之可自动化项;真 Server 演示归集成/手工)
- Modify: `README.md`(ch08 节:演示命令两 Server+主服务、env 说明、已知边界)
- Modify: `dev-notes/ch08.md`(完结段)

**Interfaces:** Consumes 全部;Produces: 验收 4/5/6 自动化钉 + 1/2/3 演示脚本口径(P9)。

- [ ] **Step 1: 写 e2e**——C4(confirm→tickets 表 FakeCrud 计数+工单号 token)、C5(cancel→审计「权限拒绝」+不建单)、C6(超时:timeout=0.01+max_retries=2 → 审计 retry_count=2/状态「超时」/duration_ms 非空;write 超时恒 0)、C2 近似(快照含 mcp 源件被 bind)、C1/C3 手工脚本在 README 钉步骤。
- [ ] **Step 2: 全量回归** `pytest -q`(单元全绿)+ `pytest -m integration`(含 T6;输出落文件)。
- [ ] **Step 3: README ch08 节 + dev-notes 完结段(四样)+ 提交**,随后 executing-plans 终审批段(批评审 M 批 + fresh 终审 + fix pass)按 skill 走;finishing 菜单必停等人拍板(规约:merge/finishing 留人)。

---

## 任务间接口预检(executing-plans setup 时逐行核 ledger)

- T2 `ToolSpec/BUILTIN_SPECS/snapshot_tools` ← T3/T4/T5/T7 消费(名字逐字)。
- T3 `ToolContext.audit_sink/ticket_confirmed`、`ToolOutcome.awaiting_confirmation` ← T4/T7。
- T5 `STATUS_LABELS/format_result` ← T6 集成断言引用。
- T7 `stream_graph_turn(resume_value=)`、`("ticket_preview", payload)` ← T8/T9。
- T2 settings 键名 `mcp_logistics_url/mcp_aftersale_url` ← T6 测。
