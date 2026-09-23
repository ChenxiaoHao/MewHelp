# ch04 RAG 进阶(混合检索+重排+评估体系+生成质量控制)实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 query_faq 从 dense 单路升级为「Milvus 原生 BM25 + hybrid_search RRF + SiliconFlow bge-reranker-v2-m3 精排 + 首尾排布」,配前置双闸拒答与低置信度池;消费老师 300 题评估集出四策略对比报告与忠实度台账;前端做引用弹窗、满意度反馈采集与 faith_cases 台账页。

**Architecture:** 离线建库沿用 ch03 两段双写、第二段升级为集合 v2(text 随向量入库供服务端 BM25 函数生成 sparse);在线 `retriever.retrieve(strategy=…)` 同一代码路径承载四策略+闸1,`tool_chat_service` 增闸2 与固定拒答出口;评估两个新 runner 只读消费老师 CSV;前端 REST 增 `/api/chunks/{id}` 与 `/api/faith_cases` GET/PATCH。设计权威 = spec,本计划与其冲突时以 spec 为准并停下向用户报告。

**Tech Stack:** Milvus v3.0.0 standalone + pymilvus 3.0.2(原生 BM25/chinese analyzer/hybrid_search+RRFRanker)、SiliconFlow `/v1/rerank`(BAAI/bge-reranker-v2-m3,httpx 直连)、FastAPI + sse + pydantic v2、SQLAlchemy 2 async + aiomysql、LangChain 1.x(@tool + RunnableConfig)、uv/pytest(`-m integration` 标记)。

**Spec:** `docs/superpowers/specs/2026-09-23-ch04-hybrid-rerank-eval-design.md`(§0 决策 13 条 + §1-§12 + 附录 A/B/C 全读;本计划逐节对应)

## Global Constraints(每个任务隐含包含)

- **依赖冻结**:pyproject 零新增运行时依赖(httpx 已在)。若实施中发现必须新增,停下向用户报告(spec §10)。
- **老师材料只读**:`knowledge/*.md`(6 份)、`evals/run_rag.py`(300 题 CSV)、spec 附录 A 的 DDL 原文——任何实现迁就数据,不反向改数据(§0-5/§0-6/§0-12)。材料自身 3 处标注出入如实进报告(附录 B),禁止为达标改数据/藏结果/强行调参(§0-9)。
- **兼容红线**:ch01/ch02 既有测试全绿;SSE 五帧语义逐字符不变(tool_result 仅增可选 `citations`);query_faq **工具签名**(模型可见 schema,`keyword: str`)不变;`knowledge_chunks`/`qa_extraction_staging` 零改列(§12)。
- **基线事实**(2026-09-23 实测):`uv run pytest` = **130 passed / 4 failed**,4 个失败全在 `tests/test_corpus_ch03.py`(老师语料替换后必炸,附录 C 预期红)——Task 3 重锚定后归零;任何任务不得让其它测试变红。
- **dev-notes 逐段留痕**:每任务完成提交时,同步在 `dev-notes/ch04.md` 追记一段四要素(用户原话/关键产出/拒绝纠偏/翻车返工),与该任务 commit 同车;不许收尾补记。
- **Context7 核对点**(spec §11):①②③ 已在本计划编写时预查(pymilvus create_schema/Function/AnnSearchRequest/RRFRanker 形状已确认;rerank 请求字段为 `documents`(spec §4.3 预写的 `texts` 由核对点③授权修正)、响应 `results[].index/relevance_score`);**现场冒烟销账仍不可省**(Task 2/5 集成测试即冒烟);④RunnableConfig 注入有 create_ticket 先例,T7 复核;⑤FastAPI 422 在 T10 写前查;⑥中文 ENUM 读写 T10 集成实测。落码前对不确定的 API 一律先 Context7 再动手。
- **命令惯例**:一律 `uv run …`;集成测试 `uv run pytest -m integration -q`;提交信息前缀 `feat/test/docs(ch04):`,尾部空行后加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- **拒答/降级语义不许自行加码**:闸2 失败=放行(§0-2);评估不设阻断硬线(§0-9);评估跑分前先报成本(--limit 成本闸)。
- Windows/Git Bash;文件一律 UTF-8;新 `.py` 首行中文 docstring(spec 风格)。

## 文件总图(创建/修改一览)

```
app/core/config.py            改:ch04 11 字段          app/rag/query_understanding.py  新增
app/rag/milvus_store.py       重写为 v2 门面            app/rag/reranker.py             新增
app/rag/indexer.py            改 vectorize_pending     app/rag/retriever.py            重写(策略化)
app/rag/hit_format.py         新增(纯函数)            app/prompts/query_rewrite.py    新增
app/services/self_check.py    新增                    app/prompts/self_check.py       新增
app/services/refusals.py      新增                    app/prompts/customer_service.py 改造
app/prompts/faith_judge.py    新增
app/db/models.py              增两 ORM                app/db/crud.py                  增池/台账/chunk 函数
app/tools/definitions.py      query_faq v2            app/tools/executor.py           citations 承载
app/schemas/chat.py           ToolResultEvent 增列    app/schemas/knowledge.py        新增
app/services/tool_chat_service.py  闸2+拒答出口       app/api/routes.py               三新端点+exclude_none
db/init/05_ch04_schema.sql    新增(spec 附录A截取)   static/index.html               Vibe(引用/反馈)
static/faith.html             新增 Vibe(台账页)     evals/teacher_csv.py            新增
evals/run_strategy_eval.py    新增                    evals/run_faith_eval.py         新增
evals/smoke_query_rewrite.py  新增                    evals/run_rag_eval.py           标 deprecated
tests/: ch04 新测试×12 + 重锚定(test_corpus→ch04、retriever 系、routes/schemas/tools 最小适配)
README.md                     ch04 节                 .env.example                    ch04 块
```

---

### Task 1: ch04 配置 11 项 + `05_ch04_schema.sql`(附录 A 程序化截取)+ 两表 ORM

**Files:**
- Modify: `app/core/config.py`(`qa_mine_batch_conversations: int = 5` 行之后追加)
- Modify: `.env.example`(文件末尾追加 ch04 块)
- Create: `db/init/05_ch04_schema.sql`(**只能由脚本从 spec 附录 A 生成,禁止手抄**)
- Modify: `app/db/models.py`(文件末尾追加两个模型)
- Test: `tests/test_config_ch04.py`(Create)、`tests/test_models_ch04.py`(Create)

**Interfaces:**
- Consumes: spec 附录 A(DDL 原文,哨兵 8/8 已验);`app/db/models.py` 既有 `_pk()`/`Base` 惯例
- Produces: `Settings` 新字段 `rerank_api_base/rerank_api_key/rerank_model/rerank_timeout_seconds/hybrid_recall_k/rrf_k/rerank_top_n/retrieval_low_conf_threshold/self_check_enabled/query_rewrite_enabled/faith_judge_model`;ORM `LowConfidenceQuestion`、`FaithCase`(后续任务全按这两个类名用)

- [ ] **Step 1: 写失败测试 `tests/test_config_ch04.py`**

```python
"""ch04 Settings 默认值/env 覆盖/构造不破坏红线(守卫测试,同 test_config_ch03 惯例)。"""

import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")


def test_ch04_defaults(env):
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.rerank_api_base == "https://api.siliconflow.cn/v1"
    assert s.rerank_api_key == ""
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_timeout_seconds == 5.0
    assert s.hybrid_recall_k == 50
    assert s.rrf_k == 60
    assert s.rerank_top_n == 10
    assert s.retrieval_low_conf_threshold == 0.3  # D 桶校准前初值(§6.2)
    assert s.self_check_enabled is True
    assert s.query_rewrite_enabled is True
    assert s.faith_judge_model == ""


def test_ch04_env_override(env, monkeypatch):
    monkeypatch.setenv("RERANK_API_KEY", "sk-test")
    monkeypatch.setenv("HYBRID_RECALL_K", "30")
    monkeypatch.setenv("SELF_CHECK_ENABLED", "false")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.rerank_api_key == "sk-test"
    assert s.hybrid_recall_k == 30
    assert s.self_check_enabled is False


def test_ch01_to_ch03_still_untouched(env):
    from app.core.config import Settings

    s = Settings(_env_file=None, openai_base_url="http://fake/v1",
                 openai_api_key="fake-key", model_name="fake-model")
    assert s.rag_score_threshold == 0.3 and s.milvus_collection == "knowledge"
    assert s.history_token_budget == 4000 and s.tool_timeout_seconds == 5.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_config_ch04.py -q`
Expected: FAIL(字段不存在;`test_ch01_to_ch03_still_untouched` 已过——它只守卫旧字段)

- [ ] **Step 3: `app/core/config.py` 追加字段**

在 `qa_mine_batch_conversations: int = 5` 之后插入:

```python
    # --- ch04: 混合检索+重排+评估(全部带默认值,不破坏 ch01-ch03 构造) ---
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_timeout_seconds: float = 5.0
    hybrid_recall_k: int = 50      # dense/BM25 双腿各召回 Top-50
    rrf_k: int = 60                # RRFRanker(k=60)
    rerank_top_n: int = 10         # 精排后喂给模型的证据数
    retrieval_low_conf_threshold: float = 0.3  # 闸1;Task 12 D 桶校准后回写终值
    self_check_enabled: bool = True            # 闸2 总开关(评估对照/省调用)
    query_rewrite_enabled: bool = True         # 查询理解开关
    faith_judge_model: str = ""                # 空=model_name
```

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_config_ch04.py -q` → PASS(3 个全过)

- [ ] **Step 5: `.env.example` 末尾追加**

```
# --- ch04: 混合检索+重排+评估 ---
# RERANK_API_KEY 需注册 SiliconFlow 申请(https://cloud.siliconflow.cn/account/ak),不配则重排降级、hybrid_rerank 臂失真
RERANK_API_BASE=https://api.siliconflow.cn/v1
RERANK_API_KEY=sk-your-siliconflow-key
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TIMEOUT_SECONDS=5
HYBRID_RECALL_K=50
RRF_K=60
RERANK_TOP_N=10
RETRIEVAL_LOW_CONF_THRESHOLD=0.3
SELF_CHECK_ENABLED=true
QUERY_REWRITE_ENABLED=true
FAITH_JUDGE_MODEL=
```

同时提醒用户本机 `.env` 需补 `RERANK_API_KEY`(T5/T11/T12 演示前若无 key 再次索要)。

- [ ] **Step 6: 程序化截取生成 `db/init/05_ch04_schema.sql`**(ch03 Task 5 教训:定稿文本不信任手指)

```bash
uv run python - <<'EOF'
import re
from pathlib import Path
spec = Path("docs/superpowers/specs/2026-09-23-ch04-hybrid-rerank-eval-design.md").read_text(encoding="utf-8")
m = re.search(r"## 附录 A.*?```sql\n(.*?)```", spec, re.S)
assert m, "附录 A sql 块未命中"
Path("db/init/05_ch04_schema.sql").write_text(m.group(1), encoding="utf-8", newline="\n")
print("written bytes:", len(m.group(1).encode("utf-8")))
EOF
```

- [ ] **Step 7: 哨兵机械验证**(首行应输出 2;八条哨兵全部 ≥1;任一为 0 → 截取失败回查 spec,**禁止手改补**)

```bash
grep -c "CREATE TABLE" db/init/05_ch04_schema.sql
for s in "SET NAMES utf8mb4;" "FOREIGN KEY (conversation_id) REFERENCES conversations (id)" "UNIQUE KEY uk_eval_id (eval_id)" "ENUM('retrieval_low_conf','self_check','user_feedback')" "ENUM('未解决','已解决','无需解决')" "VARCHAR(300)" "citations" "INT UNSIGNED"; do grep -cF "$s" db/init/05_ch04_schema.sql; done
```

- [ ] **Step 8: 活库执行(老库升级路径,§3.2;新库首启由 docker 自动跑 01-05)**

```bash
docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp < db/init/05_ch04_schema.sql
docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp -e "SHOW TABLES LIKE '%faith%'; SHOW TABLES LIKE 'low_conf%';"
```
Expected: `faith_cases`、`low_confidence_questions` 两行输出。

- [ ] **Step 9: 写失败测试 `tests/test_models_ch04.py`**

```python
"""ORM ↔ 05_ch04_schema.sql(=spec 附录 A)列级对账;活库对账走文末集成(ch03 同法)。"""

import pytest


def test_low_confidence_question_columns():
    from app.db.models import LowConfidenceQuestion

    t = LowConfidenceQuestion.__table__
    assert t.name == "low_confidence_questions"
    assert [c.name for c in t.columns] == [
        "id", "conversation_id", "raw_question", "source", "reason", "created_at",
    ]
    cols = t.columns
    assert cols["conversation_id"].nullable is True
    assert {fk.target_fullname for fk in cols["conversation_id"].foreign_keys} == {"conversations.id"}
    assert set(cols["source"].type.enums) == {"retrieval_low_conf", "self_check", "user_feedback"}
    assert cols["raw_question"].nullable is False and cols["source"].nullable is False


def test_faith_case_columns():
    from app.db.models import FaithCase

    t = FaithCase.__table__
    assert t.name == "faith_cases"
    assert [c.name for c in t.columns] == [
        "id", "eval_id", "bucket", "query", "strategy", "answer", "reason", "citations",
        "judge_model", "status", "seen_count", "first_seen_at", "last_seen_at",
        "resolution", "resolved_at",
    ]
    cols = t.columns
    assert cols["eval_id"].unique is True and cols["eval_id"].nullable is False
    assert cols["query"].type.length == 512 and cols["resolution"].type.length == 300
    assert set(cols["status"].type.enums) == {"未解决", "已解决", "无需解决"}
    assert cols["citations"].nullable is True and cols["judge_model"].nullable is True
    assert cols["seen_count"].nullable is False
    assert cols["resolved_at"].nullable is True


@pytest.mark.integration
async def test_live_schema_matches_ch04_orm():
    """需 mysql 容器 + 已执行 05(本任务 Step 8)。"""
    from sqlalchemy import inspect

    from app.core.config import get_settings
    from app.db.engine import dispose_engine, get_engine, init_engine
    from app.db.models import FaithCase, LowConfidenceQuestion

    init_engine(get_settings())
    try:
        def _reflect(sync_conn):
            insp = inspect(sync_conn)
            return {
                name: sorted(c["name"] for c in insp.get_columns(name))
                for name in (LowConfidenceQuestion.__tablename__, FaithCase.__tablename__)
            }

        async with get_engine().connect() as conn:
            live = await conn.run_sync(_reflect)
        for model in (LowConfidenceQuestion, FaithCase):
            cols = {c.name for c in model.__table__.columns}
            assert set(live[model.__tablename__]) == cols, model.__tablename__
    finally:
        await dispose_engine()
```

- [ ] **Step 10: `app/db/models.py` 末尾追加两模型**

```python
class LowConfidenceQuestion(Base):
    """低置信度问题池;DDL 以 05_ch04_schema.sql(用户原文逐字)为准,本章两写方(§0-12)。"""

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = _pk()
    conversation_id: Mapped[int | None] = mapped_column(
        BIGINT(unsigned=True), ForeignKey("conversations.id"), nullable=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(
        Enum("retrieval_low_conf", "self_check", "user_feedback", name="lcq_source"),
        nullable=False,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class FaithCase(Base):
    """忠实度编造个案台账;一题一行 uk_eval_id,复发流转见 crud.upsert_faith_case。"""

    __tablename__ = "faith_cases"

    id: Mapped[int] = _pk()
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(
        String(24), nullable=False, server_default="hybrid_rerank"
    )
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_status"),
        nullable=False,
        server_default="未解决",
    )
    seen_count: Mapped[int] = mapped_column(
        INTEGER(unsigned=True), nullable=False, server_default=text("1")
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
```

并把顶部 `from sqlalchemy.dialects.mysql import BIGINT` 改为 `from sqlalchemy.dialects.mysql import BIGINT, INTEGER`;核对 `text`/`JSON`/`String`/`DateTime`/`func`/`ForeignKey`/`Enum` 已在既有 import 中(ch03 已全有,缺则补)。

- [ ] **Step 11: 单测 + 集成对账**

Run: `uv run pytest tests/test_models_ch04.py tests/test_config_ch04.py -q` → PASS
Run: `uv run pytest -m integration tests/test_models_ch04.py -q` → PASS(顺带销核对点⑥前半:中文 ENUM 表建成形)

- [ ] **Step 12: 全量回归不破**

Run: `uv run pytest -q` → Expected: **135 passed, 4 failed**(本任务逐字新增单测 = 3 config + 2 model;`test_live_schema_matches_ch04_orm` 带 integration 标记被默认 addopts deselect,另在集成步骤跑;4 红=仅 test_corpus_ch03 预期红。2026-09-23 实施 T1 实测修订,原写 138 为误算)

- [ ] **Step 13: dev-notes 追记 + 提交**

`dev-notes/ch04.md` 末尾追加「计划落盘与评审通过」段与「Task 1 完成」段(四要素;评审通过段在用户批准计划后补写亦可——**落盘段现在就写,不等收尾**)。

```bash
git add app/core/config.py .env.example db/init/05_ch04_schema.sql app/db/models.py tests/test_config_ch04.py tests/test_models_ch04.py dev-notes/ch04.md docs/superpowers/plans/2026-09-23-ch04-hybrid-rerank-eval.md
git commit -m "feat(ch04): ch04 配置11项+05建表DDL(附录A程序化截取+哨兵复核)+两表ORM列对账

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: `milvus_store` 集合 v2 门面(BM25 Function/双索引/hybrid_search)+「MH-LP100」冒烟

**Files:**
- Rewrite: `app/rag/milvus_store.py`
- Rewrite: `tests/test_milvus_store.py`(v2 门面 fake 全套)
- Rewrite: `tests/test_milvus_store_integration.py`(旧版用 upsert_vectors+dim=4,整体换 v2)
- Modify: `app/rag/indexer.py` 一行 + `tests/test_indexer.py` 两处打桩签名(最小适配,防提交瞬间全红;完整语义在 T3)

**Interfaces:**
- Consumes: `Settings.hybrid_recall_k/rrf_k`(T1)
- Produces(门面契约,T3/T6 与集成测试全按这套):
  - `ensure_collection(client, name, dim)`(签名不变,内部 v2 建集;`TEXT_MAX_LENGTH = 8192` 模块常量)
  - `upsert_rows(client, name, rows: list[dict]) -> int`(取代 `upsert_vectors`,后者删除)
  - `search_vectors(client, name, vector, top_k, expr=None) -> list[tuple[int, float]]`
  - `bm25_search(client, name, text, top_k, expr=None) -> list[tuple[int, float]]`
  - `hybrid_search(client, name, vector, bm25_text, *, limit, recall_k, rrf_k, expr=None) -> list[tuple[int, float]]`
  - `get_client/health_ok/drop_collection/flush/count_rows/all_ids` 不变

- [ ] **Step 1: 重写 `tests/test_milvus_store.py`(失败测试先行)**

```python
"""门面纯逻辑测试 v2:假 client 记录 schema/调用形状(不碰真 Milvus)。
真实形状以集成冒烟为准——核对点①双保险(同 ch03 惯例)。"""


class FakeSchema:
    def __init__(self):
        self.fields = []      # [(name, data_type, kwargs)]
        self.functions = []

    def add_field(self, name, data_type, **kw):
        self.fields.append((name, data_type, kw))

    def add_function(self, fn):
        self.functions.append(fn)


class FakeIndexParams:
    def __init__(self):
        self.indexes = []     # [(field, kwargs)]

    def add_index(self, field, **kw):
        self.indexes.append((field, kw))


class FakeClient:
    def __init__(self, *, has=False, search_out=None, query_out=None, hybrid_out=None, fail=False):
        self.calls = []
        self.has = has
        self.search_out = search_out if search_out is not None else [[]]
        self.query_out = query_out if query_out is not None else []
        self.hybrid_out = hybrid_out if hybrid_out is not None else [[]]
        self.fail = fail
        self.schema = FakeSchema()
        self.index_params = FakeIndexParams()

    def list_collections(self):
        if self.fail:
            raise RuntimeError("boom")
        return ["knowledge"]

    def has_collection(self, collection_name):
        self.calls.append(("has", collection_name))
        return self.has

    def create_schema(self, **kw):
        self.calls.append(("create_schema", kw))
        return self.schema

    def prepare_index_params(self):
        return self.index_params

    def create_collection(self, **kw):
        self.calls.append(("create", kw))

    def drop_collection(self, collection_name):
        self.calls.append(("drop", collection_name))

    def flush(self, collection_name):
        self.calls.append(("flush", collection_name))

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))
        return type("R", (), {"upsert_count": len(kw["data"])})()

    def search(self, **kw):
        self.calls.append(("search", kw))
        return self.search_out

    def query(self, **kw):
        self.calls.append(("query", kw))
        return self.query_out

    def hybrid_search(self, **kw):
        self.calls.append(("hybrid", kw))
        return self.hybrid_out


def _field_kw(client):
    return {name: kw for name, _dt, kw in client.schema.fields}


def test_ensure_collection_declares_bm25_schema():
    from pymilvus import DataType, FunctionType

    from app.rag.milvus_store import ensure_collection

    c = FakeClient(has=False)
    ensure_collection(c, "knowledge", 1024)
    names = [n for n, _dt, _kw in c.schema.fields]
    assert names == ["chunk_id", "text", "sparse", "embedding", "category", "content_type"]
    text = _field_kw(c)["text"]
    assert text["enable_analyzer"] is True and text["analyzer_params"] == {"type": "chinese"}
    assert text["max_length"] == 8192
    assert c.schema.fields[0][1] is DataType.INT64 and c.schema.fields[0][2]["is_primary"] is True
    assert c.schema.fields[2][1] is DataType.SPARSE_FLOAT_VECTOR
    assert _field_kw(c)["embedding"]["dim"] == 1024
    assert _field_kw(c)["category"]["max_length"] == 765
    fn = c.schema.functions[0]
    assert fn.type is FunctionType.BM25 and fn.name == "bm25_fn"  # 2026-09-23 T2 实测:pymilvus 3.0.2 公开访问器是 .type(.function_type 仅构造参数名,Context7 复核)
    assert list(fn.input_field_names) == ["text"] and list(fn.output_field_names) == ["sparse"]
    assert (c.index_params.indexes[0][0], c.index_params.indexes[0][1]["metric_type"]) == ("embedding", "COSINE")
    assert (c.index_params.indexes[1][0], c.index_params.indexes[1][1]["index_type"]) == ("sparse", "SPARSE_INVERTED_INDEX")
    kw = [call for call in c.calls if call[0] == "create"][0][1]
    assert kw["collection_name"] == "knowledge" and kw["schema"] is c.schema
    c2 = FakeClient(has=True)
    ensure_collection(c2, "knowledge", 1024)
    assert not any(k == "create" for k, *_ in c2.calls)


def test_upsert_rows_passthrough_and_empty_guard():
    from app.rag.milvus_store import upsert_rows

    assert upsert_rows(FakeClient(), "knowledge", []) == 0
    c = FakeClient()
    rows = [{"chunk_id": 5, "text": "t5", "embedding": [0.1], "category": "c", "content_type": "policy"}]
    assert upsert_rows(c, "knowledge", rows) == 1
    kw = [call for call in c.calls if call[0] == "upsert"][0][1]
    assert kw["data"] == rows  # 仅 {chunk_id,text,embedding,category,content_type};sparse 服务端函数生成
    assert all("sparse" not in r for r in kw["data"])


def test_search_dense_and_bm25_shapes():
    from app.rag.milvus_store import bm25_search, search_vectors

    c = FakeClient(search_out=[[{"chunk_id": 2, "distance": 0.9}, {"chunk_id": 1, "distance": 0.31}]])
    assert search_vectors(c, "knowledge", [0.0], 2) == [(2, 0.9), (1, 0.31)]
    kw = [call for call in c.calls if call[0] == "search"][0][1]
    assert kw["data"] == [[0.0]] and "filter" not in kw
    c2 = FakeClient(search_out=[[{"chunk_id": 7, "distance": 12.5}]])
    assert bm25_search(c2, "knowledge", "MH-LP100", 3, expr='category == "商品参数"') == [(7, 12.5)]
    kw2 = [call for call in c2.calls if call[0] == "search"][0][1]
    assert kw2["data"] == ["MH-LP100"] and kw2["anns_field"] == "sparse"
    assert kw2["filter"] == 'category == "商品参数"'


def test_hybrid_search_builds_two_legs_with_rrf():
    from pymilvus import RRFRanker

    from app.rag.milvus_store import hybrid_search

    c = FakeClient(hybrid_out=[[{"chunk_id": 3, "distance": 0.032}]])
    out = hybrid_search(c, "knowledge", [0.1, 0.2], "猫砂盆 清理",
                        limit=10, recall_k=50, rrf_k=60, expr='category == "x"')
    assert out == [(3, 0.032)]
    kw = [call for call in c.calls if call[0] == "hybrid"][0][1]
    reqs = kw["reqs"]
    assert (reqs[0].anns_field, reqs[0].limit, reqs[0].expr) == ("embedding", 50, 'category == "x"')
    assert (reqs[1].anns_field, reqs[1].limit, reqs[1].data) == ("sparse", 50, ["猫砂盆 清理"])
    assert isinstance(kw["ranker"], RRFRanker)
    assert kw["ranker"].dict()["params"] == {"k": 60}  # 2026-09-23 T2 实测:RRFRanker 无公开 .k 属性,文档口径=dict() 序列化(Context7 复核)
    assert kw["collection_name"] == "knowledge" and kw["limit"] == 10


def test_health_drop_flush_count_ids():
    from app.rag import milvus_store as ms

    assert ms.health_ok(FakeClient()) is True
    assert ms.health_ok(FakeClient(fail=True)) is False
    c = FakeClient(has=True)
    ms.drop_collection(c, "knowledge")
    assert ("drop", "knowledge") in c.calls
    ms.drop_collection(FakeClient(has=False), "knowledge")  # 不存在不报错
    ms.flush(FakeClient(), "knowledge")
    assert ms.count_rows(FakeClient(query_out=[{"count(*)": 5}]), "knowledge") == 5
    assert ms.all_ids(FakeClient(query_out=[{"chunk_id": 1}, {"chunk_id": 3}]), "knowledge") == [1, 3]
```

Run: `uv run pytest tests/test_milvus_store.py -q` → FAIL(符号不存在)

- [ ] **Step 2: 重写 `app/rag/milvus_store.py`**

```python
"""Milvus 同步门面 v2(spec §3.1)。ch03「只存 id+向量」已推翻:原生 BM25 必须存 text(§0-7),
sparse 由服务端 BM25 Function 生成、客户端不写;原文权威仍是 MySQL,回查/孤儿过滤语义不变。

ch03 实测语义沿用:PK 回显键名=主键字段名 `chunk_id`;COSINE distance 越大越相似;写完必 flush。
ch04 核对点①②(建集确切写法/VARCHAR 字节上限/中文分词)由集成冒烟现场销账。
"""

from __future__ import annotations

TEXT_MAX_LENGTH = 8192  # 核对点②:三格拼接 chunk ≈ ≤2.4KB UTF-8,3 倍冗余;超限实测后再调


def get_client(uri: str, timeout: float = 10.0):
    from pymilvus import MilvusClient  # 延迟 import:纯为启动快,本项目已装

    return MilvusClient(uri=uri, timeout=timeout)


def health_ok(client) -> bool:
    try:
        client.list_collections()
        return True
    except Exception:  # noqa: BLE001 —— 探活语义:任何异常都算不健康
        return False


def ensure_collection(client, name: str, dim: int) -> None:
    """集合 v2:chunk_id PK + text(chinese analyzer)+ sparse(BM25 函数输出)
    + embedding(FLOAT_VECTOR)+ category/content_type(标量过滤)。建集 schema 不可改,迁移=drop 重建。"""
    if client.has_collection(name):
        return
    from pymilvus import DataType, Function, FunctionType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field(
        "text", DataType.VARCHAR, max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True, analyzer_params={"type": "chinese"},
    )
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("category", DataType.VARCHAR, max_length=765)
    schema.add_field("content_type", DataType.VARCHAR, max_length=32)
    schema.add_function(Function(
        name="bm25_fn", function_type=FunctionType.BM25,
        input_field_names=["text"], output_field_names=["sparse"],
    ))
    index_params = client.prepare_index_params()
    index_params.add_index("embedding", index_type="AUTOINDEX", metric_type="COSINE")
    index_params.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


def drop_collection(client, name: str) -> None:
    if client.has_collection(name):
        client.drop_collection(name)


def flush(client, name: str) -> None:
    """核对点(ch03 实测):upsert 是缓冲的,search/count 前必须 flush。"""
    client.flush(name)


def upsert_rows(client, name: str, rows: list[dict]) -> int:
    """rows 每项 {chunk_id,text,embedding,category,content_type};sparse 缺席(函数列服务端生成)。
    按主键幂等 upsert(§5 语义核心):同 chunk_id 再写 = 覆盖。"""
    if not rows:
        return 0
    res = client.upsert(collection_name=name, data=rows)
    return getattr(res, "upsert_count", None) or len(rows)


def search_vectors(client, name: str, vector: list[float], top_k: int,
                   expr: str | None = None) -> list[tuple[int, float]]:
    """dense 腿。COSINE 相似度(越大越相似)。expr=标量过滤(需求3)。"""
    kw = dict(collection_name=name, data=[vector], limit=top_k)
    if expr:
        kw["filter"] = expr
    res = client.search(**kw)
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def bm25_search(client, name: str, text: str, top_k: int,
                expr: str | None = None) -> list[tuple[int, float]]:
    """BM25 腿:查询文本原样进 data(服务端 analyzer 分词),distance=BM25 分数(越大越相关)。"""
    kw = dict(collection_name=name, data=[text], anns_field="sparse", limit=top_k)
    if expr:
        kw["filter"] = expr
    res = client.search(**kw)
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def hybrid_search(client, name: str, vector: list[float], bm25_text: str, *,
                  limit: int, recall_k: int, rrf_k: int,
                  expr: str | None = None) -> list[tuple[int, float]]:
    """§4.2-1:双腿各 Top-recall_k → RRFRanker(k=rrf_k) 融合,返回 [(chunk_id, rrf_score)]。"""
    from pymilvus import AnnSearchRequest, RRFRanker

    reqs = [
        AnnSearchRequest(data=[vector], anns_field="embedding",
                         param={"metric_type": "COSINE"}, limit=recall_k, expr=expr),
        AnnSearchRequest(data=[bm25_text], anns_field="sparse",
                         param={}, limit=recall_k, expr=expr),
    ]
    res = client.hybrid_search(collection_name=name, reqs=reqs, ranker=RRFRanker(k=rrf_k),
                               limit=limit, output_fields=["chunk_id"])
    hits = res[0] if res else []
    return [(h["chunk_id"], float(h["distance"])) for h in hits]


def count_rows(client, name: str) -> int:
    res = client.query(collection_name=name, filter="", output_fields=["count(*)"])
    return int(res[0]["count(*)"]) if res else 0


def all_ids(client, name: str) -> list[int]:
    """demo 规模一次性拉全 id,供 --check 对账差集。"""
    res = client.query(collection_name=name, filter="chunk_id > 0", output_fields=["chunk_id"])
    return sorted(r["chunk_id"] for r in res)
```

- [ ] **Step 3: 重写 `tests/test_milvus_store_integration.py`(冒烟=核对点①②现场销账)**

```python
"""集合 v2 集成冒烟(spec §4.2-6 硬断言):chinese analyzer 裸 BM25 命中型号/中文词、
标量过滤、hybrid_search 形状、flush 语义——实测形状记 dev-notes,核对点①②在此销账。"""

import pytest

from app.core.config import get_settings
from app.rag import milvus_store

pytestmark = pytest.mark.integration

COLL = "knowledge_it"  # 独立测试集合,不碰演示用 knowledge
DIM = 4

ROWS = [
    {"chunk_id": 1, "text": "智能猫砂盆 Pro(MH-LP100)支持 App 远程监控,废砂盒建议 5 至 7 天清理一次",
     "embedding": [1.0, 0.0, 0.0, 0.0], "category": "商品参数", "content_type": "policy"},
    {"chunk_id": 2, "text": "冻干猫粮喂食量:幼猫每天 3-4 次,按体重计算",
     "embedding": [0.0, 1.0, 0.0, 0.0], "category": "商品参数", "content_type": "policy"},
    {"chunk_id": 3, "text": "退货政策:签收后 7 天内无理由退货,不影响二次销售",
     "embedding": [0.9, 0.1, 0.0, 0.0], "category": "退货政策", "content_type": "policy"},
]


def test_collection_v2_bm25_hybrid_roundtrip():
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    assert milvus_store.health_ok(client)
    milvus_store.drop_collection(client, COLL)
    try:
        milvus_store.ensure_collection(client, COLL, DIM)
        assert milvus_store.upsert_rows(client, COLL, ROWS) == 3
        milvus_store.flush(client, COLL)
        # 硬断言①:型号 token 裸 BM25 命中 = chinese analyzer 对拉丁型号可用(验收2 地基)
        hits = milvus_store.bm25_search(client, COLL, "MH-LP100", 3)
        assert hits and hits[0][0] == 1
        # 硬断言②:中文词命中(分词生效)
        zh = milvus_store.bm25_search(client, COLL, "废砂盒", 3)
        assert zh and zh[0][0] == 1
        dense = milvus_store.search_vectors(client, COLL, [1.0, 0.0, 0.0, 0.0], 3)
        assert dense[0][0] == 1
        # 标量过滤(需求3):品类 expr 只放行该品类块。
        # 2026-09-23 T2 实测修正:原「猫→只{3}」物理不成立(row3 无「猫」token),改用 1/3 行共有「7 天」,
        # 无过滤先证 {1,3} 都在、过滤后只放行 3——才真正测到「排除其它品类」语义。
        assert {cid for cid, _ in milvus_store.bm25_search(client, COLL, "7 天", 3)} == {1, 3}
        filt = milvus_store.bm25_search(client, COLL, "7 天", 3, expr='category == "退货政策"')
        assert {cid for cid, _ in filt} == {3}
        # hybrid:dense 腿偏 2、BM25 腿偏 3 → 融合含两者且形状 [(int, float)]
        hyb = milvus_store.hybrid_search(client, COLL, [0.0, 1.0, 0.0, 0.0], "7 天无理由退货",
                                         limit=3, recall_k=3, rrf_k=60)
        assert hyb and all(isinstance(cid, int) and isinstance(s, float) for cid, s in hyb)
        # 2026-09-23 T2 实测修正:原「=={2,3}」不成立——recall_k=3 在 3 行小库=全库进融合,
        # chunk1 双腿中游(dense 第3+BM25 第2)RRF 反超单腿冠军 chunk2;实测序 [3,1,2] 与
        # RRFRanker(k=60) 公式 1/(k+rank) 逐项吻合(0.0325/0.0320/0.0164),按「融合含两者」本意改 ⊇。
        assert {2, 3} <= {cid for cid, _ in hyb}
        assert [cid for cid, _ in hyb] == [3, 1, 2]
        # 同 PK 覆写幂等 + flush 语义沿用
        assert milvus_store.upsert_rows(client, COLL, [dict(ROWS[0])]) == 1
        milvus_store.flush(client, COLL)
        assert milvus_store.count_rows(client, COLL) == 3
    finally:
        milvus_store.drop_collection(client, COLL)
```

- [ ] **Step 4: 单测 + 集成实跑(Milvus 容器需 `docker compose up -d`)**

Run: `uv run pytest tests/test_milvus_store.py -q` → PASS
Run: `uv run pytest -m integration tests/test_milvus_store_integration.py -q` → PASS
**形状不符预期(键名/RRF 分数语义/字节上限)→ 只改门面适配,不改断言语义;实测形状记入 dev-notes(核对点①②销账)。改 analyzer 写法前必须 Context7 复查。**

- [ ] **Step 5: indexer 一行最小适配 + 测试打桩签名(防 T2 提交瞬间全红)**

`app/rag/indexer.py`:`milvus_store.upsert_vectors(client, st.milvus_collection, list(zip(ids, vectors)))` 替换为(完整语义 T3):

```python
        rows = [
            {"chunk_id": cid, "text": t, "embedding": vec,
             "category": r.category, "content_type": r.content_type or ""}
            for cid, t, vec, r in zip(ids, texts, vectors, batch)
        ]  # text 与 embed 输入同源三格拼接(§3.1);r 来自 batch(ORM 行)
        milvus_store.upsert_rows(client, st.milvus_collection, rows)
```

`tests/test_indexer.py`:`def upsert(c, name, pairs)` / `lambda c, n, p: len(p)` 两处打桩签名改 `(c, name, rows)` / `(c, n, r)`(计数断言 `[10, 10, 5]` 不变)。若 batch 变量名与现有循环不吻合,以现有 `for batch in ...` 结构为准对齐变量,**不留注释悬案**。

- [ ] **Step 6: 全量回归 + 提交 + dev-notes**

Run: `uv run pytest -q` → Expected: 除 test_corpus_ch03 4 红外全绿
dev-notes 追记「Task 2」段(含集成实测形状记录),随 commit:

```bash
git add app/rag/milvus_store.py tests/test_milvus_store.py tests/test_milvus_store_integration.py app/rag/indexer.py tests/test_indexer.py dev-notes/ch04.md
git commit -m "feat(ch04): milvus_store 集合v2门面(BM25函数/双索引/hybrid_search)+MH-LP100中文冒烟销核对点①②

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: 集合 v2 全量重灌 + 语料测试重锚定(test_corpus_ch03 → ch04)+ 块数实测锚

**Files:**
- Modify: `app/rag/indexer.py`(仅 docstring,v2 行形状 T2 已就位)
- Delete: `tests/test_corpus_ch03.py`(git rm)
- Create: `tests/test_corpus_ch04.py`
- Modify: `tests/test_indexer_integration.py`(追加块数锚)

**Interfaces:**
- Consumes: T2 `upsert_rows`(已在 indexer 生效)
- Produces: 活库 `knowledge` 集合 = v2 schema、行数 = 块数锚 `EXPECTED_TOTAL`;全库可供 T6 集成与 T11/T12 评估

- [ ] **Step 1: 写 `tests/test_corpus_ch04.py`(重锚老师语料,断言全部对实料核过)**

```python
"""老师材料替换后的语料级断言(spec §0-5/附录 B):6 份正式文档过 chunker 的形态必须
支撑四策略评估与验收地基;老师材料自身出入(运费跨文档矛盾、银卡折扣)如实入锚——
数据只读,测试描述现状,不为指标改数据。"""

from pathlib import Path

from app.rag.chunking import split_markdown

DOCS = Path("knowledge")
FILES = ["after-sales-manual.md", "billing-shipping.md", "member-benefits.md",
         "product-faq.md", "product-specs.md", "returns-policy.md"]


def _drafts(name: str):
    return split_markdown((DOCS / name).read_text(encoding="utf-8"))


def _all_drafts():
    out = []
    for f in sorted(DOCS.glob("*.md")):
        out.extend(split_markdown(f.read_text(encoding="utf-8")))
    return out


def test_six_docs_all_chunkable_and_vector_text_rule():
    for name in FILES:
        assert _drafts(name), f"{name} 切不出块"
    for d in _all_drafts():
        assert d.vector_text() == "\n".join([d.category, d.questions, d.answer])


def test_product_specs_has_13_model_sections_with_lp100_anchor():
    drafts = _drafts("product-specs.md")
    models = [d for d in drafts if "型号 MH-" in d.questions]
    assert len(models) == 13  # LP100/LP50/LP200/W20/W40/W60/CT30/HP12/HP20/FD10/FD30/CAM1/NEST20
    lp100 = [d for d in drafts if "MH-LP100" in d.section_path]
    assert lp100 and "废砂盒" in lp100[0].answer  # 验收2 硬锚(BM25 型号命中靶块)


def test_faq_10yuan_vs_billing_6yuan_conflict_recorded():
    """附录 B-3:老师语料自身跨文档矛盾(faq 收 10 元 vs billing 收 6 元)。
    断言「两者并存在库」而非择一为真;报告评估时如实说明。"""
    faq = [d for d in _drafts("product-faq.md") if d.questions == "运费怎么算"]
    billing = [d for d in _drafts("billing-shipping.md") if "运费与包邮" in d.section_path]
    assert faq and "10 元" in faq[0].answer
    assert billing and "6 元" in billing[0].answer


def test_member_silver_9off_actual_wording_anchored():
    """附录 B-2:老师标注 A22/E9 写银卡 95 折,语料实为银卡 9 折(金卡才 95 折)。
    裁判以语料为准,此锚锁死措辞现状。"""
    silver = [d for d in _drafts("member-benefits.md") if "银卡" in d.answer and "9 折" in d.answer]
    assert silver and "95 折" in silver[0].answer  # 同句「银卡享商品 9 折,金卡享商品 95 折」


def test_aftersale_time_limit_table_header_copied_per_block():
    drafts = [d for d in _drafts("after-sales-manual.md") if "常见问题处理时限" in d.section_path]
    assert drafts
    assert {d.answer.splitlines()[0] for d in drafts} == {"| 问题类型 | 首次响应 | 处理时限 |"}
    assert all(d.answer.splitlines()[1].startswith("| ---") for d in drafts)
```

Run: `uv run pytest tests/test_corpus_ch04.py -q` → PASS(切块器零改动,锚点已对实料逐条核过;若某断言意外红,先 `uv run python -c` 打印实际 draft 核对,**不许改语料**)

- [ ] **Step 2: 删除旧语料测试(4 个基线红随之归零)**

```bash
git rm tests/test_corpus_ch03.py
```

- [ ] **Step 3: `app/rag/indexer.py` docstring 升级 v2**(首行到闭合 `"""` 整块替换)

```python
"""ch03 两段双写、ch04 集合 v2(spec §3/§5)。

Stage1 ingest_docs:默认全量重建(清 MySQL 表 + drop Milvus 集合 → 重灌 pending);
  --skip-existing:不清表,sha1 指纹跳过重复,只追加新块。
Stage2 vectorize_pending:扫 pending → embed_batch → upsert_rows(
  {chunk_id,text,embedding,category,content_type})→ 回填 done。
  text 与 embed 输入同源三格拼接(§3.1);sparse 列由服务端 BM25 Function 生成,客户端不写。
  任一点崩溃:行仍 pending 或已 done 但向量同 pk 可覆写,重跑即自愈——「按主键幂等」。
fault_after:N 之后(批粒度)SystemExit(42),验收 2 的注入。
"""
```

- [ ] **Step 4: 全量重灌(旧 ch03 集合 drop → v2 重建;老师 6 语料入 Milvus)**

```bash
docker compose up -d
uv run python -m app.jobs.build_knowledge
uv run python -m app.jobs.build_knowledge --check
```
Expected: 6 行 `[ingest] *.md: N 块…`;`[vectorize] … done (N/N)`;**抄下最终总块数 N(记入 dev-notes,下一步用)**;`--check` 输出 `pending=0 … 差集=∅ → OK`。

- [ ] **Step 5: 块数实测锚追加到 `tests/test_indexer_integration.py` 末尾**

```python
@pytest.mark.integration
async def test_corpus_block_count_pinned():
    """实测锚(附录 C 重锚定):当前老师语料全量重建后总块数 = EXPECTED_TOTAL。
    语料再被替换时此测试红 → 人工确认后更新数字并在 dev-notes 说明(回归绊线,非可调阈值)。"""
    from app.core.config import get_settings
    from app.db import crud
    from app.db.engine import dispose_engine, get_session_factory, init_engine
    from app.rag import milvus_store

    EXPECTED_TOTAL = 57  # ← Step 4 实测输出 N 替换本行数字(实施时唯一运行时填值,来源已在 Step 4 说明)
    st = get_settings()
    init_engine(st)
    try:
        async with get_session_factory()() as session:
            counts = await crud.count_chunks_by_status(session)
        assert counts.get("done", 0) == EXPECTED_TOTAL and counts.get("pending", 0) == 0
        client = milvus_store.get_client(st.milvus_uri)
        assert milvus_store.count_rows(client, st.milvus_collection) == EXPECTED_TOTAL
    finally:
        await dispose_engine()
```

- [ ] **Step 6: 全量单测归零基线红 + 集成盘点**

Run: `uv run pytest -q` → Expected: **全部 passed、0 failed**(基线 4 红随 test_corpus_ch03 删除收口)
Run: `uv run pytest -m integration -q` → Expected: 本任务相关全绿;**`test_retriever_integration` 允许红**(仍锚 ch03 旧章节路径「运费说明」,T6 重锚定;`test_mine_qa_integration` 若红也记 T15 处理)——把红项与原因记 dev-notes,不顺手改。

- [ ] **Step 7: dev-notes 追记 + 提交**

```bash
git add app/rag/indexer.py tests/test_corpus_ch04.py tests/test_indexer_integration.py dev-notes/ch04.md
git commit -m "feat(ch04): indexer v2文档化+老师6语料全量重灌+语料测试重锚(块数绊线+标注出入如实入锚)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: `query_understanding`(改写+同义扩展,失败降级原话)+ rewrite prompt + 标注样例冒烟

**Files:**
- Create: `app/prompts/query_rewrite.py`
- Create: `app/rag/query_understanding.py`
- Create: `tests/test_query_understanding.py`
- Create: `evals/smoke_query_rewrite.py`(prompt 类任务替代 TDD 的标注样例验证,§9 红线)

**Interfaces:**
- Consumes: `Settings.query_rewrite_enabled`(T1);conftest `FakeChatModel`
- Produces(T6/T11/T12 按这套用):
  - `UnderstandResult(standard_query: str, synonyms: list[str], degraded: bool)` + 属性 `.bm25_text -> str`
  - `parse_rewrite(raw) -> tuple[str, list[str]] | None`(纯函数,不合 schema → None)
  - `async understand_query(query: str, st: Settings, *, model=None) -> UnderstandResult`(异常/超时 → 原话+degraded,永不抛)

- [ ] **Step 1: 写失败测试 `tests/test_query_understanding.py`**

```python
"""parse_rewrite 纯函数 TDD + understand_query 三分支(禁用/成功/降级)。
prompt 输出效果不在单测断言(那是标注样例冒烟的事,§9)。"""

import pytest

from app.core.config import Settings
from app.rag.query_understanding import UnderstandResult, parse_rewrite, understand_query


@pytest.fixture
def st():
    return Settings(_env_file=None, openai_base_url="http://fake/v1",
                    openai_api_key="fake", model_name="fake")


class _Reply:
    def __init__(self, content):
        self.content = content


class OkModel:
    """(PROMPT | model) 链里 model 的最小替身:收到渲染后的消息、回固定文本。"""

    def __init__(self, content):
        self.content = content

    async def ainvoke(self, _inp):
        return _Reply(self.content)


class BoomModel:
    async def ainvoke(self, _inp):
        raise RuntimeError("llm down")


# ---- parse_rewrite 纯函数 ----

def test_parse_rewrite_accepts_dict_and_json_str():
    assert parse_rewrite({"standard_query": "退货运费谁承担", "synonyms": ["运费", "退货"]}) == (
        "退货运费谁承担", ["运费", "退货"])
    assert parse_rewrite('{"standard_query": "a", "synonyms": []}') == ("a", [])


def test_parse_rewrite_strips_code_fence():
    assert parse_rewrite('```json\n{"standard_query": "a", "synonyms": ["x"]}\n```') == ("a", ["x"])


def test_parse_rewrite_cleans_synonyms():
    raw = {"standard_query": "猫粮", "synonyms": ["猫粮", " 主粮 ", "", "主粮", "冻干", "湿粮", "零食", "处方粮"]}
    std, syn = parse_rewrite(raw)
    assert syn == ["主粮", "冻干", "湿粮", "零食"]  # 去重/去空/去等于标准问法/≤4 截断(← 实施 T4 实测修订:清洗后剩 [主粮,冻干,湿粮,零食,处方粮] 5 项,截断只可能裁尾丢「处方粮」;原计划期望值与本节实现及其注释自相矛盾)


def test_parse_rewrite_rejects_garbage():
    assert parse_rewrite("not json at all") is None
    assert parse_rewrite({"standard_query": "  ", "synonyms": []}) is None
    assert parse_rewrite({"standard_query": "a", "synonyms": "notalist"}) is None
    assert parse_rewrite([1, 2]) is None
    assert parse_rewrite(None) is None


# ---- understand_query 三分支 ----

async def test_disabled_returns_verbatim_without_llm_call(st):
    st = st.model_copy(update={"query_rewrite_enabled": False})
    r = await understand_query("包邮吗", st, model=BoomModel())  # 禁用时模型炸与否都到不了它
    assert r == UnderstandResult(standard_query="包邮吗", synonyms=[], degraded=False)


async def test_success_path(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("东西不想要了还能退不", st,
                               model=OkModel('{"standard_query": "退货政策是什么", "synonyms": ["退款", "退换"]}'))
    assert r.standard_query == "退货政策是什么" and r.synonyms == ["退款", "退换"] and not r.degraded
    assert r.bm25_text == "退货政策是什么 退款 退换"


async def test_model_boom_degrades_to_verbatim(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("包邮吗", st, model=BoomModel())
    assert r.standard_query == "包邮吗" and r.synonyms == [] and r.degraded is True


async def test_bad_output_degrades(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("包邮吗", st, model=OkModel("抱歉我改写不了"))
    assert r.degraded is True and r.standard_query == "包邮吗"
```

Run: `uv run pytest tests/test_query_understanding.py -q` → FAIL(module not found)

- [ ] **Step 2: 写 `app/prompts/query_rewrite.py`**

```python
"""ch04 §4.1 查询理解 prompt(教学演示版:约束写全、少示例)。"""

from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商客服知识库的检索查询理解器。把用户的口语问题改写为一句标准问法,并给出至多 4 个检索同义词。
规则:
- standard_query:保留全部关键信息(型号数字/金额/时限一个字不能丢),去掉语气词与口水话,输出单句;
- synonyms:只给实词(名词/型号/术语),不给整句、不给虚词;想不出就输出空数组;
- 同义词只服务关键词召回,不得引入问题里没有的实体或立场;
- 只输出 JSON,不加解释:{{"standard_query": "...", "synonyms": ["..."]}}"""),  # ← 实施 T4 实测修订:ChatPromptTemplate 按 format 模板解析,示例 JSON 的花括号必须 {{ }} 转义,否则渲染即 KeyError;转义后模型可见文本与未转义原文逐字节一致
    ("human", "{question}"),
])
```

- [ ] **Step 3: 写 `app/rag/query_understanding.py`**

```python
"""ch04 查询理解(spec §4.1):一次 LLM 调用 → {标准问法, ≤4 同义词};失败降级原话直检。

同义词只查询侧、不入库拆存(§0-10):dense 腿吃 standard_query,BM25 腿吃 bm25_text。
parse_rewrite 纯函数 TDD;prompt 改写质量走标注样例冒烟(evals/smoke_query_rewrite.py)。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from app.core.config import Settings
from app.prompts.query_rewrite import REWRITE_PROMPT

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SECONDS = 15.0
MAX_SYNONYMS = 4
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```\s*$", re.M)


@dataclass
class UnderstandResult:
    standard_query: str
    synonyms: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def bm25_text(self) -> str:
        return " ".join([self.standard_query, *self.synonyms])


def parse_rewrite(raw) -> tuple[str, list[str]] | None:
    """dict/JSON 字符串皆可;不合 schema → None(调用方降级)。容忍 ```json 围栏。"""
    if isinstance(raw, str):
        try:
            data = json.loads(_FENCE.sub("", raw.strip()))
        except ValueError:
            return None
    else:
        data = raw
    if not isinstance(data, dict):
        return None
    std = data.get("standard_query")
    syns = data.get("synonyms")
    if not isinstance(std, str) or not std.strip() or not isinstance(syns, list):
        return None
    out: list[str] = []
    for s in syns:
        if isinstance(s, str) and (s := s.strip()) and s != std.strip() and s not in out:
            out.append(s)
        if len(out) >= MAX_SYNONYMS:
            break
    return std.strip(), out


async def understand_query(query: str, st: Settings, *, model=None) -> UnderstandResult:
    if not st.query_rewrite_enabled:
        return UnderstandResult(standard_query=query)
    try:
        from app.services.chat_service import get_model  # 延迟 import:rag 层不反向拖 services 依赖

        m = model or get_model(st)
        # ← 实施 T4 实测修订:两段式等价于 REWRITE_PROMPT | m —— installed langchain-core 1.x 的 `|`
        # 要求右端 RunnableLike,会拒测试用最小替身(纯类,非 Runnable,TypeError 被 except 吞成
        # degraded=True → test_success_path 假红);format_messages → ainvoke 与 RunnableSequence
        # 末步一致,评审用 RunnableLambda 对照实验证实模型可见输入完全相同。
        messages = REWRITE_PROMPT.format_messages(question=query)
        resp = await asyncio.wait_for(m.ainvoke(messages), timeout=LLM_TIMEOUT_SECONDS)
        parsed = parse_rewrite(getattr(resp, "content", resp))
        if parsed is None:
            raise ValueError(f"rewrite 输出不合 schema: {str(resp)[:200]}")
        std, syn = parsed
        return UnderstandResult(standard_query=std, synonyms=syn)
    except Exception:  # noqa: BLE001 —— §8:理解层断了宁可退回原话,不许拖垮检索
        logger.warning("query 理解失败,原话直检: %s", query, exc_info=True)
        return UnderstandResult(standard_query=query, degraded=True)
```

Run: `uv run pytest tests/test_query_understanding.py -q` → PASS(8 个全过;← 实施 T4 实测修订:本节逐字测试文件实含 8 个用例「4 parse + 4 understand」,原「7」为计划笔误)

- [ ] **Step 4: 写 `evals/smoke_query_rewrite.py` 并实跑(替代 TDD 的标注样例验证)**

```python
"""查询理解标注样例冒烟(spec §9:prompt 类任务不套 TDD):6 道老师题库原题手工核对改写质量。
uv run python evals/smoke_query_rewrite.py —— 输出逐行 JSON,人工判「关键信息丢没丢、同义词是不是实词」,
结论与样例输出贴进 dev-notes;不达标 → 只改 prompt 再跑,不改断言。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.rag.query_understanding import understand_query  # noqa: E402

SAMPLES = [
    "东西不想要了还能退不",
    "钱退给我要等到什么时候啊",
    "我在新疆下单 80 块钱,运费怎么算,会员的免运费能不能抵",
    "MH-LP100 的废砂盒多久倒一次",
    "夜里十一点客服还在线吗",
    "猫粮拆封了还能退吗",
]


async def main() -> None:
    st = get_settings()
    for q in SAMPLES:
        r = await understand_query(q, st)
        print(json.dumps({"原话": q, "标准问法": r.standard_query,
                          "同义词": r.synonyms, "降级": r.degraded}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
```

Run: `uv run python evals/smoke_query_rewrite.py` → 6 行输出;**人工核对要点:③题的「新疆/80 块/会员免运费」三个关键信息是否全在,④题型号是否原样保留**;输出与结论进 dev-notes。

- [ ] **Step 5: 回归 + 提交**

Run: `uv run pytest -q` → 全绿(T3 后基线)
```bash
git add app/prompts/query_rewrite.py app/rag/query_understanding.py tests/test_query_understanding.py evals/smoke_query_rewrite.py dev-notes/ch04.md
git commit -m "feat(ch04): query_understanding 改写+同义扩展(失败降级原话)+6题标注样例冒烟

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: `reranker`(SiliconFlow /v1/rerank 门面,httpx MockTransport TDD,降级 None)

**Files:**
- Create: `app/rag/reranker.py`
- Create: `tests/test_reranker.py`
- Create: `tests/test_reranker_integration.py`(skipif 无 key)

**Interfaces:**
- Consumes: `Settings.rerank_api_base/rerank_api_key/rerank_model/rerank_timeout_seconds/rerank_top_n`(T1)
- Produces: `async rerank(query: str, texts: list[str], st: Settings, *, transport=None) -> list[tuple[int, float]] | None` —— 按相关性降序的 `(原文下标, relevance_score)`;**无 key / 空 texts / 超时 / HTTP 错 / 响应不合形状 → None**(调用方按 §4.3 降级);`transport` 参数 = 测试注入缝
- ⚠ 核对点③预查结论:请求体字段是 **`documents`**(spec §4.3 预写的 `texts` 按核对点授权修正)、响应 `{"results": [{"index", "relevance_score"}]}`;集成用例就是对这结论的现场销账

- [ ] **Step 1: 写失败测试 `tests/test_reranker.py`**

```python
"""rerank 门面:MockTransport 断言请求形状(documents/top_n/Bearer)+ 五种降级路径 → None。"""

import json

import httpx
import pytest

from app.core.config import Settings
from app.rag import reranker as rr


@pytest.fixture(autouse=True)
def _reset_warn():
    rr._warned = False


def _st(**kw):
    base = dict(openai_base_url="http://fake/v1", openai_api_key="fake", model_name="fake",
                rerank_api_key="sk-test", rerank_api_base="https://api.example.cn/v1")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_success_returns_index_scores_sorted_passthrough():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.42}]})

    out = await rr.rerank("猫砂盆清理", ["a", "b", "c"], _st(), transport=_transport(handler))
    assert out == [(2, 0.91), (0, 0.42)]  # 服务端已降序,门面原样透出
    assert seen["url"] == "https://api.example.cn/v1/rerank"
    assert seen["auth"] == "Bearer sk-test"
    body = seen["body"]
    assert body["model"] == "BAAI/bge-reranker-v2-m3" and body["query"] == "猫砂盆清理"
    assert body["documents"] == ["a", "b", "c"]  # 核对点③:字段名 documents,不是 texts
    assert body["top_n"] == 3 and body["return_documents"] is False  # ← 实施 T5 实测修订:3 候选文档下实现按 min(rerank_top_n, len(texts)) 送 3;原期望 10 与本章实现及姊妹用例 test_top_n_capped_to_candidate_count(2→2)自相矛盾,==3 反而把 min 计算值钉死(丢 min 即红)


async def test_top_n_capped_to_candidate_count():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["top_n"] == 2
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.5},
                                                     {"index": 0, "relevance_score": 0.1}]})

    out = await rr.rerank("q", ["a", "b"], _st(rerank_top_n=10), transport=_transport(handler))
    assert out and out[0][0] == 1


async def test_no_key_returns_none_without_http():
    def handler(request):  # 被调即失败
        raise AssertionError("不该发请求")

    assert await rr.rerank("q", ["a"], _st(rerank_api_key=""), transport=_transport(handler)) is None


async def test_empty_texts_returns_none():
    assert await rr.rerank("q", [], _st(), transport=_transport(lambda r: pytest.fail("不该发"))) is None


@pytest.mark.parametrize("make_resp", [
    lambda: httpx.Response(500, json={"error": "boom"}),
    lambda: httpx.Response(200, json={"no_results": []}),
    lambda: httpx.Response(200, text="not-json"),
])
async def test_bad_responses_degrade_to_none(make_resp):
    out = await rr.rerank("q", ["a"], _st(), transport=_transport(lambda r: make_resp()))
    assert out is None


async def test_timeout_degrades_to_none():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert await rr.rerank("q", ["a"], _st(), transport=_transport(handler)) is None
```

Run: `uv run pytest tests/test_reranker.py -q` → FAIL

- [ ] **Step 2: 写 `app/rag/reranker.py`**

```python
"""SiliconFlow /v1/rerank 门面(spec §4.3)。BAAI/bge-reranker-v2-m3 云托管,httpx 直连零新依赖。

降级语义(§8):无 key/超时/HTTP 错/响应不合形状 → None,调用方用 RRF 前序续跑并跳过闸1;
WARN 每进程只响一次(评估批量跑 300 题时防刷屏)。请求体字段 documents/top_n 形状经 Context7
预核(核对点③),集成用例现场销账。
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

_warned = False


def _warn_degraded(exc: Exception) -> None:
    global _warned
    if not _warned:
        logger.warning("rerank 不可用,降级 RRF 序(WARN 只响一次): %s", exc)
        _warned = True
    else:
        logger.info("rerank 再次失败(已降级): %s", exc)


async def rerank(query: str, texts: list[str], st: Settings, *,
                 transport: httpx.AsyncBaseTransport | None = None
                 ) -> list[tuple[int, float]] | None:
    if not st.rerank_api_key or not texts:
        return None
    try:
        kw = {"transport": transport} if transport else {}
        async with httpx.AsyncClient(timeout=st.rerank_timeout_seconds, **kw) as client:
            resp = await client.post(
                f"{st.rerank_api_base.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {st.rerank_api_key}"},
                json={"model": st.rerank_model, "query": query, "documents": texts,
                      "top_n": min(st.rerank_top_n, len(texts)), "return_documents": False},
            )
            resp.raise_for_status()
            results = resp.json()["results"]
        return [(int(r["index"]), float(r["relevance_score"])) for r in results]
    except Exception as exc:  # noqa: BLE001 —— §8:重排永不炸检索链路
        _warn_degraded(exc)
        return None
```

- [ ] **Step 3: 写 `tests/test_reranker_integration.py`(核对点③现场销账)**

```python
"""真云连通:配了 RERANK_API_KEY 才跑。顺序合理性(相关文档 index 排前)即 API 形状确认。"""

import pytest

from app.core.config import get_settings
from app.rag.reranker import rerank

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="未配置 RERANK_API_KEY,跳过云连通")
async def test_siliconflow_rerank_live():
    st = get_settings()
    out = await rerank(
        "猫砂盆的废砂盒多久清理一次",
        ["智能猫砂盆 Pro(MH-LP100)废砂盒建议 5 至 7 天清理一次,集尘袋容量 8L",
         "全景看护摄像头支持 360 度云台与夜视",
         "退货政策:签收后 7 天内无理由退货"],
        st,
    )
    assert out, "云 API 未返回结果(检查 key/模型名)"
    assert out[0][0] == 0, f"最相关文档应排第一,实际 {out}"
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in out)
    print(f"[rerank-live] {out}")  # 形状与分值量级记入 dev-notes(闸1 阈值初值 0.3 的依据)
```

- [ ] **Step 4: 单测全绿;集成若有 key 实跑并把 `[rerank-live]` 输出行记 dev-notes(核对点③销账+0.3 初值依据);无 key → 红着不行、skip 可接受**

Run: `uv run pytest tests/test_reranker.py -q` → PASS(8 个;← 实施 T5 实测修订:逐字文件实含 8 用例,原「7」与 T4 节同族笔误)
Run: `uv run pytest -m integration tests/test_reranker_integration.py -q` → PASS 或 SKIPPED

- [ ] **Step 5: 回归 + 提交**

Run: `uv run pytest -q` → 全绿
```bash
git add app/rag/reranker.py tests/test_reranker.py tests/test_reranker_integration.py dev-notes/ch04.md
git commit -m "feat(ch04): reranker SiliconFlow /v1/rerank 门面(httpx MockTransport TDD+降级None+云连通销核对点③)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: `retriever` v2——策略化四臂 + 闸1 + expr 白名单 + head_tail 纯函数 + 集成重锚

**Files:**
- Rewrite: `app/rag/retriever.py`
- Rewrite: `tests/test_retriever.py`
- Rewrite: `tests/test_retriever_integration.py`(T3 允许红的旧锚在此收口)

**Interfaces:**
- Consumes: T2 门面五函数、T4 `UnderstandResult/understand_query`、T5 `reranker.rerank`、`crud.fetch_chunks_by_ids`(ch03 已有)
- Produces(T7/T11/T12 全按这套):
  - `retrieve(query, *, strategy="hybrid_rerank", category=None, settings=None, understood=None) -> RetrieveResult`
  - `RetrieveResult(chunks: list[ScoredRow], refused: bool = False, note: str = "")`
  - `ScoredRow(chunk_id: int, score: float, row)`(row=KnowledgeChunk ORM,`.section_path/.questions/.answer/.category` 可用)
  - `head_tail_indices(n) -> list[int]`、`apply_head_tail(items) -> list`、`vector_text(row) -> str`
  - `build_category_expr(category|None) -> str|None`(白名单外 ValueError)
  - 临时兼容 shim `retrieve_hits(query, settings=None)`(**T7 删除**,本任务保留以不破 definitions import)

- [ ] **Step 1: 重写 `tests/test_retriever.py`(先失败)**

```python
"""retriever v2 单测:假腿/假回查/假重排,断言策略分派、阈值作用域、闸1、降级、排布纯函数。"""

from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.rag import retriever as r
from app.rag.query_understanding import UnderstandResult

ST = Settings(_env_file=None, openai_base_url="http://f/v1", openai_api_key="f", model_name="f")
U = UnderstandResult(standard_query="标准问", synonyms=["同1", "同2"])


def _row(i):
    return SimpleNamespace(id=i, category=f"c{i}", questions=f"q{i}\nq{i}b", answer=f"a{i}",
                           section_path=f"手册 > 节{i}", content_type="policy")


@pytest.fixture
def legs(monkeypatch):
    calls = {}
    monkeypatch.setattr(r, "_get_client", lambda st: object())

    async def fake_embed(text, st):
        calls["embed"] = text
        return [0.1, 0.2]

    async def fake_fetch(ids):
        return [_row(i) for i in ids]

    monkeypatch.setattr(r, "_embed", fake_embed)
    monkeypatch.setattr(r, "_fetch_rows", fake_fetch)

    def dense(c, st, vec, expr):
        calls["dense"] = (vec, expr)
        return [(1, 0.9), (2, 0.2)]

    def bm25(c, st, text, expr):
        calls["bm25"] = (text, expr)
        return [(2, 5.0), (3, 1.0)]

    def hyb(c, st, vec, text, expr):
        calls["hyb"] = (vec, text, expr)
        return [(3, 0.03), (1, 0.02), (2, 0.01)]

    monkeypatch.setattr(r, "_dense_sync", dense)
    monkeypatch.setattr(r, "_bm25_sync", bm25)
    monkeypatch.setattr(r, "_hybrid_sync", hyb)

    async def no_rerank(query, texts, st):
        calls["rerank_texts"] = texts
        return None

    monkeypatch.setattr(r.reranker, "rerank", no_rerank)
    return calls


async def test_dense_arm_uses_threshold_only_here(legs):
    res = await r.retrieve("q", strategy="dense", settings=ST, understood=U)
    assert legs["embed"] == "标准问"
    assert [c.chunk_id for c in res.chunks] == [1]  # 0.2 < 0.3 被纯 dense 腿阈值砍掉
    assert not res.refused


async def test_bm25_arm_sends_built_text_no_threshold(legs):
    res = await r.retrieve("q", strategy="bm25", settings=ST, understood=U)
    assert legs["bm25"][0] == "标准问 同1 同2"
    assert [c.chunk_id for c in res.chunks] == [2, 3]  # 低分不砍:bm25 分数量级与 COSINE 无关


async def test_hybrid_arm_legs_and_no_refusal(legs):
    res = await r.retrieve("q", strategy="hybrid", settings=ST, understood=U)
    assert legs["hyb"] == ([0.1, 0.2], "标准问 同1 同2", None)
    assert [c.chunk_id for c in res.chunks] == [3, 1, 2]
    assert not res.refused


async def test_hybrid_rerank_degrades_to_rrf_when_rerank_none(legs):
    res = await r.retrieve("q", settings=ST, understood=U)  # legs 里 rerank → None
    assert [c.chunk_id for c in res.chunks] == [3, 1, 2]
    assert not res.refused and res.note == ""
    assert legs["rerank_texts"][0] == "c3\nq3\nq3b\na3"  # 重排候选文本 = 三格拼接(vector_text 同源)


async def test_hybrid_rerank_success_orders_by_rerank(legs, monkeypatch):
    async def ok_rerank(query, texts, st):
        return [(2, 0.95), (0, 0.80)]  # cands=[3,1,2] → id2 最相关、id3 次之

    monkeypatch.setattr(r.reranker, "rerank", ok_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert [(c.chunk_id, round(c.score, 2)) for c in res.chunks] == [(2, 0.95), (3, 0.8)]
    assert not res.refused


async def test_gate1_low_top1_refuses(legs, monkeypatch):
    async def weak_rerank(query, texts, st):
        return [(1, 0.05)]

    monkeypatch.setattr(r.reranker, "rerank", weak_rerank)
    res = await r.retrieve("q", settings=ST, understood=U)
    assert res.refused and res.chunks == [] and "置信度不足" in res.note


async def test_gate1_zero_hits_refuses(legs, monkeypatch):
    monkeypatch.setattr(r, "_hybrid_sync", lambda c, st, vec, text, expr: [])
    res = await r.retrieve("q", settings=ST, understood=U)
    assert res.refused and "无命中" in res.note


async def test_category_expr_flows_into_all_legs(legs):
    await r.retrieve("q", strategy="hybrid", category="退货政策", settings=ST, understood=U)
    assert legs["hyb"][2] == 'category == "退货政策"'


@pytest.mark.parametrize("bad", ['x" or 1=1', "a;b", "drop table", "括号（）测", ""])
def test_build_category_expr_whitelist(bad):
    assert r.build_category_expr(None) is None
    assert r.build_category_expr("商品参数") == 'category == "商品参数"'
    with pytest.raises(ValueError):
        r.build_category_expr(bad)


def test_head_tail_indices_pure():
    assert r.head_tail_indices(10) == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]
    assert r.head_tail_indices(0) == [] and r.head_tail_indices(1) == [1]
    assert r.head_tail_indices(4) == [1, 3, 4, 2]
    assert r.apply_head_tail(list("abcdef")) == ["a", "c", "e", "f", "d", "b"]
    assert sorted(map(str, r.apply_head_tail([i for i in range(1, 11)]))) == sorted(map(str, range(1, 11)))


async def test_unknown_strategy_raises(legs):
    with pytest.raises(ValueError):
        await r.retrieve("q", strategy="magic", settings=ST, understood=U)
```

Run: `uv run pytest tests/test_retriever.py -q` → FAIL
(注:degrade 用例里那行带 `and True or True` 的弱断言**不要求实现**,实施时直接删掉它、只保留下一行 `rerank_texts` 强断言。)

- [ ] **Step 2: 重写 `app/rag/retriever.py`**(全文替换)

```python
"""ch04 在线检索核心(spec §4/§5):query 理解 → 策略化召回(四臂同一代码路径)→ MySQL 回查 → 重排 → 闸1。

strategy 是唯一开关:评估四臂=在线一路,差异只在配置——rag_score_threshold 只作用纯 dense 腿
(§4.2-3);闸1 只在 hybrid_rerank 且重排成功时生效(§4.3)。
集合 v2 起 Milvus 带 text(BM25 服务端分词必需,§0-7),原文权威仍是 MySQL:命中 id 回查整行,
孤儿向量被回查天然过滤(ch03 语义)。
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings, get_settings
from app.db import crud
from app.db.engine import get_session_factory
from app.rag import milvus_store, reranker
from app.rag.embeddings import EmbeddingClient, build_embeddings
from app.rag.query_understanding import UnderstandResult, understand_query

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
_clients: dict[str, Any] = {}  # uri → MilvusClient 进程级缓存(grpc 通道不宜每请求新建,ch03 同款)

# §4.2-4 白名单:品类值实测若含全角括号致集成红,只允许加「（）」两个字符并记 dev-notes,其余照 spec 原样
_ALLOWED_FILTER = re.compile(r"^[\w一-鿿 >()×/,-]+$")


def build_category_expr(category: str | None) -> str | None:
    """白名单校验通过才拼等值 expr——引号/反斜杠进不来,表达式不可注入。"""
    if category is None:
        return None
    if not _ALLOWED_FILTER.match(category):
        raise ValueError(f"非法 category 过滤值: {category!r}")
    return f'category == "{category}"'


@dataclass
class ScoredRow:
    chunk_id: int
    score: float
    row: Any  # KnowledgeChunk ORM(原文权威在 MySQL;score 语义随策略:COSINE/BM25/RRF/relevance)


@dataclass
class RetrieveResult:
    chunks: list[ScoredRow]  # 相关性降序(重排/RRF 序),未过首尾排布
    refused: bool = False
    note: str = ""


def _get_client(st: Settings):
    client = _clients.get(st.milvus_uri)
    if client is None:
        client = _clients[st.milvus_uri] = milvus_store.get_client(st.milvus_uri, timeout=5.0)
    if not client.has_collection(st.milvus_collection):
        raise RuntimeError(f"知识库集合 {st.milvus_collection} 不存在,先跑 python -m app.jobs.build_knowledge")
    return client


def vector_text(row) -> str:
    """与 embed 输入、集合 text 列同源三格拼接(§3.1)。"""
    return f"{row.category}\n{row.questions}\n{row.answer}"


def _dense_sync(client, st: Settings, vec: list[float], expr: str | None):
    return milvus_store.search_vectors(client, st.milvus_collection, vec, st.hybrid_recall_k, expr=expr)


def _bm25_sync(client, st: Settings, text: str, expr: str | None):
    return milvus_store.bm25_search(client, st.milvus_collection, text, st.hybrid_recall_k, expr=expr)


def _hybrid_sync(client, st: Settings, vec, text, expr):
    return milvus_store.hybrid_search(client, st.milvus_collection, vec, text,
                                      limit=st.hybrid_recall_k, recall_k=st.hybrid_recall_k,
                                      rrf_k=st.rrf_k, expr=expr)


async def _embed(text: str, st: Settings) -> list[float]:
    return (await EmbeddingClient(build_embeddings(st)).embed_texts([text]))[0]


async def _fetch_rows(ids: list[int]) -> list[Any]:
    async with get_session_factory()() as session:
        return await crud.fetch_chunks_by_ids(session, ids)


async def retrieve(query: str, *, strategy: str = "hybrid_rerank", category: str | None = None,
                   settings: Settings | None = None,
                   understood: UnderstandResult | None = None) -> RetrieveResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略: {strategy}")
    st = settings or get_settings()
    expr = build_category_expr(category)
    if understood is None:
        understood = await understand_query(query, st)
    client = _get_client(st)
    pairs: list[tuple[int, float]] = []
    if strategy == "bm25":
        pairs = await asyncio.to_thread(_bm25_sync, client, st, understood.bm25_text, expr)
    elif strategy == "dense":
        vec = await _embed(understood.standard_query, st)
        pairs = await asyncio.to_thread(_dense_sync, client, st, vec, expr)
        pairs = [(cid, s) for cid, s in pairs if s >= st.rag_score_threshold]  # §4.2-3 只作用这条腿
    else:
        vec = await _embed(understood.standard_query, st)
        pairs = await asyncio.to_thread(_hybrid_sync, client, st, vec, understood.bm25_text, expr)
        if strategy == "hybrid_rerank":
            return await _rerank_stage(st, pairs, understood)
    return await _attach(st, pairs)


async def _attach(st: Settings, pairs) -> RetrieveResult:
    if not pairs:
        return RetrieveResult(chunks=[])
    rows = {r.id: r for r in await _fetch_rows([cid for cid, _ in pairs])}
    chunks = [ScoredRow(cid, s, rows[cid]) for cid, s in pairs if cid in rows]
    return RetrieveResult(chunks=chunks)


async def _rerank_stage(st: Settings, pairs, understood: UnderstandResult) -> RetrieveResult:
    res = await _attach(st, pairs)
    if not res.chunks:
        return RetrieveResult(chunks=[], refused=True, note="知识库无命中")
    cands = res.chunks[: st.hybrid_recall_k]
    scores = await reranker.rerank(understood.standard_query,
                                   [vector_text(c.row) for c in cands], st)
    if scores is None:  # §4.3-3:重排不可用 → RRF 前 N 序、闸1 跳过(WARN 已在 reranker 内,note 不外泄运维噪声)
        return RetrieveResult(chunks=cands[: st.rerank_top_n])
    reranked = [ScoredRow(cands[i].chunk_id, s, cands[i].row)
                for i, s in scores if 0 <= i < len(cands)][: st.rerank_top_n]
    if not reranked:
        return RetrieveResult(chunks=[], refused=True, note="知识库无命中")
    top1 = reranked[0].score
    if top1 < st.retrieval_low_conf_threshold:  # 闸1(§5.2)
        return RetrieveResult(chunks=[], refused=True,
                              note=f"证据置信度不足(top1={top1:.2f} < 阈值 {st.retrieval_low_conf_threshold})")
    return RetrieveResult(chunks=reranked)


# ---- §4.4 首尾排布纯函数(query_faq 与评估共用,[n] = 排布后 1-based 位置) ----

def head_tail_indices(n: int) -> list[int]:
    """奇数升序铺前段、偶数降序铺后段:n=10 → [1,3,5,7,9,10,8,6,4,2],首尾最相关。"""
    odds = list(range(1, n + 1, 2))
    evens = list(range(n if n % 2 == 0 else n - 1, 1, -2))
    return odds + evens


def apply_head_tail(items: list) -> list:
    return [items[i - 1] for i in head_tail_indices(len(items))]


async def retrieve_hits(query: str, settings: Settings | None = None) -> list:
    """ch03 兼容 shim(返回 ORM 行列表):T7 切 query_faq v2 后即删,勿增新调用方。"""
    res = await retrieve(query, strategy="hybrid", settings=settings)
    return [c.row for c in res.chunks]
```

Run: `uv run pytest tests/test_retriever.py -q` → PASS

- [ ] **Step 3: 重写 `tests/test_retriever_integration.py`(旧「运费说明」锚 → 老师语料「运费族」锚)**

```python
"""检索在线集成:需要 T3 全量重灌完成的 knowledge 集合 + 已配 embedding key。
(混合检索臂不依赖 rerank key:无 key 走 RRF 降级序,断言用 bm25/hybrid 腿不受影响。)"""

import asyncio

import pytest

from app.core.config import get_settings
from app.rag import milvus_store, retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", autouse=True)
def ensure_knowledge_built():
    """库空则现场全量重建(自举;CI 单跑此文件也成立)。"""
    st = get_settings()
    client = milvus_store.get_client(st.milvus_uri)
    if client.has_collection(st.milvus_collection) and milvus_store.count_rows(client, st.milvus_collection) > 0:
        return
    from app.jobs import build_knowledge as bk

    asyncio.run(bk.main([]))  # 若 main 签名不带 argv,改为 asyncio.run 两段:ingest_docs+vectorize_pending


async def test_bm25_arm_hits_model_number():
    """验收2 地基:型号题 BM25 腿 top-1 就是该型号节。"""
    res = await retriever.retrieve("MH-LP100 的废砂盒多久倒一次", strategy="bm25")
    assert res.chunks and "MH-LP100" in res.chunks[0].row.section_path


async def test_hybrid_ship_family_top3():
    """口语「邮费」经 hybrid 腿命中运费族章节(ch03 锚「运费说明」→ 老师语料族)。"""
    res = await retriever.retrieve("邮费是多少", strategy="hybrid")
    assert any("运费" in c.row.section_path for c in res.chunks[:3])


async def test_category_filter_restricts_results():
    res = await retriever.retrieve("MH-LP100", strategy="hybrid", category="商品规格手册")
    assert res.chunks, "该品类下应命中型号内容"
    assert all(c.row.category == "商品规格手册" for c in res.chunks)


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="闸1 需 rerank 分数,无 key 时降级路径不判拒答")
async def test_gate1_refuses_absent_topic():
    """验收4(检索侧):知识库没有的问题 → 闸1 refused+note。"""
    res = await retriever.retrieve("喵星人太空电梯门票多少钱")
    assert res.refused and res.chunks == [] and res.note
```

`ensure_knowledge_built` 里 `bk.main` 的确切入口以 `app/jobs/build_knowledge.py` 现有 CLI 函数为准(实施者先读该文件,用其真实入口或直接 `ingest_docs+vectorize_pending` 两段 await;两分支都写在注释里是刻意的,选定后删另一支)。

- [ ] **Step 4: 全量回归 + 集成**

Run: `uv run pytest -q` → 全绿(definitions 仍走 shim)
Run: `uv run pytest -m integration -q` → 除 mine_qa(记 T15)外全绿;**若 category 测试因全角括号白名单红,按 retriever 内注释执行唯一允许改法并记 dev-notes**

- [ ] **Step 5: 提交 + dev-notes**

```bash
git add app/rag/retriever.py tests/test_retriever.py tests/test_retriever_integration.py dev-notes/ch04.md
git commit -m "feat(ch04): retriever v2 策略化四臂+闸1+expr白名单+head_tail纯函数,集成锚换老师语料运费族/MH-LP100

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 7: `query_faq` 契约 v2 + 低置信池写入 + citations 上帧(executor/ToolResultEvent/routes)+ 受影响测试适配

**Files:**
- Create: `app/rag/hit_format.py`、`app/services/refusals.py`
- Modify: `app/tools/definitions.py`、`app/tools/executor.py`、`app/schemas/chat.py`、`app/services/tool_chat_service.py`(仅 tool_result payload)、`app/api/routes.py`、`app/db/crud.py`、`app/rag/retriever.py`(删 shim)
- Test: Create `tests/test_hit_format.py`、`tests/test_refusals.py`、`tests/test_crud_ch04.py`、`tests/test_executor_citations.py`、`tests/test_pool_refusal_integration.py`;Modify `tests/test_tools.py`(两 query_faq 用例+schema 断言)、`tests/test_schemas_ch02.py`(帧形状)、`tests/test_routes_ch02.py`(新增 citations 帧用例)

**Interfaces:**
- Consumes: T6 `retrieve/apply_head_tail`、T1 `LowConfidenceQuestion`、`get_session_factory`
- Produces:
  - `hit_format.format_hits(chunks: list[ScoredRow]) -> list[dict]`(hits v2 元素 `{n,id,question,answer,category,section_path}`,question=首行)
  - `hit_format.build_citations(hits: list[dict]) -> list[dict]`(`{n,chunk_id,section_path,question,answer}`;无 `n` 键的旧形状 → 空列表)
  - `refusals.REFUSAL_ANSWER: str`(固定拒答文案,T8 编排层与 T12 评估共用)、`async pool_low_confidence(conversation_id, raw_question, source, reason) -> None`(自开 session、任何异常 WARN 吞掉)
  - `crud.add_low_confidence_question(session, *, conversation_id, raw_question, source, reason) -> None`
  - query_faq 返回 v2:`{"keyword","hits":[…],"refused":bool,"note":str}`;refused 时工具侧已落池
  - `ToolOutcome` 尾部新字段 `citations: list | None = None`;tool_result 帧仅在其非空时多 `citations` 键

- [ ] **Step 1: 写失败测试 `tests/test_hit_format.py`**

```python
from types import SimpleNamespace

from app.rag.hit_format import build_citations, format_hits
from app.rag.retriever import ScoredRow


def _row(i):
    return SimpleNamespace(questions=f"q{i}首行\nq{i}次行", answer=f"a{i}", category=f"c{i}",
                           section_path=f"手册 > 节{i}")


def test_format_hits_numbers_and_first_line():
    hits = format_hits([ScoredRow(7, 0.9, _row(7)), ScoredRow(9, 0.5, _row(9))])
    assert hits[0] == {"n": 1, "id": 7, "question": "q7首行", "answer": "a7",
                       "category": "c7", "section_path": "手册 > 节7"}
    assert hits[1]["n"] == 2 and hits[1]["question"] == "q9首行"


def test_build_citations_maps_and_filters_legacy():
    hits = format_hits([ScoredRow(7, 0.9, _row(7))]) + [{"id": 1, "question": "旧", "answer": "旧", "category": "旧"}]
    cites = build_citations(hits)
    assert cites == [{"n": 1, "chunk_id": 7, "section_path": "手册 > 节7", "question": "q7首行", "answer": "a7"}]
    assert build_citations([]) == []
```

Run → FAIL。**实现 `app/rag/hit_format.py`:**

```python
"""命中组装纯函数(spec §5.1):query_faq / T12 评估 runner 共用一份,契约单源。"""

from __future__ import annotations

from typing import Any


def format_hits(chunks) -> list[dict]:
    """ScoredRow 列表(已过首尾排布)→ hits v2;n = 列表 1-based 位置(§4.4 [n] 语义)。"""
    return [
        {"n": i + 1, "id": c.chunk_id, "question": c.row.questions.splitlines()[0],
         "answer": c.row.answer, "category": c.row.category,
         "section_path": c.row.section_path or ""}
        for i, c in enumerate(chunks)
    ]


def build_citations(hits: list[dict]) -> list[dict]:
    """hits → citations(台账与原文弹窗回放集,answer 全文随带,附录 A 注)。"""
    return [
        {"n": h["n"], "chunk_id": h["id"], "section_path": h.get("section_path", ""),
         "question": h.get("question", ""), "answer": h.get("answer", "")}
        for h in hits if "n" in h
    ]
```

- [ ] **Step 2: 写失败测试 `tests/test_refusals.py` + `tests/test_crud_ch04.py`**

`test_refusals.py`:

```python
"""池写入永不阻断拒答:成功转发 / 引擎未初始化 WARN 吞掉(spec §8)。"""

from app.services import refusals


async def test_pool_forwards_to_crud(monkeypatch):
    seen = {}

    class FakeSession:
        def __init__(self):
            self.committed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def commit(self):
            self.committed = True

    async def fake_add(session, **kw):
        seen.update(kw)
        assert session.committed is False  # crud 里才 commit,这里传的是 session
    monkeypatch.setattr(refusals.crud, "add_low_confidence_question", fake_add)
    monkeypatch.setattr(refusals, "get_session_factory", lambda: (lambda: FakeSession()))
    await refusals.pool_low_confidence(5, "退货运费谁出", "retrieval_low_conf", "note-x")
    assert seen == {"conversation_id": 5, "raw_question": "退货运费谁出",
                    "source": "retrieval_low_conf", "reason": "note-x"}


async def test_pool_swallows_engine_uninitialized():
    # 单测环境引擎没 init → get_session_factory 抛 RuntimeError → 必须被吞
    await refusals.pool_low_confidence(None, "问题", "self_check", "r")  # 不抛即过
```

`test_crud_ch04.py`(仿 test_crud_ch03 的 FakeSession 惯例):

```python
"""池写入 crud:一行 add + 一次 commit。"""

import pytest

from app.db import crud


class FakeSession:
    def __init__(self):
        self.added, self.commits = [], 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


@pytest.mark.asyncio
async def test_add_low_confidence_question():
    from app.db.models import LowConfidenceQuestion

    s = FakeSession()
    await crud.add_low_confidence_question(s, conversation_id=None, raw_question="q?",
                                           source="user_feedback", reason=None)
    assert s.commits == 1 and len(s.added) == 1
    row = s.added[0]
    assert isinstance(row, LowConfidenceQuestion) and row.source == "user_feedback"
```

Run → FAIL。**实现** `app/services/refusals.py`:

```python
"""统一拒答出口与低置信池写入(spec §5.2/§8)。写池失败只 WARN 不阻断(与 persister 同风格)。"""

from __future__ import annotations

import logging

from app.db import crud
from app.db.engine import get_session_factory

logger = logging.getLogger(__name__)

REFUSAL_ANSWER = (
    "抱歉喵,这个问题我在知识库里没有找到足够可靠的依据,不能凭空作答。"
    "您可以换个说法再问一次,或者让我帮您创建人工工单,由客服跟进处理。"
)


async def pool_low_confidence(conversation_id: int | None, raw_question: str,
                              source: str, reason: str) -> None:
    try:
        async with get_session_factory()() as session:
            await crud.add_low_confidence_question(
                session, conversation_id=conversation_id, raw_question=raw_question,
                source=source, reason=reason,
            )
    except Exception:  # noqa: BLE001 —— 池是复盘材料,丢一行不配打断用户
        logger.warning("低置信池写入失败(不阻断拒答): %s", raw_question, exc_info=True)
```

`app/db/crud.py` 末尾追加:

```python
async def add_low_confidence_question(session, *, conversation_id: int | None,
                                      raw_question: str, source: str,
                                      reason: str | None) -> None:
    """低置信问题池;conversation_id 可空(评估 runner 无会话)。"""
    session.add(LowConfidenceQuestion(conversation_id=conversation_id, raw_question=raw_question,
                                      source=source, reason=reason))
    await session.commit()
```

(models import 区补 `LowConfidenceQuestion`——该文件从 `app.db.models` 的具体 import 列表按现状追加。)

- [ ] **Step 3: `definitions.py` 切 v2(整块替换 query_faq 与 import 区)**

import 区:`from app.rag.retriever import retrieve_hits` 一行替换为:

```python
from app.rag.hit_format import format_hits
from app.rag.retriever import apply_head_tail, retrieve
from app.services.refusals import pool_low_confidence
```

模块 docstring 第 2 行 `query_faq 走向量语义检索(app/rag/retriever.py,ch03)` 改为 `query_faq 走混合检索+重排(app/rag/retriever.py,ch04)`。

query_faq 整函数(现文件 79-89 行)替换为:

```python
@tool
async def query_faq(keyword: str, config: RunnableConfig) -> dict:
    """语义检索平台知识库,回答规则、政策、费用与商品使用类问题(退换货政策、运费与包邮门槛、售后流程、积分等)。用户咨询任何平台规则、政策、费用、商品用法类问题时,必须先调用本工具再作答,即使你认为自己知道通用答案。keyword: 用户的原始问题完整句子(语义检索按整句匹配,请勿自行拆词)。返回的 hits 按相关性首尾排布:[1] 与末位最相关,回答引用时用其 n 编号;refused=true 表示证据不足,此时必须拒答不得编造。"""
    res = await retrieve(keyword)
    if res.refused:
        conversation_id = (config.get("configurable") or {}).get("conversation_id")
        await pool_low_confidence(conversation_id, keyword, "retrieval_low_conf", res.note)
        return {"keyword": keyword, "hits": [], "refused": True, "note": res.note}
    return {"keyword": keyword, "hits": format_hits(apply_head_tail(res.chunks)),
            "refused": False, "note": ""}
```

- [ ] **Step 4: `tests/test_tools.py` 两 query_faq 用例替换 + 工具 schema 断言新增**

删 `test_query_faq_hits`/`test_query_faq_miss_shape` 两函数,替换为:

```python
async def test_query_faq_contract_v2(monkeypatch):
    """契约 v2:refused/note 顶层键、hits 带 n 与 section_path、首尾排布生效、question=首行。"""
    from types import SimpleNamespace

    from app.rag.retriever import RetrieveResult, ScoredRow
    from app.tools import definitions as d

    def row(i):
        return SimpleNamespace(id=i, category="c", questions=f"q{i}a\nq{i}b", answer=f"a{i}",
                               section_path=f"手册 > 节{i}", content_type="policy")

    chunks = [ScoredRow(i, 1.0 - i / 10, row(i)) for i in range(1, 11)]

    async def fake_retrieve(query, **kw):
        assert query == "幼猫喂几次"
        return RetrieveResult(chunks=chunks)

    monkeypatch.setattr(d, "retrieve", fake_retrieve)
    out = await d.query_faq.ainvoke({"keyword": "幼猫喂几次"})
    assert out["refused"] is False and out["note"] == "" and len(out["hits"]) == 10
    assert [h["id"] for h in out["hits"]] == [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]  # §4.4 排布
    assert [h["n"] for h in out["hits"]] == list(range(1, 11))
    assert out["hits"][0]["question"] == "q1a" and out["hits"][0]["section_path"] == "手册 > 节1"


async def test_query_faq_refused_pools_and_keeps_shape(monkeypatch):
    from app.rag.retriever import RetrieveResult
    from app.tools import definitions as d

    seen = {}

    async def fake_retrieve(query, **kw):
        return RetrieveResult(chunks=[], refused=True, note="证据置信度不足(top1=0.05 < 阈值 0.3)")

    async def fake_pool(cid, q, source, reason):
        seen.update(cid=cid, q=q, source=source)

    monkeypatch.setattr(d, "retrieve", fake_retrieve)
    monkeypatch.setattr(d, "pool_low_confidence", fake_pool)
    out = await d.query_faq.ainvoke({"keyword": "太空电梯门票"},
                                    config={"configurable": {"conversation_id": 42}})
    assert out == {"keyword": "太空电梯门票", "hits": [], "refused": True,
                   "note": "证据置信度不足(top1=0.05 < 阈值 0.3)"}
    assert seen == {"cid": 42, "q": "太空电梯门票", "source": "retrieval_low_conf"}


def test_query_faq_visible_args_still_only_keyword():
    """核对点④复核:config: RunnableConfig 不进模型可见 schema,签名兼容红线。"""
    from app.tools.registry import get_tool

    assert sorted(get_tool("query_faq").args.keys()) == ["keyword"]
```

Run: `uv run pytest tests/test_tools.py -q` → PASS(fake_pool 用例证明工具是唯一池写方,T8 编排层不再重复写)。

- [ ] **Step 5: executor / chat schema / service payload / routes 帧**

`app/tools/executor.py`:import 区加 `from app.rag.hit_format import build_citations`;`ToolOutcome` 尾部加字段:

```python
    citations: list | None = None  # ch04: query_faq 命中集引用(前端弹窗数据);其余工具恒 None
```

成功 return(现 80 行)替换:

```python
            citations = None
            if name == "query_faq" and isinstance(result, dict):
                citations = build_citations(result.get("hits", [])) or None
            return ToolOutcome(name, tool_call_id, True, result,
                               make_summary(name, result), citations=citations)
```

`app/schemas/chat.py` ToolResultEvent 尾部加:

```python
    citations: list[dict] | None = None  # ch04 可选增列;routes 帧 dump 用 exclude_none,无引用时键不出现
```

`app/services/tool_chat_service.py` 现 111-119 行 yield 块替换:

```python
        payload = {
            "id": outcome.tool_call_id,
            "name": outcome.name,
            "ok": outcome.ok,
            "summary": outcome.summary,
        }
        if outcome.citations:  # 仅 query_faq 且有命中时加键,其余帧与旧结构逐字符一致
            payload["citations"] = outcome.citations
        yield ("tool_result", payload)
```

`app/api/routes.py` 现 95 行 dump 替换:

```python
                    data=ToolResultEvent(**payload).model_dump(exclude_none=True), event="tool_result"
```

- [ ] **Step 6: 受影响旧测试适配(逐字给法)**

`tests/test_schemas_ch02.py::test_event_models_match_spec_frame_shape` 里 ToolResultEvent 两条断言替换为:

```python
    assert ToolResultEvent(
        id="call_x", name="query_logistics", ok=True, summary="运输中"
    ).model_dump(exclude_none=True) == {"id": "call_x", "name": "query_logistics", "ok": True, "summary": "运输中"}
    # ch04: citations 可选键——不带时帧形状逐字符不变(兼容红线),带时追加
    assert "citations" not in ToolResultEvent(
        id="c", name="query_faq", ok=True, summary="s").model_dump(exclude_none=True)
    assert ToolResultEvent(id="c", name="query_faq", ok=True, summary="s",
                           citations=[{"n": 1, "chunk_id": 7}]
                           ).model_dump(exclude_none=True)["citations"] == [{"n": 1, "chunk_id": 7}]
```

`tests/test_routes_ch02.py` 末尾新增(旧 `test_tool_frames_order_and_shape` 本体不动——它的 fake outcome citations 缺省 None,exclude_none 保帧逐字旧形):

```python
async def test_tool_result_frame_carries_citations(client, monkeypatch):
    """ch04: query_faq 命中帧带 citations 键;前端弹窗数据链路。"""
    from app.api import routes as routes_mod
    from app.main import app
    from app.services import tool_chat_service as svc
    from app.tools.executor import ToolOutcome

    cites = [{"n": 1, "chunk_id": 7, "section_path": "手册 > 节1", "question": "q", "answer": "a"}]

    async def fake_execute(name, args, tcid, ctx):
        return ToolOutcome(name, tcid, True, {"keyword": "k", "hits": [{"n": 1}], "refused": False, "note": ""},
                           "命中 1 条", citations=cites)

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    r = await client.post("/api/chat/stream",
                          json={"messages": [{"role": "user", "content": "退货政策"}]})
    tr_line = [l for l in r.text.splitlines() if l.startswith("data:") and '"citations"' in l][0]
    assert json.loads(tr_line[len("data:"):].strip())["citations"] == cites
```

(`ScriptedToolModel`/`FakeCrud` 沿用文件内已有 helper;`_override` 与 conversation 帧在该测试里按 test_tool_frames_order_and_shape 同款补齐。)

- [ ] **Step 7: 删 `retriever.py` 末尾 `retrieve_hits` shim + 全链路引用扫描**

```bash
grep -rn "retrieve_hits" app tests evals || echo CLEAN
uv run pytest -q
```
Expected: grep CLEAN(definitions 已在 Step 3 切换);全绿。

- [ ] **Step 8: 池写入+拒答集成(验收4 数据侧证据)**

Create `tests/test_pool_refusal_integration.py`:

```python
"""验收4 集成:无 key 时闸1 不判 → 用例 skipif;有 key:问知识库没有的 → query_faq refused + 池行数 +1。"""

import pytest
from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.engine import dispose_engine, get_engine, init_engine
from app.db.models import LowConfidenceQuestion
from app.tools import definitions as d

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="闸1 依赖 rerank 分数")
async def test_absent_question_refused_and_pooled():
    st = get_settings()
    init_engine(st)
    try:
        async with get_engine().connect() as conn:
            before = (await conn.execute(select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        out = await d.query_faq.ainvoke({"keyword": "请问月球基地的喵星人会员费多少钱一个月"},
                                        config={"configurable": {"conversation_id": None}})
        assert out["refused"] is True and out["hits"] == []
        async with get_engine().connect() as conn:
            after = (await conn.execute(select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        assert after == before + 1
    finally:
        await dispose_engine()
```

Run: `uv run pytest -m integration tests/test_pool_refusal_integration.py -q` → PASS 或 SKIPPED(无 key)

- [ ] **Step 9: 提交 + dev-notes**

```bash
git add app/rag/hit_format.py app/services/refusals.py app/tools/definitions.py app/tools/executor.py app/schemas/chat.py app/services/tool_chat_service.py app/api/routes.py app/db/crud.py app/rag/retriever.py tests/ dev-notes/ch04.md
git commit -m "feat(ch04): query_faq 契约v2+低置信池写入+citations 上帧(exclude_none 保旧帧逐字不变)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 8: 闸2 生成前自评 + 编排层固定拒答出口

**Files:**
- Create: `app/prompts/self_check.py`、`app/services/self_check.py`
- Modify: `app/services/tool_chat_service.py`(import 区 + 工具循环整块替换)
- Test: Create `tests/test_self_check.py`;Modify `tests/test_tool_chat_service.py`(新增 4 用例 + `faq_chunk` helper)、`tests/test_routes_ch02.py`(旧帧序测试注入 fake_check 一行)

**Interfaces:**
- Consumes: T7 `REFUSAL_ANSWER/pool_low_confidence`(经 svc 模块命名空间,测试可 monkeypatch)、hits v2 结构
- Produces:
  - `self_check.EvidenceCheck(BaseModel){sufficient: bool, reason: str = ""}`
  - `async evaluate_evidence(question, hits, settings, *, model=None) -> EvidenceCheck | None`(任何异常/超时 → None = 放行,§0-2)
  - 编排新行为:`query_faq refused → 不进第二轮、REFUSAL_ANSWER 单 token 出口并落库`;`hits 非空且 self_check_enabled 且判不足 → 落池(self_check)+ 同出口`;`self_check_enabled=False → 整闸关闭`

- [ ] **Step 1: 写失败测试 `tests/test_self_check.py`**

```python
"""闸2 判定与降级:成功透出 structured 结果;一切异常 → None(放行)。"""

import pytest

from app.core.config import Settings
from app.services.self_check import EvidenceCheck, evaluate_evidence


@pytest.fixture
def st():
    return Settings(_env_file=None, openai_base_url="http://f/v1", openai_api_key="f", model_name="f")


HITS = [{"n": 1, "id": 5, "question": "退货政策是什么", "answer": "签收后 7 天内无理由退货",
         "category": "退货政策", "section_path": "手册 > 退货"}]


class _M:
    def __init__(self, result=None, boom=False):
        self.result, self.boom, self.seen = result, boom, {}

    def with_structured_output(self, schema):
        outer = self

        class _Chain:
            async def ainvoke(self, inp):
                outer.seen = inp
                if outer.boom:
                    raise RuntimeError("llm down")
                return outer.result

        return _Chain()


async def test_sufficient_verdict_passthrough_with_rendered_evidence(st):
    m = _M(result=EvidenceCheck(sufficient=True, reason="覆盖核心诉求"))
    out = await evaluate_evidence("退货要几天", HITS, st, model=m)
    assert out.sufficient is True
    assert m.seen["question"] == "退货要几天"
    assert "[1] 退货政策是什么" in m.seen["evidence"]  # 证据渲染含编号与问题


async def test_boom_degrades_to_none(st):
    assert await evaluate_evidence("q", HITS, st, model=_M(boom=True)) is None


async def test_long_answer_excerpted(st):
    hits = [{"n": 1, "id": 5, "question": "q", "answer": "长" * 500, "category": "c", "section_path": "s"}]
    m = _M(result=EvidenceCheck(sufficient=True))
    await evaluate_evidence("q", hits, st, model=m)
    assert len(m.seen["evidence"]) < 500  # 证据截断(§5.2:控制自评 token)
```

Run → FAIL。**实现** `app/prompts/self_check.py`:

```python
"""ch04 §5.2 闸2 提示词(判定标准写死「答不中问题=false」,拿不准从严)。"""

from langchain_core.prompts import ChatPromptTemplate

SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是客服回答前的证据充分性审查员。判断「仅凭下列知识库证据,能否直接回答用户问题」。
标准:证据必须覆盖问题的核心诉求;只是话题沾边、答不中问题 → sufficient=false;
部分覆盖也判 false(宁可拒答转人工,不许半编造)。
输出字段:sufficient(布尔)、reason(一句话,说明缺什么,将展示给用户复盘)。"""),
    ("human", "用户问题:{question}\n候选证据:\n{evidence}"),
])
```

`app/services/self_check.py`:

```python
"""闸2:生成前证据自评(spec §5.2)。失败/超时 → None = 放行(§0-2 用户确认的降级方向)。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from app.core.config import Settings
from app.prompts.self_check import SELF_CHECK_PROMPT

logger = logging.getLogger(__name__)

EVIDENCE_TIMEOUT_SECONDS = 15.0
_ANSWER_EXCERPT = 200


class EvidenceCheck(BaseModel):
    sufficient: bool
    reason: str = ""


def _render_hits(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        ans = (h.get("answer") or "").replace("\n", " ")[:_ANSWER_EXCERPT]
        lines.append(f"[{h.get('n')}] {h.get('question', '')} | {h.get('section_path', '')} | {ans}")
    return "\n".join(lines)


async def evaluate_evidence(question: str, hits: list[dict], settings: Settings,
                            *, model: Any | None = None) -> EvidenceCheck | None:
    try:
        from app.services.chat_service import get_model  # 延迟 import,同 query_understanding 惯例

        m = model or get_model(settings)
        chain = SELF_CHECK_PROMPT | m.with_structured_output(EvidenceCheck)
        result = await asyncio.wait_for(
            chain.ainvoke({"question": question, "evidence": _render_hits(hits)}),
            timeout=EVIDENCE_TIMEOUT_SECONDS,
        )
        return EvidenceCheck.model_validate(result)
    except Exception:  # noqa: BLE001 —— §0-2:自评挂了不许拦生成
        logger.warning("证据自评失败,放行生成: %s", question, exc_info=True)
        return None
```

(_render_hits 行格式 `[n] question | section_path | answer` 与测试断言 `"[1] 退货政策是什么" in evidence` 对齐。)

Run: `uv run pytest tests/test_self_check.py -q` → PASS

- [ ] **Step 2: 写失败的编排测试(`tests/test_tool_chat_service.py` 追加)**

helper 与 4 用例(仿文件内 `tc_chunk`/`FakeToolModel`/`FakePersister` 现成件):

```python
def faq_chunk(args_fragment, first=False):
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": "query_faq" if first else None, "args": args_fragment,
                           "id": "call_f" if first else None, "index": 0, "type": "tool_call_chunk"}],
    )


def faq_outcome(refused=False, hits=None, note=""):
    return ToolOutcome(
        "query_faq", "call_f", True,
        {"keyword": "退货款几天到账", "hits": hits or [], "refused": refused, "note": note},
        "未命中" if not hits else f"命中 {len(hits)} 条",
    )


HIT1 = [{"n": 1, "id": 5, "question": "q", "answer": "a", "category": "c", "section_path": "s"}]


async def test_gate1_refused_exits_with_fixed_refusal_no_round2(settings, monkeypatch):
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="不该走到第二轮")],
    )

    async def fake_execute(name, args, tcid, ctx):
        return faq_outcome(refused=True, note="证据置信度不足")

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    p = FakePersister()
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=9, persister=p))
    assert [k for k, _ in events] == ["tool_call", "tool_result", "token"]
    assert events[-1][1] == svc.REFUSAL_ANSWER
    assert model.calls == 1
    assert [c[0] for c in p.calls] == ["tool_calls", "tool_result", "final"]
    assert p.calls[-1][2] == svc.REFUSAL_ANSWER  # 固定文案落库(与 yield 同源,§5.2)


async def test_gate2_insufficient_refuses_and_pools(settings, monkeypatch):
    from app.services.self_check import EvidenceCheck

    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')], round2=[])
    seen = {}

    async def fake_execute(name, args, tcid, ctx):
        return faq_outcome(hits=HIT1)

    async def fake_check(question, hits, st, *, model=None):
        seen["q"] = question
        return EvidenceCheck(sufficient=False, reason="证据只沾边")

    async def fake_pool(cid, q, source, reason):
        seen.update(cid=cid, source=source, reason=reason)

    monkeypatch.setattr(svc, "execute_tool", fake_execute)
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)
    monkeypatch.setattr(svc, "pool_low_confidence", fake_pool)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=11, persister=None))
    assert events[-1] == ("token", svc.REFUSAL_ANSWER)
    assert model.calls == 1  # 不进第二轮
    assert seen == {"q": "退货款几天到账", "cid": 11, "source": "self_check", "reason": "证据只沾边"}


async def test_gate2_verdict_none_passes_through(settings, monkeypatch):
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="最终回答")],
    )
    monkeypatch.setattr(svc, "execute_tool",
                        lambda *a: _ret(faq_outcome(hits=HIT1)))
    async def fake_check(question, hits, st, *, model=None):
        return None  # 自评失败 = 放行(§0-2)
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), settings, model, conversation_id=None, persister=None))
    assert model.calls == 2 and events[-1] == ("token", "最终回答")


async def test_gate2_disabled_skips_check_entirely(settings, monkeypatch):
    st = settings.model_copy(update={"self_check_enabled": False})
    model = FakeToolModel(
        round1=[faq_chunk('{"keyword": ', first=True), faq_chunk('"退货款几天到账"}')],
        round2=[AIMessageChunk(content="最终回答")],
    )
    monkeypatch.setattr(svc, "execute_tool", lambda *a: _ret(faq_outcome(hits=HIT1)))

    async def never_called(*a, **k):
        raise AssertionError("开关关闭时不得触发自评")

    monkeypatch.setattr(svc, "evaluate_evidence", never_called)
    events = await collect(svc.stream_chat_with_tools(
        user_msg("退货款几天到账"), st, model, conversation_id=None, persister=None))
    assert events[-1] == ("token", "最终回答")
```

`_ret` helper(把同步值包成 awaitable,给 monkeypatch 的 lambda 用):

```python
async def _ret(v):
    return v
```

Run: `uv run pytest tests/test_tool_chat_service.py -q` → 新 4 用例 FAIL(出口不存在)

- [ ] **Step 3: `tool_chat_service.py` 接线**

import 区追加:

```python
from app.services.refusals import REFUSAL_ANSWER, pool_low_confidence
from app.services.self_check import evaluate_evidence
```

从 `round2: list = [*messages, full]` 行(现 107 行)到文件尾整块替换:

```python
    round2: list = [*messages, full]  # 第一轮完整 AIMessage（含 tool_calls）回传上游
    faq_refused = False
    faq_hits: list | None = None
    for tc in tool_calls:
        yield ("tool_call", {"id": tc["id"], "name": tc["name"], "args": tc.get("args") or {}})
        outcome = await execute_tool(tc["name"], tc.get("args") or {}, tc["id"], ctx)
        payload = {
            "id": outcome.tool_call_id,
            "name": outcome.name,
            "ok": outcome.ok,
            "summary": outcome.summary,
        }
        if outcome.citations:  # 仅 query_faq 且有命中时加键,其余帧与旧结构逐字符一致
            payload["citations"] = outcome.citations
        yield ("tool_result", payload)
        round2.append(
            ToolMessage(
                content=json.dumps(outcome.result, ensure_ascii=False, default=str),
                tool_call_id=outcome.tool_call_id,
            )
        )
        await _persist(persister, "on_tool_result", conversation_id, outcome)
        if outcome.name == "query_faq" and outcome.ok:
            if outcome.result.get("refused"):
                faq_refused = True
            if outcome.result.get("hits"):
                faq_hits = outcome.result["hits"]

    # ---- 固定拒答出口(spec §5.2):闸1/闸2 任一触发 → 不进第二轮,落库=出口同源 ----
    question = chat_messages[-1].content
    if faq_refused:
        yield ("token", REFUSAL_ANSWER)
        await _persist(persister, "on_final_answer", conversation_id, REFUSAL_ANSWER)
        return
    if faq_hits and settings.self_check_enabled:
        verdict = await evaluate_evidence(question, faq_hits, settings)
        if verdict is not None and not verdict.sufficient:
            await pool_low_confidence(conversation_id, question, "self_check", verdict.reason)
            yield ("token", REFUSAL_ANSWER)
            await _persist(persister, "on_final_answer", conversation_id, REFUSAL_ANSWER)
            return

    # ---- 第二轮：裸 model（不 bind_tools）+ ch01 stream_chat，物理保证单轮收敛 ----
    parts: list[str] = []
    async for t in stream_chat(round2, model):
        parts.append(t)
        yield ("token", t)
    await _persist(persister, "on_final_answer", conversation_id, "".join(parts))
```

(闸1 的池写入在 query_faq 工具内已完成——本出口**不重复落池**,一个写方原则,T7 测试已锁。)

- [ ] **Step 4: `tests/test_routes_ch02.py` 两处注入放行(防真实自评触网)**

在 `test_tool_frames_order_and_shape` 与 `test_tool_result_frame_carries_citations`(T7 新增)**两处**的 `monkeypatch.setattr(svc, "execute_tool", fake_execute)` 行后各插入:

```python
    async def fake_check(question, hits, settings, *, model=None):
        return None  # 闸2 = 不判即放行(降级语义);路由帧断言不触真实自评
    monkeypatch.setattr(svc, "evaluate_evidence", fake_check)
```

(`evaluate_evidence` 设计上永不抛、失败返回 None,不注入也能过——但会真发一次失败 HTTP 重试,拖慢测试,故显式放行。)

- [ ] **Step 5: 全量回归 + 提交**

Run: `uv run pytest -q` → 全绿
```bash
git add app/prompts/self_check.py app/services/self_check.py app/services/tool_chat_service.py tests/test_self_check.py tests/test_tool_chat_service.py tests/test_routes_ch02.py dev-notes/ch04.md
git commit -m "feat(ch04): 闸2 生成前自评+固定拒答出口(失败放行/开关可关),编排四用例锚定

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 9: System Prompt 三改([n] 引用规范 / 禁止承诺清单 / 拒答行为)+ tool_routing eval 回归

**Files:**
- Modify: `app/prompts/customer_service.py`(SYSTEM_PROMPT 整块替换)
- Test: Create `tests/test_prompts_ch04.py`;既有 `tests/test_prompts_ch02.py` **必须不动且保持绿**(锚定词清单兼容)

**Interfaces:**
- Consumes: T7 契约 v2(refused 键与 [n] 编号的语义在 prompt 里被引用)
- Produces: 新 SYSTEM_PROMPT(唯一改动物;工具行为代码零变更)

- [ ] **Step 1: 写失败测试 `tests/test_prompts_ch04.py`**

```python
"""ch04 System Prompt 锚定词(spec §5.3 三改;ch02 锚必须原样保住=兼容红线)。"""

from app.prompts.customer_service import SYSTEM_PROMPT


def test_ch04_new_anchors():
    assert "[n]" in SYSTEM_PROMPT                      # ①引用规范
    assert "refused" in SYSTEM_PROMPT                  # ③拒答行为(对齐契约 v2)
    assert "不承诺" in SYSTEM_PROMPT                    # ②禁止承诺清单
    assert "工作日" in SYSTEM_PROMPT                    # 承诺清单点名到账时效
    assert "转人工" in SYSTEM_PROMPT                    # 拒答收敛去向

def test_ch02_anchors_survive():
    for anchor in ("喵帮", "客服", "订单号", "工具", "严禁编造", "人工工单"):
        assert anchor in SYSTEM_PROMPT
    assert "没有接入任何订单/物流查询系统" not in SYSTEM_PROMPT
```

Run: `uv run pytest tests/test_prompts_ch04.py -q` → FAIL

- [ ] **Step 2: 整块替换 `SYSTEM_PROMPT`**(现文件只此一常量;保留 build_messages 引用路径不动)

```python
SYSTEM_PROMPT = """你是「喵帮」电商平台的智能客服喵喵,负责解答购物、订单、物流、售后相关问题。

行为约束:
1. 只回答与电商购物、订单、物流、售后相关的问题;无关问题(如写作业、闲聊八卦)礼貌拒绝并引回主题。
2. 不越权承诺(硬红线):不承诺退款/补发的具体到账工作日数字,不承诺「一定能退/一定赔」,不给「几天必到」类时效承诺;涉及此类诉求只说明流程并建议转人工客服。
3. 处理售后问题时,主动引导用户提供订单号,以便调用工具查询。
4. 工具使用指引:订单状态、物流轨迹、商品信息、平台政策等具体数据只能来自工具返回结果——涉及这些信息时(包括邮费、寄递费用、包邮门槛等平台规则咨询),即使你认为可以给出通用或条件式回答,也必须先调用相应工具查询再作答,严禁编造工具结果,不得在未调用工具时声称"已查到""已核实"。调用 query_faq 时 keyword 传用户原话的完整句子,语义检索按整句匹配,不要自行拆成关键词。工具无结果或执行失败时,如实告知用户暂时查询不到,并建议创建人工工单(转人工)跟进。
5. 引用规范:凡来自 query_faq 返回证据的事实句,必须在句末以 [n] 标注来源编号,n 只能是本轮工具结果 hits 里实际存在的编号;一句可连引多个(如 [2][7]);没有证据支撑的内容禁止带编号。
6. 证据不足拒答:当 query_faq 返回 refused=true 或 hits 为空时,如实告诉用户知识库里查不到,建议转人工工单,严禁凭常识编造平台规则。
7. 语气友好、简洁,使用中文回答,适当使用「喵」保持品牌风格但不堆砌。"""
```

Run: `uv run pytest tests/test_prompts_ch04.py tests/test_prompts_ch02.py -q` → 双绿

- [ ] **Step 3: 标注样例回归——tool_routing eval 重跑(prompt 改动影响路由行为的验证方式)**

Run: `uv run python evals/run_tool_routing_eval.py`
Expected: 全部样本 PASS(ch02 基线行为不因三改回退);若样本因新增 [n]/拒答措辞导致判定漂移,只允许在 dev-notes 记录并如实报告,**不放宽 eval 判定器**。输出贴 dev-notes。

- [ ] **Step 4: 全量回归 + 提交**

Run: `uv run pytest -q` → 全绿
```bash
git add app/prompts/customer_service.py tests/test_prompts_ch04.py dev-notes/ch04.md
git commit -m "docs(ch04): System Prompt 三改([n]引用规范/禁止承诺清单/证据不足拒答行为)+tool_routing eval 回归

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 10: REST——`/api/chunks/{id}` 原文回查 + `/api/faith_cases` GET/PATCH(台账 upsert 复发流转)

**Files:**
- Create: `app/schemas/knowledge.py`
- Modify: `app/db/crud.py`(末尾追加 5 函数)、`app/api/routes.py`(import 区 + 末尾 3 端点)
- Test: Create `tests/test_knowledge_routes.py`、`tests/test_faith_crud_ch04.py`、`tests/test_ledger_integration.py`

**Interfaces:**
- Consumes: T1 `FaithCase`/`KnowledgeChunk` ORM、`dep_db_session` None 降级惯例
- Produces(T13/T14 前端与 T12 runner 全按这套):
  - `GET /api/chunks/{chunk_id}` → `ChunkOut{id,section_path,category,questions,answer,content_type,prev_chunk_id,next_chunk_id}`;404/503
  - `GET /api/faith_cases?status=&bucket=` → `list[FaithCaseOut]`(last_seen_at 降序)
  - `PATCH /api/faith_cases/{case_id}` body=`FaithCasePatch{status: 未解决|已解决|无需解决, resolution?}`;非「未解决」缺 resolution → **422**;404;「未解决」强制清 resolution
  - `crud.get_chunk / upsert_faith_case / list_faith_cases / get_faith_case / set_faith_case_status`
- **核对点⑤**:写路由测试前 Context7 查 FastAPI 文档确认「请求体模型 model_validator 抛错 → RequestValidationError → 422」(ch02 `ChatRequest` 已有 422 先例,确认即销账);**核对点⑥**:集成用例实测中文 ENUM 读写往返

- [ ] **Step 1: 写失败测试 `tests/test_faith_crud_ch04.py`(FakeSession 纯逻辑:upsert 三态与流转)**

```python
"""台账 upsert 复发流转(spec §3.3):created/updated/reactivated 三态;
复发自动退回未解决+清 resolution;resolved_at 保留;seen_count 递增;uk_eval_id 一题一行。"""

from datetime import datetime

import pytest

from app.db import crud
from app.db.models import FaithCase


class FakeResult:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row


class FakeSession:
    def __init__(self, existing=None, get_row=None):
        self.existing, self.added, self.commits, self._get = existing, [], 0, get_row

    async def execute(self, _stmt):
        return FakeResult(self.existing)

    async def get(self, _model, _cid):  # session.get 是协程,假件必须 async
        return self._get

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


ARGS = dict(eval_id="A22", bucket="A_policy", query="银卡打折吗", answer="打了95折",
            reason="语料为9折", citations=[{"n": 1}], judge_model="m")


async def test_upsert_creates_when_absent():
    s = FakeSession(existing=None)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "created" and s.commits == 1
    row = s.added[0]
    assert row.status == "未解决" and row.seen_count == 1 and row.first_seen_at == row.last_seen_at


async def test_upsert_updates_unresolved_keeps_first_seen():
    old = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="未解决", seen_count=3,
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 2))
    s = FakeSession(existing=old)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "updated" and old.seen_count == 4 and old.first_seen_at == datetime(2026, 1, 1)
    assert old.status == "未解决" and old.query == "银卡打折吗"


async def test_reactivation_resets_status_and_clears_resolution():
    old = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="已解决", seen_count=2, resolution="老师标注有误",
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 2),
                    resolved_at=datetime(2026, 1, 3))
    s = FakeSession(existing=old)
    out = await crud.upsert_faith_case(s, **ARGS)
    assert out == "reactivated"
    assert old.status == "未解决" and old.resolution is None
    assert old.resolved_at == datetime(2026, 1, 3)  # 保留做复发显示(附录 A 注释语义)


async def test_set_status_resolve_marks_and_unresolve_clears():
    row = FaithCase(eval_id="A22", bucket="A_policy", query="q", strategy="hybrid_rerank",
                    answer="a", reason="r", status="未解决", seen_count=1,
                    first_seen_at=datetime(2026, 1, 1), last_seen_at=datetime(2026, 1, 1))
    s = FakeSession(get_row=row)
    out = await crud.set_faith_case_status(s, 1, "已解决", "语料确实写9折")
    assert out.status == "已解决" and out.resolution == "语料确实写9折" and out.resolved_at is not None
    out2 = await crud.set_faith_case_status(s, 1, "未解决", None)
    assert out2.status == "未解决" and out2.resolution is None and out2.resolved_at == out.resolved_at
```

Run → FAIL。**实现 `app/db/crud.py` 追加**(import 区补 `FaithCase, LowConfidenceQuestion`(LowConfidenceQuestion 已在 T7 加)、`select` 已有则不重复):

```python
# ---- ch04: 原文回查 + 忠实度台账 ----

async def get_chunk(session, chunk_id: int):
    return await session.get(KnowledgeChunk, chunk_id)


async def upsert_faith_case(session, *, eval_id: str, bucket: str, query: str, answer: str,
                            reason: str, citations, judge_model,
                            strategy: str = "hybrid_rerank") -> str:
    """一题一行(uk_eval_id):created/updated/reactivated。复发即退回未解决、清 resolution,
    resolved_at 保留;first_seen_at 不覆写、seen_count+1、重判字段刷新(§3.3)。"""
    row = (await session.execute(select(FaithCase).where(FaithCase.eval_id == eval_id))).scalar_one_or_none()
    now = datetime.now()
    if row is None:
        session.add(FaithCase(eval_id=eval_id, bucket=bucket, query=query, strategy=strategy,
                              answer=answer, reason=reason, citations=citations, judge_model=judge_model,
                              status="未解决", seen_count=1, first_seen_at=now, last_seen_at=now))
        await session.commit()
        return "created"
    result = "reactivated" if row.status in ("已解决", "无需解决") else "updated"
    if result == "reactivated":
        row.resolution = None
    row.status, row.bucket, row.query, row.strategy = "未解决", bucket, query, strategy
    row.answer, row.reason, row.citations, row.judge_model = answer, reason, citations, judge_model
    row.seen_count += 1
    row.last_seen_at = now
    await session.commit()
    return result


async def list_faith_cases(session, *, status: str | None = None, bucket: str | None = None,
                           limit: int = 200):
    stmt = select(FaithCase).order_by(FaithCase.last_seen_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(FaithCase.status == status)
    if bucket:
        stmt = stmt.where(FaithCase.bucket == bucket)
    return list((await session.execute(stmt)).scalars().all())


async def get_faith_case(session, case_id: int):
    return await session.get(FaithCase, case_id)


async def set_faith_case_status(session, case_id: int, status: str, resolution: str | None):
    row = await session.get(FaithCase, case_id)
    if row is None:
        return None
    row.status = status
    if status == "未解决":
        row.resolution = None  # resolved_at 保留(复发显示)
    else:
        row.resolution = (resolution or "").strip() or None
        row.resolved_at = datetime.now()
    await session.commit()
    return row
```

(crud 文件若未 import `datetime`,补 `from datetime import datetime`。)

Run: `uv run pytest tests/test_faith_crud_ch04.py -q` → PASS

- [ ] **Step 2: 写失败测试 `tests/test_knowledge_routes.py`**

```python
"""三端点单测:fake crud 转发/404/503/PATCH 校验 422(核对点⑤)。"""

import pytest

from app.api import routes as routes_mod
from app.main import app


def _override(app_obj, dep, val):
    app_obj.dependency_overrides[dep] = lambda: val


class FakeCrud:
    def __init__(self, chunk=None, cases=None, patched="ROW"):
        self.chunk, self.cases, self.patched = chunk, cases or [], patched
        self.calls = []

    async def get_chunk(self, session, chunk_id):
        self.calls.append(("chunk", chunk_id))
        return self.chunk

    async def list_faith_cases(self, session, *, status, bucket, limit=200):
        self.calls.append(("list", status, bucket))
        return self.cases

    async def set_faith_case_status(self, session, case_id, status, resolution):
        self.calls.append(("patch", case_id, status, resolution))
        return self.patched


def _faith_row(case_id=1, status="未解决"):
    from datetime import datetime

    from app.db.models import FaithCase

    return FaithCase(id=case_id, eval_id="A22", bucket="A_policy", query="银卡打折吗",
                     strategy="hybrid_rerank", answer="a", reason="r", citations=[{"n": 1}],
                     judge_model="m", status=status, seen_count=1,
                     first_seen_at=datetime(2026, 9, 23), last_seen_at=datetime(2026, 9, 23),
                     resolution=None, resolved_at=None)


def _chunk_row():
    from app.db.models import KnowledgeChunk

    return KnowledgeChunk(id=7, category="商品规格手册", questions="q", answer="a",
                          vector_id="7", status="done", section_path="商品规格手册 > 节",
                          content_type="policy", prev_chunk_id=None, next_chunk_id=8)


async def test_get_chunk_ok_404_503(client, monkeypatch):
    fake = FakeCrud(chunk=_chunk_row())
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.get("/api/chunks/7")
    assert r.status_code == 200
    assert r.json() == {"id": 7, "section_path": "商品规格手册 > 节", "category": "商品规格手册",
                        "questions": "q", "answer": "a", "content_type": "policy",
                        "prev_chunk_id": None, "next_chunk_id": 8}
    monkeypatch.setattr(routes_mod, "crud", FakeCrud(chunk=None))
    assert (await client.get("/api/chunks/999999")).status_code == 404
    _override(app, routes_mod.dep_db_session, None)
    assert (await client.get("/api/chunks/7")).status_code == 503
    app.dependency_overrides.clear()


async def test_faith_list_and_filter(client, monkeypatch):
    fake = FakeCrud(cases=[_faith_row()])
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.get("/api/faith_cases", params={"status": "未解决", "bucket": "A_policy"})
    assert r.status_code == 200 and r.json()[0]["eval_id"] == "A22"
    assert fake.calls[-1] == ("list", "未解决", "A_policy")
    app.dependency_overrides.clear()


async def test_faith_patch_validation_422(client, monkeypatch):
    fake = FakeCrud()
    monkeypatch.setattr(routes_mod, "crud", fake)
    _override(app, routes_mod.dep_db_session, object())
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决"})  # 缺处置说明
    assert r.status_code == 422
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决", "resolution": "  "})
    assert r.status_code == 422  # 空白同缺
    r = await client.patch("/api/faith_cases/1", json={"status": "随便"})
    assert r.status_code == 422  # Literal 枚举外
    r = await client.patch("/api/faith_cases/1", json={"status": "已解决", "resolution": "老师标注出入"})
    assert r.status_code == 200
    assert fake.calls[-1] == ("patch", 1, "已解决", "老师标注出入")
    monkeypatch.setattr(routes_mod, "crud", FakeCrud(patched=None))
    assert (await client.patch("/api/faith_cases/2",
                               json={"status": "未解决"})).status_code == 404
    app.dependency_overrides.clear()
```

Run → FAIL。**实现 `app/schemas/knowledge.py`**:

```python
"""ch04 原文回查与忠实度台账的 REST 契约(spec §5.4)。"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ChunkOut(BaseModel):
    id: int
    section_path: str
    category: str
    questions: str
    answer: str
    content_type: str
    prev_chunk_id: int | None
    next_chunk_id: int | None


class FaithCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


class FaithCasePatch(BaseModel):
    status: Literal["未解决", "已解决", "无需解决"]
    resolution: str | None = None

    @model_validator(mode="after")
    def resolution_required(self):
        if self.status != "未解决" and not (self.resolution or "").strip():
            raise ValueError("标已解决/无需解决必须填写处置说明")
        return self
```

**实现 `app/api/routes.py`**:import 区补 `from app.schemas.knowledge import ChunkOut, FaithCaseOut, FaithCasePatch`;文件末尾追加:

```python
@router.get("/api/chunks/{chunk_id}", response_model=ChunkOut)
async def get_chunk(chunk_id: int, session=Depends(dep_db_session)):
    """引用弹窗原文(T13)。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.get_chunk(session, chunk_id)
    if row is None:
        raise HTTPException(status_code=404, detail="chunk 不存在")
    return row


@router.get("/api/faith_cases", response_model=list[FaithCaseOut])
async def list_faith_cases(status: str | None = None, bucket: str | None = None,
                           session=Depends(dep_db_session)):
    """台账页列表(T14):last_seen_at 降序,可按状态/桶过滤。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    return await crud.list_faith_cases(session, status=status, bucket=bucket)


@router.patch("/api/faith_cases/{case_id}", response_model=FaithCaseOut)
async def patch_faith_case(case_id: int, req: FaithCasePatch, session=Depends(dep_db_session)):
    """处置流转:非「未解决」必须带 resolution(schema 422);「未解决」强制清处置说明。"""
    if session is None:
        raise HTTPException(status_code=503, detail="数据库未初始化")
    row = await crud.set_faith_case_status(session, case_id, req.status, req.resolution)
    if row is None:
        raise HTTPException(status_code=404, detail="个案不存在")
    return row
```

Run: `uv run pytest tests/test_knowledge_routes.py tests/test_faith_crud_ch04.py -q` → PASS

- [ ] **Step 3: 活库集成(核对点⑥中文 ENUM + 流转全真)** Create `tests/test_ledger_integration.py`:

```python
"""台账活库往返:中文 ENUM 读写(核对点⑥)、uk_eval_id upsert、JSON citations、复发流转。
每次用随机 eval_id 避免重跑撞 uk;结束自清理。"""

import uuid
from datetime import datetime

import pytest

from app.core.config import get_settings
from app.db import crud
from app.db.engine import dispose_engine, get_session_factory, init_engine

pytestmark = pytest.mark.integration


async def test_ledger_roundtrip_live():
    init_engine(get_settings())
    eid = f"IT{uuid.uuid4().hex[:10]}"
    try:
        async with get_session_factory()() as session:
            assert await crud.upsert_faith_case(
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开",
                reason="应拒答未拒", citations=[{"n": 1, "chunk_id": 7, "section_path": "s",
                                                "question": "q", "answer": "a"}],
                judge_model="judge-x") == "created"
            assert await crud.upsert_faith_case(
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开2",
                reason="复发", citations=None, judge_model="judge-x") == "updated"
            row = (await crud.list_faith_cases(session, status="未解决", bucket="D_absent"))
            mine = [r for r in row if r.eval_id == eid][0]
            assert mine.seen_count == 2 and mine.citations[0]["n"] == 1
            await crud.set_faith_case_status(session, mine.id, "已解决", "老师标注出入,语料可答")
            assert await crud.upsert_faith_case(  # 复发 → reactivated
                session, eval_id=eid, bucket="D_absent", query="纸质发票", answer="能开3",
                reason="再判编造", citations=[], judge_model="judge-x") == "reactivated"
            again = await crud.get_faith_case(session, mine.id)
            assert again.status == "未解决" and again.resolution is None
            assert again.resolved_at is not None and again.seen_count == 3
            assert isinstance(again.first_seen_at, datetime)
    finally:
        from sqlalchemy import delete

        from app.db.models import FaithCase

        async with get_session_factory()() as session:
            await session.execute(delete(FaithCase).where(FaithCase.eval_id == eid))
            await session.commit()
        await dispose_engine()
```

Run: `uv run pytest -m integration tests/test_ledger_integration.py -q` → PASS(**中文 status 值读写往返成功即核对点⑥销账**;失败不得改表结构迁就——先 Context7 查 aiomysql charset 参数,根因是连接层就修连接串 `charset=utf8mb4`)

- [ ] **Step 4: 全量回归 + 提交**

Run: `uv run pytest -q` → 全绿
```bash
git add app/schemas/knowledge.py app/db/crud.py app/api/routes.py tests/test_knowledge_routes.py tests/test_faith_crud_ch04.py tests/test_ledger_integration.py dev-notes/ch04.md
git commit -m "feat(ch04): /api/chunks+faith_cases GET/PATCH、台账upsert复发流转、中文ENUM销核对点⑥、422销核对点⑤

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 11: 老师题库解析器+指标纯函数 TDD + `run_strategy_eval` 四臂对比 + D 桶校准 + 旧 eval 标 DEPRECATED

**Files:**
- Create: `evals/teacher_csv.py`、`evals/run_strategy_eval.py`、`tests/test_teacher_csv.py`
- Modify: `evals/run_rag_eval.py`(仅 docstring 加 DEPRECATED 头,保留不删——附录 C)

**Interfaces:**
- Consumes: 老师 CSV(只读)、T6 `retriever.retrieve(strategy=…)`、T4 `UnderstandResult`
- Produces:
  - `EvalQuestion{id,bucket,query,groups,expect_points,should_refuse}`、`load_questions(path=CSV_PATH)`、`parse_expect_section(raw)`、`eval_question(groups, ranked_paths, ks)`(纯函数,T12 复用)
  - `evals/run_strategy_eval.py` CLI:`--limit N --buckets 逗号 --arms 逗号 --no-cache`;产物 `evals/reports/ch04_strategy_report.md`
  - `evals/cache/rewrite_cache.json`(rewrite 结果缓存:四臂同输入、重跑零漂移、省 LLM 调用,§4.1-4)

- [ ] **Step 1: 写失败测试 `tests/test_teacher_csv.py`(样例行全部逐字取自老师 CSV)**

```python
"""解析器与指标纯函数。样例行逐字抄自 evals/run_rag.py(附录 B 语法);真文件全量对账。"""

from evals.teacher_csv import eval_question, load_questions, parse_expect_section

SAMPLE = """id,桶(bucket),问题(query),期望章节(expect_section),标准要点(expect_points),应拒答(should_refuse)
A5,A_policy,"满多少钱包邮,不满怎么收运费",运费怎么算,满 99 元包邮 | 10 元运费,否
A23,A_policy,偏远地区能配送吗,运费与包邮 | 配送范围,偏远地区,否
E2,E_multi,退货要怎么弄,退货政策 | 适用范围 + 退换货运费承担 + 退款时效,步骤,否
D3,D_absent,能不能开纸质发票邮寄,,,是
"""


def test_parse_expect_section_grammar():
    assert parse_expect_section("运费与包邮 | 配送范围") == [["运费与包邮", "配送范围"]]  # 组内 OR
    assert parse_expect_section("A + B") == [["A"], ["B"]]                                # 组间 AND
    assert parse_expect_section("会员 / 积分规则") == [["会员 > 积分规则"]]                 # 路径分隔符归一
    assert parse_expect_section("") == [] and parse_expect_section(None) == []


def test_load_sample_rows(tmp_path):
    f = tmp_path / "q.csv"
    f.write_text(SAMPLE, encoding="utf-8")
    qs = load_questions(f)
    assert len(qs) == 4
    assert qs[0].query == "满多少钱包邮,不满怎么收运费"  # 引号字段内逗号不拆
    assert qs[0].groups == [["运费怎么算"]] and qs[0].should_refuse is False
    assert qs[2].groups == [["退货政策", "适用范围"], ["退换货运费承担"], ["退款时效"]]  # |与+混排
    assert qs[3].should_refuse is True and qs[3].groups == [] and qs[3].bucket == "D_absent"


def test_real_teacher_file_shape():
    qs = load_questions()  # 默认 evals/run_rag.py
    assert len(qs) == 300
    from collections import Counter

    c = Counter(q.bucket for q in qs)
    assert set(c) == {"A_policy", "B_model", "C_colloquial", "D_absent", "E_multi"}
    assert all(v == 60 for v in c.values())
    assert all(q.should_refuse for q in qs if q.bucket == "D_absent")
    assert all(not q.should_refuse for q in qs if q.bucket != "D_absent")


def test_eval_question_metrics_group_semantics():
    groups = [["运费"], ["包邮"]]
    m = eval_question(groups, ["x > 退货", "y > 包邮与运费", "z"], ks=(3, 10))
    assert m["hit@3"] == 1.0 and m["recall@3"] == 1.0
    assert abs(m["mrr@10"] - 0.5) < 1e-9  # 两组都 rank2 → (1/2+1/2)/2


def test_eval_question_window_and_empty():
    m = eval_question([["深层"]], ["a", "b深层c"], ks=(1, 10))
    assert m["hit@1"] == 0.0 and m["recall@1"] == 0.0 and m["hit@10"] == 1.0
    assert eval_question([], ["anything"]) == {}  # D 桶不参与排序指标
    assert eval_question([["x"]], [])["mrr@10"] == 0.0
```

Run: `uv run pytest tests/test_teacher_csv.py -q` → FAIL。**实现 `evals/teacher_csv.py`**:

```python
"""老师 300 题评估 CSV 的解析器与指标纯函数(spec §6.1/附录 B;evals/run_rag.py 是只读数据文件)。

期望章节语法:「+」=AND 组、「|」=组内 any-of、「 / 」≡「 > 」=路径层级;
原子按 section_path **子串**匹配。D 桶期望列空、应拒答=是:不参与排序指标,
只进闸1阈值校准(数字在 run_strategy_eval,T12 终值回写 Settings)。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

CSV_PATH = Path("evals/run_rag.py")  # 扩展名与内容不符(实为 UTF-8 CSV)——盘点结论,附录 B


@dataclass
class EvalQuestion:
    id: str
    bucket: str
    query: str
    groups: list[list[str]] = field(default_factory=list)  # AND 组,组内 OR 原子
    expect_points: str = ""
    should_refuse: bool = False


def parse_expect_section(raw) -> list[list[str]]:
    raw = (raw or "").strip()
    if not raw:
        return []
    groups: list[list[str]] = []
    for g in raw.split("+"):
        atoms = [a.replace(" / ", " > ").strip() for a in g.split("|") if a.strip()]
        if atoms:
            groups.append(atoms)
    return groups


def load_questions(path: Path = CSV_PATH) -> list[EvalQuestion]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            EvalQuestion(
                id=row["id"].strip(),
                bucket=row["桶(bucket)"].strip(),
                query=row["问题(query)"].strip(),
                groups=parse_expect_section(row["期望章节(expect_section)"]),
                expect_points=row["标准要点(expect_points)"].strip(),
                should_refuse=row["应拒答(should_refuse)"].strip() == "是",
            )
            for row in csv.DictReader(f)
        ]


def eval_question(groups: list[list[str]], ranked_paths: list[str],
                  ks: tuple[int, ...] = (3, 10)) -> dict:
    """单题指标:组间 AND(全中才 hit)、组内 OR(任一原子最早命中定 rank);
    MRR = 命中组 1/r 的组间平均(未命中组按 0 计入母),窗口 max(ks);D 桶空组 → {}。"""
    if not groups:
        return {}
    ranks = []
    for atoms in groups:
        r = next((i + 1 for i, p in enumerate(ranked_paths) if any(a in p for a in atoms)), None)
        ranks.append(r)
    out: dict[str, float] = {}
    kmax = max(ks)
    for k in ks:
        within = [r for r in ranks if r is not None and r <= k]
        out[f"hit@{k}"] = 1.0 if len(within) == len(ranks) else 0.0
        out[f"recall@{k}"] = len(within) / len(ranks)
    inv = [1.0 / r for r in ranks if r is not None and r <= kmax]
    out[f"mrr@{kmax}"] = sum(inv) / len(ranks)
    return out
```

Run → PASS(5 个全过)

- [ ] **Step 2: 写 `evals/run_strategy_eval.py`**

```python
"""ch04 四策略对比评估(spec §6.2):老师题库 A/B/C/E × {dense,bm25,hybrid,hybrid_rerank}
分桶出 Hit@3/10、Recall@3/10、MRR@10;D 桶跑 hybrid_rerank 观测 top1 分数做闸1校准表。

uv run python evals/run_strategy_eval.py [--limit N] [--buckets A_policy,...] [--arms dense,...] [--no-cache]
产物:evals/reports/ch04_strategy_report.md(重跑覆盖)。评估臂关两个在线阈值
(rag_score_threshold/retrieval_low_conf_threshold → 0.0):测排序质量,不测在线拒答。
无阻断线:数字如实出,劣化进失败样例分析(spec §0-9);--limit 是成本闸。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, init_engine  # noqa: E402
from app.rag import milvus_store, retriever  # noqa: E402
from app.rag.query_understanding import UnderstandResult, understand_query  # noqa: E402
from evals.teacher_csv import eval_question, load_questions  # noqa: E402

CACHE_FILE = Path("evals/cache/rewrite_cache.json")
REPORT_FILE = Path("evals/reports/ch04_strategy_report.md")
ANSWER_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
ALL_ARMS = ("dense", "bm25", "hybrid", "hybrid_rerank")
METRIC_KEYS = ("hit@3", "hit@10", "recall@3", "recall@10", "mrr@10")

TEACHER_NOTES = """### 老师材料出入说明(只读原则,不改数据)
1. D3「能不能开纸质发票邮寄」标为应拒答,但语料 billing-shipping《可开票类型》实际可答——该题会拉低「正确拒答率」,如实呈现。
2. A22/E9 标准要点写银卡 95 折,语料为银卡 9 折(金卡才 95 折)——按语料评估。
3. FAQ 未满 99 收 10 元 与 billing《运费与包邮》6 元为跨文档矛盾——两文档并存入库,引用哪块都算命中(期望章节按原子子串匹配)。"""


def _load_cache() -> dict:
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


async def _understand(query: str, st, cache: dict) -> UnderstandResult:
    hit = cache.get(query)
    if hit is not None:
        return UnderstandResult(**hit)
    r = await understand_query(query, st)
    cache[query] = {"standard_query": r.standard_query, "synonyms": r.synonyms, "degraded": r.degraded}
    return r


def _mean(xs) -> float:
    return statistics.fmean(xs) if xs else 0.0


def _calibrate(d_scores: list[float], ans_scores: list[float]):
    cands = sorted({round(s, 3) for s in d_scores + ans_scores})
    if cands:
        cands = cands[:: max(1, len(cands) // 12)]
    table, best = [], (float("inf"), 0.3)
    for t in cands or [0.3]:
        false_refuse = _mean([1.0 if s < t else 0.0 for s in ans_scores])   # 可答题被误拒
        over_conf = _mean([1.0 if s >= t else 0.0 for s in d_scores])       # D 题仍自信
        table.append((t, false_refuse, over_conf))
        if false_refuse + over_conf < best[0]:
            best = (false_refuse + over_conf, t)
    return table, round(best[1], 3)


async def run(args) -> int:
    st = get_settings()
    st_eval = st.model_copy(update={"rag_score_threshold": 0.0,
                                    "retrieval_low_conf_threshold": 0.0})
    init_engine(st)
    cache = {} if args.no_cache else _load_cache()
    questions = load_questions()
    buckets = args.buckets.split(",") if args.buckets else list(ANSWER_BUCKETS)
    arms = args.arms.split(",") if args.arms else list(ALL_ARMS)
    assert set(arms) <= set(ALL_ARMS) and set(buckets) <= set(ANSWER_BUCKETS)
    answerable = [q for q in questions if q.bucket in buckets]
    if args.limit:
        answerable = answerable[: args.limit]
    d_questions = [q for q in questions if q.bucket == "D_absent"][: args.limit or 60]

    rows: dict[tuple, list] = defaultdict(list)
    fails: list[str] = []
    ans_top1: dict[str, float] = {}
    started = datetime.now()
    for q in answerable:
        u = await _understand(q.query, st, cache)
        for arm in arms:
            res = await retriever.retrieve(q.query, strategy=arm, settings=st_eval, understood=u)
            paths = [c.row.section_path for c in res.chunks]
            m = eval_question(q.groups, paths, ks=(3, 10))
            rows[(q.bucket, arm)].append(m)
            if arm == "hybrid_rerank" and res.chunks:
                ans_top1[q.id] = res.chunks[0].score
            if m and m["hit@10"] == 0.0 and len(fails) < 40:
                fails.append(f"- `{q.id}` [{q.bucket}/{arm}] {q.query}\n  - 期望组: {q.groups}\n  - 实际 Top-3: {paths[:3]}")
        print(f"[strategy-eval] {q.id} ok", flush=True)
    d_top1 = []
    for q in d_questions:
        u = await _understand(q.query, st, cache)
        res = await retriever.retrieve(q.query, strategy="hybrid_rerank", settings=st_eval, understood=u)
        d_top1.append(res.chunks[0].score if res.chunks else 0.0)
    _save_cache(cache)

    client = milvus_store.get_client(st.milvus_uri)
    total_blocks = milvus_store.count_rows(client, st.milvus_collection)
    table, best_t = _calibrate(d_top1, list(ans_top1.values()))
    lines = [
        "# ch04 四策略对比评估报告",
        f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 题库:老师 `evals/run_rag.py`(300 题,只读) · 集合 `knowledge` 块数:{total_blocks} · 用时:{(datetime.now() - started).seconds // 60} 分",
        f"- 参数:arms={arms} buckets={buckets} limit={args.limit or '全量'} · rerank key:{'已配置' if st.rerank_api_key else '**未配置→hybrid_rerank 实为 RRF 降级序,D 桶校准数字不可用**'}",
        f"- 在线阈值在评估中关闭(两阈值→0.0),本表测**排序质量**;`rag_score_threshold` 本就只作用 dense 腿,四臂同输入。",
        "- **小库声明**:语料切块后仅约 " + str(total_blocks) + " 块,双腿 Top-50≈全库,Recall 天然偏高、**MRR 更有分辨力**(spec §6.2)。",
        "- **无阻断线声明**:未达观察目标不构成失败;劣化桶与失败样例见下,禁止为达标改数据/藏结果(spec §0-9,用户 2026-09-23 纠偏)。",
        "",
        "## 分桶 × 策略 指标均值",
        "",
        "| 桶 | 策略 | n | " + " | ".join(METRIC_KEYS) + " |",
        "|---|---|---|" + "---|" * len(METRIC_KEYS),
    ]
    for b in buckets:
        for a in arms:
            ms = [m for m in rows[(b, a)] if m]
            lines.append(f"| {b} | {a} | {len(ms)} | " +
                         " | ".join(f"{_mean([m[k] for m in ms]):.3f}" for k in METRIC_KEYS) + " |")
    lines += ["", "## 未命中样例(hit@10=0,最多 40 条)", ""] + (fails or ["(无)"])
    lines += ["", "## D 桶闸1阈值校准表", "",
              "top1 分数分布 + 误拒/误自信权衡(`--calibrate` 候选阈值;终值由 Task 12 回写 Settings 默认):", "",
              "| 阈值 | 可答题误拒率 | D题仍自信率 | 合计 |", "|---|---|---|---|"]
    lines += [f"| {t:.3f} | {f_:.3f} | {o:.3f} | {f_ + o:.3f} |" for t, f_, o in table]
    lines += [f"", f"**建议阈值:{best_t}**", "", TEACHER_NOTES, ""]
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"[strategy-eval] 报告 → {REPORT_FILE}")
    await dispose_engine()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ch04 四策略对比评估")
    parser.add_argument("--limit", type=int, default=0, help="题量成本闸(各桶截断)0=全量")
    parser.add_argument("--buckets", type=str, default="", help="逗号分隔桶名,默认 A/B/C/E 四答桶")
    parser.add_argument("--arms", type=str, default="", help="逗号分隔策略,默认四臂")
    parser.add_argument("--no-cache", action="store_true", help="禁 rewrite 缓存(重测改写漂移)")
    sys.exit(asyncio.run(run(parser.parse_args())))
```

- [ ] **Step 3: 冒烟(小成本验管线)**

Run: `uv run python evals/run_strategy_eval.py --limit 2 --buckets A_policy --arms dense,bm25`
Expected: `[strategy-eval] A1 ok / A2 ok` + 报告生成;打开 `evals/reports/ch04_strategy_report.md` 检查表结构、D 校准表占位在无 rerank key 时的警示行。

- [ ] **Step 4: 全量四臂跑(成本闸已开;预计 20-40 分钟 / 数百次 embedding+rerank 调用)**

Run: `uv run python evals/run_strategy_eval.py`
Expected: 报告覆盖为全量数字。**无论结果如何都不回改数据、不调指标定义**(§0-9);若 hybrid_rerank 未优于 dense,把「哪些桶、哪些样例、为什么」写进报告失败样例行即可(表本身已按桶拆)。无 rerank key 时先跑 `--arms dense,bm25,hybrid` 全量,dev-notes 记「rerank 臂与 D 校准待 key」,拿到 key 后重跑补全。

- [ ] **Step 5: 旧 eval 标记(保留不删不跑,附录 C)**

打开 `evals/run_rag_eval.py`,在其模块 docstring **首行开头**插入:
`[DEPRECATED · ch04] 老师语料换代后本 12 题期望路径失效,保留作 ch03 历史参照,不再运行;本章评估入口 = evals/run_strategy_eval.py(spec 附录 C)。`

- [ ] **Step 6: 回归 + 提交**

Run: `uv run pytest -q` → 全绿(runner 不 import 进 app 链)
```bash
git add evals/teacher_csv.py evals/run_strategy_eval.py evals/run_rag_eval.py tests/test_teacher_csv.py evals/reports/ch04_strategy_report.md dev-notes/ch04.md
git commit -m "feat(ch04): 老师题库解析器+指标纯函数TDD(逐字样例行锚定)+run_strategy_eval 四臂报告+D桶校准表

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 12: Faithfulness LLM-as-judge + faith_cases 台账落盘 + 闸1阈值校准终值回写

**Files:**
- Create: `app/prompts/faith_judge.py`、`evals/run_faith_eval.py`
- Modify(校准终值回写):`app/core/config.py`(一行默认值)、`.env.example`、`tests/test_config_ch04.py`(对应断言)
- 产物:`evals/reports/ch04_faith_report.md`

**Interfaces:**
- Consumes: `stream_chat_with_tools(chat_messages, settings, model, *, conversation_id=None, persister=None) -> AsyncIterator[ToolEvent]`(实测签名 `("token",str)|("tool_call",dict)|("tool_result",dict)`);T7 tool_result 帧可选 `citations`;T10 `crud.upsert_faith_case`;`crud.create_or_get_conversation(session, None, user_id)`;T1 `faith_judge_model`/`get_model` 读 `settings.model_name`
- Produces: `FaithVerdict{verdict: Literal["faithful","fabricated"], reason: str}`(with_structured_output,同 T8 EvidenceCheck 模式);终值后的 `retrieval_low_conf_threshold` 默认(验收④与在线行为以此为准)

- [ ] **Step 1: 写 `app/prompts/faith_judge.py`**

```python
"""忠实度裁判 Prompt(spec §6.3):只依本轮链路产出的证据集判生成,不用世界知识补证。"""

FAITH_JUDGE_SYSTEM = """你是 RAG 忠实度评审,只依据给定的【证据】判断【候选回答】。

规则:
1. 候选回答中每一个事实性断言(数字、时效、条件、流程)都必须能由某条证据推出;推不出、或与证据矛盾的句子即编造。
2. 只要存在至少一句编造,整体判 fabricated,reason 点名是哪句、缺什么依据;全部可推出、或回答是明确拒答/纯流程引导(不含平台规则事实)→ faithful。
3. 【证据】为空时,候选回答凡涉及平台规则/政策/商品参数的事实句一律按编造处理。
4. 不使用你自己的世界知识为回答补证据;不评语气与格式。

输出 JSON:{"verdict": "faithful" 或 "fabricated", "reason": "≤80字中文理由"}"""
```

- [ ] **Step 2: 写 `evals/run_faith_eval.py`**

```python
"""ch04 忠实度评估(spec §6.3):真实链路生成 → LLM-as-judge → fabricated 落 faith_cases。

uv run python evals/run_faith_eval.py [--sample 30] [--d-limit 60] [--seed 42] [--recheck 0]
- 可答桶按桶分层抽 --sample 题,每题走完整 stream_chat_with_tools(query_faq 真检索+双闸;
  persister=None 零消息落库,conversation_id 用本次创建的 eval_faith 会话 → 池写可落);
- REFUSAL_ANSWER 开头的答案计「显式拒答」不送判;其余全部送判(证据=本轮帧 citations,温度 0 可复现);
- D 桶全量测正确拒答率;未拒答且判 fabricated → 台账 bucket='D_absent'(验收④反向证据);
- --recheck N:台账最近 N 条「已解决」复判,观察处置稳定性(只打印不自动翻状态);
- 报告 evals/reports/ch04_faith_report.md(重跑覆盖)+ 末尾打印低置信池本次增量。
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel  # noqa: E402
from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db import crud  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import LowConfidenceQuestion  # noqa: E402
from app.prompts.faith_judge import FAITH_JUDGE_SYSTEM  # noqa: E402
from app.schemas.chat import ChatMessage  # noqa: E402
from app.services.chat_service import get_model  # noqa: E402
from app.services.refusals import REFUSAL_ANSWER  # noqa: E402
from app.services.tool_chat_service import stream_chat_with_tools  # noqa: E402
from evals.teacher_csv import load_questions  # noqa: E402

REPORT_FILE = Path("evals/reports/ch04_faith_report.md")
ANSWER_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")


class FaithVerdict(BaseModel):
    verdict: Literal["faithful", "fabricated"]
    reason: str = ""


def sample_answerable(questions, per_bucket: int, seed: int) -> list:
    by: dict[str, list] = defaultdict(list)
    for q in questions:
        if q.bucket in ANSWER_BUCKETS:
            by[q.bucket].append(q)
    rng = random.Random(seed)
    out = []
    for b in ANSWER_BUCKETS:
        out.extend(rng.sample(by[b], min(per_bucket, len(by[b]))))
    return out


async def run_chain(question: str, st, conversation_id: int) -> tuple[str, list]:
    """真实链:token 聚合成答案,query_faq 帧的 citations 作评审证据。"""
    model = get_model(st)
    parts, citations = [], []
    async for evt, payload in stream_chat_with_tools(
        [ChatMessage(role="user", content=question)], st, model, conversation_id=conversation_id,
    ):
        if evt == "token":
            parts.append(payload)
        elif evt == "tool_result" and payload.get("name") == "query_faq":
            citations = payload.get("citations") or citations
    return "".join(parts), citations


def _evidence_block(citations: list) -> str:
    return "\n".join(
        f"[{c['n']}] {c.get('section_path', '')} | {c.get('question', '')} | {(c.get('answer') or '')[:400]}"
        for c in citations)


async def judge(question: str, answer: str, citations: list, st) -> FaithVerdict:
    key = st.faith_judge_model or st.model_name
    judge_st = st.model_copy(update={"model_name": key, "temperature": 0})
    model = get_model(judge_st).with_structured_output(FaithVerdict)
    user = (f"【问题】{question}\n【证据】\n{_evidence_block(citations) or '(本轮无知识库证据命中)'}"
            f"\n【候选回答】{answer}")
    return await asyncio.wait_for(
        model.ainvoke([("system", FAITH_JUDGE_SYSTEM), ("user", user)]), timeout=60)


async def run(args) -> int:
    st = get_settings()
    init_engine(st)
    questions = load_questions()
    started = datetime.now()
    async with get_session_factory()() as session:
        conv = await crud.create_or_get_conversation(session, None, "eval_faith")
        cid = conv.id
        pool_before = (await session.execute(
            select(func.count()).select_from(LowConfidenceQuestion))).scalar()

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fabricated_rows: list[str] = []
    d_refused = d_total = 0
    recheck_lines: list[str] = []
    try:
        for q in sample_answerable(questions, args.sample, args.seed):
            answer, citations = await run_chain(q.query, st, cid)
            if answer.startswith(REFUSAL_ANSWER[:8]):
                stats[q.bucket]["refused"] += 1
                continue
            v = await judge(q.query, answer, citations, st)
            stats[q.bucket][v.verdict] += 1
            if v.verdict == "fabricated":
                fabricated_rows.append(f"- `{q.id}` [{q.bucket}] {q.query}\n  - 裁判:{v.reason[:80]}\n  - 回答:{answer[:120]}")
                async with get_session_factory()() as session:
                    await crud.upsert_faith_case(
                        session, eval_id=q.id, bucket=q.bucket, query=q.query, answer=answer,
                        reason=v.reason, citations=citations, judge_model=st.faith_judge_model or st.model_name)
            print(f"[faith] {q.id} → {v.verdict}", flush=True)

        for q in [q for q in questions if q.bucket == "D_absent"][: args.d_limit]:
            d_total += 1
            answer, citations = await run_chain(q.query, st, cid)
            if answer.startswith(REFUSAL_ANSWER[:8]):
                d_refused += 1
                stats["D_absent"]["refused"] += 1
                continue
            v = await judge(q.query, answer, citations, st)
            stats["D_absent"][v.verdict] += 1
            if v.verdict == "fabricated":
                fabricated_rows.append(f"- `{q.id}` [D_absent 应拒未拒] {q.query}\n  - 裁判:{v.reason[:80]}\n  - 回答:{answer[:120]}")
                async with get_session_factory()() as session:
                    await crud.upsert_faith_case(
                        session, eval_id=q.id, bucket="D_absent", query=q.query, answer=answer,
                        reason="应拒答未拒:" + v.reason, citations=citations,
                        judge_model=st.faith_judge_model or st.model_name)
            print(f"[faith] {q.id} D→{v.verdict}", flush=True)

        if args.recheck:
            async with get_session_factory()() as session:
                rows = [r for r in await crud.list_faith_cases(session, status="已解决")][: args.recheck]
            for r in rows:
                v = await judge(r.query, r.answer, r.citations or [], st)
                recheck_lines.append(f"- 复判 `{r.eval_id}`:{v.verdict} | {v.reason[:60]}")

        async with get_session_factory()() as session:
            pool_now = (await session.execute(
                select(func.count()).select_from(LowConfidenceQuestion))).scalar()

        lines = [
            "# ch04 忠实度评估报告",
            f"- 生成:{datetime.now():%Y-%m-%d %H:%M} · 抽样:每桶 {args.sample} + D 桶 {d_total}(seed={args.seed}) · 裁判:{st.faith_judge_model or st.model_name} · 用时:{(datetime.now() - started).seconds // 60} 分",
            "- 链路与生产一致(temperature=生产值);仅裁判温度 0 保可复现。REFUSAL 开头答案计显式拒答不送判。",
            "",
            "| 桶 | faithful | fabricated | 显式拒答 |", "|---|---|---|---|",
        ]
        for b in ANSWER_BUCKETS + ("D_absent",):
            s = stats[b]
            lines.append(f"| {b} | {s['faithful']} | {s['fabricated']} | {s['refused']} |")
        lines += [
            "",
            f"D 桶正确拒答率:{d_refused}/{d_total}(D3 老师标注出入见策略报告,只读如实呈现)",
            f"低置信池本次增量:{pool_now - pool_before} 行(验收④证据链;演练另见 evals/demo_ch04.py)",
            "",
            "## fabricated 个案(已入 faith_cases 台账)", "",
        ] + (fabricated_rows or ["(无)"])
        if recheck_lines:
            lines += ["", "## 复判抽验(已解决个案,仅观察)", ""] + recheck_lines
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        print(f"[faith] 报告 → {REPORT_FILE}")
        return 0
    finally:
        await dispose_engine()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ch04 忠实度评估")
    parser.add_argument("--sample", type=int, default=30, help="每答桶抽样题数(四臂成本闸)")
    parser.add_argument("--d-limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recheck", type=int, default=0, help="复判 N 条已解决个案")
    sys.exit(asyncio.run(run(parser.parse_args())))
```

- [ ] **Step 3: 冒烟**

Run: `uv run python evals/run_faith_eval.py --sample 2 --d-limit 2 --seed 7`
Expected: 若干 `[faith] …` 行 + 报告生成;核对报告表格数字与 console 一致、台账页(浏览器开 `/static/index.html`,T14 完成前直接 `curl localhost:8000/api/faith_cases`)能看到 fabricated 行入库。**跑前先起 uvicorn 不是必须**(runner 直调服务层,不经 HTTP)。

- [ ] **Step 4: 全量跑**

Run: `uv run python evals/run_faith_eval.py`(≈(4×30+60) 次链路生成 + 等量裁判调用,成本与耗时记 dev-notes)
Expected: 报告覆盖为全量;无 rerank key 时闸1不写,池增量行主要来自闸2——如实标注。

- [ ] **Step 5: 闸1阈值校准终值回写(spec §4.4:0.3 是占位默认,终值=实测)**

1. 打开 `evals/reports/ch04_strategy_report.md` 「D 桶闸1阈值校准表」,抄「建议阈值」t* 与其误拒/误自信两率。
2. 若 t* ∉ [0.1, 0.6]:停下问用户(选型权力边界),不自行拍。
3. 在范围内则:`app/core/config.py` 把 `retrieval_low_conf_threshold: float = 0.3` 的默认值改为 t*(仅此一行);`.env.example` ch04 段同步;`tests/test_config_ch04.py` 里该字段默认值断言改为 t*。
4. Run: `uv run pytest tests/test_config_ch04.py -q` → PASS;校准表 + 依据(两率)贴 dev-notes,标「终值回写完成」。

- [ ] **Step 6: 复判抽验**

Run: `uv run python evals/run_faith_eval.py --sample 0 --d-limit 0 --recheck 5`(台账有已解决个案时;没有则跳并记 dev-notes)
Expected: 复判行只打印不翻状态;若多数翻 fabricated,dev-notes 记「裁判假阳观察」。

- [ ] **Step 7: 回归 + 提交**

Run: `uv run pytest -q` → 全绿
```bash
git add app/prompts/faith_judge.py evals/run_faith_eval.py app/core/config.py .env.example tests/test_config_ch04.py evals/reports/ch04_faith_report.md dev-notes/ch04.md
git commit -m "feat(ch04): faithfulness LLM裁判+fabricated落台账+D桶正确拒答率+闸1阈值终值回写

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 13: 前端——引用角标弹窗 + 👍/👎 一次性反馈(Vibe Coding,不套 TDD)

**Files:**
- Modify: `static/index.html`(唯一聊天前端,508 行;SSE 解析在 `send()` 内 `event === 'tool_result'` 分支,弹窗可仿既有抽取 overlay)

**Interfaces:**
- Consumes: tool_result 帧可选 `citations: [{n,chunk_id,section_path,question,answer}]`(T7);`GET /api/chunks/{id}`(T10)
- Produces: 纯前端行为;localStorage key 格式(数据飞轮读侧约定):`mewhelp.feedback.v1:{conversationId}:{assistantSeq}` → `{"v":"up"|"down","ts":"ISO"}`

**工作方式(用户工作要求①前端例外):** 不写单测;按下面契约与验收清单 Vibe 实现;每改一处**手动刷新页面走一遍清单**,清单结果与偏差逐条记 dev-notes(④翻车段照记)。

- [ ] **Step 1: citations 暂存与角标渲染** — `send()` 里 `tool_result` 分支把 `payload.citations`(有则)暂存到本轮闭包变量;最终答案文本渲染后,用正则 `/\[(\d+)\]/g` 替换为可点 `<sup class="cite" data-cid="{chunk_id}">[n]</sup>`(n 不在 citations 里 → 保留原文不渲染角标,防模型错标);气泡左下追加 👍/👎 按钮组。

- [ ] **Step 2: 原文弹窗** — 点角标 → `fetch('/api/chunks/'+cid)` → 弹层显示:面包屑 `category > section_path`、问法行 questions、正文 answer、相邻块 prev/next 按钮(有则换发 cid 重开)。样式沿用页面既有 overlay 基调,不引任何新依赖。

- [ ] **Step 3: 反馈一次性锁定** — 点击任一拇指:写 localStorage(上列 key;conversationId 用页面既有的会话 id 状态量,seq=该气泡在本会话中的第几条 assistant 消息)→ 两钮禁用、被点钮点亮、旁边出现「已反馈」;**不发任何后端请求**(纯采集,留飞轮入口);页面加载/历史渲染时按 key 回显锁定态。

- [ ] **Step 4: 兼容红线自查** — 除新增 `tool_result` 分支内取值外,不得改既有帧处理逻辑(ch01/ch02 五帧协议消费路径零变更);`resetChat` 与新会话按钮不清 localStorage(锁定态跟人跟会话)。

- [ ] **Step 5: 验收清单(全过才提交,结果贴 dev-notes)**

1. 问「退货要寄回去吗,邮费谁出」→ 回答带 [n] → 角标可点,弹窗显示原文+章节路径(**验收③前端侧**);
2. 问知识库里没有的 → 收到拒答文本,无角标、无报错;
3. 模型错标 `[99]` → 无角标渲染、原样文本、控制台不抛;
4. 点👍 → 点亮+「已反馈」、再点👎无效;刷新页面仍锁定;
5. F12 确认反馈点击零网络请求;弹窗是唯一新增请求(`GET /api/chunks/{id}`);
6. `uv run pytest -q` 仍全绿(没碰后端)。

- [ ] **Step 6: 提交**

```bash
git add static/index.html dev-notes/ch04.md
git commit -m "feat(ch04-ui): 引用角标弹窗(章节路径+原文)+👍/👎一次性锁定(localStorage 纯前端)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 14: 前端——faith_cases 轻量台账页 `static/faith.html`(Vibe Coding)

**Files:**
- Create: `static/faith.html`(经既有 `/static` 挂载即达 `/static/faith.html`,main.py 零改动)
- Modify: `static/index.html`(顶栏加一个「忠实度台账」链接)

**Interfaces:**
- Consumes(T10 契约,页面按此写死):`GET /api/faith_cases?status=&bucket=` → `[{id,eval_id,bucket,query,strategy,answer,reason,citations,judge_model,status,seen_count,first_seen_at,last_seen_at,resolution,resolved_at}]`;`PATCH /api/faith_cases/{id}` body `{status, resolution?}`(非「未解决」缺 resolution → 422)

- [ ] **Step 1: 列表** — 两个筛选下拉(状态:全部/未解决/已解决/无需解决;桶:A_policy/B_model/C_colloquial/D_absent/E_multi)+ 表格列:eval_id、桶、问题、裁判理由(截断 hover 全显)、复发次数 seen_count、最近发现、状态徽章(未解决红/已解决绿/无需解决灰);默认 last_seen_at 降序(服务端已排,前端不再排)。

- [ ] **Step 2: 处置弹层** — 行内「处置」按钮 → 弹层:完整 answer、reason、citations 展平(n + section_path + answer 截断)、状态下拉、处置说明 textarea(必填红星:非「未解决」时)→ 提交 PATCH;状态非 2xx:422 → 提示「填写处置说明后再保存」;其他 → 显示后端 detail。成功 → 行内徽章即时更新。

- [ ] **Step 3: 验收清单(结果贴 dev-notes)**

1. 跑过 Task 12 后打开页面能看到个案、按「未解决+D_absent」过滤正确;
2. 置「已解决」不填说明 → 前端提示(服务端 422 兜底,手工 curl 复验一次);
3. 填说明保存 → 徽章变绿、显示处置说明;
4. 重跑 `run_faith_eval.py` 该题再判编造 → 刷新后行自动回「未解决」、seen_count+1(spec §3.3 复发流转,**核对 T10 已测、此处端到端再验**);
5. 窄屏(浏览器缩到手机宽)表格横向滚动不破版;零新依赖。

- [ ] **Step 4: 提交**

```bash
git add static/faith.html static/index.html dev-notes/ch04.md
git commit -m "feat(ch04-ui): faith_cases轻量台账页(过滤/处置流转/复发退回),index顶栏入口

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 15: mine_qa 集成重锚定 + README ch04 节 + 全量终跑 + 验收四条演练 + dev-notes 完结

**Files:**
- Modify: `tests/test_mine_qa_integration.py`(一行断言重锚定)、`README.md`(追加 ch04 节)
- Create: `evals/demo_ch04.py`(验收演练脚本)

- [ ] **Step 1: mine_qa 集成测试锚机制不锚话题词(附录 C 重锚定项)**

`tests/test_mine_qa_integration.py:38` 现断言 `"運費險" in joined and "积分" in joined` 锚的是 ch03 自造语料话题词,老师语料换代后失效。替换该行为:

```python
        assert mined >= 4 and all(q.strip() and a.strip() for q, a in ins)  # 机制锚:条数+完整性;话题词锚随语料换代废止(附录C)
```

Run: `uv run pytest -m integration tests/test_mine_qa_integration.py -q`
Expected: PASS。若语义闸真的拦掉某话题样本致 `mined < 4`:允许把数量下限按实测放宽(如 `>=3`),**只准改数量、禁止恢复话题词锚定**;实际数字记 dev-notes。

- [ ] **Step 2: 写 `evals/demo_ch04.py`(验收②③数据面/④ 一条命令演练)**

```python
"""ch04 验收演练(README 引用)。前置:docker mysql/milvus 起、.env key 配好、build_knowledge 跑过。
验收①=run_strategy_eval 报告;验收③的 UI 侧在聊天页人工点(T13)。"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.engine import dispose_engine, get_session_factory, init_engine  # noqa: E402
from app.db.models import LowConfidenceQuestion  # noqa: E402
from app.rag import retriever  # noqa: E402
from app.rag.hit_format import format_hits  # noqa: E402


async def main() -> None:
    st = get_settings()
    init_engine(st)
    try:
        r = await retriever.retrieve("MH-LP100 猫粮保质期多久", strategy="bm25", settings=st)
        print(f"[验收2] BM25 型号题 top1 → {r.chunks[0].row.section_path}")
        h = await retriever.retrieve("退货要寄回去吗,邮费谁出", strategy="hybrid_rerank", settings=st)
        for x in format_hits(retriever.apply_head_tail(h.chunks))[:3]:  # 与 query_faq 同排布,[n] 编号才与生产一致
            print(f"[验收3] 引用[{x['n']}] chunk_id={x['id']} → {x['section_path']}")
        g = await retriever.retrieve("我家猫想吃没有的东西", strategy="hybrid_rerank", settings=st)
        print(f"[验收4] refused={g.refused} note={g.note}")
        async with get_session_factory()() as session:
            n = (await session.execute(
                select(func.count()).select_from(LowConfidenceQuestion))).scalar()
        print(f"[验收4] low_confidence_questions 池现有 {n} 行(UI 问同类问题后此数 +1)")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
```

Run: `uv run python evals/demo_ch04.py`
Expected: 四行打印;top1 章节含 `MH-LP100`、验收4 `refused=True`(无 rerank key 时闸1不写、该行可能 False——报告里如实注明,先补 key 再演)。

- [ ] **Step 3: README 追加 ch04 节(按既有逐章风格放文末;命令与本计划实测一致)**

```markdown
## ch04 RAG 进阶:混合检索 + 重排 + 评估体系

- **混合检索**:Milvus 原生 BM25(text 字段挂 BM25 Function、内置 chinese analyzer)+ dense 向量各 Top-50 → `hybrid_search` + RRF(k=60)融合;品类元数据先过滤再检索。
- **重排**:SiliconFlow `bge-reranker-v2-m3` 精排 Top-10;组装 prompt 用首尾排布(奇数位升序+偶数位降序),最相关在两端。
- **Query 理解**:口语→标准问法改写 + 检索侧同义词扩展(不入库);评估走 `evals/cache/rewrite_cache.json` 保四臂同输入。
- **前置双闸拒答**:检索侧 top1 置信 < `retrieval_low_conf_threshold`(实测校准终值)→ 拒答+进池;生成前证据自评不足 → 拒答+进池(`low_confidence_questions`,source 标写方)。
- **引用与反馈**:回答 `[n]` ↔ tool_result 帧 `citations` → 聊天页点角标弹窗看原文+章节路径(`GET /api/chunks/{id}`);每条回答 👍/👎 一次性锁定(纯前端 localStorage,数据飞轮入口)。
- **忠实度**:LLM-as-judge(`evals/run_faith_eval.py`)判 fabricated → `faith_cases` 台账(`static/faith.html`,GET/PATCH `/api/faith_cases`,复发自动退回)。
- **评估**:`evals/run_strategy_eval.py` 四策略×分桶 Hit/Recall@3、@10 + MRR@10 + D 桶阈值校准表 → `evals/reports/`。

​```bash
uv run python -m app.jobs.build_knowledge   # 老师 6 文档重建(knowledge_chunks + Milvus 双写)
uv run python evals/run_strategy_eval.py    # 验收①:四策略对比报告
uv run python evals/run_faith_eval.py       # 忠实度评估 + 台账落盘
uv run python evals/demo_ch04.py            # 验收②③④数据面演练
uv run pytest -q                            # 全量单测
uv run pytest -m integration -q             # 集成(需 docker mysql/milvus + key)
​```
```

(写盘时把围栏里两个零宽空格 `​` 去掉——它们是防嵌套转义,README 正文用普通 ```` ```bash ````。)

- [ ] **Step 4: 全量终跑**

Run: `uv run pytest -q` → 全绿;`uv run pytest -m integration -q` → 全绿(环境前提在输出顶部注明)
Expected: 0 failed。任何红:回到对应 Task 的处置规则修,**禁止**删测试/降断言强度迁就(§0-9 精神)。

- [ ] **Step 5: dev-notes 完结段(四要素)**

`dev-notes/ch04.md` 追记「计划执行与交付完结」段:①执行期用户关键指令原话 ②关键产出(各 Task commit 号、两份评估报告路径与核心数字、验收四条逐条证据)③执行中被拒/纠偏(如有,原话+处置)④翻车与返工(逐条,含集成允许红转绿的真实过程)。交付三件套在段尾显式列出:**演示命令**(README ch04 节 6 条)、**测试结果**(pytest 终跑输出)、**dev-notes 路径**。

- [ ] **Step 6: 提交**

```bash
git add tests/test_mine_qa_integration.py evals/demo_ch04.py README.md dev-notes/ch04.md
git commit -m "docs(ch04): README ch04节+验收演练demo+mine_qa集成重锚定(机制锚)+dev-notes完结交付

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## 计划落盘后自查记录(writing-plans Self-Review)

- **Spec 覆盖对账**:§3.1→T1/T7/T8;§3.2/附录A DDL→T1;§4.1 Query理解→T4;§4.2 混合检索/过滤→T2/T6;§4.3 重排→T5;§4.4 双闸→T6/T7/T8(终值→T12-Step5);§5.1 契约v2→T7;§5.2 首尾排布→T6/T7;§5.3 Prompt三改→T9;§6.1 解析器→T11;§6.2 四策略报告→T11;§6.3 Faithfulness→T12;§7 REST(原文回查/台账)→T10;§8 前端引用/反馈→T13;§8 台账页→T14;§9 无阻断线纪律→Global Constraints+T11/T12 报告声明行;§10 验收四条→T11-4/T2-4/T13/T15;§11 核对点①-⑥→T1(⑤DDL哨兵并入)/T2①/T4②/T5③/T9④/T10⑤⑥;附录C 重锚定→T3(corpus)/T6(retriever)/T11(rag_eval 弃用)/T15(mine_qa)。
- **占位符扫描**:全文无 TBD/TODO/「类似 Task N」;两处运行时实测值EXPECTED_TOTAL(T3-Step4)与校准终值(T12-Step5)均为「实测+来源+哨兵」步骤,非占位。
- **跨任务签名一致性**:`UnderstandResult(standard_query,synonyms,degraded)`(T4↔T6/T11)、`retrieve(...)→RetrieveResult`(T6↔T7/T11/T12经链路)、`format_hits/build_citations`(T7↔T13)、`upsert_faith_case(...)->str` 三态(T10↔T12)、`evaluate_evidence→EvidenceCheck|None`(T8↔T7服务侧)、`REFUSAL_ANSWER`(T7↔T8/T12)、tool_result 帧 citations 条件注入(T7↔T13/T14)。

**End of plan — 15 tasks。**

