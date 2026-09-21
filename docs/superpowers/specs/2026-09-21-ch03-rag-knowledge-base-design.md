# ch03 设计稿：RAG 知识库——query_faq 换芯向量语义检索

> 日期：2026-09-21 ｜ 分支：master ｜ 前置：ch02 Function Calling 工具链（已交付）
> 一句话：把 `query_faq` 的内部实现从 faq 表 SQL LIKE 升级为「text-embedding-v4 + Milvus dense 单路语义检索」，工具入参出参契约逐字段不变；配套离线建库（文档切分双写 + 历史对话挖 QA）。

## §0 决策记录

| # | 决策 | 结论 | 来源 |
|---|------|------|------|
| 1 | 嵌入模型 | **text-embedding-v4**（DashScope 兼容模式，复用现有 key）；技术栈段的 BGE-M3 被用户末尾改令覆盖 | 用户定稿（AskUserQuestion 确认） |
| 2 | Milvus 部署 | docker-compose 加 **standalone 服务**（宿主 19530）。Milvus Lite 原生不支持 Windows（milvus-lite 无 win wheel，官方 issue #176），嵌入文件方案排除 | 用户拍板 |
| 3 | 挖 QA「定时任务」形态 | job 函数 + **CLI 手动触发**为主，README 标注生产可挂 cron/APScheduler | 用户拍板 |
| 4 | 建库语料 | **自造样例**：三份 Markdown（含运费/包邮一节）+ 8 通历史对话种子 | 用户拍板 |
| 5 | 旧 faq 表 LIKE | **彻底替换，不回退**；向量无命中返回空 hits，LLM 按 Prompt 转工单兜底 | 用户拍板 |
| 6 | 流水线路线 | **A 自研轻管线**：chunker 纯函数 + `langchain_openai.OpenAIEmbeddings` 指 DashScope + 裸 `pymilvus.MilvusClient`（三条切分定制需求无现成 splitter 覆盖） | 用户拍板 |
| 7 | is_key_clause 标注 | **文档约定标记**：标题后缀「（关键条款）」，chunker 解析置 1；不用 LLM 判断 | 用户拍板 |
| 8 | 双写幂等机制 | 建库拆**两段**：入库段全量重建（无唯一键下的最简幂等）；向量化段扫 pending + Milvus 以 chunk.id 为主键 upsert——重跑捡漏。DDL 未给内容唯一键，此为本设计的解释，逐节评审通过 | 设计决策，用户认可 |
| 9 | Milvus 集合内容 | 只存 `chunk_id + embedding`，不回存文本；在线命中后回查 MySQL（权威源），孤儿向量天然过滤 | 设计决策，用户认可 |
| 10 | 章节不做 | 关键词召回、混合检索、重排——本章 dense 单路 | 用户需求 |

## §1 目标与范围

**功能需求（对应用户五条）**
1. 离线文档处理：Markdown 按标题层级结构感知切分；超长递归切；块间重叠裁到最近句号不留半截话；大表格按行切且每块复制表头
2. 对话挖知识：历史客服对话分批喂 LLM 抽 QA → `qa_extraction_staging` → 整体去重 → `knowledge_chunks`；CLI 触发、可中断重跑
3. 落库结构：`category + questions + answer` 三格拼向量文本；faq 填真实问法，policy/manual 的 questions=章节标题、category=上级标题路径；`section_path`/`content_type`/`is_key_clause`/`prev,next_chunk_id` 四类元数据只存不进向量
4. 双写落库：MySQL `knowledge_chunks` 原文权威源 + Milvus `knowledge` 集合；先 MySQL 记 pending，再 Milvus 回填 `vector_id` 翻 done；按主键幂等可重跑
5. 在线检索：query 向量化 → Milvus Top-K 相似度 → 替换 query_faq 的 LIKE 实现

**验收标准**
- 验收 1：「邮费是多少」换说法能召回运费说明并答对（FAQ 表红线无邮费字样，只有知识库命中一条路径）
- 验收 2：中断建库再重跑，漏向量化的 pending 块被捡起补齐，最终 MySQL done 数 == Milvus entity 数

**明确不做**：混合检索/稀疏向量/reranker；`faq` 表迁移工具（旧表保留不再被查）；APScheduler 常驻；向量库鉴权/TLS；文档级软删增量。

## §2 架构与模块布局

```
 ┌─ 离线 CLI job（python -m app.jobs.*）──────────────────────────────────┐
 │ knowledge/*.md ─► chunking.py(纯函数) ─► indexer.ingest ─► MySQL pending │
 │                                              │                           │
 │ messages 历史会话 ─► mine_qa 抽取 ─► staging ─► 三道闸去重 ─► MySQL       │
 │                                       (extracted)      (kept/discarded)  │
 │                                              ▼                           │
 │            indexer.vectorize_pending：embed → Milvus upsert → 回填 done  │
 └──────────────────────────────────────────────────────────────────────────┘
 ┌─ 在线（FastAPI，ch02 工具链不变）───────────────────────────────────────┐
 │ query_faq(keyword) ─► embeddings.embed_query ─► milvus_store.search      │
 │                       ─► retriever 回查 MySQL 组装 hits（原契约）         │
 └──────────────────────────────────────────────────────────────────────────┘
```

| 模块 | 职责 | 依赖 | 备注 |
|------|------|------|------|
| `app/rag/chunking.py` | Markdown → `ChunkDraft` 列表，全部规则纯函数 | 无 I/O | TDD 主战场 |
| `app/rag/embeddings.py` | `embed_texts(list)/embed_query`，批切+重试 | langchain_openai | API 型，单测 mock |
| `app/rag/milvus_store.py` | collection 初始化/upsert/search/count/health | pymilvus | 集成测试 |
| `app/rag/indexer.py` | 两段双写 + `--skip-existing` 指纹跳重 + 对账 | chunking+embeddings+milvus+crud | |
| `app/rag/retriever.py` | query → Top-K → 回查 MySQL → 结构化命中 | embeddings+milvus+crud | |
| `app/db/crud.py` 扩展 | knowledge_chunks / staging 读写 | models | 沿用 async sessionmaker |
| `app/jobs/build_knowledge.py` | 建库 CLI：ingest→vectorize，`--fault-after N` 中断注入、`--check` 对账 | indexer | 验收 2 入口 |
| `app/jobs/mine_qa.py` | 挖 QA CLI：抽取→去重入库→顺带 vectorize_pending | embeddings+milvus+crud+LLM | |
| `app/prompts/qa_mining.py` | 抽 QA Prompt | — | eval 验证（非 TDD） |
| `app/tools/definitions.py` | **只改 `query_faq` 函数体与 docstring**，签名与返回 dict 结构不变 | retriever | |
| `app/main.py` | startup 加 Milvus health 探活（WARN 不阻塞） | milvus_store | 同 ch02 工具探活风格 |

**docker-compose 新增**：etcd + minio + milvus standalone 三服务（官方 Windows 指南结构，镜像版本 pin 为 §11-② 核对点），端口 19530/9091，named volume 持久化。`knowledge/` 目录放三份样例文档。

## §3 数据层

### 3.1 MySQL

DDL 见**附录 A（用户提供，原样落盘 `db/init/03_ch03_schema.sql`）**：`knowledge_chunks` + `qa_extraction_staging` 两表。要点：
- `id` 为双方共享主键：Milvus 的 `chunk_id` 恒等于 `knowledge_chunks.id`（`vector_id = str(id)`）
- `vectorize_status ENUM('pending','done')` 是双写幂等状态的唯一依据；`idx_vectorize_status` 支撑捡漏扫描
- `prev/next_chunk_id` 自引用外键 `ON DELETE SET NULL`：链在文档内相邻 chunk 间构建（入库拿 id 后回填）
- **无内容唯一键**——幂等靠 §5 的两段机制与内存指纹，不靠 DB 约束（这是用户给定 DDL 的既有事实，本设计在其约束内成立）
- 老库升级：`docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp < db/init/03_ch03_schema.sql`（新库首启自动跑 01-04）；`db/init/04_ch03_seed.sql` 灌 8 通历史对话（附录 D）
- ORM：`KnowledgeChunk`、`QaExtractionStaging` 两个模型进 `app/db/models.py`，与附录 A 逐列对齐（ch02 惯例：模型仅供映射，建表以 .sql 为准）

### 3.2 Milvus

集合 `knowledge`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | INT64, primary_key, auto_id=False | = MySQL id |
| `embedding` | FLOAT_VECTOR(dim=1024) | text-embedding-v4 默认维度，`Settings.embedding_dimensions` 可调 |

索引/度量：AUTOINDEX + **COSINE**（具体创建 API 以 §11-① 核对待定版本 pymilvus 文档为准）。demo 规模（数百 entity）不压测不做分区。

## §4 chunker 细则（纯函数，TDD）

输入：`(file_text, doc_name)`；输出：`ChunkDraft` 列表，字段即 MySQL 列（除 id/prev/next/vector 状态，由 indexer 补）。

1. **结构感知**：按 `#`/`##`/`###` 标题栈切 section，`section_path` = 各级标题以 ` > ` 连接；标题后缀「（关键条款）」→ `is_key_clause=1`（后缀从 section_path 中剥离）
2. **超长递归**：section 正文 > `chunk_size`（默认 500 字，按 `len()` 字符计）时，依次按 段落(`\n\n`) → 行(`\n`) → 句 三级递归到 ≤chunk_size；句边界 = `。！？；` 及 `.`+空格+非数字
3. **重叠裁句**：相邻块之间加 `chunk_overlap`（默认 80 字）重叠。取**前块末尾能凑成的整句后缀**（从句子边界起、总长 ≤ `overlap×1.5`，优先 ≥ `overlap×0.5`），前置进后块开头；凑不出整句则**宁缺毋滥不重叠**——两个方向都不产生半截话
4. **表格按行**：连续 `|` 行构成表块；表块超 chunk_size 时按数据行分组建块，每块复制**表头行+分隔行**；表块 questions=所在节标题（同 policy/manual 规则）
5. **三格填法**：

| content_type | category | questions | answer |
|---|---|---|---|
| `faq`（商品 FAQ 文档） | 所在 H2 节名（商品分类） | 该组的真实问法，多条 `\n` 分隔 | 答案 |
| `policy` / `manual` | 上级标题路径（=section_path 去末级） | 所在章节标题 | 节正文 |
| `qa_mined` | 固定 `客服对话问答` | 真实问法（合并同类） | 客服答案 |

6. **向量文本**：`"\n".join([category, questions, answer])`——只在 embed 时拼接，不落单独列
7. **文档类型**：由文件头 front-matter `content_type:` 声明（return-policy.md→policy、product-faq.md→faq、aftersale-manual.md→manual）；chunk 顺序 = 文档内阅读顺序
8. **FAQ 文档解析**（`content_type: faq` 专属）：一行以 `**Q：` 开启一个 chunk；组内连续多条 Q 行（同义问法）并入该 chunk 的 questions；`**A：` 行至下一个 Q/标题为该 chunk 的 answer；FAQ 文档内的非问答内容（表格等）按 policy 三格规则处理（questions=所在节标题），content_type 仍记 `faq`

`ChunkDraft` 间 prev/next 链由 indexer 入库后按序回填（同文档首尾 NULL）。

## §5 双写 indexer（两段幂等）

```
Stage 1 ingest：清表 knowledge_chunks（自引用 FK 下 TRUNCATE 的处置见 §11-⑤）
                → 逐文档 chunk → 批量 INSERT(pending)
                → 回填 prev/next
  （--skip-existing 模式：不清表，读全表现存行在内存算
    sha1(category|questions|answer) 指纹，同指纹跳过——demo 规模全表进内存）
Stage 2 vectorize_pending：SELECT * WHERE vectorize_status='pending'
     按 embedding_batch_size 分批：embed → upsert(chunk_id=id) → 本批回填
     vector_id=str(id) + status='done'（每批一次 commit，批内 MySQL 事务）
```

- **幂等论证**：Stage 1 中断→重跑整段（表要么旧要么新，无半态污染）；Stage 2 中断→embed/upsert/回填任一点断掉，行仍 pending，重跑以同主键 upsert 覆盖 Milvus——「按主键幂等」的精确含义。**upsert 成功但回填前崩溃**是有意允许的窗口：重跑再 upsert 一次无害
- **全量重建的副作用（明示）**：`qa_mined` 行也在清除之列，Milvus 旧向量成孤儿（在线回查 MySQL 时天然过滤，无正确性问题）；重建后想保留挖出的知识，用 `--skip-existing` 或重跑 mine_qa
- `--fault-after N`：Stage 2 向量化满 N 块后 `sys.exit(42)`，验收 2 的中断注入
- `--check`：输出 `pending 数 / done 数 / Milvus count / 差集`，双端对账
- embedding 失败：批内指数退避重试 3 次，仍败→整轮退出（pending 语义保证可续）

## §6 对话挖 QA（mine_qa，两阶段一条 CLI）

```
阶段一 抽取：候选 = messages 中 role∈{user,assistant} 的会话，且 source_ref
  ="conv:{id}" 不在 staging 任何行（会话级记账即幂等）。按 --batch-size（默认 5
  通）分批，批号 batch_no=时间戳+随机后缀；每通会话单独一次 LLM 调用
  （qwen-plus + app/prompts/qa_mining.py + with_structured_output，§11-④ 核对），
  抽 [{question, answer}] → 逐行写 staging(status=extracted)。
  LLM 输出不合 pydantic → 抛错重试 1 次，再失败整通不落账（source_ref 无记录，
  重跑自然再抽，不自造重试状态机）

阶段二 去重入库（整体执行；--dedup-only 可单跑）：
  闸1 规范化精确去重：NFKC + 去空白标点归一；同问法多行 → 合并进同一 chunk 的
     questions 多行（answer 取首次出现的）
  闸2 语义去重（批内）：两两 cosine ≥ qa_dedup_threshold(0.92) 归并
  闸3 语义去重（对库内）：embed 后与 Milvus 现有向量 search，≥ 阈值 → discarded
  过闸 → 一批内 INSERT knowledge_chunks(qa_mined, pending) + 对应 staging 行
  翻 kept 放同一事务（断→回滚→行仍 extracted→幂等）；未过闸翻 discarded，
  行保留可追溯（--clear-staging 才物理清）
  阶段二尾 → 顺带跑 indexer.vectorize_pending()（挖出的新知即刻可检索）
```

阈值 0.92「宁严勿宽」：误杀靠调低救回，误放仅致近似重复命中。

## §7 在线检索与 query_faq 改造

**retriever**：`embed_query` → `milvus_store.search(top_k=rag_top_k(默认 5))` → 过滤 `score < rag_score_threshold` → chunk_ids 回查 MySQL（`id IN (...)`）→ 按相似度序回排 → 返回命中对象列表。Milvus 有 id 而 MySQL 无行（孤儿）→ 跳过。

**query_faq（契约逐字段不变）**：

```python
@tool
async def query_faq(keyword: str) -> dict:
    # docstring 调整：「keyword: 用户的原始问题完整句子」（语义检索吃整句；
    # 参数名、类型、返回结构均不动）
    hits = await retrieve_hits(keyword)
    return {"keyword": keyword,
            "hits": [{"id": c.id,
                      "question": c.questions.splitlines()[0],
                      "answer": c.answer,
                      "category": c.category} for c in hits]}
```

- 空命中 `hits: []`，**不回退 LIKE**；`crud.search_faq` 与 faq 表保留（ch02 既有测试引用），仅 query_faq 不再调用
- 异常向上抛，executor 统一包装（5s 超时、重试 1 次沿用 ch02）
- `app/prompts/customer_service.py`：工具指引补「query_faq 按语义匹配，传用户原话整句，勿自行拆词」；「政策类必须先调工具」红线不动
- `app/main.py` startup：Milvus `health()` 探活，失败 WARN 不阻塞

## §8 错误处理矩阵

| 故障点 | 策略 | 恢复路径 |
|---|---|---|
| embedding 单批失败 | 指数退避 3 次→整轮退出 | 行保持 pending，重跑自捡 |
| Milvus 不可用（离线） | 建库/挖 QA 开始即 health() fail-fast | 不起半库 |
| Milvus 不可用（在线） | 启动 WARN；运行期 query_faq 抛错 | executor 错误帧→LLM 转工单 |
| LLM 抽取不合 schema | 校验+重试 1 次→整通不落账 | source_ref 无记录，重跑再抽 |
| upsert 后回填前崩溃 | 允许窗口存在 | 同主键再 upsert 无害 |
| 去重入库中断 | chunk 插入+kept 同事务 | 回滚→extracted→重跑幂等 |
| MySQL 不可用 | aiomysql 异常直抛 | 同 ch02 行为 |

## §9 测试与评估

**pytest（TDD 适用代码）**
- `test_chunking.py`：标题栈/section_path、（关键条款）剥离、递归降档、重叠整句裁切（含「凑不出整句不重叠」分支）、表头复制、三格填法、向量文本拼接、front-matter 类型
- `test_embeddings.py`（mock）、`test_indexer.py`（mock embed+milvus，真 sqlite 不行——MySQL 方言，用 mock session；真库行为进集成）、`test_retriever.py`（mock）、`test_mine_qa.py`（三道闸纯逻辑 + 事务回滚 mock）、`test_models_ch03.py`（ORM vs 附录 A 逐列，ch02 惯例）
- **ch02 回归红线**：除 query_faq 相关断言（改为 mock retriever 验契约结构）外全绿；tool_routing eval 不动

**集成（`@pytest.mark.integration`，默认 skip，demo 环境 `uv run pytest -m integration`）**：真 Milvus+DashScope 跑 §10 验收序列。

**评估集（Prompt/数据类任务替代 TDD，沿用 evals/ 惯例）**
- `evals/rag_retrieval_samples.json`：**≥12 条**换说法 query→期望命中（附录 C），`run_rag_eval.py` 报 hit-rate@3，目标 **≥10/12**；`rag_score_threshold` 初值 0.3 在此**实测标定**（标注为校准值）
- `evals/qa_mining_samples.json`：5-8 段对话→期望 QA 要点，`run_qa_mining_eval.py` 报漏抽/幻觉
- `rag_retrieval` 中「邮费/运费」组样例是验收 1 的量化形态，**必须全过**

**演示/验收命令（README ch03 节）**
```bash
docker compose up -d
uv run python -m app.jobs.build_knowledge            # 全量重建（放在 mine_qa 之前跑，见 §5 副作用）
uv run python -m app.jobs.mine_qa                    # 挖对话知识
uv run python -m app.jobs.build_knowledge --skip-existing   # 增量补向量（可选演示）
# 验收1：起 uvicorn 问「邮费是多少」→ query_faq 徽章 + 命中运费说明
# 验收2：--fault-after 3 打断 → SELECT pending → 重跑 → --check 双端对账
uv run python -m app.jobs.build_knowledge --check
```

## §10 配置与依赖新增

`Settings`（全带默认值，不破坏 ch01/ch02 构造）：`milvus_uri="http://127.0.0.1:19530"`、`milvus_collection="knowledge"`、`embedding_model="text-embedding-v4"`、`embedding_dimensions=1024`、`embedding_batch_size=10`（上限值待 §11-③ 按官方文档校准）、`chunk_size=500`、`chunk_overlap=80`、`rag_top_k=5`、`rag_score_threshold=0.3`（评估集校准值）、`qa_dedup_threshold=0.92`、`qa_mine_batch_conversations=5`。embedding 复用 `openai_base_url/openai_api_key`，不另设密钥。

`pyproject` 新增：`pymilvus>=2.5,<3`（版本经 §11-① 核对后 pin）。dev 组无新增。`.env.example` 同步。

## §11 实施时硬性二次核对点（工作要求 3：Context7 先查再写）

1. **pymilvus**：`MilvusClient` create_collection schema/`upsert`/`search`(output_fields、distance→COSINE 相似度方向与取值域)/`health`/count 的当前签名 → 决定 milvus_store 全部门面方法与 score 过滤实现
2. **Milvus standalone compose**：官方 Windows 指南当前 pin 的镜像版本组合（milvus/etcd/minio）与必需 env；健康检查写法
3. **DashScope text-embedding-v4**：兼容模式 `/embeddings` 的 batch 上限、`dimensions` 参数支持、OpenAIEmbeddings 直连的坑（`check_embedding_ctx_length` 分词路径、encoding_format）→ 定 embedding_batch_size 与 embeddings.py 实现
4. **LangChain 结构化抽取**：当前版本 `with_structured_output` + qwen-plus（DashScope 兼容模式）的 function/tool-call 路径是否可靠 → 定 mine_qa 抽取实现（不可靠则回落 JSON-mode+pydantic 校验，属实现细节非选型变更）
5. **ENUM 列 + aiomysql + TRUNCATE**：staging/knowledge_chunks 两个 ENUM 读写在 async session 下的语句形态（沿用 ch02 已验模式）。**已知坑**：`knowledge_chunks` 有自引用外键，MySQL 对带 FK 的表直接 `TRUNCATE` 报 1701——Stage 1 清表实现为 session 内 `SET FOREIGN_KEY_CHECKS=0` → TRUNCATE → 复原（或退化 `DELETE FROM`），实施时实测验证，禁止凭记忆写

## §12 ch02 兼容红线

- 工具五件套**名称、参数名、返回 dict 结构**全部不变；仅 query_faq 函数体与 docstring 变
- SSE 帧协议（token/tool_call/tool_result/done/error）零改动；前端零改动
- ch01/ch02 既有测试除 §9 声明的 query_faq 断言最小必要适配外不动且全绿；README ch02 三条 curl 验收 1、2 继续成立
- **验收 3 语义翻转**：ch02 的「邮费是多少→预期漏召回」是本章立的目标，README ch02 处加一行「(ch03 起此问已能命中,见 ch03)」注记——这是刻意演进,不算破坏兼容
- 聊天主链路（无工具 chat_service）零改动

## 附录 A：建表 DDL（用户提供，原样落盘 `db/init/03_ch03_schema.sql`）

```sql
-- =============================================================
-- ch03 · RAG 基础 · 建表 DDL
-- 本章新建:knowledge_chunks(知识库原文权威源)
-- 向量落 Milvus 集合 knowledge(非 MySQL,DDL 不含);MySQL 存原文 + 双写状态
-- category + questions + answer 三格拼成向量化文本;其余字段是元数据,只存不进向量
-- =============================================================

SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```

（用户原文含首条 `SET NAMES utf8mb4` 前的注释块「确保中文 COMMENT…」一并保留于落盘文件。）

## 附录 B：知识样例文档

**B.1 `knowledge/return-policy.md`（policy，验收 1 数据源，全文定稿）**：

```markdown
---
content_type: policy
---
# 退货政策

## 退换货条件（关键条款）
签收后 7 天内支持无理由退换，商品需保持未洗涤、未使用且吊牌完整。
15 天内出现质量问题可享受免费修换，需提供订单号与问题照片。
定制类商品（如喵帮定制猫爬架）不支持无理由退换，质量问题除外。

## 运费说明
退货运费：7 天无理由退换由买家承担寄回运费；质量问题换货由我方承担双向运费。
发货运费：全场单笔订单实付满 99 元包邮，未满 99 元收取基础运费 8 元。
偏远地区（新疆、西藏、内蒙古部分地区）包邮门槛为 199 元，未达门槛运费 15 元。
秒杀与一元试用商品不享受包邮优惠，运费下单时按地区实时计算。

## 退款到账
验收通过后 1-3 个工作日按原支付路径退回，到账时间以支付渠道为准。

## 不支持退换的情形（关键条款）
人为损坏、洗涤后、吊牌剪失、超过 15 天质量窗口、内服类拆封商品不支持退换。
```

**B.2 `knowledge/product-faq.md`（faq，真实问法体）**：front-matter `content_type: faq`；H1「商品 FAQ」，H2 为商品分类（猫粮/用品/清洁）；每个 H2 节内以「**Q：问法** A：答案」行构成多问法 chunk（≥8 组）；含一张 5 行「冻干猫粮喂食量表」（列：月龄/体重/每日克数）演示表块处理。

**B.3 `knowledge/aftersale-manual.md`（manual，长文+大表）**：H2 小节 ≥4（售后服务时间/转人工流程/破损件处理/物流异常赔付标准），其中「物流异常赔付标准」放 12+ 行长表触发**按行切+表头复制**；「转人工流程」「破损件处理」标（关键条款）。

## 附录 C：检索评估集标注（`evals/rag_retrieval_samples.json`，12 条）

| # | query | 期望命中（section_path/关键词） |
|---|-------|------|
| 1 | 邮费是多少 | return-policy > 运费说明 |
| 2 | 寄回来要出运费吗 | return-policy > 运费说明 |
| 3 | 买多少才包邮 | return-policy > 运费说明 |
| 4 | 新疆的快递费怎么算 | return-policy > 运费说明（偏远门槛） |
| 5 | 退货要自己付快递吗 | return-policy > 运费说明 |
| 6 | 收到东西破了不想留 | return-policy > 退换货条件（7 天无理由） |
| 7 | 猫粮开封后能吃吗拆了还能退吗 | return-policy > 不支持退换的情形 |
| 8 | 多久能收到退款 | return-policy > 退款到账 |
| 9 | 幼猫一天喂多少合适 | product-faq > 冻干猫粮喂食量表 |
| 10 | 饮水机怎么清洗 | product-faq（用品节） |
| 11 | 快递把管子压扁了赔不赔 | aftersale-manual > 物流异常赔付标准 |
| 12 | 晚上十点能找人工吗 | aftersale-manual > 售后服务时间 / 转人工流程 |

通过线：hit-rate@3 ≥ 10/12，且 1-5（运费组）**5/5 全过**（验收 1）。

## 附录 D：历史对话种子（`db/init/04_ch03_seed.sql`，8 通）

| 会话 | 话题 | 期望挖掘 |
|---|---|---|
| C1 | 运费三连问（包邮门槛/退货邮费谁出） | 与文档重复 → 闸3 语义去重 discarded |
| C2 | 退换流程咨询 | 与文档重复 → discarded |
| C3 | 「一元秒杀怎么不包邮」客服答同文档 | 语义重复 → discarded（闸3 边界样例） |
| C4 | 猫粮搭配冻干能否混喂 | **增量知识** → kept（qa_mined） |
| C5 | 饮水机滤芯多久一换 | **增量知识** → kept |
| C6 | 积分怎么获得抵扣 | **增量知识** → kept（文档完全无此话题） |
| C7 | 运费险能不能买（客服：暂不支持） | **增量知识** → kept（问法近运费但答案新，去重不误杀） |
| C8 | 同一通会话里用户换两种说法问同一件事，客服同义作答 | LLM 抽出两条近义 QA → 闸2 批内归并，kept 一条（questions 双行） |

去重演示预期：8 通抽出的 QA 中 ≥3 条 discarded、≥4 条 kept——C7 是「语义去重别误杀」的反例锚点。
