# ch04 设计稿:RAG 进阶——混合检索+重排+评估体系+生成质量控制

> 日期:2026-09-23 ｜ 分支:ch04 ｜ 前置:ch03 RAG 知识库(已交付 @ e5cbc78)
> 一句话:query_faq 从「dense 单路」升级为「Milvus 原生 BM25 + hybrid_search RRF + SiliconFlow bge-reranker-v2-m3 精排 + 首尾排布」,配前置双闸拒答与低置信度池;消费老师 300 题评估集出四策略对比报告与忠实度台账;前端做引用弹窗、满意度反馈采集与 faith_cases 台账页。

## §0 决策记录

| # | 决策 | 结论 | 来源 |
|---|------|------|------|
| 1 | 重排模型部署 | **SiliconFlow 云 API** 托管 `BAAI/bge-reranker-v2-m3`(`/v1/rerank`,httpx 直连,零新依赖);模型名用户定死,部署形态 AskUserQuestion 拍板 | 用户选推荐项 |
| 2 | 拒答判定 | **前置双闸**:闸1 rerank top1 < 阈值 → 拒答落池 `retrieval_low_conf`;闸2 生成前自评不足 → 拒答落池 `self_check`;闸2 调用失败 → WARN 降级放行生成 | 用户选推荐项+批次B确认 |
| 3 | 元数据过滤暴露面 | **仅 retriever API 参数**(category/content_type),在线 query_faq 不暴露给 LLM | 用户选推荐项 |
| 4 | faith_cases 台账 | **轻量台账页** static/faith.html + REST(列表/状态按钮/处置说明必填) | 用户选推荐项,兑现 DDL「人工点按钮」注释 |
| 5 | 知识库来源 | 老师 knowledge/ 六份 md 为**唯一来源**;ch03 三份自建语料作废;验收演示**不跑 mine_qa**(纯文档库) | 用户材料替代令 |
| 6 | 评估题库 | 老师 `evals/run_rag.py`(实为 300 题 **CSV** 数据,无任何代码)为唯一题库,数据零修改,runner 我方新写;ch03 12 题集(deprecated 标记保留)不再作为本章验收依据 | 用户材料替代令 |
| 7 | ch03「Milvus 只存 id+向量不回存文本」决策 | **推翻**:原生 BM25 必须在集合内存 text;原文权威源仍是 MySQL,Milvus 只是检索冗余(回查/孤儿过滤语义不变) | §1-#1 技术栈必然推导,批次A确认 |
| 8 | 引用编号 [n] | 用**首尾排布后的列表位置**编号([1][10] 最强,[5][6] 最弱),prompt 顺序与编号自洽;前端不做 rank 暗示 | 批次B确认 |
| 9 | 评估目标 | 不设「达不到不进交付」硬线;四策略报告如实出数,B 桶 bm25 臂 Recall@10 ≥90% 作为验收2 数字证据,**不达标如实报因不拦截** | 用户批次C纠偏 |
| 10 | 同义词扩展位置 | 只做在**查询侧**(BM25 腿吃「标准问法+同义词拼接」);入库侧一份原文,不拆存多份 | 用户功能需求4 |
| 11 | D 桶编造入台账 | 应拒未拒且编造 → faith_cases 记 bucket='D_absent'(超出 DDL 注释所列四桶的正当扩展,台账价值优先) | 批次C确认 |
| 12 | user_feedback 入池 | 本章枚举**预留无写方**;👍/👎 纯前端 localStorage 采集(ch09 数据飞轮入口) | 用户功能需求7 |
| 13 | 本章不做 | 指代消解、多轮改写;faq 表迁移;向量库鉴权;文档级增量 | 用户「本章不做」+YAGNI |

## §1 目标与范围

**功能需求 → 章节映射**

| 用户功能需求 | 落点 |
|---|---|
| 1 混合检索(Milvus BM25,chinese analyzer,dense+BM25 各 Top-50,hybrid_search+RRF) | §3、§4.2 |
| 2 重排 bge-reranker-v2-m3 出 Top-10;首尾排布 | §4.3、§4.4 |
| 3 元数据过滤(品类先过滤再检索) | §4.2 |
| 4 Query 理解(口语→标准问法;同义词只查询侧) | §4.1 |
| 5 生成质量控制(引用编号/显式拒答/自评落池/负面知识清单) | §5 |
| 6 评估体系(难度梯度题库/Recall@K/MRR/Faithfulness/四策略/分桶报告) | §6 |
| 7 前端(引用可点弹窗/👍👎一次性锁定) | §7 |

**验收标准(用户四条)**
1. 四策略对比报告能跑出数字 → `run_strategy_eval.py` 产物 §6.4
2. 问带具体型号的问题,BM25 那路能命中 → B 桶 bm25 臂 Recall@10 数字 + 集成测试「MH-LP100 裸 BM25 命中」§6.1/§9
3. 答案引用编号能定位回原文,聊天页点引用看到来源原文 → §5.1/§7
4. 问知识库没有的内容,明确拒答且问题进低置信度池 → D 桶拒答正确率 + `low_confidence_questions` 落库实查 §6.3

**明确不做**:见 §0-13。

## §2 架构与模块布局

```
 ┌─ 离线(沿用 ch03 两段框架,升级第二段)────────────────────────────────┐
 │ knowledge/*.md(6) → chunking.py(零改动)→ MySQL pending              │
 │ → vectorize_pending v2:embed(三格拼接)+ upsert{chunk_id,text,        │
 │   embedding,category,content_type}(sparse 由 BM25 Function 服务端生成)│
 │ → 回填 done;全量重建 drop 旧集合 → 新 schema 重建(迁移即重建)       │
 └──────────────────────────────────────────────────────────────────────┘
 ┌─ 在线 query_faq(签名不变,返回契约 v2)──────────────────────────────┐
 │ keyword → query_understanding(LLM→{standard,synonyms},失败降级原话) │
 │  → retriever(strategy=hybrid_rerank):dense Top50 ⊕ BM25 Top50       │
 │     → hybrid_search(RRFRanker(60))                                   │
 │  → reranker(SiliconFlow,失败降级 RRF 序+闸1跳过)→ Top10+relevance    │
 │  → 闸1:top1<low_conf_threshold 或零命中 → refused=True+落池,不生成   │
 │  → head_tail_order([1,3,5,7,9,10,8,6,4,2])→ hits 带 [n] 编号返回     │
 │ 编排层 tool_chat_service:                                            │
 │   query_faq 有证据 → 闸2 自评 {sufficient,reason};false → 固定拒答   │
 │     话术+落池(self_check),不进第二轮                                │
 │   true/放行 → 第二轮生成(System Prompt:[n] 引用规范+禁止承诺清单)    │
 │ SSE:tool_result 帧新增可选 citations[](其余帧语义不变)              │
 └──────────────────────────────────────────────────────────────────────┘
 ┌─ 评估(新 runner,消费老师题库)─────────────────────────────────────┐
 │ run_strategy_eval.py:240 题 × dense|bm25|hybrid|hybrid_rerank        │
 │   Recall@3/@10+MRR 分桶;D桶=low_conf 阈值校准样本(分数分布)        │
 │ run_faith_eval.py:hybrid_rerank 全链生成 → 忠实度裁判 → faith_cases  │
 │   upsert;D 桶拒答正确率                                              │
 └──────────────────────────────────────────────────────────────────────┘
 ┌─ 前端(Vibe Coding)──────────────────────────────────────────────────┐
 │ 聊天页:[n] 角标可点 → 弹窗(GET /api/chunks/{id},原文+section_path   │
 │   +上一块/下一块);每条回答左下 👍/👎 一次性锁定(localStorage)        │
 │ static/faith.html 台账页:列表/筛选/三态按钮(resolution 必填)/复发   │
 │   标记/citations 展开 —— GET+PATCH /api/faith_cases                   │
 └──────────────────────────────────────────────────────────────────────┘
```

| 模块 | 动作 | 职责 |
|------|------|------|
| `app/rag/query_understanding.py` | 新增 | LLM 改写+同义扩展;JSON 解析/降级纯函数 |
| `app/rag/reranker.py` | 新增 | SiliconFlow /v1/rerank 客户端(超时/降级) |
| `app/rag/retriever.py` | 重构 | 策略化 dense/bm25/hybrid/hybrid_rerank;闸1;元数据过滤;首尾排布挂点 |
| `app/rag/milvus_store.py` | 升级 | 全 schema 建集(Function/双索引);hybrid_search 门面;单腿方法 |
| `app/rag/indexer.py` | 小改 | vectorize_pending v2 写 text/category;drop-recreate 迁移 |
| `app/services/tool_chat_service.py` | 增闸2 | 生成前自评、拒答出口、落池钩子 |
| `app/tools/definitions.py` | 改 query_faq | 返回契约 v2;`config: RunnableConfig` 取 conversation_id |
| `app/prompts/customer_service.py` | 改造 | [n] 引用规范、禁止承诺负面清单、拒答行为 |
| `app/prompts/` 新增 rewrite/self_check/faith_judge 三个 | 新增 | 数据类任务,eval 样例验证替代 TDD |
| `app/db/models.py`+`crud.py` | 扩展 | 两新表 ORM;入池/台账 upsert/状态流转 |
| `app/api/routes.py`+`schemas` | 扩展 | /api/chunks/{id}、/api/faith_cases GET/PATCH;citations 帧字段 |
| `static/index.html` | 前端 | 引用弹窗+反馈锁 |
| `static/faith.html` | 前端 | 台账页 |
| `evals/run_strategy_eval.py`+`run_faith_eval.py` | 新增 | 评估 runner(老师 `run_rag.py` 数据文件原样不动) |

## §3 数据层

### 3.1 Milvus `knowledge` 集合 v2

建集不可改 schema → 迁移 = `build_knowledge` 全量重建自动 drop-recreate(既有路径)。

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunk_id` | INT64, PK, auto_id=False | = MySQL `knowledge_chunks.id`,ch03 恒等关系不变 |
| `text` | VARCHAR(max_length=定值,核对点②), `enable_analyzer=True`, `analyzer_params={"type":"chinese"}`(确切写法以核对点①实测为准) | = 三格拼接 `category\nquestions\nanswer`,与 dense 同源,两腿吃同一份证据 |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 Function 输出,**客户端不写** |
| `embedding` | FLOAT_VECTOR(1024) | text-embedding-v4,不变 |
| `category` | VARCHAR(765) | 元数据过滤标量字段(需求3) |
| `content_type` | VARCHAR(32) | 预留过滤维度(faq/policy/manual/qa_mined) |

`Function(name="bm25_fn", input=["text"], output=["sparse"], function_type=BM25)`;索引:dense AUTOINDEX+COSINE;sparse SPARSE_INVERTED_INDEX+BM25。upsert 数据形状(text 必填/sparse 缺席)按核对点①定。flush 语义沿用(ch03 核对点①实测)。

### 3.2 MySQL

- 老师 DDL(附录 A)**逐字原样**落 `db/init/05_ch04_schema.sql`(含全部注释与 `SET NAMES utf8mb4`);老库升级 `docker compose exec -i mysql ... < 05_...sql`,新库首启自动跑 01-05
- ORM 新增 `LowConfidenceQuestion`、`FaithCase`(逐列对齐,ch02/ch03 惯例:建表以 .sql 为准,模型仅供映射;`test_models_ch04` 列对账)
- 老师 DDL 语义实现要点:
  - `low_confidence_questions.source` 三值枚举本章只有两写方(§0-12);`conversation_id` 可空 FK→conversations
  - `faith_cases` **一题一行(uk_eval_id)**:重判 → 更新 answer/reason/citations 快照 + `seen_count+1` + `last_seen_at`;状态已解决/无需解决又被判 → 自动退回「未解决」且 `resolution` 清空、`resolved_at` 保留(复发标记依据 DDL 注释);`citations` JSON = 该轮喂模型 Top-K **全集**(含未被引用者)
- `knowledge_chunks` 零改列

## §4 检索层

### 4.1 Query 理解(`query_understanding.py`)
1. 一次 LLM 调用(默认 `model_name`,可配 `query_rewrite_model`)→ JSON `{standard_query, synonyms[]}`(≤4 个,须是实词,禁整句改写)
2. dense 腿 embed `standard_query`;BM25 腿检索文本 = `standard_query + " " + " ".join(synonyms)`;同义词只查询侧(§0-10)
3. 失败/超时/不合 schema → 原话直检,WARN(理解层不许断链路)
4. 评估共用同一预处理:rewrite 结果缓存 `evals/cache/rewrite_cache.json`,四臂同输入、重跑零漂移

### 4.2 混合检索与过滤(`milvus_store.py` 门面)
1. `hybrid_search(client, vec, bm25_text, limit, expr=None)`:双腿 `AnnSearchRequest`(各 limit=`hybrid_recall_k`=50)→ `RRFRanker(k=rrf_k=60)`;返回 `[(chunk_id, rrf_score)]`
2. `dense_only`/`bm25_only` 单腿方法:评估臂与在线**同一代码路径**(策略=开关,不存在第二实现)
3. `rag_score_threshold`(0.3,cosine 域)**只作用于 dense 腿**;BM25/RRF 分数不同域,低置信判定由闸1(rerank 分数)承担
4. 元数据过滤:值经白名单 `^[\w一-鿿 >()×/,-]+$`(拒引号/反斜杠)后拼 `category == "v"` expr;非法值 ValueError;retriever 参数 `category=None`
5. 回查 MySQL 组装/孤儿过滤/相似度序沿用 ch03
6. 新库冒烟硬断言:「MH-LP100」裸 BM25 命中「自动猫砂盆 Pro」节 = chinese analyzer 生效现场证据

### 4.3 重排(`reranker.py`)
1. httpx async `POST {rerank_api_base}/rerank`,body `{model, query, texts, top_n, return_documents:false}`(字段以核对点③为准);返回 `[(候选下标, relevance_score∈0~1)]`
2. 候选=RRF Top-50 全量精排,取 `rerank_top_n`=10
3. key 未配置:启动探活 WARN;运行期 超时/HTTP 错 → **降级用 RRF 前 10 序继续 + 闸1 跳过**(WARN 一次)
4. 评估 dense/bm25/hybrid 三臂不经重排(策略定义即不含)

### 4.4 首尾排布(`head_tail_order` 纯函数,TDD)
- 输入相关性降序 rank 1..n → 输出 `[1,3,5,7,9,10,8,6,4,2]`(n≠10 泛化:奇数升序铺前段、偶数降序铺后段)
- hits 列表按此序给模型,**引用编号 [n] = 列表位置**(§0-8)

## §5 生成质量控制

### 5.1 query_faq 契约 v2(签名不变)
```python
{"keyword": str,
 "hits": [{"n": 1..10, "id": int, "question": str, "answer": str,
           "category": str, "section_path": str}],          # 首尾序
 "refused": bool,          # 闸1/零命中
 "note": str}              # 拒答机器可读理由(入池 reason)
```
executor 从结果提取 `citations:[{n, chunk_id, section_path, question}]` 挂 tool_result 帧(只增字段,旧前端兼容)。

### 5.2 闸1 检索侧低置信(source=`retrieval_low_conf`)
- 判定点 retriever(hybrid_rerank 下):零命中/全被过滤 或 **rerank top1 < `retrieval_low_conf_threshold`**(初值 0.3,D 桶校准 §6.2)→ `refused=True, hits=[]`
- query_faq 写池:raw_question=keyword(用户原话整句,ch03 约定),conversation_id 经 `RunnableConfig`(LangChain 注入不进 LLM schema,create_ticket 先例,核对点④复核)
- 编排层见 refused → 跳过闸2 直接拒答出口(不消耗生成)

### 5.3 闸2 生成前自评(source=`self_check`)
- 时机:本轮有 query_faq 证据、第二轮生成**前**;输入=用户原话+Top-K 证据摘要;输出 `{sufficient, reason}`;prompt 明确「只是相关但答不中问题=不够」
- false → 流式固定拒答话术(同闸1 出口)→ 写池(reason=模型理由)→ assistant 落库+done 正常收流,**不进第二轮**
- 自评调用失败 → **WARN 降级放行生成**(§0-2);`self_check_enabled=False` 整体关闭(评估对照/省调用)
- 写池失败 → WARN 不阻断拒答(与 persister 同风格)

### 5.4 System Prompt 改造(`customer_service.py`)
1. **引用规则**:来自知识库的事实句必须以 `[n]` 结尾,n 只许用 hits 实存编号;一句可连引 `[2][7]`;无依据内容禁止带编号
2. **禁止承诺负面清单**(硬红线,语料显式要求):不承诺退款/补发**具体到账工作日数字**、不承诺「一定能退/一定赔」、不给「几天必到」时效承诺;此类诉求只说流程+建议转人工
3. **拒答行为**:refused=true 或 hits 空 → 如实说查不到+建议工单,严禁编造(闸外的模型自觉层,双保险)
4. 保留 ch03 红线:政策必先调工具、keyword 传整句

### 5.5 新增 REST
- `GET /api/chunks/{id}` → {id, section_path, category, questions, answer, content_type, prev_chunk_id, next_chunk_id};404 兜底
- `GET /api/faith_cases`(status/bucket 筛选 + eval_id 搜索)
- `PATCH /api/faith_cases/{id}` → status ∈ {未解决,已解决,无需解决};**标已解决/无需解决时 resolution 必填否则 422**(DDL 注释「空处置=没交代」);退回未解决清空 resolution

## §6 评估体系(老师题库 300 题,runner 新写)

### 6.1 题面解析
- `evals/run_rag.py` 按 CSV 读取(列:id,bucket,问题,期望章节,标准要点,应拒答);**数据文件一字不改**
- 期望章节语法:顶层 `+`=AND(每组都要命中),组内 `|`=any-of;原子与 `section_path` **子串匹配**;归一 `" / "`→`" > "`、strip
- 指标:**Hit@K**(全部 AND 组在 top-K 各有 ≥1 命中)/ **Recall@K**(覆盖组数/需求组数)/ **MRR**(每 AND 组取其任一 OR 原子最早命中 r → 1/r,组间平均);K∈{3,10}

### 6.2 四策略对比(`run_strategy_eval.py`)
- 臂=dense|bm25|hybrid|hybrid_rerank;题=A/B/C/E 240 题;query 理解共用预处理(§4.1 缓存);评估臂**不套在线余弦阈值**(测排序质量)
- 输出:按桶 × 臂的 Hit/Recall@3、@10、MRR 表 + 总表 → `evals/reports/ch04_strategy_report.md`(重跑覆盖)
- D 桶 60 题(无期望章节):hybrid_rerank 臂 top1 rerank 分数分布分位数表 → `retrieval_low_conf_threshold` 校准(使可答题召回损失最小且 D 桶误自信率最低);校准终值写回 Settings 默认
- 报告如实声明:新语料仅数十块,Top-50≈全库,Recall 偏宽松、MRR 更有分辨力
- 观察项(非硬线,§0-9):hybrid_rerank 应优于 dense;B 桶 bm25 臂 Recall@10 ≥90% = 验收2 数字证据,不达标如实报因

### 6.3 生成段(`run_faith_eval.py`)
- 全链(hybrid_rerank+双闸)跑 240 题 → {答案, citations, 拒答态};D 桶 60 题跑 → 拒答判定
- 忠实度裁判(LLM,`faith_judge_model` 默认=model_name):输入=问题+**喂给模型的 Top-K 证据全集**(含未引用条目)+答案 → `{verdict: faithful|fabricated, reason}`
- fabricated → upsert faith_cases(eval_id 幂等、seen_count、复发退回、citations 快照,§3.2 语义)
- D 桶:正确拒答=pass;未拒答且编造 → 台账 bucket='D_absent'(§0-11);出**拒答正确率**
- 汇总:Faithfulness 率、D 桶拒答正确率、分桶表 → `evals/reports/ch04_faith_report.md`;终跑后实查 `low_confidence_questions` 有 D 桶条目(验收4 证据链)
- 成本闸:`--limit N` / `--bucket X`;全量 ~600 调用仅终跑

### 6.4 Prompt/数据类任务验证(替代 TDD,用户工作要求1)
rewrite/self_check/faith_judge 三 Prompt:标注样例(从老师题库存样例+C/E 桶抽)+ 冒烟跑验证;faith 裁判用已判个案复判抽验。

## §7 前端(Vibe Coding,接口契约如上)

1. **引用角标**:回答文本 `[n]` 渲染可点像素小方块(仅该条回答有 citations 时);点击弹窗:section_path 面包屑+category+原文+上一块/下一块(`GET /api/chunks/{id}`,弹窗按 id 缓存);citations 随 tool_result 帧存该条回答 JS 状态,新对话清空
2. **满意度反馈**:每条 AI 回答左下角 👍/👎;点击点亮所选+显示「已反馈」+**一次性锁定**(再点无效);持久 `localStorage`(`fb:{conversationId}:{序号}`);零后端(§0-12)
3. **台账页 `static/faith.html`**(同款复古风):列表+状态/桶筛选+题号搜索;列 eval_id/bucket/query/seen_count(≥2 标「复发」)/strategy/judge_model/first·last_seen_at/status;行按钮已解决/无需解决/退回未解决(前两者弹框强制 resolution);行展开 citations 全集并标答案实际引用项

## §8 错误处理矩阵

| 故障点 | 策略 | 恢复 |
|---|---|---|
| rewrite LLM 失败 | 原话直检 WARN | 链路不断 |
| BM25 腿异常(Milvus) | 抛错→executor 错误帧→转工单(ch02 语义) | 同 ch03 |
| rerank 超时/报错/无key | 降级 RRF 前10序+闸1跳过,WARN | 闸2 兜底 |
| 闸1 判低置信 | 直接拒答+落池,不生成 | 台账/池复盘 |
| 闸2 自评失败 | WARN **放行**生成 | prompt 层拒答红线兜底 |
| 落池/台账写失败 | WARN 不阻断 | 演示可重跑 |
| Milvus 启动不可达 | WARN 探活(ch03 语义不变) | — |
| faith PATCH 校验失败 | 422 | 台账页提示 |

## §9 测试与回归红线

**单元(TDD)**:head_tail_order、expr 白名单、老师 CSV 解析器(引号逗号/`|`+`+` 混排/` / ` 归一/D 桶空期望边界)、query_understanding 纯部分、reranker 门面(httpx mock)、milvus_store v2(fake client:schema/Function/索引声明形状)、retriever 策略分派+闸1、闸2 编排(fake:false→拒答 token+入池钩子+不进第二轮)、crud 两新表 upsert/复发流转、citations 帧 schema、PATCH 422。
**集成(@integration)**:集合 v2 往返、「MH-LP100」裸 BM25 命中、hybrid_search 形状、SiliconFlow 连通(断网 skip)、入池链路实写、建库全量重灌(6 文件,块数实测锚定)。
**回归重锚定清单**(冲突处理,均老师材料优先,详见附录 C):`test_corpus_ch03.py`→新语料锚定版;`test_retriever_integration.py`(「运费说明」→新语料运费组 top-3);`test_indexer_integration`(19 块→实测值);`evals/rag_retrieval_samples.json`+`run_rag_eval.py` 标 deprecated 不删;mine_qa 集成锚机制不锚条数;README ch04 节新增。
**兼容红线**:ch01/ch02 既有测试全绿;SSE 五帧语义不变(仅 tool_result 增可选 citations);query_faq 工具签名不变;tool_routing eval 重跑。

## §10 配置与依赖

Settings 新增(全带默认,不破坏 ch01-ch03 构造,守卫测试同法):`rerank_api_base="https://api.siliconflow.cn/v1"`、`rerank_api_key=""`、`rerank_model="BAAI/bge-reranker-v2-m3"`、`rerank_timeout_seconds=5.0`、`hybrid_recall_k=50`、`rrf_k=60`、`rerank_top_n=10`、`retrieval_low_conf_threshold=0.3`(§6.2 校准前初值)、`self_check_enabled=True`、`query_rewrite_enabled=True`、`faith_judge_model=""`。`.env.example` 同步;**用户需注册 SiliconFlow key 方可演示/评估**。pyproject 预计零新增(httpx 已在;若核对点③要求别的 SDK 再报)。

## §11 实施时硬性二次核对点(先 Context7 再落码)

1. **pymilvus 3.0.x 全 schema 建集**:`Function`+`add_function`、`analyzer_params` 中文内置分词在 v3.0.0 server 的确切写法(含查询侧是否需 `analyzer_name`)、`AnnSearchRequest(expr=...)`、`client.hybrid_search` 返回 hit 形状(distance=RRF 分数?entity 键名?)、**含 Function 字段的 upsert 数据形状**(text 写、sparse 不写)、index_params 双索引声明
2. **VARCHAR max_length 字节语义**:中文×3B → text 字段上限定值;冒烟验证「MH-LP100」tokenization
3. **SiliconFlow /v1/rerank**:请求/响应字段名(relevance_index/relevance_score?)、top_n/return_documents 语义、错误码
4. **LangChain @tool+RunnableConfig**:当前版本注入不污染 LLM 可见 schema(create_ticket 已验,复核)
5. **FastAPI/pydantic v2**:PATCH body 校验 + 422 形态(仅写新接口时查)
6. **ENUM 中文列 + aiomysql**:两新表(含中文 ENUM 值)读写——沿用 ch02/ch03 已验模式,落码时实测

## §12 ch01-ch03 兼容红线

- 五工具名称/参数名不变;query_faq **返回增列**(hits 增 n/section_path、顶层增 refused/note)——tool_routing eval 与相关测试最小适配,其余断言不动
- SSE token/tool_call/conversation/done/error 五帧逐字符不变;tool_result 仅增可选字段
- chat_service 纯闲聊链路、extract 接口零改动;ch03 双写幂等/check/--skip-existing/--fault-after 语义不变(indexer 仅扩展写入形状)
- `knowledge_chunks`/`qa_extraction_staging` 表与既有 crud 零改列零改函数
- 老师六份语料与 300 题数据文件**只读**:任何实现迁就数据,不反向改数据(唯一例外:若语料需 front-matter 才能分型——实测不需要,chunker 缺省 policy 兼容)

## 附录 A:ch04 建表 DDL(用户提供,逐字落盘 `db/init/05_ch04_schema.sql`)

```sql
-- =============================================================
-- ch04 · RAG 进阶 · 建表 DDL
-- 本章新建:low_confidence_questions(低置信度问题池)、faith_cases(编造个案台账)
-- 生成阶段模型自评知识不够答就拒答,把原话落进这张池子,是 ch09 数据飞轮的入口
-- 本章只靠 useful 自评判入池(useful=false 才入,不单独存该字段);ch09 会给这张表 ALTER 加归并字段
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

CREATE TABLE low_confidence_questions (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '来源会话',
  raw_question    TEXT            NOT NULL                COMMENT '用户原话,带情绪口语',
  source          ENUM('retrieval_low_conf','self_check','user_feedback') NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决',
  reason          TEXT            NULL                    COMMENT '判不能的原因,留作复盘',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '入池时间',
  PRIMARY KEY (id),
  KEY idx_source (source),
  KEY idx_created_at (created_at),
  CONSTRAINT fk_lcq_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='低置信度问题池';

-- ch04 编造个案台账:忠实度裁判判出的每条编造,不只留在这一轮的报告里,进表长期管理。
--
-- 为什么要一张表:报告是产物,重跑一次就被覆盖,上一轮判出的个案连带它的处置状态一起没了。
-- 而这些个案的价值恰恰在跨轮追溯——同一道题反复被判编造,说明库里那一格一直没补对。
--
-- 一题一行(uk_eval_id):同一道题再次被判编造不新增行,只把答案/理由/角标快照更新成最近一次、
-- seen_count 加一。已经标「已解决」的题又被判出来,状态自动退回「未解决」——那是复发,
-- 不是新问题,得让它重新出现在待处理列表里。
--
-- citations 存的是这一轮喂给模型的 Top-K 证据**全集**(答案里的角标 [n] 就是这份列表的序号)。
-- 裁判说「证据里没有」,追溯时必须能当场看到当时喂进去的到底是什么,不能只留一句结论;
-- 而且要看得出「手里有哪几条、实际只引了哪几条」——没被引用的那些同样是判断依据。
CREATE TABLE faith_cases (
  id            BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '主键',
  eval_id       VARCHAR(16)     NOT NULL                COMMENT '评估集题号,如 A43;一题一行',
  bucket        VARCHAR(24)     NOT NULL                COMMENT '题目所属桶:A_policy / B_model / C_colloquial / E_multi',
  query         VARCHAR(512)    NOT NULL                COMMENT '用户问题原文',
  strategy      VARCHAR(24)     NOT NULL DEFAULT 'hybrid_rerank' COMMENT '产出这条答案的检索策略',
  answer        TEXT            NOT NULL                COMMENT '被判编造的那版生成答案原文',
  reason        TEXT            NOT NULL                COMMENT '裁判给的理由:编在哪一句',
  citations     JSON            NULL                    COMMENT '这一轮喂给模型的 Top-K 证据全集快照:[{n,chunk_id,section_path,question,answer}];答案里的角标 [n] 就是这份列表的序号,答案通常只引用其中两三条;老数据没记为 NULL',
  judge_model   VARCHAR(64)     NULL                    COMMENT '判这条的裁判模型',
  status        ENUM('未解决','已解决','无需解决') NOT NULL DEFAULT '未解决' COMMENT '处置状态,人工点按钮改',
  seen_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '被判编造的累计次数(跨轮)',
  first_seen_at DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '第一次被判编造的时间',
  last_seen_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '最近一次被判编造的时间',
  resolution    VARCHAR(300)    NULL                    COMMENT '处置说明:标已解决要写清怎么解决的,标无需解决要写清为什么不用改;退回未解决时清空。空着的处置在台账上等于没有交代',
  resolved_at   DATETIME        NULL                    COMMENT '最近一次被标为已解决/无需解决的时间;复发后仍保留,用来标「复发」',
  PRIMARY KEY (id),
  UNIQUE KEY uk_eval_id (eval_id),
  KEY idx_status (status),
  KEY idx_last_seen_at (last_seen_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ch04 忠实度编造个案台账';
```

注:DDL 头注释提到「useful 自评判入池」——本章实现按用户功能需求 5 与 AskUserQuestion 定稿的前置双闸语义落池(retrieval_low_conf/self_check 两写方),`user_feedback` 为 ch09 预留;表结构本身不因注释措辞改动。

## 附录 B:老师题库格式与已知标注出入清单

**文件**:`evals/run_rag.py`(扩展名 .py,**实为 CSV**,301 行含表头,300 题=5 桶×60)。列:`id,桶(bucket),问题(query),期望章节(expect_section),标准要点(expect_points),应拒答(should_refuse)`。数据文件**只读**。

**期望章节语法实测归纳**(runner 解析器按此实现,单元用例逐条覆盖):
- `+` = AND 组:`E1 "运费怎么算 + 配送范围 + 会员运费权益"`(全部组都要命中)
- `|` = 组内 any-of:`A23 "运费与包邮 | 配送范围"`
- ` / ` = 章节路径分隔符(对应我方 section_path 的 ` > `):`E4 "… + 换货政策 / 换货运费"`
- 原子=章节名子串(如「处理时限」⊂「常见问题处理时限」;「MH-LP100」⊂「智能猫砂盆 Pro(型号 MH-LP100)」)
- D 桶期望列全空、应拒答=是(60 题)

**已知标注与语料出入(数据不改,报告如实,§0-9)**:
1. **D3**「能不能开纸质发票邮寄给我」标应拒答,但 billing-shipping「可开票类型」实有纸质发票条款——裁判按拒答判,若系统给出有据回答将计「应拒未拒」,如实呈现
2. **A22/E9** 期望要点含「银卡 95 折」,语料 member-benefits 实为「银卡 9 折、金卡 95 折」——要点为 `|` 分隔的 any-of 语义,其余要点可命中,不影响检索段
3. **跨文档矛盾(疑似老师埋的 confusable 雷)**:product-faq「运费怎么算」= 未满 99 **收 10 元**,billing-shipping「运费与包邮」= 未满 **收 6 元**;A5/E41 期望按 10 元版——忠实度裁判只裁「答案是否忠于所给证据」,不裁哪个数字对;检索层两条腿都可能召回双块,由重排与首尾排布消化

**规模事实**:6 文档切块后集合仅数十块(实测锚定,预估 40-60),各腿 Top-50 ≈ 全库——Recall 数值偏宽松、MRR 更有分辨力,报告显式声明。

## 附录 C:ch03 资产重锚定清单(冲突处理逐条)

| 资产 | 冲突 | 处置 |
|---|---|---|
| `knowledge/return-policy.md`、`aftersale-manual.md` | 已被删除(老师语料替代) | 不恢复;git 历史留档 |
| `knowledge/product-faq.md` | 被老师版覆盖(旧 **Q/A 体→H2 问句体) | chunker 兼容实测(缺 front-matter→policy 默认,questions=H2 标题,无需改切分规则) |
| `tests/test_corpus_ch03.py` | 断言旧文件/违禁词闸全炸 | 重写为 `test_corpus_ch04.py`:锚评估依赖的最小语料事实(12 型号节存在、运费两节数字矛盾存在、处理时限表、section_path 形态) |
| `tests/test_retriever_integration.py` | 断言「邮费是多少 top-1=运费说明」 | 重锚:运费族章节(「运费怎么算」「运费与包邮」任一)进 top-3 |
| `tests/test_indexer_integration*` | 19 块断言 | 更新为 6 文件实测块数 |
| `evals/rag_retrieval_samples.json` + `run_rag_eval.py` | 期望路径在新语料不存在 | 文件头标 DEPRECATED(保留 ch03 历史,不删不跑) |
| `db/init/04_ch03_seed.sql` + mine_qa | 对话种子话题(水垢/混喂/积分)与新语料部分重叠,闸3 靶语义漂移 | 种子与 job 保留;ch04 演示不跑 mine_qa(§0-5);mine 集成测试改锚机制断言(幂等/找回),不锚条数 |
| 库内旧数据(19 doc chunk + mined) | schema 过期 | 全量重建一次完成迁移(drop 集合+清表重灌) |
| README | — | 新增 ch04 节(演示/评估/验收命令,全部真跑后落笔) |
