# ch09 · 可观测性与数据飞轮 · 设计定稿

> 2026-10-05 经用户批准。拍板方式:编号清单 8 项一次问完,用户复「全默认」,全部按推荐项执行并在此逐项记死。
> 上游实据:三路摸底(图管线/数据层/前端 API 面)+ Context7 Langfuse 核对一次(结论只见「观测链」一节,不再重翻)。

## 决策与拍板记录

| # | 开放点 | 拍板 |
|---|---|---|
| 1 | 👎 回捞的当轮召回快照落点 | A:`messages` 表 ALTER 加 `retrieval_snapshot JSON` 列,进 agent 的轮随 assistant 行落库 |
| 2 | 飞轮流水线触发时机 | A:落池即 fire-and-forget 异步任务 + `app/jobs/flywheel.py` CLI 补扫 `matched_review_id IS NULL` |
| 3 | 升级闸作用面 | A:同一 `evidence_confidence` 函数生效于 gate 与 refund_gate 两实例;阈值 300 题 CSV 校准,禁拍脑袋 |
| 4 | 核准答案回写形态 | A:人工核准即终审,绕开 mine_qa 三道查重闸,复用 `indexer` 层直写 knowledge_chunks + 向量化,category 沿用「客服对话问答」 |
| 5 | 评估轮次形态 | A:`app/jobs/eval_cycle.py` CLI;「定期」靠 Windows 任务计划器/手动,不引入进程内调度依赖 |
| 6 | 👎 协议与 👍 去向 | A:`POST /api/feedback`;👎 落池进流水线,👍 维持纯前端不落后端(YAGNI) |
| 7 | Langfuse 部署栈 | A:官方 v3 全套进 docker-compose(web/worker/clickhouse/postgres/redis,S3 复用既有 minio 新开桶) |
| 8 | 意图成本呈现面 | A:trace 打 `intent:<八类>` tag + CLI `--report` 从 Langfuse API 聚合,双保险保验收 5 |

**本章不做**(用户 prompt 钉死):低置信问题按主题归类的微调分类器。另按范围裁剪:👍 后端化、进程内调度器、Langfuse prompt 管理/人工标注面、审计表保留策略均不入本章。

## 范围与非目标

两条新链路装进现有客服系统:

1. **观测链**:Langfuse 自部署,LangGraph 编译处挂一次回调;每请求一条完整 trace 树(节点 prompt/工具调用/检索结果/token/耗时自动入树);意图打进 trace 标签,支持按意图看 token 花销。
2. **飞轮链**:ch05 简版置信度闸升级为正式 `evidence_confidence` 闸;三个入口(置信闸/生成自评/用户👎)统一落 `low_confidence_questions` 并带当轮召回片段快照;落池触发「标准化→查重→入队」流水线;`review_queue` 后台审核页人工通过/驳回;通过者走 ch03 落库管线回写知识库;评估流水线每轮分数落 `eval_runs` 连趋势。

**非目标**:不改路由/工具面/上下文分层(ch06/ch07/ch08 行为);不动 ch04 评估集原文;不做多用户鉴权(单用户 demo 形制,IDOR 同律)。

## 架构总览

新增模块与其唯一职责(依赖方向只进不出):

| 单元 | 文件 | 职责 | 依赖 |
|---|---|---|---|
| 观测挂接 | `app/workflows/graph.py`(build_graph cfg 处)+ `app/services/observability.py` | 构造 CallbackHandler(env 缺键→短路返回 None)、意图→trace 归因、trace_id 兜底通道 | langfuse SDK |
| 置信度闸升级 | `app/workflows/nodes.py`(gate 函数替换)+ `app/rag/confidence.py` | 纯函数 `evidence_confidence(signals)` 与判定;校准定值从 config 读 | 无 |
| 召回快照随行 | `app/services/persistence.py` / logging 落库处 | 有 evidence 且过闸进 agent 的轮,assistant 行写 `retrieval_snapshot` | db/crud |
| 飞轮流水线 | `app/services/flywheel.py` + `app/prompts/flywheel.py` + `app/schemas/flywheel.py` | 标准化/示例答案/查重判定的 LLM 调用与落队/归并写库;异常全吞只 WARN | db/crud、model |
| 落池统一漏斗 | `app/services/refusals.py`(扩展) | 三入口共用:写 LCQ 行(含快照/新列)→ `asyncio.create_task` 触发流水线 | flywheel |
| 反馈端点 | `app/api/routes.py` + `app/schemas/chat.py` | `POST /api/feedback`:定位当轮、回捞快照、落池(source=user_feedback) | refusals |
| 审核 API | `app/api/routes.py`(review-queue 三条) | 列表/详情(归并原话+快照)/核准(触发回写知识库)/驳回 | crud、indexer、flywheel |
| 知识库回写 | `app/services/flywheel.py` 内 `publish_approved()` | 复用 `app/rag/indexer` 写 chunk+向量;失败则状态不置「通过」 | indexer |
| 飞轮批扫 CLI | `app/jobs/flywheel.py` | 扫 `matched_review_id IS NULL` 补跑流水线(崩溃/漏网兜底) | flywheel |
| 评估周期 CLI | `app/jobs/eval_cycle.py` | 跑检索段+生成段指标,一行写 `eval_runs`;`--trend` 打趋势;`--report` 意图成本聚合(走 Langfuse API) | evals 既有核、crud、langfuse SDK |
| 审核页 | `static/review.html`(+ index.html 顶栏入口、👎 发请求改造) | Vibe Coding 豁免面:列表+详情弹层+通过/驳回;faith.html 形制 | 审核 API |

数据流(一 turn 内):`retrieve → gate(evidence_confidence) → [fail: 兜底+落池(带快照)+异步流水线] / [pass: agent → logging 落 assistant 行(带 retrieval_snapshot) → trace 意图归因]`;离线:`low_confidence_questions → 流水线 → review_queue → 审核页 → indexer → Milvus/knowledge_chunks → 再问即答对`;评估:`eval_cycle → eval_runs 一行 → --trend 趋势`。

## 观测链:Langfuse 挂接与意图成本

**API 核对结论**(Context7,langfuse-docs,一次成文,评审不重翻):

- SDK v3:`from langfuse.langchain import CallbackHandler`;挂在 config `callbacks` 即自动铺 LangGraph 全树(节点 span、LLM generation 含 prompt/completion、token 与成本自动从 `usage_metadata` 采——本项目 `ChatOpenAI` 走 DashScope compatible-mode,usage 随 AIMessage 回,无需手采)。
- trace 级属性可在 invoke config 的 metadata 里用保留键 `langfuse_session_id` / `langfuse_user_id` / `langfuse_tags` 传入。
- 运行中改 trace 用 `langfuse.update_current_trace(...)`(SDK v3);跨请求边界处用底层 `Langfuse().api.trace.update(...)`。

**挂接口径(「编译时挂一次」的字面兑现)**:`build_graph()` 末尾 compile 后 `with_config({"callbacks":[handler]})`,handler 由 `observability.py` 工厂出:`LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY` 任缺其一 → 工厂返回 None → 不挂,**图行为与 ch08 终态逐字节等价**(现 448 单元基线必须原绿)。cfg 其余键(`configurable` 四件套)不动;`stream_graph_turn` 传 config 时补 `langfuse_session_id = f"conv-{cid}"`(anon 轮用 `anon-{uuid}` 值)、`run_name="chat_turn"`;confirm 续播 `Command(resume)` 同 cfg 面(同一 session 同树语义,续播轮自身是一棵新 trace——挂账可接受)。

**意图归因(验收 5 的锚)**:`intent` 节点出结果后打 tag `intent:<八类中文名>` 与 metadata `{intent, intent_confidence}`。主道=节点体内 `update_current_trace`;若实现验证 contextvar 在 LangGraph 节点内不透传(核对文档见官方 LangGraph cookbook 同形用法,预期可用),**备道=routes 层流末拿 handler 暴露的 trace_id 走底层 `api.trace.update`**——两道的验收面一致:UI trace 列表可按 intent tag 过滤。实测定一道、删另一道,不留双路。

**成本面(拍板 8A)**:①UI:`Traces` 按 tag 过滤 + `Cost/Daily` 视图人工看;②CLI:`eval_cycle.py --report` 调 Langfuse trace API 拉窗口内 traces,按 `intent:*` tag 聚合 totalTokens/totalCost 打表(UTF-8 文件,GBK 红线),哪类意图最烧钱一眼可看。Langfuse 不可达时 `--report` 明确报错退避,不影响评估主行落库。

**部署(拍板 7A)**:`docker-compose.yml` 增 `langfuse-web`(端口 3001,宿主 3000 已被 minio console 占)、`langfuse-worker`、`clickhouse`、`langfuse-postgres`、`langfuse-redis`;S3 blob 复用既有 `minio` 新开桶 `langfuse`(建桶命令进 README 部署步骤)。clickhouse 所需的少量配置挂载若遇 Windows bind-mount 权限坑(06 文件 cnf 前科),改打薄镜像内置配置——记为部署风险,不阻塞设计。**flush 语义**:CLI 任务(eval_cycle/flywheel 批扫)退出前显式 flush;uvicorn 常驻由 SDK 后台批处理自管。

## 置信度闸升级:evidence_confidence

**位置与行为不变**(用户钉死):仍卡 retrieve 之后、进 agent 之前;拦下 → `REFUSAL_ANSWER` 兜底 + 转人工建议 + 落池 + fail→logging。`make_confidence_gate_node` 换判定核,gate 与 refund_gate 两实例同函数生效(拍板 3A)。

**信号与判定**:从 `state["evidence"]`(已含 rerank 相关性分)算三信号——`top1`、`n_eff`(score ≥ floor_eff 的证据条数)、`gap`(top1−top2,单证据时 gap=top1)。`evidence_confidence = f(top1, n_eff, gap)` 为纯函数,规则式或加式和由校准脚本二选一实测定,**参数(floor_eff、θ、组合权重/内阈)一律出自 300 题老师 CSV 扫参**:D_absent 桶 60 题=应拦面,其余 240 题=应放面;定线=D 桶漏放 ≤5%、非 D 误拦 ≤ ch05 基线 4.2% + 2pp 容差;两目标冲突时优先误拦不恶化、漏放收紧到可达最优点,校准报告落 `evals/reports/`,定值写死 config(`evidence_conf_*` 键组),spec→plan 期间不预设数字。

**降级带保留**:rerank 降级路径(RRF 分)不可能进新公式定标——保留现 `RRF_DEGRADED_MAX` 旁路语义(过闸不落池,记 `gate_scale="rrf_degraded"`),该旁路只在降级分域内生效的接缝测必须原样钉住。

**空证据与闸1交互不变**:retriever 内部闸1 清证据 → 新闸空证据必拦,照旧落池;reason 串记 `conf=…<θ(top1=…,n=…,gap=…)` 供复盘。

## 数据层:新表、新列、接缝测试

**新表两张**:`review_queue`、`eval_runs`——DDL 见附录一(用户原文逐字,零改动),进新文件 `db/init/09_ch09_flywheel.sql`(文件头加 `SET NAMES utf8mb4;` 与 ch09 说明注释,DDL 主体不动;09 同时含对 `low_confidence_questions` 的两列 ALTER,即附录一末段)。

**messages 加列(拍板 1A)**:`db/init/10_ch09_messages_snapshot.sql`——附录二原文。写入方=persistence 落 assistant 消息处,仅当该轮 state 有 evidence 且走 agent 生成;值为 `[{"chunk_id","score","text"}…截断]`,text 截 600 字符(常量 config `retrieval_snapshot_text_max`),无检索的轮 NULL。闲聊/纯数据/工单轮的 assistant 行恒 NULL——👎 回捞据此实现「真没走检索才空着」。

**ORM 同步**:`app/db/models.py` 加 `ReviewQueue`/`EvalRuns` 模型 + `LowConfidenceQuestion.retrieved_chunks/matched_review_id` + `Message.retrieval_snapshot`;红线照旧(模型仅供映射,建表以 db/init 为准)。中文 ENUM(`review_status`)与外键(`fk_lcq_review`)与 ch08「审计不挂外键」不冲突——归并列语义要求约束,照用户 DDL。

**接缝测试(三章先例并套)**:ch06 式 ENUM 值并集断言(中文两 ENUM)+ ch07 式列名并集断言(新表列集 ⊇ ORM、LCQ/messages 新列在位)+ ch08 式活库往返(integration 标记,逐列中文不乱码)。文件 `tests/test_db_init_seam_ch09.py`。摘文件即红演示照例做双证并记 dev-notes。

## 落池三入口与回捞

统一漏斗仍 `refusals.pool_low_confidence`,签名扩两参(`retrieved_chunks: 快照|None`、触发流水线开关),所有写方单点进出:

1. **ch05_gate / ch06_refund_gate(升级后)**:拦下即落池,快照直接取 `state["evidence"]`。池写失败照旧吞 WARN 不挡兜底话术。
2. **self_check(拍板「确认接进飞轮即可」)**:该路 `pool_low_confidence` 已被漏斗自动覆盖——触发流水线+快照(调用点手里有证据)即接通,不改代码路径;其线上不可达现状(tool_chat_service 未接生产路由)如实记 README,不假装在跑。
3. **user_feedback(新,验收 4)**:`POST /api/feedback {conversation_id, seq, vote}`(schema 对齐 `ChatRequest` 校验风格)。`vote="down"` 才处理:按 conversation 消息序反查第 seq 个 assistant 行 → 其前紧邻 user 行为 `raw_question` → 该行 `retrieval_snapshot` 回捞为快照(行在但列 NULL=真没检索,空着)→ 落池 `source="user_feedback"`(枚举空位启用,零 DDL),reason 记 seq+vote → 触发流水线。resp 200 `{pooled:true}`;反查不到(越界 seq/会话不存在)→ 404。**幂等**:同轮重复👎允许重复落池(查重归并兜住,不建轮次唯一键);seq 语义=与前端 localStorage 键同律「history 中 assistant 序号」,前端把现在只进 localStorage 的点击改为同时发请求(锁钮逻辑不变)。

**matched_review_id 生命周期**:落池即 NULL(未处理/未归并);流水线跑完后=归并目标行 id。前端审核页详情=「该缺口所有 matched 行的原话+快照」列表。

## 飞轮流水线:标准化 → 查重 → 入队

`app/services/flywheel.py::process_lcq_row(row_id)` 三段,LLM 调用均 `get_model` 同源配置、temperature=0、结构化输出 schema 进 `app/schemas/flywheel.py`:

1. **标准化**:口语原话 → `NormalizedQA{normalized_question(FAQ 式), suggested_answer(示例答案,备查)}`(答案供审核参考,不是终稿)。
2. **查重**:候选集=当前 `review_status='待审'` 全部行(≤50,超出按 updated_at 最新截断,截断记 WARN);一次 LLM 判定输出 `matched_id|None`。命中 → `occurrence_count += 1`、不新建行;未命中 → 插新行(待审, count=1, ai_suggested_answer=示例答案)。
3. **归并落点**:LCQ 行 `matched_review_id` 写回目标 id(命中=既有行,新建=新行 id)。

**触发(拍板 2A)**:落池成功后 `asyncio.create_task` 异步跑(不拖兜底话术与👎响应);任务内异常全吞→WARN,行留在未处理态。`uv run python -m app.jobs.flywheel [--limit N|--dry-run]` 补扫 `matched_review_id IS NULL` 重跑(崩溃、fire-and-forget 丢失、批量回填三态通用;幂等由「单行处理完成后才写 matched」保证,中途崩=重跑不双并——若查重命中累加先于崩可致 count 多计,演示形制接受,记已知边界)。

**Prompt 质量面(纯 Prompt 任务,不套 TDD)**:标注样例集 `evals/flywheel_samples.jsonl`(标准化≥5 条含口语/错字/指代、查重≥5 条含同义/近义陷阱)+ `evals/run_ch09_flywheel_eval.py`(run_ch08_eval 形制,判定线 ≥8/10,人工逐条核对留痕);调优只动 few-shot 不动代码。

## 审核 API 与后台页

路由三条(形状沿用 faith_cases 的 GET+PATCH 先例,`Depends(dep_db_session)`):

- `GET /api/review_queue?status=待审` → 列表(标准化问题/出现次数/示例答案/时间),按 occurrence_count desc。
- `GET /api/review_queue/{id}/detail` → 行 + 归并进来的 LCQ 原话列表(raw_question/reason/created_at)与各自 retrieved_chunks。
- `PATCH /api/review_queue/{id}` → `{status:'通过', approved_answer}` 或 `{status:'驳回'}`;仅允许 待审→通过/驳回 单向流转,其余 422;approved_answer 必填才可通过(校验同 faith resolution 风格)。

**通过=飞轮回写动作(拍板 4A)**:同请求内 `publish_approved()`——构造 QA 条目走 `indexer` 层(questions=normalized_question, answer=approved_answer, category=「客服对话问答」, 指纹幂等:同问再批不双写);**chunk+向量双落成功才置「通过」**,任一步失败 → 事务回滚状态保持待审 + 502,错误 detail 透出(审核者重试)。驳回不触发写。

**后台页(Vibe 豁免面,不套流程)**:`static/review.html` 整体照 faith.html 形制(原生 JS/CSS 内联):列表面包+状态筛选+行内「通过/驳回」;详情弹层显示归并原话与其召回片段(渲染得分与截断原文,供判断「真缺 vs 没检到」);通过弹核准答案输入框(预填 ai_suggested_answer)。顶栏入口进 `index.html`,回程链接互指。效果由用户描述驱动迭代。

## 评估流水线与 eval_runs

`uv run python -m app.jobs.eval_cycle [--trigger 手动|定时] [--limit N] [--skip-faith] [--trend] [--report]`(拍板 5A;`--trigger` 默认「手动」,「定时」留给 Windows 任务计划器调用时显式传):

- **检索段**:hybrid_rerank 臂 × 老师 CSV(默认全 300,`--limit` 等比截断),复用 `teacher_csv.eval_question` 指标函数与 strategy_eval 的 embed/rewrite 缓存;出 `recall_at_3/recall_at_10/mrr_at_10`。
- **生成段**:ch04 faith 链裁判复用,固定 seed+固定样本量(默认 30 非 D 抽样),出 `faithfulness`。
- 一行 `eval_runs`:`triggered_by/dataset_size(本轮实跑条数)/metrics JSON/created_at`。
- `--trend`:eval_runs 按 created_at 排序打 ASCII 对齐对比表(UTF-8 文件),各指标环比差值,「哪个指标下滑」一眼可看(验收 6 的拿得出的东西)。
- **可比性红线**:趋势跨轮比较须同口径;`--limit` 轮的 dataset_size 如实记录,README 写明「非满额轮仅作应急参考」。

## 配置项汇总(settings / env)

`langfuse_host/langfuse_public_key/langfuse_secret_key`(env 直读,缺任一=观测整体短路,含测试环境)、`evidence_conf_*` 校准定值组+`evidence_conf_threshold`(校准前以现 0.161 占位跑通管线,定值回填在流水线任务内完成,不起订正回合)、`retrieval_snapshot_text_max=600`、`flywheel_dedup_candidates=50`。新增 compose 服务的环境变量由各容器自持,app 面只有上述三个 langfuse 键。

## 测试策略

- **单元(离线,mock LLM/DB 或既有夹具)**:evidence_confidence 纯函数钉死+降级带旁路接缝;pool 漏斗新签名三入口形态;`/api/feedback` 端点全分支(404/幂等/回捞空);flywheel 流水线命中/未命中/LLM 崩三态;publish_approved 成功与失败回滚(状态不置通过);eval_cycle 行落库(metrics JSON 形状);观测工厂 env 缺键返回 None(cfg 零变化钉)。
- **回归红线**:无 langfuse env 时现 448 单元基线原绿(观测挂接不引入行为漂移);gate 替换后 ch05/06 既有闸测同步改钉新语义(改断言进同一 commit,不发订正回合)。
- **接缝**:test_db_init_seam_ch09 三件套(上文「数据层」节)。
- **Prompt 评估**:flywheel 样例集 ≥8/10 + 人工核对留痕。
- **集成(integration 标记,活库 3307)**:新表逐列往返+中文 ENUM 不乱码;LCQ ALTER 后旧写方仍可写。
- **e2e 验收映射**:六条各一 scenario 进 `tests/`(1 观测=人工 UI 步+trace tag 单测兜;2 问知识库没有的问题→兜底话术+队列出现+详情见原话与快照;3 审核通过→同问再答对(真模型,评估形制);4 👎→落池→队列;5 意图成本 --report 有表;6 eval_cycle 两轮→--trend 出对比)。

## 边界与已知挂账

- fire-and-forget 任务随进程死=行滞留,CLI 补扫兜底(演示当场重跑即可);查重累加先于崩溃可致 count 多计,单用户 demo 接受。
- 候选集截断 50 之外的老待审行永不参与查重(理论双行同义),队列变长后的候选策略留后续章。
- Langfuse 栈 RAM 占用高(ClickHouse);Windows 端口占用/配置挂载坑见「观测链」部署段;不装则观测面短路、飞轮面不受影响——两链解耦是本设计的有意结构。
- 意图 tag 打在 trace 级依赖 SDK contextvar 透传实测,备道已钉(观测链节),验收面不变。
- 审核通过写知识库后,Milvus 即时可见性=在线检索下次查询生效(indexer 现语义);「再问即对」演示不需重建集合。
- self_check 路在 eval 之外的线上不可达是既有现状(README 如实标注),本章不接路由。
- 审计/池/队列三表只增不清(与 ch08 挂账同律,保留策略留后续)。

## 附录一 · 用户 DDL 原文(ch09,逐字,`db/init/09_ch09_flywheel.sql` 主体)

```sql
-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 待审队列:一行 = 一个去重后的知识缺口;查重命中就累加 occurrence_count,不新建行
CREATE TABLE review_queue (
  id                  BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '缺口主键,也是查重命中要返回的 matched_question_id',
  normalized_question VARCHAR(512)    NOT NULL                COMMENT '标准化后的 FAQ 式问题',
  ai_suggested_answer TEXT            NULL                    COMMENT '模型生成的示例答案,备查',
  occurrence_count    INT UNSIGNED    NOT NULL DEFAULT 1      COMMENT '出现次数,查重命中累加,越高越该优先补',
  review_status       ENUM('待审','通过','驳回') NOT NULL DEFAULT '待审' COMMENT '人工审核状态',
  approved_answer     TEXT            NULL                    COMMENT '审核通过时补的核准答案,走 ch03 落库流程写回知识库',
  created_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次入队时间',
  updated_at          DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_review_status (review_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='飞轮待审队列';

-- 评估轮次:一行 = 评估流水线跑完的一轮,各指标分数收进 metrics JSON,按时间连起来就是趋势线
CREATE TABLE eval_runs (
  id           BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '评估轮次主键',
  triggered_by ENUM('定时','手动') NOT NULL DEFAULT '定时' COMMENT '这轮怎么起的:定时任务,或某次改动后手动跑',
  dataset_size INT UNSIGNED    NOT NULL                COMMENT '这轮跑的评估集条数',
  metrics      JSON            NOT NULL                COMMENT '各指标分数,如 {"recall_at_k":0.82,"mrr":0.71,"faithfulness":0.90}',
  created_at   DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '跑完落表时间',
  PRIMARY KEY (id),
  KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='自动化评估流水线轮次结果';

-- 原话流水记归并落点:标准化查重后指向 review_queue 里的缺口行,NULL 表示尚未处理
-- 再存一份落池当时的召回片段快照(Top 几条的原文和得分),没走检索的入口为 NULL
ALTER TABLE low_confidence_questions
  ADD COLUMN retrieved_chunks  JSON            NULL COMMENT '落池时的召回片段快照:Top 几条的原文与得分,审核页展示用;没走检索为 NULL' AFTER reason,
  ADD COLUMN matched_review_id BIGINT UNSIGNED NULL COMMENT '查重后归并到的缺口,指向 review_queue.id' AFTER retrieved_chunks,
  ADD KEY idx_matched_review_id (matched_review_id),
  ADD CONSTRAINT fk_lcq_review FOREIGN KEY (matched_review_id) REFERENCES review_queue (id) ON DELETE SET NULL;
```

## 附录二 · messages 快照列 DDL(拍板 1A,`db/init/10_ch09_messages_snapshot.sql`)

```sql
SET NAMES utf8mb4;

-- ch09 飞轮回捞数据源:进 agent 的轮把当轮证据快照随 assistant 消息行落库
-- 👎 落池时按 (conversation_id, assistant seq) 反查此行回捞;没走检索的轮恒 NULL
ALTER TABLE messages
  ADD COLUMN retrieval_snapshot JSON NULL COMMENT '当轮召回片段快照(Top 原文+得分),供事后 👎 回捞;无检索轮为 NULL' AFTER content;
```

## 附录三 · 验收对照(用户六条 → 本章钉点)

| 验收 | 钉在哪 | 验证形态 |
|---|---|---|
| 1 完整 trace 树 | 观测链节 | UI 人工 + trace tag/挂接单测兜底 |
| 2 无库问题→队列+原话+快照 | 闸升级节+流水线节+审核 API | e2e(真模型)+单测链 |
| 3 通过→再问即对(飞轮一圈) | 审核 API publish + e2e | e2e 真模型,标注「不可当回归硬基线」(dashscope temp=0 非全确定) |
| 4 👎→落池→队列 | /api/feedback + 漏斗 | 端点单测 + e2e |
| 5 意图 token 统计 | intent tag + --report | --report 输出表(UTF-8) |
| 6 两轮评估趋势 | eval_cycle + --trend | 连跑两轮出对比表,eval_runs 行数钉 |
