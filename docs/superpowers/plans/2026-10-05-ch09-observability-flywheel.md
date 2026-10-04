# ch09 可观测性与数据飞轮 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans(native,本会话逐任务实施)。Steps 用 checkbox 勾选。
> 用户已授权:M1→M4 连跑,普通实现问题自决不待批,里程碑完成即测即提交即进下一里程碑。

**Goal:** 给现有 LangGraph 客服链挂 Langfuse 自部署观测(trace 树+意图成本),并把「答不上」变成飞轮:三入口落池(带召回快照)→标准化查重→待审队列→人工核准回写知识库→再问即对;评估轮次落 `eval_runs` 成趋势。

**Architecture:** 两条新链解耦——观测链只在 `build_graph` compile 处挂一次回调(env 缺键整体短路,行为=ch08 终态);飞轮链以 `refusals.pool_low_confidence` 为唯一漏斗(写 LCQ 含快照→异步触发 `flywheel.process_lcq_row`),审核 API 的通过动作复用 `app/rag/indexer` 直写知识库。详见 spec。

**Tech Stack:** langfuse SDK v3(langchain CallbackHandler)、docker-compose 自部署(langfuse v3 栈+clickhouse/postgres/redis,minio 复用)、SQLAlchemy async/aiomysql、FastAPI、原生 JS 页。

**Spec:** `docs/superpowers/specs/2026-10-05-ch09-observability-flywheel-design.md`(附录一为用户 DDL 原文)

## Global Constraints

- 用户 DDL 原文逐字不改,进 `db/init/09_ch09_flywheel.sql`(附录一主体,文件头可加注释与 `SET NAMES utf8mb4;`);messages 快照列进 `db/init/10_ch09_messages_snapshot.sql`(附录二)。活库手工 `docker compose exec -i mysql mysql -uroot -pmewhelp_dev mewhelp < db/init/09...sql` 应用。
- 一任务一 commit:代码+测试+本计划勾选+dev-notes/ch09.md 阶段追加进同一笔;禁单独「计划订正」commit。
- GBK 红线:控制台只出 ASCII 摘要,中文明细一律 UTF-8 落文件。
- 测试经 `uv run pytest`;打活库/真模型用例标 `@pytest.mark.integration`(默认 deselected)。
- 新依赖仅 `langfuse`(python SDK);不引入进程内调度库;前端零框架。
- trace tag 口径:`intent:<八类中文名>`(物流/订单/商品咨询/退款退货/售后/投诉/闲聊/其他)。
- 纯 Prompt 任务(标准化/查重)验收=标注样例集 eval 跑批,调优只动 few-shot。
- 前端(review.html、👎 接线、index 顶栏)= Vibe Coding 豁免面:不套 TDD/code review,效果用户后验。
- 评审按里程碑批量跑(fresh reviewer),只对行为/正确性/安全立 finding。

## Review Focus(跨任务盲区,各钉入归属任务)

1. langfuse 未配置/服务不可达 → 图行为与 ch08 终态零漂移,cfg 不含 callbacks。(T1)
2. `down -v && up` 重建后 09/10 缺位 → 池写/快照写静默丢行(1265/截断被 WARN 吞)。ENUM 并集+列并集 seam 测兜。(T3)
3. rerank 降级路 RRF 分不在新公式标定域 → 降级带旁路语义原样保持(过闸不落池,`gate_scale="rrf_degraded"`)。(T4)
4. 👎 的 seq 与 DB 消息序错位:`role='tool'` 行不得计入 assistant 序号。(T6)
5. 审核「通过」的半成功:KB 写失败不得置「通过」(502+状态留待审);重复通过/非法流转 422。(T9)
6. fire-and-forget 随进程死/中途崩 → CLI 补扫以 `matched_review_id IS NULL` 为界幂等,不双并新行(命中累加可重,新建行以「同事务写回 matched」封口)。(T7)

---

## M1 观测链

### Task 1:Langfuse 挂接与观测基座
**Files:** Modify `pyproject.toml`(加 langfuse)、`app/core/config.py`(langfuse 三键读 env 判定)、Create `app/services/observability.py`、Modify `app/workflows/graph.py`(build_graph compile 处 + stream_graph_turn cfg)、`docker-compose.yml`(langfuse-web 3001/worker/clickhouse/langfuse-postgres/langfuse-redis,S3 指既有 minio 桶 `langfuse`)、Create `tests/test_observability_ch09.py`;README 部署段(建桶+env 样例)留 T12 统一写,本任务先 `dev-notes` 记命令。
**Interfaces:** Produces `observability.build_handler() -> CallbackHandler|None`(env 任一键缺→None);`observability.enabled() -> bool`。
- [x] Step1 RED:`test_factory_returns_none_without_env`(monkeypatch 三键缺→None;有键→handler)、`test_build_graph_mounts_handler_once`(env 有键时 compiled graph 的 config 含该 handler;缺键时 cfg 与基线逐项相等——钉 ch08 终态形状)。
- [x] Step2 实现:工厂 + `build_graph` 末尾 `with_config` 单点挂;`stream_graph_turn`/confirm 预检 cfg 补 `metadata.langfuse_session_id=conv-{id}|anon-{uuid}`、`run_name="chat_turn"`(仅 enabled 时)。
- [x] Step3 compose v3 栈落盘,`docker compose up -d` 实起(langfuse-web 宿主端口避开 3000/3307/9000/19530/8101/8102,取 3001);.env 样例三键进 README 素材。
- [x] Step4 GREEN + 全量单元回归(基线 448 原绿) + commit(并入 `feat(ch09-m1)`,理由见 T2 Step3)。

### Task 2:意图归因进 trace
**Files:** Modify `app/workflows/nodes.py`(intent 节点尾)、`app/services/observability.py`(归因函数)、Create `tests/test_trace_intent.py`。
**Interfaces:** Produces `observability.record_intent_to_trace(intent: str, confidence: float)` → tag `intent:<X>` + metadata `{intent,intent_confidence}`;主道 `update_current_trace`,实测 contextvar 在 LangGraph 节点内不透传则切实spec 备道(底层 `api.trace.update`+routes 末拿 trace_id),定一道删另一道,不留双路。
- [x] Step1 实现+真跑一轮:活 langfuse 零真模型(闲聊快路)整轮 → Langfuse API 轮询断言 trace 带 `intent:*` tag(真模型面并入 T12 e2e)。
- [x] Step2 GREEN + 单元面:enabled=False/无 tid/无 intent 三前提均静默 no-op 不抛。
- [x] Step3 commit(与 T1 合并为一笔 `feat(ch09-m1)`:T1 提交凭据重构后未落,T2 代码已长在同文件,拆 hunks 风险大于收益,ledger 记裁决);M1 批评审(fresh reviewer 审 t1–t2 diff)→ 修复进同里程碑。

## M2 数据底座 + 闸升级 + 落池

### Task 3:DDL 09/10 + ORM + 接缝测试
**Files:** Create `db/init/09_ch09_flywheel.sql`(附录一逐字+头注)、`db/init/10_ch09_messages_snapshot.sql`(附录二)、`tests/test_db_init_seam_ch09.py`;Modify `app/db/models.py`(ReviewQueue/EvalRuns + LCQ 两列 + Message.retrieval_snapshot)。
**Interfaces:** Produces ORM:`ReviewQueue(id,normalized_question,ai_suggested_answer,occurrence_count,review_status,approved_answer,created_at,updated_at)`、`EvalRuns(id,triggered_by,dataset_size,metrics,created_at)`、`LowConfidenceQuestion.retrieved_chunks(JSON)/matched_review_id(FK SET NULL)`、`Message.retrieval_snapshot(JSON)`。
- [x] Step1 RED seam:ch06 式中文 ENUM 并集(review_status 三值/triggered_by 二值)+ ch07 式列名并集(新表列 ⊇ ORM;LCQ/messages 新列经 ALTER 在位)+ 摘 09 文件即红双证记 dev-notes。
- [x] Step2 活库应用两文件(integration:逐列往返+中文不乱码;LCQ ALTER 后旧写方仍可写;活库对账文件 `tests/test_db_ch09_roundtrip.py`,ch04 列序断言随本 commit 改钉)。
- [x] Step3 GREEN + commit `feat(ch09-t3): 飞轮数据底座——09/10 DDL+ORM+三件套接缝`。

### Task 4:evidence_confidence 闸升级(含校准)
**Files:** Create `app/rag/confidence.py`、`evals/calibrate_confidence.py`;Modify `app/workflows/nodes.py`(gate 判定核替换,两实例共用)、`app/core/config.py`(定值键组)。
**Interfaces:** Produces `confidence.signals_from_evidence(evidence) -> (top1,n_eff,gap)`、`confidence.evaluate(evidence, cfg) -> EvidenceVerdict(ok, conf, detail:str)`;config `evidence_conf_*` 组(floor_eff/θ/组合参,校准回填前以现 0.161 等效占位)。判定式形态(规则式 vs 加式和)由校准脚本扫参择优,报告 UTF-8 落 `evals/reports/`。
- [x] Step1 RED:`test_confidence_signals`(空证据/单证据 gap=top1/n_eff 计数)、`test_gate_uses_confidence`(gate 节点 mock:拦下→REFUSAL+落池带快照参+fail;RRF≤0.04 旁路过闸不落池)、ch05/06 既有闸测同步改钉新语义(断言随本 commit 改,不另起订正回合)。
- [x] Step2 校准:脚本跑 300 题 CSV(D 桶 60=应拦面,余=应放面;embed/rewrite 缓存复用),定线=D 漏放≤5% 且非 D 误拦≤6.2%,冲突优先误拦不恶化;定值回填 config+记报告路径。实跑结果:sum 形 w=(0.9,0.05,0.05) θ=0.168 floor=0.1,漏放 5.000%/误拦 4.167%,报告 `evals/reports/ch09_confidence_calibration.md`。
- [x] Step3 GREEN + integration 抽验闸活链 + commit `feat(ch09-t4): evidence_confidence 正式闸——三信号+300题校准定值,gate/refund_gate 双实例`。(活链测选题取侧车两端裕量 A36/D24,单 loop 跑通)

### Task 5:当轮召回快照随 assistant 行落库
**Files:** Modify assistant 消息落库处(`app/services/persistence.py` / logging 节点,实施时定位唯一写点)、`app/db/crud.py`;Create `tests/test_snapshot_ch09.py`。
**Interfaces:** Produces 落库语义:进 agent 且有 evidence 的轮 → 该行 `retrieval_snapshot=[{chunk_id,score,text截600}…]`;闲聊/数据/工单轮 NULL。text 截断常量 config `retrieval_snapshot_text_max=600`。
- [ ] Step1 RED:三形态测(knowledge 轮含快照/闲聊轮 NULL/text 超限截断)。
- [ ] Step2 实现 GREEN + commit `feat(ch09-t5): retrieval_snapshot 随行落库(👎 回捞数据源)`。

### Task 6:POST /api/feedback(👎 落池+回捞)
**Files:** Modify `app/api/routes.py`、`app/schemas/chat.py`;Create `tests/test_feedback_api_ch09.py`;Modify `app/services/refusals.py`(签名扩 `retrieved_chunks=None`,写 LCQ 新列;触发钩子留 T7 接)。
**Interfaces:** Consumes `Message.retrieval_snapshot`、`crud.add_low_confidence_question(+快照参)`;Produces `POST /api/feedback {conversation_id≥1, seq≥0, vote:'up'|'down'}` → down:反查该会话第 seq+1 个 `role='assistant'` 行(仅数 assistant,Review Focus 4)→ raw_question=其前紧邻 user 行 → 快照回捞(列 NULL=空着)→ 落池 source=`user_feedback`,reason 记 seq → 200 `{pooled:true}`;越界/会话缺→404;up→204 不落。重复👎允许重复落池。
- [ ] Step1 RED:全分支(404/up 204/down 落池形态与回捞空/越界 seq)。
- [ ] Step2 实现 GREEN + commit `feat(ch09-t6): 👎 后端化——user_feedback 枚举空位启用+当轮回捞`;M2 批评审→修复批。

## M3 飞轮流水线 + 审核

### Task 7:flywheel 流水线服务 + 触发 + 补扫 CLI
**Files:** Create `app/services/flywheel.py`、`app/prompts/flywheel.py`、`app/schemas/flywheel.py`、`app/jobs/flywheel.py`、`tests/test_flywheel_ch09.py`;Modify `app/services/refusals.py`(落池成功后 `asyncio.create_task(process_lcq_row(row_id))`,异常吞→WARN)、`app/db/crud.py`(队列读写/累加/matched 写回)。
**Interfaces:** Produces `flywheel.process_lcq_row(row_id:int) -> None`(标准化 NormalizedQA{normalized_question,suggested_answer} → 查重 DedupMatch(matched_id|None),候选=待审全量≤50(超按 updated_at 截,WARN)→ 命中累加/新建行 → LCQ.matched_review_id 同事务写回);`crud.append_review_queue(...)`;CLI `uv run python -m app.jobs.flywheel [--limit N|--dry-run]`。LLM 走 `get_model` temperature=0 structured output。
- [ ] Step1 RED(mock LLM):命中累加不新建/未命中新建且 matched 指新行/candidate 截断/LLM 抛异常行留 NULL 不外泄/新建行「同事务写回 matched」封口(Focus 6)/补扫只吃 NULL 行幂等。
- [ ] Step2 实现 GREEN + commit `feat(ch09-t7): 飞轮流水线——标准化查重入队+异步触发+CLI 补扫`。

### Task 8:流水线 Prompt 评估(真模型)
**Files:** Create `evals/flywheel_samples.jsonl`(标准化≥5:口语/错字/指代;查重≥5:同义命中/近义陷阱/新问不误并)、`evals/run_ch09_flywheel_eval.py`(run_ch08 形制:ASCII 控制台+UTF-8 JSON 明细+exit code)。
- [ ] Step1 跑批 ≥8/10 + 人工逐条核对留痕 dev-notes;不达标只调 few-shot 重跑。
- [ ] Step2 commit `test(ch09-t8): 飞轮标准化/查重标注样例集+评估跑批结论`。

### Task 9:审核 API + 通过回写知识库
**Files:** Modify `app/api/routes.py`、`app/db/crud.py`;Create `app/services/review.py`(或并入 flywheel,实施自决)、`tests/test_review_api_ch09.py`。
**Interfaces:** Consumes `app/rag/indexer`(直写 chunk+向量化,category=「客服对话问答」,指纹幂等);Produces 三路由:`GET /api/review_queue?status=`(list,occurrence desc)/`GET /api/review_queue/{id}/detail`(行+matched 的 LCQ 原话/快照列表)/`PATCH /api/review_queue/{id}`(仅 待审→通过|驳回;通过必带 approved_answer 否则 422;其余流转 422;通过=同请求内 `publish_approved` 双落成功才置状态,失败 502 状态不动——Focus 5)。
- [ ] Step1 RED(mock indexer 成败两路):流转表全覆盖+半成功 502+detail 组装。
- [ ] Step2 实现 GREEN + integration 活库过一路 + commit `feat(ch09-t9): 审核 API——状态机+核准回写(过=写成才置态)`;M3 批评审→修复批。

### Task 10:后台审核页 + 👎 接线(Vibe,豁免流程)
**Files:** Create `static/review.html`(faith.html 形制:列表/状态筛选/通过·驳回/详情弹层含归并原话与召回片段/通过弹核准答案框预填 ai_suggested_answer);Modify `static/index.html`(顶栏入口;👎 点击同时 POST /api/feedback,👍 不发;锁钮与 localStorage 回显逻辑保持)。
- [ ] Step1 实现+手测闭环(列表→详情→通过→再问);commit `feat(ch09-t10): 审核后台页+👎 接线(Vibe)`。

## M4 评估 + 收口

### Task 11:eval_cycle 评估流水线
**Files:** Create `app/jobs/eval_cycle.py`、`tests/test_eval_cycle_ch09.py`;复用 `teacher_csv`/strategy_eval 缓存/faith 裁判核(必要时提函数不改原脚本行为)。
**Interfaces:** Produces `--trigger 手动|定时 --limit N --skip-faith --trend --report`;一行 `eval_runs{metrics:{recall_at_3,recall_at_10,mrr_at_10,faithfulness},dataset_size=实跑条数}`;`--trend` UTF-8 对比表(环比差值);`--report` Langfuse trace API 按 intent tag 聚合 token/cost 打表(不可达明确报错,不拖主流程)。
- [ ] Step1 RED:行落库形状/trend 排序/report 聚合纯函数(mock API 响应)。
- [ ] Step2 真跑两轮(integration,满额或 --limit 同口径)出 trend + commit `feat(ch09-t11): eval_cycle——两指标段落库+trend+意图成本 report`。

### Task 12:e2e 验收 + README + 完结
**Files:** Create `tests/test_e2e_ch09.py`(验收 2/3/4/5/6 DB 面钉;验收 1 UI 手测步 README 记)、README ch09 节(部署含 Langfuse 栈+建桶、演示命令、验收话术 P 步骤)、dev-notes 完结段。
- [ ] Step1 e2e:知识库没有的问题→兜底+队列出现+详情原话/快照;审核通过→同问再答对(真模型,标注非回归硬基线);👎 链→队列;eval_runs ≥2 行 trend 出;report 文件成。
- [ ] Step2 全量:unit 回归 + `-m integration` + e2e;commit `test(ch09-t12): 验收 e2e+README ch09 节+完结段`。
- [ ] Step3 终审批(fresh reviewer 全分支 diff)→ C/I 一次修复批,Minor deferred 入最终汇报;finishing-a-development-branch 三菜单交用户。

## 自审记录(成文后跑)
- Spec 覆盖:观测(T1/T2)、闸升级(T4)、三入口(T6;ch05/06 闸=T4;self_check=T7 漏斗覆盖+README 如实标注)、快照(T5)、流水线(T7/T8)、队列审核(T9/T10)、评估(T11/T12)、部署(T1/T12)、验收(T12)。无缺口。
- 类型一致性:`process_lcq_row/retrieved_chunks/retrieval_snapshot/EvidenceVerdict` 各任务口径已互锁。
- Review Focus 六条均已落归属任务 Step1。
