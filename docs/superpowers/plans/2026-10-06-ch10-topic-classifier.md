# ch10 多标签主题分类器微调 · 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 微调 RoBERTa-wwm-ext 17 类多标签主题分类器,旁路批量归类飞轮低置信度池,落 `topic_classifications` 并在后台出主题分布页。

**Architecture:** `finetune/` 数据管道+训练+评测(离线面),`app/` 旁路推理 job+只读 API(在线面),`static/topic.html` 后台页;实时主链路零侵入。spec=`docs/superpowers/specs/2026-10-06-ch10-topic-classifier-design.md`(定稿权威)。

**Tech Stack:** hfl/chinese-roberta-wwm-ext(全参,不用 LoRA)、torch(cu121)+transformers+scikit-learn(新增)、FastAPI/SQLAlchemy 既有栈、LLM 造数/预标走 `app.services.chat_service.get_model`。

**Spec:** `docs/superpowers/specs/2026-10-06-ch10-topic-classifier-design.md`

## Global Constraints

- 密钥真值只进 gitignored .env;永不 `docker compose down -v` / `git clean -fdx`;
- 控制台 ASCII,中文明细 UTF-8 落文件(GBK 红线);`uv run python` / `uv run pytest` 永远;
- ch04 三恒 M 文件不入库;提交精确列文件;一任务一 commit(代码+测试+勾选+dev-notes 同笔);
- 既有池 138 行与演示资产只读不动;`topic_classifications` 新表可随意;
- 选型定死,矛盾即停即问;Context7 每库至多一次(transformers 已核:TrainingArguments 用
  `eval_strategy`,多标签头 `problem_type="multi_label"`,float labels);
- 非单测任务(造数/预标/训练/评测实跑)以「样例或评估集跑一遍验证」替代 TDD 环节,其余步骤照走;
- 前端 T11 走 Vibe Coding(用户描述效果直改,不套 TDD/评审)。

## API 核对结论(实现期不再翻)

- `AutoModelForSequenceClassification.from_pretrained(path, num_labels=17, problem_type="multi_label", id2label=…, label2id=…)` → 自动 BCEWithLogits;labels 传 `float tensor`(multi-hot);
- `TrainingArguments(output_dir, learning_rate=2e-5, per_device_train_batch_size=16, num_train_epochs=10, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1_macro")` + `EarlyStoppingCallback(early_stopping_patience=3)`;
- 推理:`model(**inputs).logits.sigmoid()` 自算阈值,不用 pipeline(阈值要可配)。

---

### Task 1: DDL + ORM `TopicClassification`

**Files:** Create `sql/10_ch10_topic_classifications.sql`(用户 DDL 原文逐字);Modify `app/db/models.py`(尾追加 `class TopicClassification(Base)`);Test `tests/test_topic_model_ch10.py`(integration 活库)。

**Interfaces:** Produces 表 `topic_classifications(id, question_id uk→low_confidence_questions.id, labels JSON, classified_at)`;ORM 字段与其余表同律(`_pk()`、`BIGINT(unsigned=True)`)。

- [ ] Step1 落 sql 文件(原文逐字,含 SET NAMES utf8mb4)。
- [ ] Step2 活库执行建表:`docker compose exec -T mysql mysql -uroot -p"$(grep -E '^MYSQL_ROOT_PASSWORD=' .env | cut -d= -f2 -)" mewhelp < sql/10_ch10_topic_classifications.sql`(口令从 .env 变量取,不回显明文)。Expected: `SHOW TABLES LIKE 'topic_classifications'` 命中。
- [ ] Step3 RED:integration 测试——向真池任一行 insert 归类行成功、二次 insert 同 question_id 触发 `IntegrityError`(uk 幂等),测后删行。Expected FAIL(类不存在)。
- [ ] Step4 GREEN:models.py 加类;测试转绿。
- [ ] Step5 commit `feat(ch10-t1): topic_classifications DDL+ORM+uk幂等钉`。

### Task 2: 术语表 `finetune/glossary.py`

**Files:** Create `finetune/glossary.json`、`finetune/glossary.py`;Test `tests/test_glossary_ch10.py`(unit)。

**Interfaces:** Produces `CLASSES: list[str]`(17 类定序,唯一权威)、`load_glossary() -> dict[str, dict]`(name → {boundary, synonyms})、`assert_valid_labels(labels) -> list[str]`。

- [ ] Step1 RED:钉 17 类名单与顺序(退换货领头、其他收尾)、JSON 每类 boundary 非空、三对近邻边界句在对应类 boundary 原文中出现(修/退、运费/物流、价保/优惠)、synonyms 每类 ≥3。Expected FAIL。
- [ ] Step2 GREEN:glossary.json 按 spec 类目契约撰写(边界说明原文进 boundary);glossary.py 加载+缓存+校验。
- [ ] Step3 commit `feat(ch10-t2): 17类权威术语表——边界说明+同义词表`。

### Task 3: 清洗 `finetune/clean.py`

**Files:** Create `finetune/clean.py`;Test `tests/test_clean_ch10.py`(unit)。

**Interfaces:** Produces `clean(text: str) -> str`(= `desensitize` ∘ `normalize`);手机号→`<PHONE>`、订单号/长数字串→`<NUM>`、连续空白折叠、全角 ASCII 归半角、首尾修剪。错别字**不在**本模块(拍板 2 进预标)。

- [ ] Step1 RED:脱敏三例+格式三例+幂等(clean(clean(x))==clean(x))+空串。Expected FAIL。
- [ ] Step2 GREEN 实现;跑一遍真实池:`uv run python -m finetune.clean --dump finetune/audit/pool_cleaned.csv`(138 行 UTF-8,人可查)——样例验证替代步骤,记 dev-notes。
- [ ] Step3 commit `feat(ch10-t3): 清洗——脱敏+格式规范,池全量清洗落盘可查`。

### Task 4: 造数 `finetune/synth.py`

**Files:** Create `finetune/synth.py`;Test `tests/test_synth_ch10.py`(unit)。

**Interfaces:** Consumes glossary、chat_service.get_model;Produces `build_prompt(cls_name, glossary, n) -> str`(含术语表全文+边界+多诉求 directive)、`parse_questions(reply) -> list[str]`(行拆+去序号+校验非空/长度)、`gap_plan(counts, target_per_class) -> dict[str, int]`(缺口=目标-已有,真实池与已造都算已有)。CLI:`-m finetune.synth --target 100 --out finetune/drafts/synth.jsonl`(断点续造:已有行按类计数,只补缺口)。

- [ ] Step1 RED:build_prompt 含 17 类名+该类目标词+「≥2 标签」示例句要求;parse_questions 抗「1. 」「- 」前缀与空行;gap_plan 只补缺口、不超造。Expected FAIL。
- [ ] Step2 GREEN(fake LLM 单测全绿后)**实跑 LLM 造数**至 ~1560+(1700-138 缺口,多诉求句 directive 常驻 prompt)→ 验证:类分布统计 ASCII 打印 + 抽查 20 条 UTF-8 落 `finetune/audit/synth_check.csv`(样例验证替代 TDD 尾步)。
- [ ] Step3 commit `feat(ch10-t4): 术语表驱动造数——缺口续算+多诉求directive,~1700语料成形`。

### Task 5: 预标 + 抽审面 `finetune/prelabel.py`

**Files:** Create `finetune/prelabel.py`、`finetune/apply_audit.py`;Test `tests/test_prelabel_ch10.py`(unit + 1 integration)。

**Interfaces:** Consumes clean、glossary、synth 草稿、真池;Produces `PRELABEL_SYSTEM`(prompt:照术语表打标+顺手修错别字+只回 JSON `{"labels":[...]}`)、`parse_labels(reply) -> list[str]`(JSON 解+白名单过滤+空则归「其他」+「一个不多一个不少」校验去重)、`run_audit_export(dataset, per_class=0.1, seed) -> Path`(每类 10%+全部多标签 → `finetune/audit/annotation_review.csv`)、`apply_audit(csv, dataset)`(人工改标回写,labels 列以 `|` 分隔)。真池读取 integration:LEFT 无归类要求的 raw_question 全量入标。

- [ ] Step1 RED:parse_labels 对脏回复(JSON 包裹/越类名/重复/空)四类行为钉死;audit_export 固定 seed 确定性+多标签 100% 入审;apply_audit 回改生效+非法类名拒收。Expected FAIL。
- [ ] Step2 GREEN;**实跑全量预标**(1700 条,qwen 温度 0,并发≤4)→ `finetune/drafts/labeled.jsonl`;抽审 CSV 导出给用户(人工环节异步:用户不回改即以预标定稿,记 ledger 与 dev-notes)。
- [ ] Step3 commit `feat(ch10-t5): LLM预标(含错别字归一)+10%/全多标签抽审面+回改apply`。

### Task 6: 分层切分 + 增强 `finetune/split.py` `augment.py`

**Files:** Create 两文件;Test `tests/test_split_augment_ch10.py`(unit)。

**Interfaces:** Produces `stratified_split(rows, ratios=(.8,.1,.1), seed) -> dict[str, list]`(按 **label-组合** 分层,类内等比;组合稀有到 <3 条时并入「其他组合」桶轮转分配,断言只钉「每类 train/valid/test 计数 ≥ 该类总数 70%/5%/5% 且无一行跨集重复」);`augment_train(train, glossary) -> list`(同义词替换 ×1 + 句式模板改写 ×0.5,新增行带 `aug:true`;**绝不触碰 valid/test**)。产物 `finetune/dataset/{train,valid,test}.jsonl`。

- [ ] Step1 RED:比例容差、无跨集泄漏(行 text 集合互斥)、增强只进 train、valid/test 文件哈希前后不变、增强行数 ∈ [原始×1.2, 原始×2.2]。Expected FAIL。
- [ ] Step2 GREEN + 实跑出定稿数据集,ASCII 统计(每类 train/valid/test 计数)进 dev-notes。
- [ ] Step3 commit `feat(ch10-t6): 分层80/10/10+同义词/句式增强只扩train,dataset定稿`。

### Task 7: 依赖 + 模型资产 + `train.py`

**Files:** Modify `pyproject.toml`(新增 `torch`(cu121 index)、`transformers`、`scikit-learn`——uv 习惯用 `[[tool.uv.index]]` extra,不升既有包);Create `finetune/train.py`;Test `tests/test_train_smoke_ch10.py`(integration:tiny 数据 20 条 + `--max-steps 2` CPU smoke,断言产出目录含 config.json/model.safetensors/tokenizer 三件)。

**Interfaces:** Consumes dataset jsonl+glossary;Produces `load_dataset(path) -> datasets-like list`、`compute_metrics(eval_pred) -> {"f1_macro","f1_micro"}`(sigmoid@0.5 起步)、CLI `uv run python -m finetune.train [--epochs 10] [--max-steps N] [--smoke]`;产物 `models/ch10_topic/`(模型+tokenizer)+ `models/ch10_topic/topic_config.json`(classes 序、threshold、best_valid_f1、train 时间戳)。阈值扫:训练毕在 valid 上对 [0.3,0.35,…,0.7] 扫 macro-F1 取最优写 config(spec 拍板 5)。

- [ ] Step1 装依赖+下载 `hfl/chinese-roberta-wwm-ext`(`HF_ENDPOINT=https://hf-mirror.com`)。Expected:`uv run python -c "import torch,transformers;print(torch.cuda.is_available())"` → True;模型目录本地存在。
- [ ] Step2 RED smoke 测试(产出不存在即 FAIL)。
- [ ] Step3 GREEN train.py(全参、无 LoRA;早停 patience=3 monitor f1_macro;fp16 走 TrainingArguments `fp16=True`——4060 支持,显存告急即报不换方案);smoke 绿。
- [ ] Step4 **正式训练实跑**(机时 1-2h,`--limit` 不砍数据)→ 日志 ASCII 落 `finetune/reports/train_log.txt`;最优 checkpoint+阈值配置落盘。样例/资产验证替代 TDD 尾步。
- [ ] Step5 commit `feat(ch10-t7): RoBERTa-wwm-ext全参微调——早停+valid扫阈值,模型资产落盘`(models/ 与 pip cache 进 .gitignore,pyproject+代码进库)。

### Task 8: 评测 `finetune/evaluate.py`

**Files:** Create;Test `tests/test_evaluate_ch10.py`(unit:合成 pred/gold 小矩阵钉逐类 P/R/F1 与二值混淆四格已知值;threshold 加载生效)。

**Interfaces:** Consumes `models/ch10_topic/`+test.jsonl;Produces `binary_confusion(preds, golds, classes) -> dict[str, dict]`(每类 TP/FP/FN/TN)、`report_text(...) -> str`(逐类表+micro/macro+阈值+多标签命中例);CLI `-m finetune.evaluate`。落 `finetune/reports/ch10_eval_report.md`(UTF-8)+ `misclassified.csv`;控制台一行 ASCII `macro_f1=… micro_f1=… n=…`。

- [ ] Step1 RED:四格数学+空类(support=0)不炸+阈值单调影响格数。Expected FAIL。
- [ ] Step2 GREEN;**test 集封存首跑**出报告(验收 1);多诉求句命中例(「买大了想退」→ 尺码|退换货)进报告样例节(验收 3 前半);错例人工抽判 20 条,结论(真错/标错比例)记 dev-notes(验收 3 复核)。
- [ ] Step3 commit `test(ch10-t8): 逐类PRF1+每类二值混淆矩阵+错例台账,test封存首跑落报告`。

### Task 9: 旁路推理 `app/services/topic_classifier.py` + 攒批 job

**Files:** Create 两文件;Test `tests/test_topic_classify_ch10.py`(unit:mock classifier 断言选池 SQL 只要未归类行、批切分、insert/upsert 调用面;integration:真模型 `--limit 3` 活库真写+重跑幂等)。

**Interfaces:** Consumes 池表+`models/ch10_topic/topic_config.json`;Produces `TopicClassifier(settings).classify_batch(texts: list[str]) -> list[list[str]]`(加载一次、批推理、threshold 来自 config)、CLI `-m app.jobs.topic_classify [--limit N] [--rerun] [--dry-run]`;job 输出 ASCII 行计数。Langfuse trace 一条(name=`ch10_topic_classify`,tags=["ch10"])。

- [ ] Step1 RED→Step2 GREEN(mock 面全绿)。
- [ ] Step3 integration 真写:`--limit 3` 首跑 insert 3 行,重跑(无 --rerun)处理数 0,`--rerun --limit 3` 刷新同 3 行(uk 不新增)。测后**保留**这 3 行作演示资产?——保留(验收 2 素材),dev-notes 记行 id。
- [ ] Step4 commit `feat(ch10-t9): 旁路批量归类job——未选池攒批/uk幂等/--rerun刷新,主链路零侵入`。

### Task 10: `GET /api/topics/distribution`

**Files:** Modify `app/api/routes.py`、`app/schemas/chat.py`(新响应模型);Test `tests/test_topics_api_ch10.py`(unit:fake crud 断言 days 下传+shape;integration:真库有归类行时计数=预期,pct 和=100±0.1)。

**Interfaces:** Produces `TopicStat(label, count, pct)`;`GET /api/topics/distribution?days=30` → 按 count 降序数组;labels JSON 展开计数在 Python 层(crud `topic_distribution(session, days)`,join LCQ.created_at 过滤)。

- [ ] Step1 RED(fake)→ Step2 GREEN → Step3 integration(依赖 T9 已写的演示行)→ 全绿。
- [ ] Step4 commit `feat(ch10-t10): 主题分布只读API——labels展开+days窗+pct`。

### Task 11: 主题分布页 `static/topic.html`(Vibe Coding)

**Files:** Create `static/topic.html`、可选 `finetune/export_onnx.py`(spec 拍板 4 可选件:预算有余才做,缺失不阻塞任何验收——ledger 记一笔即可);Modify `static/review.html`(头部一行入口链接)。

Vibe 直做:fetch `/api/topics/distribution`,17 条纯 CSS 横向条形(count 定宽,pct 标条尾),降序;days 下拉(7/30/全部);风格贴 review.html(猫系文案一句)。**不走 TDD/评审**(用户工作规约 1 例外);浏览器打开 `http://localhost:8000/static/topic.html` 真数据截图口述。commit `feat(ch10-t11): 飞轮后台主题分布页(Vibe)+审核页入口`。

### Task 12: e2e 演示 + README + 完结

- [ ] Step1 e2e 钉(integration):攒批→分布 API→计数单调;多标签句(验收 3 终钉:一句真实多诉求进池→归类 labels≥2)。
- [ ] Step2 README「ch10」节(演示命令:`-m finetune.train/evaluate`、`-m app.jobs.topic_classify`、分布页 URL、新依赖安装)。
- [ ] Step3 dev-notes 完结段+里程碑批评审收口;终审(review-package → fresh reviewer 高配)→ fix pass(RED→GREEN)→ finishing-a-development-branch(合并前问用户 push 与否——远端仍无,预期本地 merge)。
- [ ] commit `test(ch10-t12): 验收三钉+README ch10节+完结留痕`。

## Review Focus

spec 隐含但任务测试不全覆盖、终审须专查的输入类:

1. **主链路零侵入**:任何 ch10 新代码路径都不得出现在 chat/stream 调用链(import 面+运行面)——查 `app/services/topic_classifier.py` 的使用者只有 job;
2. **池行只读**:job/API 不得 UPDATE/DELETE `low_confidence_questions`(唯一例外 matched_review_id 旧逻辑,与本表无关);
3. **标签词表一致性**:glossary CLASSES、DDL 注释、topic_config.json classes、分布页图例四处同名同序——漂移即静默错位统计;
4. **增强泄漏**:valid/test 任何形式被增强/被造数句近似重复(近似去重按 text 精确+空白归一后哈希);
5. **阈值语义**:训练扫阈、推理用阈、评测报阈三者同源 config,任一硬编码 0.5 即 finding;
6. **GBK/控制台**:所有 CLI 打印 ASCII-only,中文进文件。

## 每任务通用验证

任务收尾跑 `uv run pytest`(unit 全量)+ 本任务 integration 文件;里程碑批评审(规约 5)只审行为/正确/安全。
