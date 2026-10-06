# ch10 模型微调:多标签主题分类器 · 设计规格(spec)

日期:2026-10-06 · 分支:`ch10-topic-classifier`(自 master 45245b8)· 状态:待用户过目批准

## 目标与定位

把 ch09 飞轮攒下的低置信度问题按主题批量归类,回答「先补哪块知识」。核心是微调一个
自有模型:RoBERTa-wwm-ext 全参数多标签分类器(17 类),以**旁路批量 job** 形式消费
`low_confidence_questions` 池,结果写 `topic_classifications`,飞轮后台新增主题分布页。
实时对话主链路**不写不读**本章新表,零侵入。

本章不做:LoRA/QLoRA、embedding 微调、意图识别微调(教材明示不用/不做)。

## 现状事实(2026-10-06 实测)

- 池内真实问题 138 条:retrieval_low_conf 58 / self_check 65 / user_feedback 3 / ch05_gate 12;
- `topic_classifications` 表不存在;仓库无既有术语表文件,本章新建为唯一词表;
- 算力:RTX 4060 Laptop 8GB 本机直跑;torch/transformers/scikit-learn 未装,本章新增依赖;
- LLM 造数/预标走项目 .env 已配的 DashScope 通道,不占本对话账户预算。

## 类目契约(17 类,定死)

权威类目表落 `finetune/glossary.json`,全系统唯一,标注/造数/增强三处共用:

退换货、物流、尺码、发票、质量问题、运费、优惠活动、价保、支付、订单修改、库存补货、
商品信息、保修维修、账号、会员积分、评价、其他。

四大类(退换货/物流/尺码/发票)领头,每类配一句边界说明;近邻边界(进边界说明原文):
修归保修维修、退归退换货;运费管钱、物流管货;价保是补差价、优惠活动是券和满减。
每类另带同义词表(供数据增强做同义词替换,也作造数的类内词面)。

## 数据管道(finetune/)

四步按序,产物全部落盘可复查:

1. **清洗** `finetune/clean.py`:代码做确定性面——脱敏(手机号/订单号/地址等 → 占位符)、
   全半角/空白/标点规范。**错别字修正并入 LLM 预标同一次调用**(拍板 2,省一半往返)。
2. **语料构成** `finetune/synth.py`:真实池 138 条全量入集;每类不足→大模型照术语表补造。
   目标规模 ~1700 条(均值 100/类,「其他」略少),多诉求句(字面命中 ≥2 标签)占比 ~25%
   (拍板 1)。造数 prompt 必须携带术语表全文(类目+边界说明)。
3. **预标+人工抽审** `finetune/prelabel.py`:大模型照术语表对全量语料出多标签
   (规则:字面提到几个诉求就打几个标签,一个不多一个不少;歧义倾向「其他」+备注)。
   抽审面(拍板 3):每类随机 10% + 全部多标签样本,导 `finetune/audit/review_*.csv`
   (UTF-8)给用户人工核;人工改标经 `finetune/apply_audit.py` 回写主数据集。
   预标结果、抽审、回改三步产物都在 git 可见,标注可追责。
4. **分层抽样** `finetune/split.py`:按类目组合分层 80/10/10 → `finetune/dataset/{train,valid,test}.jsonl`
   (行:`{"text": ..., "labels": [...]}`);各类按比例进三份,test 全程封存不参与任何调参。
5. **数据增强** `finetune/augment.py`:同义词替换(术语表词表)+句式模板微调,**只扩
   train**(目标 train ≈ 原始量 ×1.5–2),纯代码零 LLM 成本(随设计表默认批准)。
   valid/test 一行动不得——测试断言钉住。

## 训练

`finetune/train.py`:`hfl/chinese-roberta-wwm-ext` **全参微调**(不用 LoRA)。
头:`pooler_output → Linear(768,17) → Sigmoid`,损失 `BCEWithLogitsLoss`。
超参定死起步:lr 2e-5、batch 16、max_len 128、epochs ≤10、`EarlyStopping patience=3`
monitor valid **macro-F1**、weight_decay 0.01。每 epoch 结束评 valid 一次,最优 checkpoint
落 `models/ch10_topic/`(含 tokenizer+config:类目顺序、**决策阈值**)。
阈值(拍板 5):训练后在 valid 上扫 [0.3, 0.7] 取 macro-F1 最优,写进 config.json;
旁路推理与评测同用该阈值。
下载(拍板 6):torch cu121 wheel 官方镜像源;HF 模型 `HF_ENDPOINT=https://hf-mirror.com`。
`pyproject.toml` 新增 `ml` extras 或直装(实现时按 uv 习惯),不升任何既有依赖。

## 评测

`finetune/evaluate.py`,只在封存 test 上跑一次定稿报告:
- 逐类 precision/recall/F1(support)+ micro/macro 平均;
- **每类目二值混淆矩阵**(该类 TP/FP/FN/TN 四格,17 张合一份 md);
- 错例全量导 `finetune/reports/misclassified.csv`(UTF-8)供人工抽判复核;
- 控制台只出 ASCII 汇总一行(GBK 红线),中文表落 `finetune/reports/ch10_eval_report.md`。
人工复核结论(判错样本里多少是真错、多少标错)写进 dev-notes。

## 部署:旁路批量归类(不碰主链路)

- 表:`sql/10_ch10_topic_classifications.sql` = 用户 DDL 原文逐字(uk_question_id 一人一行);
  SQLAlchemy 模型 `TopicClassification` 进 `app/db/models.py`。
- 推理模块 `app/services/topic_classifier.py`:进程内加载 `models/ch10_topic/`,
  `classify_batch(texts) -> list[list[label]]`(transformers 后端;ONNX 导出脚本
  `finetune/export_onnx.py` 作为可选件落盘,服务不依赖它——拍板 4 推荐默认)。
- 攒批 job `app/jobs/topic_classify.py`(仿 eval_cycle CLI 形态):
  取池中 `LEFT JOIN topic_classifications` 未归类行 → 批量推理 → insert;
  `--rerun` 时对已归类行 upsert 刷新 labels(模型迭代场景);
  `--limit N` 限批;Langfuse trace 一条(章名+条数),不进对话主 trace 树。
- API `GET /api/topics/distribution?days=30&min_count=`:`topic_classifications` join
  `low_confidence_questions`,labels JSON 展开计数在 Python 层(行数小),返回
  `[{label, count, pct}]` 按 count 降序。只读端点。

## 飞轮后台主题分布页(Vibe Coding 例外,不套 TDD/评审)

`static/topic.html`:17 类横向条形图,纯 CSS 条(零新前端依赖),按量降序,
四大类若堆高一眼可见;顶部时间范围选择(days);`review.html` 头部加入口链接一行。
页面本身直改直验;其消费的 API 端点照常走 TDD。

## 验收映射

| 验收 | 证据 |
|---|---|
| 1 测试集各类目 F1+混淆矩阵报告 | `finetune/reports/ch10_eval_report.md` 落盘 + 演示命令 |
| 2 真实问题跑归类,后台页见分布 | `uv run python -m app.jobs.topic_classify --limit …` + topic.html 截图/口述 |
| 3 多诉求句同时命中多类 | 评测报告含多标签样例(「买大了想退」→ 尺码+退换货) + e2e 用例钉 |

## 全局约束(每任务隐含)

- 密钥真值只进 gitignored .env;compose 只 `${VAR:-placeholder}`;
- 永不 `docker compose down -v`、永不 `git clean -fdx`;
- 控制台 ASCII,中文明细 UTF-8 落文件;
- `uv run python` / `uv run pytest` 永远;
- ch04 三恒 M 文件不入库;提交精确列文件;
- 演示资产(chunk53/queue11+12/LCQ177/eval_runs 5/6/11)与既有 138 条池行**只读不动**;
- 库/框架用法 Context7 每库至多一次,故障即切官方文档;
- 选型定死,矛盾即停即问;
- dev-notes/ch10.md 每阶段追加,不收尾补记。

## 拍板记录(2026-10-06 用户「全部按默认的来吧」)

1. 数据集 ~1700 条(100/类),多标签 ~25%,真实 138 条全量入集;
2. 错别字修正并入预标调用;
3. 抽审=每类 10%+全部多标签样本;
4. 部署=进程内批量 job;ONNX 导出为可选件,独立推理服务不做;
5. 阈值=valid 扫 [0.3,0.7] 取 macro-F1 最优,写 config;
6. torch 官方镜像 + hf-mirror 下载;
7. 执行=executing-plans 内联,终审一次高配 fresh reviewer,里程碑批评审照旧。

## 里程碑

- **M1 数据管道**:DDL+模型+术语表+清洗+造数+预标+抽审回写+切分+增强 → dataset 定稿;
- **M2 训练+评测**:train/evaluate/报告 → 验收 1、3 落证;
- **M3 旁路部署**:classifier 模块+job+distribution API → 验收 2 前半;
- **M4 后台页+e2e+完结**:topic.html(Vibe)+端到端演示+README/dev-notes+finishing。
