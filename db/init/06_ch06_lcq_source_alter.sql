-- =============================================================
-- ch06 · 终审批次(Important#1):low_confidence_questions.source 补闸池枚举值
-- ch05 起置信闸拒答以 source='ch05_gate' 入池、ch06 退款闸用 'ch06_refund_gate';
-- 05 建表的三值 ENUM 不含它们 → 新环境重建后池写被 MySQL 1265 拒、refusals 吞成
-- WARN(静默丢行)。05 按「用户原文」约定不改,新章节 ALTER 前置于此文件头已预告。
-- dev 库已同步执行等价 ALTER(2026-09-28,T8)。
-- =============================================================
SET NAMES utf8mb4;

ALTER TABLE low_confidence_questions
  MODIFY COLUMN source
  ENUM('retrieval_low_conf','self_check','user_feedback','ch05_gate','ch06_refund_gate')
  NOT NULL
  COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决 / ch05 置信闸 / ch06 退款闸';
