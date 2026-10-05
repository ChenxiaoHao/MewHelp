-- ch09 飞轮回捞数据源:进 agent 的轮把当轮证据快照随 assistant 消息行落库
-- 👎 落池时按 (conversation_id, assistant seq) 反查此行回捞;没走检索的轮恒 NULL
-- (spec 附录二原文;依赖 01_schema.sql 的 messages.content 列位)
SET NAMES utf8mb4;

ALTER TABLE messages
  ADD COLUMN retrieval_snapshot JSON NULL COMMENT '当轮召回片段快照(Top 原文+得分),供事后 👎 回捞;无检索轮为 NULL' AFTER content;
