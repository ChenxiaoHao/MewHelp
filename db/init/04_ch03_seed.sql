-- ch03 历史对话种子:8 通会话(user/assistant 流水),供 mine_qa 抽取与三道闸演示。
-- 话题与期望挖掘结果 = spec 附录 D。重复行(如 C1 双问)是刻意的,别"顺手清理"。
-- 一次性执行:新库 initdb 自动跑;老库见 README「升级」段手动导入。
USE mewhelp;

INSERT INTO conversations (user_id, status) VALUES
('hist_c1', '已结束'), ('hist_c2', '已结束'), ('hist_c3', '已结束'), ('hist_c4', '已结束'),
('hist_c5', '已结束'), ('hist_c6', '已结束'), ('hist_c7', '已结束'), ('hist_c8', '已结束');

-- C1:运费三连问,"退货运费谁出"原话问两遍(仅标点差) → 闸1 精确归并靶子
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'user',  '请问买多少钱的东西才免邮寄费呀'),
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'assistant', '单笔订单实付满 99 元包邮,未满 99 元收 8 元基础运费哦。'),
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'user',  '那退货运费谁出'),
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'assistant', '7 天无理由退货寄回的运费您承担,质量问题换货我们承担双向运费。'),
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'user',  '退货运费谁出?'),
((SELECT id FROM conversations WHERE user_id='hist_c1'), 'assistant', '无理由退货运费您先垫付承担,质量问题则由我们承担来回运费。');

-- C2:换货流程,政策文档的口语化换说法(禁整句照抄是刻意的) → 闸3 discarded 靶子
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c2'), 'user',  '买的猫爬架还没拆封,想换个大的咋弄'),
((SELECT id FROM conversations WHERE user_id='hist_c2'), 'assistant', '在「我的订单」找到该订单点「申请退换」,填原因提交,审核通过后快递会上门取件。');

-- C3:秒杀不包邮,文档同义 → 闸3 边界样例
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c3'), 'user',  '1元秒杀的猫粮为啥还要派送钱'),
((SELECT id FROM conversations WHERE user_id='hist_c3'), 'assistant', '秒杀与一元试用商品不参与包邮,运费按地区下单时实时计算。');

-- C4:冻干混喂,增量知识 → kept
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c4'), 'user',  '冻干能拌在普通猫粮里一起喂吗'),
((SELECT id FROM conversations WHERE user_id='hist_c4'), 'assistant', '可以,冻干能当拌粮或零食,幼猫每天不超过 10 克,成猫 15-20 克。');

-- C5:水垢生物膜,增量知识 → kept
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c5'), 'user',  '饮水机槽里一层滑滑的东西怎么清理'),
((SELECT id FROM conversations WHERE user_id='hist_c5'), 'assistant', '那是生物膜,建议每周拆洗,用食用碱水擦洗,滤芯冲净再装回。');

-- C6:积分,文档完全无此话题 → kept
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c6'), 'user',  '积分怎么获得?能当钱花吗'),
((SELECT id FROM conversations WHERE user_id='hist_c6'), 'assistant', '每消费 1 元积 1 分,签收后到账,下单结算时可抵扣,100 积分抵 1 元。');

-- C7:运费险,问法近"运费"但答案全新 → kept 反例锚点(去重不得误杀)
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c7'), 'user',  '能买个运费险吗,退货运费太贵了'),
((SELECT id FROM conversations WHERE user_id='hist_c7'), 'assistant', '抱歉,平台暂不支持运费险;7 天无理由的寄回运费需自理,质量问题运费我们承担。');

-- C8:同通双问法问滤芯周期,客服同义作答 → 闸2 批内归并靶子(questions 双行)
INSERT INTO messages (conversation_id, role, content) VALUES
((SELECT id FROM conversations WHERE user_id='hist_c8'), 'user',  '饮水机的滤芯多久换一次'),
((SELECT id FROM conversations WHERE user_id='hist_c8'), 'assistant', '建议 30 天更换一次滤芯。'),
((SELECT id FROM conversations WHERE user_id='hist_c8'), 'user',  '那个过滤芯大概几天要换一回啊'),
((SELECT id FROM conversations WHERE user_id='hist_c8'), 'assistant', '滤芯大约 30 天换一回就可以。');
