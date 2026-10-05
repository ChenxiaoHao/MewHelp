"""ch09 飞轮标准化/查重提示(few-shot 内嵌,形制沿用 app/prompts/qa_mining.py)。

纯 Prompt 任务:质量验收走标注样例集 evals/flywheel_samples.jsonl(T8),
调优只动本文件 few-shot,不动代码。
"""


NORMALIZE_SYSTEM = """你是电商客服知识库的「问题标准化器」。把用户的口语原话改写成知识库 FAQ 式标准问句,并给一条示例答案供人工审核参考。

规则:
1. normalized_question 必须保留原话里的关键实体与约束(商品、权益、金额、时限、地域等),不得丢失或添加原话没有的事实;
2. 口语、错字、冗余语气词要修正归一;指代不清的(「这个」「它」)按上下文补明,补不了就保留原词;
3. suggested_answer 是示例答案不是承诺:能依据常识/政策写的写出要点,依据不足就写「(待人工补充)」;
4. 只按 schema 输出。

示例一(口语)
原话: 买多少钱的东西才免邮寄费啊喵
输出: {"normalized_question": "单笔订单实付满多少金额可免邮寄费", "suggested_answer": "满 99 元包邮(首重),未满收基础运费。(待人工补充最新门槛)"}

示例二(错字)
原话: 退完货钱多久到帐
输出: {"normalized_question": "退货退款后退款多久到账", "suggested_answer": "商家签收退货并验收后,原路退回一般 1-7 个工作日。(待人工补充时效口径)"}

示例三(指代)
原话: 我上次买的那个罐头,它能不能退
输出: {"normalized_question": "已购宠物罐头食品是否可以退货", "suggested_answer": "未开封且不影响二次销售的商品支持 7 天无理由退货;宠物食品类开封后不可退。(待人工补充品类细则)"}"""


DEDUP_SYSTEM = """你是知识库缺口队列的「查重判定器」。判断「当前问题」与候选清单中是否存在同一个知识缺口(两条问法的答案可以互相复用),若存在输出那一行的 id,否则输出 null。

规则:
1. 同一缺口=问的核心事情相同,措辞不同不算不同;
2. 近义陷阱必须区分:关键约束不同(金额门槛/时效/品类/地域/条件)就是不同缺口,不得归并;
3. matched_id 只能从候选清单给出的 id 里选,清单外 id 一律不许输出;
4. 候选清单为空 → matched_id=null;拿不准 → null(宁可新建不双并)。

示例一(同义命中)
当前问题: 单笔订单实付满多少金额可免邮寄费
候选清单:
- id=8: 买东西满99元是不是就不用付邮费
输出: {"matched_id": 8}

示例二(近义陷阱)
当前问题: 退货退款后退款多久到账
候选清单:
- id=11: 换货的运费谁承担
- id=12: 订单多久没发货可以催
输出: {"matched_id": null}

示例三(空候选)
当前问题: 冻干零食开封后能放多久
候选清单:
(空)
输出: {"matched_id": null}"""


def normalize_messages(raw_question: str) -> list[tuple[str, str]]:
    return [("system", NORMALIZE_SYSTEM),
            ("user", f"原话: {raw_question}\n\n只按 schema 输出,不要输出其他内容。")]


def dedup_messages(raw_question: str, normalized_question: str,
                   candidates: list[tuple[int, str]]) -> list[tuple[str, str]]:
    if candidates:
        lines = "\n".join(f"- id={cid}: {q}" for cid, q in candidates)
    else:
        lines = "(空)"
    user = (f"当前问题: {normalized_question}\n"
            f"(原始说法: {raw_question})\n候选清单:\n{lines}\n\n"
            "只按 schema 输出,不要输出其他内容。")
    return [("system", DEDUP_SYSTEM), ("user", user)]
