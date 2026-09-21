"""chunker 纯函数测试(spec §4)。规则口径全部钉在断言里,改动 chunker 先看这里。"""

from app.rag.chunking import (
    ChunkDraft,
    apply_overlap,
    iter_sections,
    parse_faq_qa,
    parse_front_matter,
    split_blocks,
    split_body,
    split_markdown,
    split_sentences,
    split_table,
    tail_sentences,
)

POLICY_DOC = """---
content_type: policy
---
# 退货政策

## 运费说明
退货运费：7 天无理由退换由买家承担寄回运费。质量问题换货由我方承担双向运费。

## 退款到账（关键条款）
验收通过后 1-3 个工作日按原支付路径退回。
"""

FAQ_DOC = """---
content_type: faq
---
# 商品 FAQ

## 猫粮

**Q：幼猫一天喂几次**
**Q：小猫咪一天要吃几顿**
**A：2-3 月龄建议少食多餐，每天 3-4 次。**

### 冻干猫粮喂食量表

| 月龄 | 体重 | 每日克数 |
| --- | --- | --- |
| 3个月 | 1-2kg | 40-60 |
"""


def test_split_sentences_boundaries():
    sents = split_sentences("满99元包邮。未满收8元！偏远199元？秒杀不算；结束. Next 2.0 版")
    assert sents == ["满99元包邮。", "未满收8元！", "偏远199元？", "秒杀不算；", "结束.", "Next 2.0 版"]
    # 「2.0」= 句号+空格+数字 → 不断句(版本号)


def test_tail_sentences_prefers_whole_suffix():
    sents = ["第一块讲包邮门槛九十元以上。", "第二块讲偏远地区另算。"]  # 14 + 11 字
    assert tail_sentences(sents, overlap=20) == "".join(sents)  # 25 ≤ 30 且 ≥ 10 → 两句都要
    assert tail_sentences(sents, overlap=8) == sents[1]  # 11 ≤ 12 且 ≥ 4 → 只取末句
    assert tail_sentences(sents, overlap=6) == ""  # 末句 11 > 9 → 凑不出整句,宁缺毋滥


def test_split_body_downgrades_paragraph_line_sentence():
    text = "。".join(["句" * 29] * 5) + "。"  # 5 句 ×30 字,无换行 → 降档到句级
    pieces = split_body(text, chunk_size=100)
    assert pieces == [text[:90], text[90:]]  # 整句成组,不留半截话


def test_split_body_paragraph_packing():
    paras = ["甲" * 60, "乙" * 60, "丙" * 60]
    assert split_body("\n\n".join(paras), chunk_size=100) == paras


def test_split_body_hard_cut_single_long_sentence():
    t = "长" * 250  # 单句无边界:三级全退化 → 硬切兜底
    pieces = split_body(t, chunk_size=100)
    assert all(len(p) <= 100 for p in pieces) and "".join(pieces) == t


def test_split_table_header_copied_per_piece():
    header = ["| 场景 | 凭证 | 赔付 |", "| --- | --- | --- |"]
    rows = [f"| 场景{i} | 运单照片 | 赔付{i * 10}元 |" for i in range(1, 6)]
    pieces = split_table([*header, *rows], chunk_size=80)
    assert [p.splitlines()[2] for p in pieces] == rows  # 行序保留,每块恰好一行(80 字口的算术结果)
    assert all(len(p.splitlines()) == 3 for p in pieces)  # 表头行+分隔行逐块复制


def test_split_blocks_separates_table_and_text():
    text = "前文一段。\n\n| 列A | 列B |\n| --- | --- |\n| 1 | 2 |\n\n后文一段。"
    pieces = split_blocks(text, chunk_size=500)
    assert pieces[0].startswith("前文") and "|" not in pieces[0]
    assert pieces[1].startswith("| 列A |")
    assert pieces[2].startswith("后文")


def test_iter_sections_strips_key_suffix_and_stacks():
    body = (
        "# 售后手册\n\n## 转人工流程（关键条款）\n第一步。\n\n### 夜间值班\n说明。\n\n"
        "## 破损件处理\n文字。\n"
    )
    paths = [(titles, key) for titles, key, _ in iter_sections(body)]
    assert paths == [
        (["售后手册", "转人工流程"], True),
        (["售后手册", "转人工流程", "夜间值班"], True),  # 祖先标记向下传导(设计决策:关键条款节内的子节同标)
        (["售后手册", "破损件处理"], False),
    ]


def test_parse_faq_qa_groups_and_leftover():
    body = (
        "**Q：幼猫一天喂几次**\n**Q：小猫咪一天要吃几顿**\n**A：2-3 月龄少食多餐，每天 3-4 次。**\n\n"
        "**Q：开封后能放多久**\n**A：密封阴凉保存，一个月内吃完。**\n\n"
        "| 月龄 | 克数 |\n| --- | --- |\n| 3个月 | 40-60 |\n"
    )
    qa, other = parse_faq_qa(body)
    assert qa[0][0] == ["幼猫一天喂几次", "小猫咪一天要吃几顿"]  # 连续 Q 行并入同组
    assert qa[0][1] == "2-3 月龄少食多餐，每天 3-4 次。"
    assert qa[1][0] == ["开封后能放多久"]
    assert "| 月龄 | 克数 |" in other and "**Q" not in other  # 表格是 leftovers,不污染 answer


def test_apply_overlap_whole_sentence_prefix():
    s1 = "第一块讲包邮门槛九十元以上。第二块讲偏远地区另算。"
    out = apply_overlap([s1, "后续内容讲秒杀不包邮。"], overlap=8)
    assert out[0] == s1  # 首块不动
    assert out[1] == "第二块讲偏远地区另算。后续内容讲秒杀不包邮。"  # 11字 ≤ 12 → 末整句前置


def test_apply_overlap_never_leaves_half_sentence():
    s1 = "这句话特别长长长到超过一点五倍重叠预算。"  # 20 字 > 6×1.5
    out = apply_overlap([s1, "下一块。"], overlap=6)
    assert out[1] == "下一块。"  # 凑不出整句 → 宁缺毋滥,不重叠


def test_split_markdown_policy_trio_and_vector_text():
    d0, d1 = split_markdown(POLICY_DOC)
    assert (d0.category, d0.questions, d0.content_type, d0.section_path) == (
        "退货政策", "运费说明", "policy", "退货政策 > 运费说明")  # §4-5: category=上级路径
    assert d0.is_key_clause is False
    assert d1.is_key_clause is True and d1.section_path == "退货政策 > 退款到账"  # 后缀剥离
    assert d0.vector_text() == "\n".join([d0.category, d0.questions, d0.answer])  # §4-6


def test_split_markdown_faq_rules():
    qa, table = split_markdown(FAQ_DOC)
    assert qa.content_type == "faq" and qa.category == "猫粮"  # §4-5: faq 的 category=H2 节名
    assert qa.questions == "幼猫一天喂几次\n小猫咪一天要吃几顿"  # §4-8: 真实问法多条 \n 分隔
    assert "少食多餐" in qa.answer
    assert table.content_type == "faq"  # §4-8: FAQ 文档内表格 content_type 仍记 faq
    assert table.questions == "冻干猫粮喂食量表"  # 非问答内容 questions=所在节标题
    assert table.category == "商品 FAQ > 猫粮"
    assert table.section_path == "商品 FAQ > 猫粮 > 冻干猫粮喂食量表"
    assert table.answer.startswith("| 月龄 |")


def test_split_markdown_ships_sentence_overlap_between_chunks():
    p1 = "第一段讲包邮门槛，实付满九十九元即可包邮。"  # 21 字
    p2 = "第二段讲偏远地区，门槛另设为一百九十九元。"
    doc = f"---\ncontent_type: policy\n---\n# 政策\n\n## 运费\n{p1}\n\n{p2}\n"
    drafts = split_markdown(doc, chunk_size=40, chunk_overlap=20)
    assert len(drafts) == 2
    assert drafts[1].answer == p1 + p2  # 前块末整句(21 ≤ 30 且 ≥ 10)前置进后块


def test_front_matter_missing_defaults_policy():
    meta, body = parse_front_matter("# 无头\n正文。\n")
    assert meta == {} and body.startswith("# 无头")
    drafts = split_markdown("# 无头\n正文一段。\n")
    assert drafts and drafts[0].content_type == "policy"
