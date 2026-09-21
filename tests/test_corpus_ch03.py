"""语料级断言:三份文档过 chunker 后的形态必须支撑附录 C/D 的验收地基(spec §4/§9)。"""

from pathlib import Path

from app.rag.chunking import split_markdown

DOCS = Path("knowledge")


def _drafts(name: str):
    return split_markdown((DOCS / name).read_text(encoding="utf-8"))


def _all_drafts():
    out = []
    for f in sorted(DOCS.glob("*.md")):
        out.extend(split_markdown(f.read_text(encoding="utf-8")))
    return out


def test_return_policy_ship_and_keys():
    drafts = _drafts("return-policy.md")
    ship = [d for d in drafts if d.questions == "运费说明"]
    assert len(ship) == 1
    assert "99" in ship[0].answer and "邮寄费" in ship[0].answer  # 词面覆盖行在块内(验收1 数据源)
    assert ship[0].section_path == "退货政策 > 运费说明"
    assert {d.questions for d in drafts if d.is_key_clause} == {"退换货条件", "不支持退换的情形"}


def test_product_faq_groups_and_anchor_table():
    drafts = _drafts("product-faq.md")
    qa = [d for d in drafts if "|" not in d.answer and d.content_type == "faq"]
    assert len(qa) >= 8  # B.2 ≥8 组
    table = [d for d in drafts if d.questions == "冻干猫粮喂食量表"]
    assert len(table) == 1 and table[0].section_path == "商品 FAQ > 猫粮 > 冻干猫粮喂食量表"
    assert table[0].content_type == "faq"  # §4-8:FAQ 文档表格仍记 faq
    wash = [d for d in drafts if d.section_path.startswith("商品 FAQ > 用品")]
    assert any("水垢" in d.answer for d in wash)  # C#10 的话题近邻锚


def test_aftersale_big_table_row_split_copies_header():
    drafts = _drafts("aftersale-manual.md")
    tbl = [d for d in drafts if d.section_path == "售后服务手册 > 物流异常赔付标准"]
    assert len(tbl) >= 2  # 超 500 字必分块
    heads = {d.answer.splitlines()[0] for d in tbl}
    assert heads == {"| 异常场景 | 判定凭证 | 赔付标准 | 处理时效 |"}  # 表头逐块复制
    assert all(d.answer.splitlines()[1].startswith("| ---") for d in tbl)
    assert {d.questions for d in drafts if d.is_key_clause} >= {"转人工流程", "破损件处理"}


def test_docs_do_not_preempt_mined_topics():
    """附录 D 的 kept 靶子(C4/C5/C6/C7/C8)不得被文档抢走答案——文档定稿的互锁红线。"""
    joined = "\n".join(d.answer + d.questions + d.category for d in _all_drafts())
    for forbidden in ("运费险", "积分", "混喂", "拌在", "饮水机", "滤芯"):
        assert forbidden not in joined, f"文档抢了增量知识靶子: {forbidden}"
    assert "换货申请流程" in joined  # C2 闸3 靶子的文档侧锚(措辞与对话不同即可,不校验字面)


def test_all_vector_text_rule():
    for d in _all_drafts():
        assert d.vector_text() == "\n".join([d.category, d.questions, d.answer])
