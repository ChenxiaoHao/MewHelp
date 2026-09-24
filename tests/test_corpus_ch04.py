"""老师材料替换后的语料级断言(spec §0-5/附录 B):6 份正式文档过 chunker 的形态必须
支撑四策略评估与验收地基;老师材料自身出入(运费跨文档矛盾、银卡折扣)如实入锚——
数据只读,测试描述现状,不为指标改数据。"""

from pathlib import Path

from app.rag.chunking import split_markdown

DOCS = Path("knowledge")
FILES = ["after-sales-manual.md", "billing-shipping.md", "member-benefits.md",
         "product-faq.md", "product-specs.md", "returns-policy.md"]


def _drafts(name: str):
    return split_markdown((DOCS / name).read_text(encoding="utf-8"))


def _all_drafts():
    out = []
    for f in sorted(DOCS.glob("*.md")):
        out.extend(split_markdown(f.read_text(encoding="utf-8")))
    return out


def test_six_docs_all_chunkable_and_vector_text_rule():
    for name in FILES:
        assert _drafts(name), f"{name} 切不出块"
    for d in _all_drafts():
        assert d.vector_text() == "\n".join([d.category, d.questions, d.answer])


def test_product_specs_has_13_model_sections_with_lp100_anchor():
    drafts = _drafts("product-specs.md")
    models = [d for d in drafts if "型号 MH-" in d.questions]
    assert len(models) == 13  # LP100/LP50/LP200/W20/W40/W60/CT30/HP12/HP20/FD10/FD30/CAM1/NEST20
    lp100 = [d for d in drafts if "MH-LP100" in d.section_path]
    assert lp100 and "废砂盒" in lp100[0].answer  # 验收2 硬锚(BM25 型号命中靶块)


def test_faq_10yuan_vs_billing_6yuan_conflict_recorded():
    """附录 B-3:老师语料自身跨文档矛盾(faq 收 10 元 vs billing 收 6 元)。
    断言「两者并存在库」而非择一为真;报告评估时如实说明。"""
    faq = [d for d in _drafts("product-faq.md") if d.questions == "运费怎么算"]
    billing = [d for d in _drafts("billing-shipping.md") if "运费与包邮" in d.section_path]
    assert faq and "10 元" in faq[0].answer
    assert billing and "6 元" in billing[0].answer


def test_member_silver_9off_actual_wording_anchored():
    """附录 B-2:老师标注 A22/E9 写银卡 95 折,语料实为银卡 9 折(金卡才 95 折)。
    裁判以语料为准,此锚锁死措辞现状。"""
    silver = [d for d in _drafts("member-benefits.md") if "银卡" in d.answer and "9 折" in d.answer]
    assert silver and "95 折" in silver[0].answer  # 同句「银卡享商品 9 折,金卡享商品 95 折」


def test_aftersale_time_limit_table_header_copied_per_block():
    drafts = [d for d in _drafts("after-sales-manual.md") if "常见问题处理时限" in d.section_path]
    assert drafts
    assert {d.answer.splitlines()[0] for d in drafts} == {"| 问题类型 | 首次响应 | 处理时限 |"}
    assert all(d.answer.splitlines()[1].startswith("| ---") for d in drafts)
