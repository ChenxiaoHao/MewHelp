"""ch10 T4:造数三钉——prompt 组装(术语表全文+多诉求 directive)、脏回复解析、缺口续算。"""

import json

from finetune.glossary import CLASSES
from finetune.synth import build_prompt, count_synth, gap_plan, parse_questions

_DIRTY = ("1. 我这个单能退吗\n- 发票咋开啊\n\n3、查一下快递到哪了呀\n```\n"
          "-换货流程是什么呀\n- 我这个单能退吗\n")


def test_build_prompt_carries_glossary_and_directive():
    p = build_prompt("价保", 12)
    assert all(c in p for c in CLASSES), "17 类词表全文必须进 prompt"
    assert "补差价" in p and "12" in p
    assert "两个及以上诉求" in p and "不要编号" in p
    assert "价保" in p.split("类目词表")[0], "目标类边界在词表前就亮明"


def test_parse_questions_strips_lists_dedups_filters():
    assert parse_questions(_DIRTY) == ["我这个单能退吗", "发票咋开啊",
                                       "查一下快递到哪了呀", "换货流程是什么呀"]


def test_gap_plan_only_deficits():
    targets = {c: 100 for c in CLASSES} | {"其他": 60}
    counts = {"物流": 40, "发票": 100, "价保": 130}
    plan = gap_plan(counts, targets)
    assert plan["物流"] == 60 and "发票" not in plan and "价保" not in plan
    assert plan["其他"] == 60 and plan["尺码"] == 100


def test_count_synth_missing_and_rows(tmp_path):
    assert count_synth(tmp_path / "nope.jsonl") == {}
    f = tmp_path / "s.jsonl"
    f.write_text('{"text": "一句够长的话喵", "main_class": "物流"}\n'
                 '{"text": "再来一句够长的话", "main_class": "物流"}\n',
                 encoding="utf-8")
    assert count_synth(f) == {"物流": 2}


def test_frozen_channel_pins_temperature_zero():
    # 终审 I-5:造数/预标必须温度 0(重跑标签稳定,与 plan 口径一致)
    from finetune.synth import get_frozen_model

    assert get_frozen_model().temperature == 0.0
