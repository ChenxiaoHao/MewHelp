"""ch07 T5:SUMMARIZE_PROMPT 形状契约(四铁律关键词在场+两占位符渲染)。

质量断言不在此——按 P7 豁免口径归 evals/smoke_ch07.py 标注样例面;本测只钉
「提示词结构没丢件」:少了哪条铁律/哪个占位符,这里先 RED。
"""

from app.prompts.summary import SUMMARIZE_PROMPT


def test_summarize_prompt_has_four_rules_and_background_rule():
    msgs = SUMMARIZE_PROMPT.format_messages(
        background="(无)", batch="用户:订单1001要退货")
    system = msgs[0].content
    # 四铁律(spec「摘要提示词四铁律」):只提炼事实诉求 / 不编造 / 不留寒暄 / 字数带
    assert "事实" in system and "诉求" in system
    assert "编造" in system
    assert "寒暄" in system
    assert "字" in system and "200" in system
    # 旧梗概仅作背景不重写、本批独立成段
    assert "背景" in system and "重写" in system


def test_summarize_prompt_holds_both_placeholders():
    msgs = SUMMARIZE_PROMPT.format_messages(
        background="早前用户咨询过订单1001", batch="用户:那后来转人工了吗")
    assert len(msgs) == 2                      # system + human 两段
    human = msgs[1].content
    assert "早前用户咨询过订单1001" in human    # {background} 渲染位
    assert "那后来转人工了吗" in human          # {batch} 渲染位
