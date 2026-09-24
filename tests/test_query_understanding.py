"""parse_rewrite 纯函数 TDD + understand_query 三分支(禁用/成功/降级)。
prompt 输出效果不在单测断言(那是标注样例冒烟的事,§9)。"""

import pytest

from app.core.config import Settings
from app.rag.query_understanding import UnderstandResult, parse_rewrite, understand_query


@pytest.fixture
def st():
    return Settings(_env_file=None, openai_base_url="http://fake/v1",
                    openai_api_key="fake", model_name="fake")


class _Reply:
    def __init__(self, content):
        self.content = content


class OkModel:
    """(PROMPT | model) 链里 model 的最小替身:收到渲染后的消息、回固定文本。"""

    def __init__(self, content):
        self.content = content

    async def ainvoke(self, _inp):
        return _Reply(self.content)


class BoomModel:
    async def ainvoke(self, _inp):
        raise RuntimeError("llm down")


# ---- parse_rewrite 纯函数 ----

def test_parse_rewrite_accepts_dict_and_json_str():
    assert parse_rewrite({"standard_query": "退货运费谁承担", "synonyms": ["运费", "退货"]}) == (
        "退货运费谁承担", ["运费", "退货"])
    assert parse_rewrite('{"standard_query": "a", "synonyms": []}') == ("a", [])


def test_parse_rewrite_strips_code_fence():
    assert parse_rewrite('```json\n{"standard_query": "a", "synonyms": ["x"]}\n```') == ("a", ["x"])


def test_parse_rewrite_cleans_synonyms():
    raw = {"standard_query": "猫粮", "synonyms": ["猫粮", " 主粮 ", "", "主粮", "冻干", "湿粮", "零食", "处方粮"]}
    std, syn = parse_rewrite(raw)
    # brief 原断言以 处方粮 收尾与其自身实现/注释矛盾(≤4 截断砍的是尾部第 5 项,
    # 非第 4 项 零食;实测 parse_rewrite 输出为 主粮/冻干/湿粮/零食)。断言改正,强度不变:
    # 5 个合法同义词进、4 个出,去重/去空/去等于标准问法/截断规则全部钉死。
    assert syn == ["主粮", "冻干", "湿粮", "零食"]  # 去重/去空/去等于标准问法/≤4 截断


def test_parse_rewrite_rejects_garbage():
    assert parse_rewrite("not json at all") is None
    assert parse_rewrite({"standard_query": "  ", "synonyms": []}) is None
    assert parse_rewrite({"standard_query": "a", "synonyms": "notalist"}) is None
    assert parse_rewrite([1, 2]) is None
    assert parse_rewrite(None) is None


# ---- understand_query 三分支 ----

async def test_disabled_returns_verbatim_without_llm_call(st):
    st = st.model_copy(update={"query_rewrite_enabled": False})
    r = await understand_query("包邮吗", st, model=BoomModel())  # 禁用时模型炸与否都到不了它
    assert r == UnderstandResult(standard_query="包邮吗", synonyms=[], degraded=False)


async def test_success_path(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("东西不想要了还能退不", st,
                               model=OkModel('{"standard_query": "退货政策是什么", "synonyms": ["退款", "退换"]}'))
    assert r.standard_query == "退货政策是什么" and r.synonyms == ["退款", "退换"] and not r.degraded
    assert r.bm25_text == "退货政策是什么 退款 退换"


async def test_model_boom_degrades_to_verbatim(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("包邮吗", st, model=BoomModel())
    assert r.standard_query == "包邮吗" and r.synonyms == [] and r.degraded is True


async def test_bad_output_degrades(st):
    st = st.model_copy(update={"query_rewrite_enabled": True})
    r = await understand_query("包邮吗", st, model=OkModel("抱歉我改写不了"))
    assert r.degraded is True and r.standard_query == "包邮吗"
