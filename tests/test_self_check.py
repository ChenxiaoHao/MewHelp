"""闸2 判定与降级:成功透出 structured 结果;一切异常 → None(放行)。"""

import pytest
from langchain_core.runnables import RunnableLambda

from app.core.config import Settings
from app.services.self_check import EvidenceCheck, evaluate_evidence


@pytest.fixture
def st():
    return Settings(_env_file=None, openai_base_url="http://f/v1", openai_api_key="f", model_name="f")


HITS = [{"n": 1, "id": 5, "question": "退货政策是什么", "answer": "签收后 7 天内无理由退货",
         "category": "退货政策", "section_path": "手册 > 退货"}]


class _M:
    """假模型:记录收到的 prompt 值。

    brief 原文用裸类 `_Chain`(非 Runnable)——`prompt | _Chain()` 在现装
    langchain_core 下 coerce_to_runnable 直接 TypeError,故按最小语义保真改为
    RunnableLambda(零新依赖)。且 `SELF_CHECK_PROMPT | structured` 管道里
    structured 一步收到的是**格式化后的 ChatPromptValue**(brief 测试原文假设
    收到 {"question","evidence"} 原始 dict,与 prompt 管道的真实数据流不符);
    断言相应改为读 human message 渲染文本——钉的是同一个语义(问题进 prompt、
    证据带编号进 prompt、answer 截断),且额外钉死了模板变量名与渲染格式,强度只增。
    """

    def __init__(self, result=None, boom=False):
        self.result, self.boom, self.seen = result, boom, None

    def with_structured_output(self, schema):
        outer = self

        async def _ainvoke(inp):
            outer.seen = inp
            if outer.boom:
                raise RuntimeError("llm down")
            return outer.result

        return RunnableLambda(_ainvoke)

    @property
    def human(self) -> str:
        return self.seen.to_messages()[-1].content


async def test_sufficient_verdict_passthrough_with_rendered_evidence(st):
    m = _M(result=EvidenceCheck(sufficient=True, reason="覆盖核心诉求"))
    out = await evaluate_evidence("退货要几天", HITS, st, model=m)
    assert out.sufficient is True
    assert "用户问题:退货要几天" in m.human  # question 进 prompt
    assert "[1] 退货政策是什么" in m.human  # 证据渲染含编号与问题


async def test_boom_degrades_to_none(st):
    assert await evaluate_evidence("q", HITS, st, model=_M(boom=True)) is None


async def test_long_answer_excerpted(st):
    hits = [{"n": 1, "id": 5, "question": "q", "answer": "长" * 500, "category": "c", "section_path": "s"}]
    m = _M(result=EvidenceCheck(sufficient=True))
    await evaluate_evidence("q", hits, st, model=m)
    evidence = m.human.split("候选证据:\n", 1)[1]
    assert len(evidence) < 500  # 证据截断(§5.2:控制自评 token)
