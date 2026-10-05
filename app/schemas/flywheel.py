"""ch09 飞轮流水线的结构化输出 schema(spec「飞轮流水线」节)。"""

from pydantic import BaseModel, Field


class NormalizedQA(BaseModel):
    normalized_question: str = Field(
        description="FAQ 式标准问句:保留商品/权益/时限等关键实体与约束,"
                    "去口语化、纠正错字、补明指代;不得编造原话没有的事实"
    )
    suggested_answer: str = Field(
        description="示例答案,仅供人工审核参考不是终稿;依据不足时写「(待人工补充)」"
    )


class DedupMatch(BaseModel):
    matched_id: int | None = Field(
        default=None,
        description="与当前问题属于同一知识缺口的候选行 id(必须从候选清单里选);"
                    "没有同缺口行或候选清单为空时返回 null"
    )
