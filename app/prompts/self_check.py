"""ch04 §5.2 闸2 提示词(判定标准写死「答不中问题=false」,拿不准从严)。"""

from langchain_core.prompts import ChatPromptTemplate

SELF_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是客服回答前的证据充分性审查员。
    判断「仅凭下列知识库证据,是否足以回答用户问题的核心诉求」。

    判断标准：
    1. 如果证据与用户问题高度相关，并且能够支持主要回答，则 sufficient=true。
    2. 不要求证据覆盖所有可能的补充细节。
    3. 缺少非核心信息（例如申请步骤、补充说明、其他可选方式）时，不应判定为证据不足。
    4. 只有以下情况返回 sufficient=false：
       - 没有相关证据；
       - 证据与问题无关；
       - 证据不足以回答用户主要诉求；
       - 证据之间存在明显冲突。
    5. 不允许根据常识补充知识库之外的信息。

    输出字段：
    sufficient(布尔)、reason(一句话说明原因)。"""),
    ("human", "用户问题:{question}\n候选证据:\n{evidence}"),
])
