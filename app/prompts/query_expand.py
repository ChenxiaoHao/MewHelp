"""Query 扩写 Prompt(ch06 需求 3,仅退款退货/售后子流程使用——FAQ 不扩写)。

库里知识只留一份,扩写发生在检索侧(P 拍板):同一诉求拆 ≤3 条侧重不同的
问法去撞同一份政策条款。输出契约:恰好一个 queries 数组,零解释。
"""

from langchain_core.prompts import ChatPromptTemplate

EXPAND_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商售后检索查询扩写器。用户有一个退款/退货诉求,请把它拆成
至多 3 条**侧重不同**的标准检索问法(如:资格与时限 / 流程与运费 / 凭证要求),
用于在同一份售后政策库里多路撞条款。

要求:
- 每条都是自包含的检索问句,不互相引用,不加编号;
- 保留用户问题为第 1 条,不要发明用户没问的新诉求;
- 只输出一个 JSON 对象,字段恰好一个,格式:
{{"queries": ["…", "…"]}}
不要任何其他字符。"""),
    ("human", "{question}"),
])
