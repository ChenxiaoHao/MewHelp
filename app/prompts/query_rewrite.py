"""ch04 §4.1 查询理解 prompt(教学演示版:约束写全、少示例)。"""

from langchain_core.prompts import ChatPromptTemplate

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商客服知识库的检索查询理解器。把用户的口语问题改写为一句标准问法,并给出至多 4 个检索同义词。
规则:
- standard_query:保留全部关键信息(型号数字/金额/时限一个字不能丢),去掉语气词与口水话,输出单句;
- synonyms:只给实词(名词/型号/术语),不给整句、不给虚词;想不出就输出空数组;
- 同义词只服务关键词召回,不得引入问题里没有的实体或立场;
- 只输出 JSON,不加解释:{{"standard_query": "...", "synonyms": ["..."]}}"""),  # {{}} 转义:f-string 模板里 JSON 示例须双花括号,渲染后回单花括号
    ("human", "{question}"),
])
