"""意图识别最简 Prompt（ch05 需求 6:单 prompt 判七类,输出 JSON;正式版留下一章）。"""

from langchain_core.prompts import ChatPromptTemplate

INTENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是电商客服意图分类器。把用户消息判为以下七类之一:
物流 | 订单 | 商品咨询 | 退款退货 | 售后 | 投诉 | 闲聊

判类口径:
- 物流:查快递/包裹位置与时效;
- 订单:查订单状态、金额、是否下单成功;
- 商品咨询:商品参数、运费、政策等知识性问题;
- 退款退货:退款/退货的申请、流程、资格;
- 售后:换货、维修、质量问题处理(未升级到投诉);
- 投诉:表达强烈不满、要求追责、点名要找人工/上级;
- 闲聊:问候、寒暄、与购物无关的闲谈。

只输出一个 JSON 对象,格式:{{"intent": "<七类之一>"}},不要任何其他字符。
(注:双花括号是 f-string 转义,渲染后为单花括号——ch04 T4 同款坑位。)"""),
    ("human", "{question}"),
])
