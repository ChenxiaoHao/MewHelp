from langchain_core.prompts import ChatPromptTemplate

EXTRACTION_SYSTEM = """你是电商售后工单信息提取器。从用户描述中提取以下字段：
- order_id: 订单号（形如 DD 开头加数字的编号）；用户未提供时输出 null，不要编造。
- issue_type: 诉求类型，只能从 [退款, 退货, 换货, 物流, 质量问题, 其他] 中选一个。
- expected_solution: 用户期望的处理方案，用一句简短中文概括。

样例1：
输入：订单 DD20260901001 的耳机右声道没声音，才买一周，我要退货退款。
输出：order_id=DD20260901001, issue_type=退货, expected_solution=退货并退款

样例2：
输入：买的杯子收到就是碎的，订单号找不到了，希望补发一个。
输出：order_id=null, issue_type=质量问题, expected_solution=补发商品"""

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACTION_SYSTEM),
        ("human", "{description}"),
    ]
)
