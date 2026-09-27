"""八意图→五出口确定性分流（ch06 spec「意图识别四件套」「七意图→出口表 v2」节）。

纯函数层：不 import LangGraph、不发网络请求——分流可测性优先。
ch06 变化：新增「其他」类与 refund 出口（退款退货/售后不再赌模型自选工具）；
兜底方向（拍板 P1）：未知/「其他」归 knowledge——带强制检索+置信度闸的防幻觉出口。
"""

import json
import re
from typing import Literal

INTENTS: tuple[str, ...] = (
    "物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他",
)

Route = Literal["knowledge", "data", "refund", "complaint", "chitchat"]

INTENT_ROUTES: dict[str, Route] = {
    "商品咨询": "knowledge",
    "其他": "knowledge",
    "物流": "data",
    "订单": "data",
    "退款退货": "refund",
    "售后": "refund",
    "投诉": "complaint",
    "闲聊": "chitchat",
}


def route_for_intent(intent: str) -> Route:
    return INTENT_ROUTES.get(intent, "knowledge")


# 快路词表:纯问候/寒暄短句精确匹配(含标点变体靠归一化吸收,不靠模糊匹配)
GREETING_PATTERNS = frozenset({
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "hey",
    "在吗", "在不在", "早上好", "下午好", "晚上好",
    "再见", "拜拜", "bye", "谢谢", "谢谢你",
})
_FAST_PATH_MAX_LEN = 12  # 护栏:超长即便命中词表也不走快路(防夹带业务诉求被吞)

_STRIP_RE = re.compile(r"[^0-9A-Za-z一-鿿]+")


def _normalize(text: str) -> str:
    return _STRIP_RE.sub("", text or "").lower()


def chitchat_fast_path(text: str) -> bool:
    """明显寒暄规则快路（拍板 D5:闲聊固定回复零额外模型调用）。"""
    n = _normalize(text)
    return bool(n) and len(n) <= _FAST_PATH_MAX_LEN and n in GREETING_PATTERNS


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


def parse_intent_json(raw: str) -> tuple[str, float] | None:
    """容错解析四件套输出:裸 JSON / 围栏 / 散文包裹皆接受;**契约收紧**——
    intent 须八类之内且 confidence 须为 [0,1] 数值(字符串数字放过),
    任一不合 = None 交调用方重试/兜底(Review Focus 3,绝不抛穿)。"""
    if not raw:
        return None
    candidates = [raw, *_JSON_OBJ_RE.findall(raw)]
    for cand in candidates:
        try:
            data = json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if not (isinstance(data, dict) and data.get("intent") in INTENTS):
            continue
        if isinstance(data.get("confidence"), bool):  # true/false 不是置信度
            continue
        try:
            conf = float(data.get("confidence"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= conf <= 1.0:
            return data["intent"], conf
    return None


# --- ch06 槽位正则(需求 6:模型不许猜单号,确定性提取) -------------------------
_SELECTION_RE = re.compile(r"^我选择订单\s*(\d{3,})$")
_ORDER_IN_TEXT_RE = re.compile(r"订单\s*[#＃:：]?\s*(\d{3,})")


def match_order_selection(text: str) -> str | None:
    """选择器协议句全匹配(前端点卡片发送的固定格式),否则 None。"""
    m = _SELECTION_RE.fullmatch((text or "").strip())
    return m.group(1) if m else None


def extract_order_id(*texts: str | None) -> str | None:
    """从任一文本内嵌提订单号(订单/订单#/订单：+ ≥3 位数字),取首个命中。"""
    for t in texts:
        m = _ORDER_IN_TEXT_RE.search(t or "")
        if m:
            return m.group(1)
    return None
