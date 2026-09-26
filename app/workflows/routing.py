"""七意图→四出口确定性分流（ch05 spec「七意图四出口映射」节，规则写死在代码）。

纯函数层：不 import LangGraph、不发网络请求——分流可测性优先。
兜底方向（拍板 P2）：未知意图归 knowledge——带强制检索+置信度闸的防幻觉出口。
"""

import json
import re
from typing import Literal

INTENTS: tuple[str, ...] = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")

Route = Literal["knowledge", "data", "complaint", "chitchat"]

INTENT_ROUTES: dict[str, Route] = {
    "商品咨询": "knowledge",
    "退款退货": "knowledge",
    "物流": "data",
    "订单": "data",
    "售后": "data",
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


def parse_intent_json(raw: str) -> str | None:
    """容错解析意图模型输出:裸 JSON / markdown 围栏 / 前后夹散文 都接受;
    缺键、非法类别、非 JSON 一律 None(调用方走兜底,绝不抛穿——Review Focus 1)。"""
    if not raw:
        return None
    candidates = [raw, *_JSON_OBJ_RE.findall(raw)]
    for cand in candidates:
        try:
            data = json.loads(cand.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("intent") in INTENTS:
            return data["intent"]
    return None
