"""ch07 T1:token 估算与上下文预算倒推(需求 4)。

估算器口径(章内 Ruling,spec 预算段同步):汉字 1 字≈1 token、ASCII 4 字≈1 token、
每条消息 +4 开销——**预算、装配 trim、ReAct 计数三处同源**(只改一处净效果是反的,
需求 4 红线)。不委托 count_tokens_approximately:实测其把 491 汉字仅计 175
(~3.45 字/token),与「中文按字数折」相悖。

compute_budgets 固定开销分项:S(人设 System 实测 543)+ 证据面(rerank_top_k ×
chunk_size)+ summary 注入预留 + 安全余量。demo env 命中验收三数 5650/3954/1695;
人设 prompt 若被改写,tests/test_budget_ch07 变红即再校准(校准门语义)。
"""

import math
from dataclasses import dataclass

from app.prompts.customer_service import SYSTEM_PROMPT

MSG_OVERHEAD = 4  # 每条消息的角色/分隔开销(估算口径,非 API 字段)


def estimate_text(text: str) -> int:
    """CJK 逐字折 1,其余 4 字折 1(向上取整)。"""
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return math.ceil(cjk + (len(text) - cjk) / 4)


def _content_text(msg) -> str:
    c = getattr(msg, "content", "") or ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def estimate_msg(msg) -> int:
    return estimate_text(_content_text(msg)) + MSG_OVERHEAD


def estimate_items(items) -> int:
    """str 与 BaseMessage 混收(trim_messages 的 token_counter 兼容形态)。"""
    return sum(estimate_msg(i) if hasattr(i, "content") else estimate_text(str(i))
               for i in items)


@dataclass(frozen=True)
class Budgets:
    sliding: int
    layer1: int
    layer2: int
    parts: dict


def _system_text() -> str:
    return SYSTEM_PROMPT


def compute_budgets(settings) -> Budgets:
    peak = settings.max_agent_steps * settings.tool_result_max_tokens
    s = estimate_text(_system_text()) + MSG_OVERHEAD          # 人设单条 System 实测 543
    evidence = settings.rerank_top_k * settings.chunk_size     # 证据条数仅预算面(P3)
    fixed = (s + evidence + settings.summary_inject_tokens
             + settings.safety_margin_tokens)
    window_side = (settings.model_context_window - settings.max_output_tokens
                   - settings.max_user_input_tokens - peak - fixed)
    sliding = max(0, min(settings.turns_to_keep * settings.steady_tokens_per_turn,
                         window_side))
    # 七三开用纯整数算术:浮点 0.7 使 5650×0.7=3954.999…,floor 少 1(浮点陷阱)
    layer1 = (sliding * 7) // 10 - 1 if sliding >= 2 else 0
    layer2 = (sliding * 3 + 9) // 10 if sliding else 0   # ceil(sliding×0.3)
    return Budgets(sliding, layer1, layer2,
                   parts={"system": s, "evidence": evidence, "peak": peak,
                          "fixed": fixed, "window_side": window_side,
                          "want": settings.turns_to_keep * settings.steady_tokens_per_turn})


def selfcheck_budget(settings) -> str | None:
    """启动自检(需求 4):连一轮稳态都装不下 → 告警文案;正常 → None。"""
    b = compute_budgets(settings)
    if b.sliding < settings.steady_tokens_per_turn:
        return (f"滑窗 {b.sliding} < 每轮稳态 {settings.steady_tokens_per_turn}"
                f"(窗口 {settings.model_context_window},固定开销 {b.parts['fixed']})")
    return None
