"""ch07 T1:预算倒推与 CJK 感知估算器(需求 4 + 验收 2 三数锚点)。

三数 (5650, 3954, 1695) = demo env 下的 spec「预算模型」验收锚点,固定开销
分项(S=人设实测543 / 证据=rerank_top_k×chunk_size / 707+1000 校准常数)以命中
此三数为校准目标——人设 prompt 若被改写,本测试变红即再校准(校准门语义)。
估算器口径(章内 Ruling):汉字 1 字≈1 token,ASCII 4 字≈1 token,每条消息 +4。
"""

from types import SimpleNamespace

from app.context.budget import compute_budgets, estimate_text, selfcheck_budget


def _demo_settings(**over):
    base = dict(model_context_window=18000, max_output_tokens=2000,
                max_user_input_tokens=2000, max_agent_steps=3,
                tool_result_max_tokens=1200, rerank_top_k=5,
                turns_to_keep=20, steady_tokens_per_turn=500,
                summary_inject_tokens=707, safety_margin_tokens=1000,
                chunk_size=500)
    base.update(over)
    return SimpleNamespace(**base)


def test_demo_env_hits_triple():
    b = compute_budgets(_demo_settings())
    assert (b.sliding, b.layer1, b.layer2) == (5650, 3954, 1695)


def test_default_window_budget():
    b = compute_budgets(_demo_settings(model_context_window=32000, max_agent_steps=6))
    assert (b.sliding, b.layer1, b.layer2) == (10000, 6999, 3000)  # want 面成小值


def test_cjk_one_char_one_token():
    assert estimate_text("汉" * 500) == 500          # 需求4:中文按字数折
    assert estimate_text("a" * 100) == 25


def test_selfcheck_warns_when_tiny():
    assert selfcheck_budget(_demo_settings(model_context_window=6000)) is not None
    assert selfcheck_budget(_demo_settings()) is None


def test_new_settings_defaults_present():
    from app.core.config import Settings

    s = Settings(_env_file=None, openai_base_url="x", openai_api_key="k", model_name="m")
    assert (s.model_context_window, s.max_output_tokens, s.max_user_input_tokens,
            s.max_agent_steps, s.tool_result_max_tokens, s.rerank_top_k) == \
        (32000, 2000, 2000, 6, 1200, 5)
    assert (s.assistant_head_chars, s.history_view_messages) == (60, 6)
