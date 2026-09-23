"""ch04 System Prompt 锚定词(spec §5.3 三改;ch02 锚必须原样保住=兼容红线)。"""

from app.prompts.customer_service import SYSTEM_PROMPT


def test_ch04_new_anchors():
    assert "[n]" in SYSTEM_PROMPT                      # ①引用规范
    assert "refused" in SYSTEM_PROMPT                  # ③拒答行为(对齐契约 v2)
    assert "不承诺" in SYSTEM_PROMPT                    # ②禁止承诺清单
    assert "工作日" in SYSTEM_PROMPT                    # 承诺清单点名到账时效
    assert "转人工" in SYSTEM_PROMPT                    # 拒答收敛去向

def test_ch02_anchors_survive():
    for anchor in ("喵帮", "客服", "订单号", "工具", "严禁编造", "人工工单"):
        assert anchor in SYSTEM_PROMPT
    assert "没有接入任何订单/物流查询系统" not in SYSTEM_PROMPT
