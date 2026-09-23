"""真云连通:配了 RERANK_API_KEY 才跑。顺序合理性(相关文档 index 排前)即 API 形状确认。"""

import pytest

from app.core.config import get_settings
from app.rag.reranker import rerank

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not get_settings().rerank_api_key, reason="未配置 RERANK_API_KEY,跳过云连通")
async def test_siliconflow_rerank_live():
    st = get_settings()
    out = await rerank(
        "猫砂盆的废砂盒多久清理一次",
        ["智能猫砂盆 Pro(MH-LP100)废砂盒建议 5 至 7 天清理一次,集尘袋容量 8L",
         "全景看护摄像头支持 360 度云台与夜视",
         "退货政策:签收后 7 天内无理由退货"],
        st,
    )
    assert out, "云 API 未返回结果(检查 key/模型名)"
    assert out[0][0] == 0, f"最相关文档应排第一,实际 {out}"
    assert all(isinstance(i, int) and isinstance(s, float) for i, s in out)
    print(f"[rerank-live] {out}")  # 形状与分值量级记入 dev-notes(闸1 阈值初值 0.3 的依据)
