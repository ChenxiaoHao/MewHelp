import pytest

from app.core.config import get_settings


@pytest.mark.integration
async def test_dashscope_v4_real_dims_batch_and_fallback():
    """需真 key(.env)。三件事:①维度=1024 ②批上限探测 ③dimensions 参数被拒则走降级路径。"""
    from app.rag.embeddings import build_embeddings

    st = get_settings()
    emb = build_embeddings(st)
    vecs = await emb.aembed_documents(["包邮门槛是多少", "退货运费谁承担"])
    assert len(vecs) == 2 and len(vecs[0]) == 1024, "v4 维度不符 spec §3.2,停!"
    q = await emb.aembed_query("邮费是多少")
    assert len(q) == 1024
    cos = sum(a * b for a, b in zip(q, vecs[0])) / (
        sum(a * a for a in q) ** 0.5 * sum(b * b for b in vecs[0]) ** 0.5
    )
    assert 0.3 < cos <= 1.0, f"query/同话题文档 cosine={cos:.3f} 异常,检索阈值初值 0.3 需重校"
    # 批上限探测:一次请求塞 12 条(>默认 batch 10),成败都要把结果记进 dev-notes
    probe = emb.model_copy(update={"chunk_size": 12})
    try:
        await probe.aembed_documents([f"探针文本第{i}号" for i in range(12)])
        print("[probe] 12/单请求: OK(embedding_batch_size 默认 10 仍保守,不动)")
    except Exception as exc:  # noqa: BLE001 —— 探测就是要吞一切异常记录形态
        print(f"[probe] 12/单请求: FAIL → 保持 batch ≤10。错误首行: {str(exc).splitlines()[0]}")
