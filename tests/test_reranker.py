"""rerank 门面:MockTransport 断言请求形状(documents/top_n/Bearer)+ 五种降级路径 → None。"""

import json

import httpx
import pytest

from app.core.config import Settings
from app.rag import reranker as rr


@pytest.fixture(autouse=True)
def _reset_warn():
    rr._warned = False


def _st(**kw):
    base = dict(openai_base_url="http://fake/v1", openai_api_key="fake", model_name="fake",
                rerank_api_key="sk-test", rerank_api_base="https://api.example.cn/v1")
    base.update(kw)
    return Settings(_env_file=None, **base)


def _transport(handler):
    return httpx.MockTransport(handler)


async def test_success_returns_index_scores_sorted_passthrough():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.42}]})

    out = await rr.rerank("猫砂盆清理", ["a", "b", "c"], _st(), transport=_transport(handler))
    assert out == [(2, 0.91), (0, 0.42)]  # 服务端已降序,门面原样透出
    assert seen["url"] == "https://api.example.cn/v1/rerank"
    assert seen["auth"] == "Bearer sk-test"
    body = seen["body"]
    assert body["model"] == "BAAI/bge-reranker-v2-m3" and body["query"] == "猫砂盆清理"
    assert body["documents"] == ["a", "b", "c"]  # 核对点③:字段名 documents,不是 texts
    # brief 原文此处断言 10,与其自身实现的 min(rerank_top_n=10, 3 候选)=3 及
    # test_top_n_capped_to_candidate_count 互斥(dev-notes ④ 记录);cap 语义照钉死
    assert body["top_n"] == 3 and body["return_documents"] is False


async def test_top_n_capped_to_candidate_count():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["top_n"] == 2
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.5},
                                                     {"index": 0, "relevance_score": 0.1}]})

    out = await rr.rerank("q", ["a", "b"], _st(rerank_top_n=10), transport=_transport(handler))
    assert out and out[0][0] == 1


async def test_no_key_returns_none_without_http():
    def handler(request):  # 被调即失败
        raise AssertionError("不该发请求")

    assert await rr.rerank("q", ["a"], _st(rerank_api_key=""), transport=_transport(handler)) is None


async def test_empty_texts_returns_none():
    assert await rr.rerank("q", [], _st(), transport=_transport(lambda r: pytest.fail("不该发"))) is None


@pytest.mark.parametrize("make_resp", [
    lambda: httpx.Response(500, json={"error": "boom"}),
    lambda: httpx.Response(200, json={"no_results": []}),
    lambda: httpx.Response(200, text="not-json"),
])
async def test_bad_responses_degrade_to_none(make_resp):
    out = await rr.rerank("q", ["a"], _st(), transport=_transport(lambda r: make_resp()))
    assert out is None


async def test_timeout_degrades_to_none():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    assert await rr.rerank("q", ["a"], _st(), transport=_transport(handler)) is None
