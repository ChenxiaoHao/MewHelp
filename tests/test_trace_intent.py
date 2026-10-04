"""ch09 T2 意图归因:trace 定向通道三钉 + 活栈集成(spec「观测链」节)。

SDK v4 实测:无 update_current_trace 模块面 → spec 备道转正为主道:
逐轮 trace_id 预生成经 trace_context 注入 CallbackHandler,流末由
observability.attach_trace_intent 低层更新 tags/metadata。
"""

import asyncio

import pytest
from app.core.config import Settings
from app.services import observability

_LF = dict(langfuse_host="http://127.0.0.1:3001",
           langfuse_public_key="pk-lf-REDACTED",
           langfuse_secret_key="sk-lf-REDACTED")


def _st(**kw):
    return Settings(_env_file=None, openai_base_url="x", openai_api_key="x",
                    model_name="m", **kw)


@pytest.fixture(autouse=True)
def _clean_langfuse_env(monkeypatch):
    for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(k, raising=False)


def test_turn_trace_id_none_without_env():
    assert observability.turn_trace_id(_st()) is None


def test_turn_trace_id_stable_with_env():
    tid = observability.turn_trace_id(_st(**_LF))
    assert isinstance(tid, str) and len(tid) >= 16
    # 同 seed 确定性:SDK create_trace_id(seed=) 可复现
    assert observability.turn_trace_id(_st(**_LF), seed="s1") == \
           observability.turn_trace_id(_st(**_LF), seed="s1")


def test_attach_disabled_is_silent(monkeypatch):
    called = []
    monkeypatch.setattr(observability, "_update_trace_tags",
                        lambda *a, **k: called.append(a))
    asyncio.run(observability.attach_trace_intent("tid", _st(), "闲聊", 1.0))
    assert called == [], "缺键不许触网"
    asyncio.run(observability.attach_trace_intent(None, _st(**_LF), "闲聊", 1.0))
    assert called == [], "无 trace_id 不许触网"
    asyncio.run(observability.attach_trace_intent("tid", _st(**_LF), None, 0.0))
    assert called == [], "state 无 intent(异常早退)跳过"
    asyncio.run(observability.attach_trace_intent("tid", _st(**_LF), "闲聊", 1.0))
    assert called == [("tid", "闲聊", 1.0)]


@pytest.mark.integration
def test_chitchat_turn_tags_trace_on_live_langfuse(monkeypatch):
    """活栈端到端:闲聊快路整轮(零真模型调用)→ trace 面带 intent tag。"""
    import time
    from types import SimpleNamespace

    import httpx

    from app.workflows.graph import stream_graph_turn

    tid = observability.turn_trace_id(_st(**_LF), seed="t2-live")
    monkeypatch.setattr(observability, "turn_trace_id",
                        lambda st, seed=None: tid)
    monkeypatch.setenv("LANGFUSE_HOST", _LF["langfuse_host"])
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", _LF["langfuse_public_key"])
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", _LF["langfuse_secret_key"])

    class _M:  # 快路不触 model;ctx/coref 触达时给空串走各自降级形
        async def ainvoke(self, msgs, **kw):
            return SimpleNamespace(content="")

    async def _run():
        msg = SimpleNamespace(role="user", content="你好")
        return [f async for f in stream_graph_turn(
            [msg], _st(**_LF), _M(), conversation_id=None)]

    frames = asyncio.run(_run())
    assert frames  # 跑通即有帧(兜底/答案)
    auth = (_LF["langfuse_public_key"], _LF["langfuse_secret_key"])
    tags = []
    seen = False
    for _ in range(15):  # SDK flush + ingestion 队列→worker→ClickHouse 秒级延迟
        r = httpx.get(f"{_LF['langfuse_host']}/api/public/traces/{tid}",
                      auth=auth, timeout=5)
        if r.status_code == 200:
            seen = True
            tags = r.json().get("tags") or []
            if any(t.startswith("intent:") for t in tags):
                break
        time.sleep(1.0)
    assert seen, f"trace {tid} 未落 Langfuse(检查挂接/flush)"
    assert any(t.startswith("intent:") for t in tags), \
        f"验收面:trace 须带 intent tag,实际 tags={tags}"
