"""ch09 T1 观测基座:Langfuse 挂接三钉。

Review Focus 1:langfuse 未配置 → build_graph 返回形状与 ch08 终态一致
(CompiledStateGraph,无 binding),cfg 不含 callbacks,全链零漂移。
"""

import pytest
from langfuse.langchain import CallbackHandler

from app.core.config import Settings
from app.services import observability
from app.workflows.graph import build_graph


@pytest.fixture(autouse=True)
def _clean_langfuse_env(monkeypatch):
    """工厂会写 os.environ(SDK 单例需要);逐测试回收,防 Settings(_env_file=None)
    经系统 env 兜底渠道把测试键漏进别的用例。"""
    for k in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(k, raising=False)


class _AnyModel:
    """build_graph 各节点工厂只捕获 model,装配期不触达。"""


def _st(**kw):
    return Settings(_env_file=None, openai_base_url="x", openai_api_key="x",
                    model_name="m", **kw)


_LF = dict(langfuse_host="http://127.0.0.1:3001",
           langfuse_public_key="pk-lf-test",
           langfuse_secret_key="sk-lf-test")


def test_factory_returns_none_without_env():
    assert observability.enabled(_st()) is False
    assert observability.build_handler(_st()) is None


def test_factory_builds_handler_with_keys():
    st = _st(**_LF)
    assert observability.enabled(st) is True
    h = observability.build_handler(st)
    assert isinstance(h, CallbackHandler)


def test_build_graph_mounts_handler_once():
    # 实测事实:Pregel.with_config 原地把 callbacks 合进 .config(非 RunnableBinding 壳)
    g = build_graph(_st(**_LF), _AnyModel())
    cbs = (getattr(g, "config", None) or {}).get("callbacks") or []
    cbs = cbs if isinstance(cbs, list) else [cbs]
    hlist = [cb for cb in cbs if isinstance(cb, CallbackHandler)]
    assert len(hlist) == 1, "编译处必须恰挂一次 handler(多挂=双树)"


def test_build_graph_baseline_shape_without_env():
    g = build_graph(_st(), _AnyModel())
    assert not (getattr(g, "config", None) or {}).get("callbacks"), \
        "缺键=ch08 终态形状,config 不许混进 callbacks"


def test_augment_turn_cfg_only_when_enabled():
    cfg = {"configurable": {"thread_id": "conv-1"}}
    observability.augment_turn_cfg(cfg, _st())
    assert cfg == {"configurable": {"thread_id": "conv-1"}}, "缺键 cfg 不许多一个键"
    observability.augment_turn_cfg(cfg, _st(**_LF))
    assert cfg["run_name"] == "chat_turn"
    assert cfg["metadata"]["langfuse_session_id"] == "conv-1"
