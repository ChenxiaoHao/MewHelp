"""ch04 Settings 默认值/env 覆盖/构造不破坏红线(守卫测试,同 test_config_ch03 惯例)。"""

import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")


def test_ch04_defaults(env):
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.rerank_api_base == "https://api.siliconflow.cn/v1"
    assert s.rerank_api_key == ""
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.rerank_timeout_seconds == 5.0
    assert s.hybrid_recall_k == 50
    assert s.rrf_k == 60
    assert s.rerank_top_n == 10
    assert s.retrieval_low_conf_threshold == 0.161  # D 桶校准终值(T12 回写:误拒 4.2%/自信 3.3%,策略报告校准表)
    assert s.self_check_enabled is True
    assert s.query_rewrite_enabled is True
    assert s.faith_judge_model == ""


def test_ch04_env_override(env, monkeypatch):
    monkeypatch.setenv("RERANK_API_KEY", "sk-test")
    monkeypatch.setenv("HYBRID_RECALL_K", "30")
    monkeypatch.setenv("SELF_CHECK_ENABLED", "false")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.rerank_api_key == "sk-test"
    assert s.hybrid_recall_k == 30
    assert s.self_check_enabled is False


def test_ch01_to_ch03_still_untouched(env):
    from app.core.config import Settings

    s = Settings(_env_file=None, openai_base_url="http://fake/v1",
                 openai_api_key="fake-key", model_name="fake-model")
    assert s.rag_score_threshold == 0.3 and s.milvus_collection == "knowledge"
    assert s.history_token_budget == 4000 and s.tool_timeout_seconds == 10.0  # ch08 T5 改基准
