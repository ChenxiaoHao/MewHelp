import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    # 屏蔽 .env 与系统环境变量:Settings(_env_file=None) 只用显式 init/monkeypatch 值


def test_ch03_defaults(env):
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.milvus_uri == "http://127.0.0.1:19530"
    assert s.milvus_collection == "knowledge"
    assert s.embedding_model == "text-embedding-v4"
    assert s.embedding_dimensions == 1024
    assert s.embedding_batch_size == 10
    assert s.chunk_size == 500
    assert s.chunk_overlap == 80
    assert s.rag_top_k == 5
    assert s.rag_score_threshold == 0.3
    assert s.qa_dedup_threshold == 0.92
    assert s.qa_mine_batch_conversations == 5


def test_ch03_env_override(env, monkeypatch):
    monkeypatch.setenv("MILVUS_URI", "http://milvus:19530")
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.55")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.milvus_uri == "http://milvus:19530"
    assert s.rag_score_threshold == 0.55


def test_ch01_ch02_still_untouched(env):
    """构造不破坏红线:ch01/ch02 字段与 fake_settings 用法原样。"""
    from app.core.config import Settings

    s = Settings(
        _env_file=None,
        openai_base_url="http://fake/v1",
        openai_api_key="fake-key",
        model_name="fake-model",
    )
    assert s.history_token_budget == 4000
    assert s.tool_timeout_seconds == 5.0
