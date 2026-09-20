import pytest


def test_get_engine_before_init_raises():
    from app.db import engine as eng

    # 单测环境从未 init_engine（除非其他测试先跑过，故先强制置空）
    eng._engine = None
    with pytest.raises(RuntimeError):
        eng.get_engine()


def test_init_engine_uses_settings_url(monkeypatch):
    from app.core.config import Settings
    from app.db import engine as eng

    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    s = Settings(_env_file=None)
    e = eng.init_engine(s)
    assert "aiomysql" in str(e.url) or "asyncmy" in str(e.url)
    assert e is eng.get_engine()
    assert eng.get_session_factory() is not None
    eng._engine = None  # 复位，避免污染其他测试


def test_init_engine_idempotent(monkeypatch):
    from app.core.config import Settings
    from app.db import engine as eng

    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")
    s = Settings(_env_file=None)
    e1 = eng.init_engine(s)
    e2 = eng.init_engine(s)
    assert e1 is e2
    eng._engine = None
