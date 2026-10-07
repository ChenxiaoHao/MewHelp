import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("MODEL_NAME", "m")


def test_database_url_assembly(env, monkeypatch):
    monkeypatch.setenv("MYSQL_PASSWORD", "pw123")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.database_url == (
        "mysql+aiomysql://root:pw123@127.0.0.1:3306/mewhelp?charset=utf8mb4"
    )


def test_ch02_defaults(env):
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.mysql_host == "127.0.0.1"
    assert s.mysql_port == 3306
    assert s.mysql_user == "root"
    assert s.mysql_password == ""  # 公开化整改:口令不再入代码默认值(真值只在 .env)
    assert s.mysql_db == "mewhelp"
    assert s.demo_user_id == "demo_user"
    assert s.tool_timeout_seconds == 10.0  # ch08 T5 基准 5→10(MCP 往返留余量)
    assert s.tool_max_retries == 1
