import pytest
from pydantic import ValidationError


def _set_required_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_NAME", "test-model")


def test_settings_reads_env_with_defaults(monkeypatch):
    _set_required_env(monkeypatch)
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.openai_base_url == "http://localhost:11434/v1"
    assert s.model_name == "test-model"
    assert s.history_token_budget == 4000
    assert s.temperature == 0.7


def test_settings_budget_overridable(monkeypatch):
    _set_required_env(monkeypatch)
    monkeypatch.setenv("HISTORY_TOKEN_BUDGET", "1200")
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.history_token_budget == 1200


def test_dotenv_file_wins_over_os_env(monkeypatch, tmp_path):
    """.env 文件优先于系统环境变量（spec: 配置全在 .env；防环境变量污染）。"""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_BASE_URL=http://from-dotenv/v1\n"
        "OPENAI_API_KEY=dotenv-key\n"
        "MODEL_NAME=from-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_NAME", "from-os-env")
    from app.core.config import Settings

    s = Settings(_env_file=env_file)
    assert s.model_name == "from-dotenv"


def test_os_env_used_when_no_dotenv(monkeypatch):
    """没有 .env 时回落到系统环境变量（部署友好）。"""
    _set_required_env(monkeypatch)
    from app.core.config import Settings

    s = Settings(_env_file=None)
    assert s.model_name == "test-model"


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
