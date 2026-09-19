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


def test_settings_missing_required_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
