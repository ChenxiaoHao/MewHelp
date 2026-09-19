from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_base_url: str
    openai_api_key: str
    model_name: str
    history_token_budget: int = 4000
    temperature: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
