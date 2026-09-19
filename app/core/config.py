from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_base_url: str
    openai_api_key: str
    model_name: str
    history_token_budget: int = 4000
    temperature: float = 0.7

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        """.env 文件优先于系统环境变量（spec §9: 地址/模型名/密钥全在 .env）。

        默认顺序是 init > 系统env > dotenv；此处调换成 init > dotenv > 系统env，
        避免机器上残留的 OPENAI_API_KEY 等环境变量悄悄覆盖项目配置。
        无 .env 时仍回落系统环境变量。
        """
        return (init_settings, dotenv_settings, env_settings, file_secret_settings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
