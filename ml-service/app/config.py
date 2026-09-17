from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql+psycopg://veritas:veritas@localhost:5432/veritas"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
