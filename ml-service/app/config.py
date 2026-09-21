from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str = "postgresql+psycopg://veritas:veritas@localhost:5432/veritas"
    environment: str = "development"

    # Optional LLM provider config (see app/services/claim_generation.py).
    # Declared here so values in .env are accepted; real env vars still win.
    llm_provider: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
