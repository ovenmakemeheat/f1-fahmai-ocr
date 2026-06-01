# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # vLLM
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model_name: str = "scb10x/typhoon-ocr-7b"
    vllm_timeout: int = 120          # seconds per request
    vllm_max_tokens: int = 4096
    vllm_max_tokens_tx: int = 8192   # bank statement transactions

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "ocr_db"
    postgres_user: str = "ocr_user"
    postgres_password: str = "change_me"

    # Dataset
    per_artifact_dir: str = "/data/fahmai/per_artifact"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def asyncpg_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
