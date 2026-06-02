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

    # Dataset
    per_artifact_dir: str = "/data/fahmai/per_artifact"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
