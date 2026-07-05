"""Runtime configuration, env-driven so the LLM provider is selectable."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM selection ---
    llm_provider: Provider = "anthropic"

    anthropic_api_key: str | None = None
    anthropic_model_fast: str = "claude-haiku-4-5"
    anthropic_model_strong: str = "claude-opus-4-8"

    openai_api_key: str | None = None
    openai_model_fast: str = "gpt-5-mini"
    openai_model_strong: str = "gpt-5"

    # --- Embeddings (hosted, OpenAI) ---
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Vector store (Qdrant) ---
    # None -> in-process local mode (":memory:"), so tests need no server.
    qdrant_url: str | None = None
    qdrant_collection: str = "jobs"

    # Max output tokens for an extraction call.
    extract_max_tokens: int = 1500
    # How many times to retry a failed extraction on the fast model before
    # escalating to the strong model (cost cascade, ADR-5).
    extract_fast_retries: int = 1

    # --- Anti-bot fetch (cloudscraper-go server) ---
    cloudscraper_server_url: str = "http://127.0.0.1:8080"

    # --- Storage ---
    cache_dir: Path = Field(default=Path(".cache/extractions"))

    def model_for(self, provider: Provider, tier: Literal["fast", "strong"]) -> str:
        if provider == "anthropic":
            return self.anthropic_model_fast if tier == "fast" else self.anthropic_model_strong
        return self.openai_model_fast if tier == "fast" else self.openai_model_strong


def load_settings() -> Settings:
    return Settings()
