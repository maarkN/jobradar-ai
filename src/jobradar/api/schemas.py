"""Request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class SourceIn(BaseModel):
    id: str
    seed_urls: list[str] = []
    anti_bot: bool = False


class RunRequest(BaseModel):
    sources: list[SourceIn]


class RunState(BaseModel):
    run_id: str
    state: str
    result: dict[str, int] | None = None
    error: str | None = None


class RankRequest(BaseModel):
    cv_text: str
    top_k: int = 20
    countries: list[str] = ["IE", "DE", "NL", "CA"]
    require_visa: bool = True
