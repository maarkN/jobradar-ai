"""Hosted (OpenAI) embeddings, behind a small protocol so tests use a fake."""

from __future__ import annotations

from typing import Protocol

from jobradar.config import Settings
from jobradar.cost import CostAccumulator, Usage
from jobradar.models import Job


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings, cost: CostAccumulator | None = None) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self.dim = settings.embedding_dim
        self.cost = cost or CostAccumulator()

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        if resp.usage is not None:
            self.cost.add(Usage(self._model, resp.usage.prompt_tokens, 0))
        return [item.embedding for item in resp.data]


def job_to_text(job: Job) -> str:
    """A compact, embedding-friendly rendering of a job posting."""
    parts = [
        f"{job.title} at {job.company}",
        f"seniority: {job.seniority}",
        f"location: {job.location or 'unknown'} ({job.country or '?'})",
        f"remote: {job.remote}",
        f"stack: {', '.join(job.stack) or 'n/a'}",
        f"visa sponsorship: {job.visa_sponsorship}",
    ]
    if job.description_summary:
        parts.append(job.description_summary)
    return "\n".join(parts)
