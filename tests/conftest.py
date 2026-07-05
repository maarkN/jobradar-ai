"""Shared test fixtures, including a fake LLM provider so tests never hit the network."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jobradar.cost import Usage
from jobradar.llm.base import LLMResponse, Tier
from jobradar.models import (
    Job,
    RemotePolicy,
    Seniority,
    VisaSponsorship,
    content_hash,
)

FIXTURES = Path(__file__).parent / "fixtures"

# A tiny deterministic vocabulary so the fake embedder's cosine similarity
# reflects real keyword overlap (a Go CV should out-rank a frontend job).
_VOCAB = [
    "go",
    "python",
    "kafka",
    "kubernetes",
    "grpc",
    "react",
    "frontend",
    "backend",
    "senior",
    "aws",
]


class FakeProvider:
    """Returns queued responses in order, recording the tiers it was called with.

    Pass one string to always return it, or a list to script a retry/escalation
    sequence (e.g. a bad payload followed by a good one).
    """

    def __init__(self, responses: str | Sequence[str]) -> None:
        self._responses: list[str] = [responses] if isinstance(responses, str) else list(responses)
        self._i = 0
        self.calls: list[Tier] = []

    def complete(
        self, system: str, user: str, tier: Tier, *, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append(tier)
        text = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return LLMResponse(
            text=text, usage=Usage(model=f"fake-{tier}", input_tokens=1000, output_tokens=200)
        )


class FakeEmbedder:
    """Deterministic bag-of-words unit vectors over a fixed vocabulary (no network)."""

    dim = len(_VOCAB)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        low = text.lower()
        raw = [1.0 if word in low else 0.0 for word in _VOCAB]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]


def _job(
    title: str, company: str, country: str, stack: list[str], visa: VisaSponsorship, sen: Seniority
) -> Job:
    return Job(
        title=title,
        company=company,
        location=f"{company} City",
        country=country,
        remote=RemotePolicy.hybrid,
        seniority=sen,
        stack=stack,
        visa_sponsorship=visa,
        description_summary=f"{title} working with {', '.join(stack)}.",
        source="fixture",
        url=f"https://example.com/{company.lower()}",
        fetched_at=datetime.now(UTC),
        content_hash=content_hash(company, title, country),
    )


@pytest.fixture
def sample_jobs() -> list[Job]:
    return [
        _job(
            "Senior Go Engineer",
            "Acme",
            "IE",
            ["Go", "Kafka", "Kubernetes", "gRPC"],
            VisaSponsorship.yes,
            Seniority.senior,
        ),
        _job(
            "Senior Python Data Engineer",
            "Beta",
            "DE",
            ["Python", "AWS"],
            VisaSponsorship.yes,
            Seniority.senior,
        ),
        _job(
            "Frontend React Developer",
            "Gamma",
            "NL",
            ["React", "Frontend"],
            VisaSponsorship.no,
            Seniority.mid,
        ),
        _job(
            "Senior Go Engineer",
            "Delta",
            "CA",
            ["Go", "AWS"],
            VisaSponsorship.unknown,
            Seniority.senior,
        ),
    ]


@pytest.fixture
def cv_text() -> str:
    return (
        "Senior Backend Engineer with 6 years of experience. Strong in Go, "
        "distributed systems, Kafka, Kubernetes and gRPC. Some Python and AWS."
    )


@pytest.fixture
def sample_html() -> str:
    return (FIXTURES / "sample_job.html").read_text(encoding="utf-8")


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "cache"
