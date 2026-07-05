from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from conftest import FakeProvider
from jobradar.cache import ExtractionCache
from jobradar.config import Settings
from jobradar.extract import html_to_text
from jobradar.extract.extractor import ExtractionError, Extractor
from jobradar.models import RemotePolicy, Seniority, VisaSponsorship

GOOD = json.dumps(
    {
        "title": "Senior Go Engineer",
        "company": "Acme Cloud",
        "location": "Dublin, Ireland",
        "country": "IE",
        "remote": "hybrid",
        "seniority": "senior",
        "stack": ["Go", "Kafka", "PostgreSQL", "Kubernetes", "gRPC"],
        "salary": {"min": 90000, "max": 120000, "currency": "EUR", "period": "year"},
        "visa_sponsorship": "yes",
        "visa_signals": ["Visa sponsorship and relocation package", "Critical Skills Permit"],
        "description_summary": "Senior Go engineer for an anti-abuse platform.",
    }
)
BAD = "Sorry, here is the info: title only, no JSON."


def _extractor(provider: FakeProvider, cache_dir: Path) -> Extractor:
    return Extractor(provider, ExtractionCache(cache_dir), Settings())


def test_happy_path_produces_validated_job(sample_html: str, tmp_cache_dir: Path) -> None:
    provider = FakeProvider(GOOD)
    ex = _extractor(provider, tmp_cache_dir)

    job = ex.extract(
        html_to_text(sample_html),
        source="irishjobs",
        url="https://irishjobs.ie/job/1",
        fetched_at=datetime.now(UTC),
    )

    assert job.seniority is Seniority.senior
    assert job.remote is RemotePolicy.hybrid
    assert job.visa_sponsorship is VisaSponsorship.yes
    assert "Go" in job.stack
    assert job.source == "irishjobs"  # provenance injected by the pipeline
    assert len(job.content_hash) == 16
    assert provider.calls == ["fast"]  # cheap model was enough
    assert ex.cost.summary()["calls"] == 1


def test_retries_then_succeeds(sample_html: str, tmp_cache_dir: Path) -> None:
    provider = FakeProvider([BAD, GOOD])  # first invalid, second valid
    ex = _extractor(provider, tmp_cache_dir)
    ex.extract(
        html_to_text(sample_html), source="s", url="https://x/1", fetched_at=datetime.now(UTC)
    )
    assert provider.calls == ["fast", "fast"]


def test_escalates_to_strong_model(sample_html: str, tmp_cache_dir: Path) -> None:
    provider = FakeProvider([BAD, BAD, GOOD])  # exhaust fast retries, then strong
    ex = _extractor(provider, tmp_cache_dir)
    ex.extract(
        html_to_text(sample_html), source="s", url="https://x/1", fetched_at=datetime.now(UTC)
    )
    assert provider.calls == ["fast", "fast", "strong"]


def test_raises_when_all_attempts_fail(sample_html: str, tmp_cache_dir: Path) -> None:
    provider = FakeProvider([BAD, BAD, BAD])
    ex = _extractor(provider, tmp_cache_dir)
    with pytest.raises(ExtractionError):
        ex.extract(
            html_to_text(sample_html), source="s", url="https://x/1", fetched_at=datetime.now(UTC)
        )


def test_cache_avoids_second_llm_call(sample_html: str, tmp_cache_dir: Path) -> None:
    provider = FakeProvider(GOOD)
    ex = _extractor(provider, tmp_cache_dir)
    text = html_to_text(sample_html)

    ex.extract(text, source="s", url="https://x/1", fetched_at=datetime.now(UTC))
    ex.extract(text, source="s", url="https://x/1", fetched_at=datetime.now(UTC))

    assert provider.calls == ["fast"]  # second call served from cache, LLM not hit again
