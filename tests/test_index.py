from __future__ import annotations

from conftest import FakeEmbedder
from jobradar.config import Settings
from jobradar.embeddings import job_to_text
from jobradar.index import QdrantJobIndex
from jobradar.models import Job, VisaSponsorship


def _build(jobs: list[Job]) -> tuple[FakeEmbedder, QdrantJobIndex]:
    emb = FakeEmbedder()
    idx = QdrantJobIndex(Settings(), emb.dim)
    idx.upsert(jobs, emb.embed([job_to_text(j) for j in jobs]))
    return emb, idx


def test_upsert_and_count(sample_jobs: list[Job]) -> None:
    _, idx = _build(sample_jobs)
    assert idx.count() == len(sample_jobs)


def test_search_returns_most_similar_first(sample_jobs: list[Job]) -> None:
    emb, idx = _build(sample_jobs)
    query = emb.embed(["Go Kafka Kubernetes backend"])[0]
    hits = idx.search(query, top_k=4)
    assert "Go" in hits[0][0].stack  # a Go role tops a Go-heavy query


def test_filter_by_visa(sample_jobs: list[Job]) -> None:
    emb, idx = _build(sample_jobs)
    query = emb.embed(["senior go"])[0]
    hits = idx.search(query, top_k=10, require_visa=True)
    assert hits
    assert all(job.visa_sponsorship is VisaSponsorship.yes for job, _ in hits)


def test_filter_by_country(sample_jobs: list[Job]) -> None:
    emb, idx = _build(sample_jobs)
    query = emb.embed(["senior go"])[0]
    hits = idx.search(query, top_k=10, countries=["IE"])
    assert hits
    assert all(job.country == "IE" for job, _ in hits)
