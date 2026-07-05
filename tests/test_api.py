from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from conftest import FakeEmbedder, FakeExtractor, FakeFetcher
from jobradar.api import deps
from jobradar.api.app import create_app
from jobradar.api.runs import RunRegistry
from jobradar.config import Settings
from jobradar.embeddings import job_to_text
from jobradar.index import QdrantJobIndex
from jobradar.models import Job, RemotePolicy, Seniority, VisaSponsorship, content_hash


def _seeded(sample_jobs: list[Job]) -> tuple[QdrantJobIndex, FakeEmbedder]:
    emb = FakeEmbedder()
    index = QdrantJobIndex(Settings(), emb.dim)
    index.upsert(sample_jobs, emb.embed([job_to_text(j) for j in sample_jobs]))
    return index, emb


def _client(
    index: QdrantJobIndex,
    emb: FakeEmbedder,
    *,
    extractor: FakeExtractor | None = None,
    fetch: FakeFetcher | None = None,
    registry: RunRegistry | None = None,
) -> TestClient:
    app = create_app()
    shared_registry = registry or RunRegistry()
    app.dependency_overrides[deps.get_index] = lambda: index
    app.dependency_overrides[deps.get_embedder] = lambda: emb
    app.dependency_overrides[deps.get_registry] = lambda: shared_registry
    if extractor is not None:
        app.dependency_overrides[deps.get_extractor] = lambda: extractor
    if fetch is not None:
        app.dependency_overrides[deps.get_fetch_factory] = lambda: lambda _sources: fetch
    return TestClient(app)


def test_health(sample_jobs: list[Job]) -> None:
    index, emb = _seeded(sample_jobs)
    resp = _client(index, emb).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "jobs": 4}


def test_list_and_filter_jobs(sample_jobs: list[Job]) -> None:
    index, emb = _seeded(sample_jobs)
    client = _client(index, emb)

    assert len(client.get("/jobs").json()) == 4
    visa_yes = {j["company"] for j in client.get("/jobs", params={"visa": "yes"}).json()}
    assert visa_yes == {"Acme", "Beta"}
    ie = client.get("/jobs", params={"country": ["IE"]}).json()
    assert [j["company"] for j in ie] == ["Acme"]


def test_get_job_by_hash(sample_jobs: list[Job]) -> None:
    index, emb = _seeded(sample_jobs)
    client = _client(index, emb)

    resp = client.get(f"/jobs/{sample_jobs[0].content_hash}")
    assert resp.status_code == 200
    assert resp.json()["company"] == "Acme"
    assert client.get("/jobs/0000000000000000").status_code == 404


def test_rank_and_export(sample_jobs: list[Job], cv_text: str) -> None:
    index, emb = _seeded(sample_jobs)
    client = _client(index, emb)

    ranked = client.post("/rank", json={"cv_text": cv_text, "require_visa": True}).json()
    assert ranked[0]["job"]["company"] == "Acme"

    csv_resp = client.post("/export/csv", json={"cv_text": cv_text})
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert "company,title,country" in csv_resp.text
    assert "Acme" in csv_resp.text


def test_run_ingests_into_index(sample_jobs: list[Job]) -> None:
    index, emb = _seeded(sample_jobs)
    new_job = Job(
        title="Senior Data Engineer",
        company="Epsilon",
        country="DE",
        remote=RemotePolicy.remote,
        seniority=Seniority.senior,
        stack=["Python", "Kafka"],
        visa_sponsorship=VisaSponsorship.yes,
        source="fixture",
        url="https://example.com/epsilon",
        fetched_at=datetime.now(UTC),
        content_hash=content_hash("Epsilon", "Senior Data Engineer", "DE"),
    )
    url = "https://board/new"
    client = _client(
        index, emb, extractor=FakeExtractor({url: new_job}), fetch=FakeFetcher({url: "<html/>"})
    )

    created = client.post("/runs", json={"sources": [{"id": "a", "seed_urls": [url]}]})
    assert created.status_code == 200
    run_id = created.json()["run_id"]

    status = client.get(f"/runs/{run_id}").json()
    assert status["state"] == "done"
    assert status["result"]["indexed"] == 1
    assert index.count() == 5  # the new job was added to the store
