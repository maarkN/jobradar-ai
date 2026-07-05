from __future__ import annotations

from fastapi.testclient import TestClient

from conftest import FakeEmbedder
from jobradar.api import deps
from jobradar.api.app import create_app
from jobradar.config import Settings
from jobradar.embeddings import job_to_text
from jobradar.index import QdrantJobIndex
from jobradar.models import Job


def _client(sample_jobs: list[Job]) -> TestClient:
    emb = FakeEmbedder()
    index = QdrantJobIndex(Settings(), emb.dim)
    index.upsert(sample_jobs, emb.embed([job_to_text(j) for j in sample_jobs]))
    app = create_app()
    app.dependency_overrides[deps.get_index] = lambda: index
    app.dependency_overrides[deps.get_embedder] = lambda: emb
    return TestClient(app)


def test_dashboard_page_renders(sample_jobs: list[Job]) -> None:
    resp = _client(sample_jobs).get("/")
    assert resp.status_code == 200
    assert "JobRadar AI" in resp.text
    assert "hx-post" in resp.text  # the HTMX form is present


def test_dashboard_rank_returns_html_fragment(sample_jobs: list[Job], cv_text: str) -> None:
    resp = _client(sample_jobs).post("/dashboard/rank", data={"cv_text": cv_text})
    assert resp.status_code == 200
    assert "<table" in resp.text
    assert "Acme" in resp.text  # top match rendered in the fragment
