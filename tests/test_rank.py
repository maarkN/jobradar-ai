from __future__ import annotations

from conftest import FakeEmbedder
from jobradar.config import Settings
from jobradar.index import QdrantJobIndex
from jobradar.models import Job
from jobradar.rank import Ranker


def _ranker() -> Ranker:
    emb = FakeEmbedder()
    return Ranker(emb, QdrantJobIndex(Settings(), emb.dim))


def test_ranks_best_cv_match_first(sample_jobs: list[Job], cv_text: str) -> None:
    ranker = _ranker()
    ranker.index_jobs(sample_jobs)

    results = ranker.rank(cv_text, top_k=10, countries=["IE", "DE", "NL", "CA"], require_visa=True)

    assert results
    assert results[0].job.company == "Acme"  # Go role, visa yes, in Ireland
    assert all(r.visa_eligible for r in results)  # non-sponsoring roles filtered out
    companies = {r.job.company for r in results}
    assert "Gamma" not in companies  # visa: no
    assert "Delta" not in companies  # visa: unknown


def test_without_visa_filter_includes_all(sample_jobs: list[Job], cv_text: str) -> None:
    ranker = _ranker()
    ranker.index_jobs(sample_jobs)

    results = ranker.rank(cv_text, top_k=10, require_visa=False)

    companies = {r.job.company for r in results}
    assert {"Acme", "Beta", "Gamma", "Delta"} <= companies


def test_scores_are_bounded_and_explained(sample_jobs: list[Job], cv_text: str) -> None:
    ranker = _ranker()
    ranker.index_jobs(sample_jobs)

    results = ranker.rank(cv_text, top_k=10, countries=["IE", "DE"], require_visa=True)
    acme = next(r for r in results if r.job.company == "Acme")

    assert 0.0 <= acme.match_score <= 1.0
    assert any("visa" in reason for reason in acme.match_reasons)
    assert any("semantic" in reason for reason in acme.match_reasons)
    others = [r.match_score for r in results if r.job.company != "Acme"]
    assert all(acme.match_score >= o for o in others)
