"""Rank jobs against a CV: semantic similarity plus explainable rule boosts."""

from __future__ import annotations

from collections.abc import Sequence

from jobradar.embeddings import Embedder, job_to_text
from jobradar.index import QdrantJobIndex
from jobradar.models import Job, RankedJob, Seniority, VisaSponsorship

_SIM_WEIGHT = 0.7
_VISA_BOOST = 0.15
_COUNTRY_BOOST = 0.10
_SENIOR_BOOST = 0.05
_SENIOR_LEVELS = {Seniority.senior, Seniority.staff, Seniority.lead}


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class Ranker:
    def __init__(self, embedder: Embedder, index: QdrantJobIndex) -> None:
        self._embedder = embedder
        self._index = index

    def index_jobs(self, jobs: Sequence[Job]) -> None:
        if not jobs:
            return
        vectors = self._embedder.embed([job_to_text(j) for j in jobs])
        self._index.upsert(jobs, vectors)

    def rank(
        self,
        cv_text: str,
        *,
        top_k: int = 20,
        countries: Sequence[str] | None = None,
        require_visa: bool = True,
    ) -> list[RankedJob]:
        cv_vector = self._embedder.embed([cv_text])[0]
        hits = self._index.search(
            cv_vector, top_k=top_k, countries=countries, require_visa=require_visa
        )
        ranked = [self._to_ranked(job, sim, countries) for job, sim in hits]
        ranked.sort(key=lambda r: r.match_score, reverse=True)
        return ranked

    def _to_ranked(self, job: Job, similarity: float, countries: Sequence[str] | None) -> RankedJob:
        sim = _clamp(similarity)
        score = _SIM_WEIGHT * sim
        reasons = [f"semantic match {sim:.2f}"]

        visa_eligible = job.visa_sponsorship is VisaSponsorship.yes
        if visa_eligible:
            score += _VISA_BOOST
            reasons.append("offers visa sponsorship")
        if countries and job.country in set(countries):
            score += _COUNTRY_BOOST
            reasons.append(f"in target country {job.country}")
        if job.seniority in _SENIOR_LEVELS:
            score += _SENIOR_BOOST
            reasons.append(f"{job.seniority}-level role")

        return RankedJob(
            job=job,
            match_score=_clamp(score),
            match_reasons=reasons,
            visa_eligible=visa_eligible,
        )
