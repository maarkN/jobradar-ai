"""Qdrant-backed job index. Defaults to in-process local mode so tests need no server."""

from __future__ import annotations

from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from jobradar.config import Settings
from jobradar.models import Job


def _point_id(content_hash: str) -> int:
    """Deterministic uint64 point id from a job's dedup hash (also gives dedup on upsert)."""
    return int(content_hash, 16)


class QdrantJobIndex:
    def __init__(self, settings: Settings, dim: int) -> None:
        self._client = QdrantClient(location=settings.qdrant_url or ":memory:")
        self._collection = settings.qdrant_collection
        self._dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
            )

    def upsert(self, jobs: Sequence[Job], vectors: Sequence[Sequence[float]]) -> None:
        points = [
            models.PointStruct(
                id=_point_id(job.content_hash),
                vector=list(vector),
                payload=job.model_dump(mode="json"),
            )
            for job, vector in zip(jobs, vectors, strict=True)
        ]
        self._client.upsert(self._collection, points=points)

    def search(
        self,
        vector: Sequence[float],
        top_k: int,
        *,
        countries: Sequence[str] | None = None,
        require_visa: bool = False,
    ) -> list[tuple[Job, float]]:
        must: list[models.FieldCondition] = []
        if countries:
            must.append(
                models.FieldCondition(key="country", match=models.MatchAny(any=list(countries)))
            )
        if require_visa:
            must.append(
                models.FieldCondition(key="visa_sponsorship", match=models.MatchValue(value="yes"))
            )
        query_filter = models.Filter(must=must) if must else None

        hits = self._client.query_points(
            self._collection, query=list(vector), limit=top_k, query_filter=query_filter
        ).points
        return [(Job.model_validate(hit.payload), float(hit.score)) for hit in hits]

    def count(self) -> int:
        return self._client.count(self._collection).count
