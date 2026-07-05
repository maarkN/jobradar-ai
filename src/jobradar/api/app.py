"""FastAPI surface over the pipeline: trigger runs, browse jobs, rank, export."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response

from jobradar.api.deps import (
    get_embedder,
    get_extractor,
    get_fetch_factory,
    get_index,
    get_registry,
)
from jobradar.api.runs import RunRegistry
from jobradar.api.schemas import RankRequest, RunRequest, RunState
from jobradar.embeddings import Embedder
from jobradar.export import ranked_to_csv
from jobradar.graph import IngestPipeline, SupportsExtract
from jobradar.index import QdrantJobIndex
from jobradar.models import Job, RankedJob
from jobradar.rank import Ranker
from jobradar.sources import FetchFn, Source


def create_app() -> FastAPI:
    app = FastAPI(title="JobRadar AI", version="0.1.0")

    @app.get("/health")
    def health(index: Annotated[QdrantJobIndex, Depends(get_index)]) -> dict[str, object]:
        try:
            return {"status": "ok", "jobs": index.count()}
        except Exception as exc:
            return {"status": "degraded", "error": str(exc)}

    @app.post("/runs", response_model=RunState)
    def create_run(
        req: RunRequest,
        background: BackgroundTasks,
        extractor: Annotated[SupportsExtract, Depends(get_extractor)],
        embedder: Annotated[Embedder, Depends(get_embedder)],
        index: Annotated[QdrantJobIndex, Depends(get_index)],
        fetch_factory: Annotated[Callable[[list[Source]], FetchFn], Depends(get_fetch_factory)],
        registry: Annotated[RunRegistry, Depends(get_registry)],
    ) -> RunState:
        run = registry.create()
        sources = [Source(id=s.id, seed_urls=s.seed_urls, anti_bot=s.anti_bot) for s in req.sources]

        def execute() -> None:
            try:
                pipeline = IngestPipeline(
                    fetch_fn=fetch_factory(sources),
                    extractor=extractor,
                    embedder=embedder,
                    index=index,
                )
                registry.complete(run.run_id, pipeline.run(sources).summary())
            except Exception as exc:
                registry.fail(run.run_id, str(exc))

        background.add_task(execute)
        return RunState(run_id=run.run_id, state=run.state)

    @app.get("/runs/{run_id}", response_model=RunState)
    def get_run(run_id: str, registry: Annotated[RunRegistry, Depends(get_registry)]) -> RunState:
        run = registry.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return RunState(run_id=run.run_id, state=run.state, result=run.result, error=run.error)

    @app.get("/jobs", response_model=list[Job])
    def list_jobs(
        index: Annotated[QdrantJobIndex, Depends(get_index)],
        country: Annotated[list[str] | None, Query()] = None,
        seniority: str | None = None,
        visa: str | None = None,
        remote: str | None = None,
        limit: int = 200,
    ) -> list[Job]:
        return index.all_jobs(
            countries=country, seniority=seniority, visa=visa, remote=remote, limit=limit
        )

    @app.get("/jobs/{content_hash}", response_model=Job)
    def get_job(content_hash: str, index: Annotated[QdrantJobIndex, Depends(get_index)]) -> Job:
        job = index.get_by_hash(content_hash)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/rank", response_model=list[RankedJob])
    def rank_jobs(
        req: RankRequest,
        embedder: Annotated[Embedder, Depends(get_embedder)],
        index: Annotated[QdrantJobIndex, Depends(get_index)],
    ) -> list[RankedJob]:
        ranker = Ranker(embedder, index)
        return ranker.rank(
            req.cv_text, top_k=req.top_k, countries=req.countries, require_visa=req.require_visa
        )

    @app.post("/export/csv")
    def export_csv(
        req: RankRequest,
        embedder: Annotated[Embedder, Depends(get_embedder)],
        index: Annotated[QdrantJobIndex, Depends(get_index)],
    ) -> Response:
        ranker = Ranker(embedder, index)
        ranked = ranker.rank(
            req.cv_text, top_k=req.top_k, countries=req.countries, require_visa=req.require_visa
        )
        return Response(
            content=ranked_to_csv(ranked),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=matches.csv"},
        )

    return app


app = create_app()
