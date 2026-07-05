"""The ingest pipeline as a LangGraph state machine.

Per URL: fetch -> parse -> extract -> index. Each node isolates its own failure
by writing an ``error`` to the state, and a conditional edge routes a failed item
straight to END, so one bad posting never takes the batch down. Indexing
deduplicates by embedding similarity (exact-key dedup is already free via the
Qdrant point id). A checkpointer makes each item's run resumable by thread id.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from jobradar.embeddings import Embedder, job_to_text
from jobradar.extract.parse import html_to_text
from jobradar.graph.state import GraphState, IngestResult
from jobradar.index import QdrantJobIndex
from jobradar.models import Job
from jobradar.sources import FetchFn, Source

log = structlog.get_logger(__name__)


class SupportsExtract(Protocol):
    def extract(self, clean_text: str, *, source: str, url: str, fetched_at: datetime) -> Job: ...


class IngestPipeline:
    def __init__(
        self,
        *,
        fetch_fn: FetchFn,
        extractor: SupportsExtract,
        embedder: Embedder,
        index: QdrantJobIndex,
        dedup_threshold: float = 0.97,
        fetch_attempts: int = 2,
    ) -> None:
        self._fetch_fn = fetch_fn
        self._extractor = extractor
        self._embedder = embedder
        self._index = index
        self._threshold = dedup_threshold
        self._fetch_attempts = fetch_attempts
        self._app = self._build()

    def _build(self) -> object:
        graph = StateGraph(GraphState)
        graph.add_node("fetch", self._fetch)
        graph.add_node("parse", self._parse)
        graph.add_node("extract", self._extract)
        graph.add_node("index", self._index_node)
        graph.set_entry_point("fetch")
        graph.add_conditional_edges("fetch", self._route, {"ok": "parse", "end": END})
        graph.add_conditional_edges("parse", self._route, {"ok": "extract", "end": END})
        graph.add_conditional_edges("extract", self._route, {"ok": "index", "end": END})
        graph.add_edge("index", END)
        return graph.compile(checkpointer=MemorySaver())

    @staticmethod
    def _route(state: GraphState) -> str:
        return "end" if state.get("error") else "ok"

    def _fetch(self, state: GraphState) -> GraphState:
        last: Exception | None = None
        for _ in range(self._fetch_attempts):
            try:
                return {"raw_html": self._fetch_fn(state["source"], state["url"])}
            except Exception as exc:
                last = exc
        return {"error": f"fetch failed: {last}"}

    def _parse(self, state: GraphState) -> GraphState:
        return {"clean_text": html_to_text(state["raw_html"])}

    def _extract(self, state: GraphState) -> GraphState:
        try:
            job = self._extractor.extract(
                state["clean_text"],
                source=state["source"],
                url=state["url"],
                fetched_at=datetime.now(UTC),
            )
            return {"job": job.model_dump(mode="json")}
        except Exception as exc:
            return {"error": f"extract failed: {exc}"}

    def _index_node(self, state: GraphState) -> GraphState:
        job = Job.model_validate(state["job"])
        vector = self._embedder.embed([job_to_text(job)])[0]
        hits = self._index.search(vector, top_k=1)
        if hits and hits[0][1] >= self._threshold:
            log.debug("ingest.duplicate", source=state["source"], url=state["url"])
            return {"is_duplicate": True}
        self._index.upsert([job], [vector])
        return {"is_duplicate": False}

    def run(self, sources: Sequence[Source]) -> IngestResult:
        result = IngestResult()
        for src in sources:
            for url in src.seed_urls:
                config = {"configurable": {"thread_id": f"{src.id}:{url}"}}
                try:
                    final: GraphState = self._app.invoke(  # type: ignore[attr-defined]
                        {"source": src.id, "url": url}, config=config
                    )
                except Exception as exc:
                    result.errors.append((src.id, url, f"graph error: {exc}"))
                    continue

                if final.get("error"):
                    result.errors.append((src.id, url, final["error"]))
                elif final.get("is_duplicate"):
                    result.duplicates += 1
                elif "job" in final:
                    result.indexed.append(Job.model_validate(final["job"]))
        return result
