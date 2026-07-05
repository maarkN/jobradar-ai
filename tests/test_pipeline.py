from __future__ import annotations

from conftest import FakeEmbedder, FakeExtractor, FakeFetcher
from jobradar.config import Settings
from jobradar.graph import IngestPipeline
from jobradar.index import QdrantJobIndex
from jobradar.models import Job
from jobradar.sources import Source

URL1 = "https://board/1"
URL2 = "https://board/2"
URL_BAD = "https://board/bad"


def _pipeline(by_url: dict[str, Job], fail: set[str]) -> tuple[IngestPipeline, QdrantJobIndex]:
    emb = FakeEmbedder()
    index = QdrantJobIndex(Settings(), emb.dim)
    fetch = FakeFetcher({URL1: "<html/>", URL2: "<html/>"}, fail=fail)
    pipe = IngestPipeline(
        fetch_fn=fetch, extractor=FakeExtractor(by_url), embedder=emb, index=index
    )
    return pipe, index


def test_ingest_dedups_and_isolates_failures(sample_jobs: list[Job]) -> None:
    by_url = {URL1: sample_jobs[0], URL2: sample_jobs[1]}  # Acme, Beta
    pipe, index = _pipeline(by_url, fail={URL_BAD})

    result = pipe.run(
        [
            Source(id="a", seed_urls=[URL1, URL_BAD]),  # one good, one that fails to fetch
            Source(id="b", seed_urls=[URL1, URL2]),  # URL1 is a cross-source duplicate
        ]
    )

    assert len(result.indexed) == 2  # Acme + Beta, each once
    assert result.duplicates == 1  # source b's URL1
    assert len(result.errors) == 1  # source a's bad URL
    assert result.errors[0][1] == URL_BAD
    assert index.count() == 2  # dedup held in the vector store too


def test_one_bad_source_does_not_stop_the_rest(sample_jobs: list[Job]) -> None:
    by_url = {URL2: sample_jobs[1]}
    pipe, _ = _pipeline(by_url, fail={URL1})

    result = pipe.run([Source(id="a", seed_urls=[URL1, URL2])])

    assert len(result.errors) == 1  # URL1 failed
    assert len(result.indexed) == 1  # URL2 still processed
    assert result.indexed[0].company == "Beta"
