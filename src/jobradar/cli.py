"""JobRadar AI command line. M1 exposes `extract`; more commands land per milestone."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from jobradar.cache import ExtractionCache, NullCache
from jobradar.config import load_settings
from jobradar.embeddings import OpenAIEmbedder
from jobradar.export import ranked_to_csv
from jobradar.extract import Extractor, html_to_text
from jobradar.fetch import AntiBotFetcher, PlainFetcher
from jobradar.graph import IngestPipeline
from jobradar.index import QdrantJobIndex
from jobradar.llm import get_provider
from jobradar.rank import Ranker
from jobradar.sources import load_sources, make_fetch_fn
from jobradar.store import load_jobs_jsonl, save_jobs_jsonl

app = typer.Typer(add_completion=False, help="JobRadar AI: scrape, extract and rank jobs.")


@app.command()
def extract(
    url: Annotated[str, typer.Option(help="Job posting URL")],
    source: Annotated[str, typer.Option(help="Adapter id, e.g. irishjobs")] = "manual",
    html_file: Annotated[
        Path | None, typer.Option(help="Read HTML from a file instead of fetching")
    ] = None,
    anti_bot: Annotated[bool, typer.Option(help="Fetch via the cloudscraper-go server")] = False,
    no_cache: Annotated[bool, typer.Option(help="Bypass the extraction cache")] = False,
) -> None:
    """Fetch (or read) a posting, extract a validated Job, and print it as JSON."""
    settings = load_settings()

    if html_file is not None:
        html = html_file.read_text(encoding="utf-8")
    else:
        fetcher = AntiBotFetcher(settings) if anti_bot else PlainFetcher()
        html = fetcher.fetch(url)

    clean_text = html_to_text(html)
    cache = NullCache() if no_cache else ExtractionCache(settings.cache_dir)
    extractor = Extractor(get_provider(settings), cache, settings)

    job = extractor.extract(clean_text, source=source, url=url, fetched_at=datetime.now(UTC))

    typer.echo(json.dumps(job.model_dump(mode="json"), indent=2, ensure_ascii=False))
    typer.echo(f"\ncost: {json.dumps(extractor.cost.summary())}", err=True)


@app.command()
def rank(
    jobs_file: Annotated[Path, typer.Option(help="JSONL corpus of extracted Jobs")],
    cv_file: Annotated[Path, typer.Option(help="CV as markdown/plain text")],
    top_k: Annotated[int, typer.Option(help="How many matches to return")] = 20,
    country: Annotated[list[str] | None, typer.Option(help="Target countries (repeatable)")] = None,
    require_visa: Annotated[bool, typer.Option(help="Only visa-sponsoring roles")] = True,
) -> None:
    """Embed a job corpus + CV, index in Qdrant, and print the top CV matches."""
    settings = load_settings()
    countries = country or ["IE", "DE", "NL", "CA"]

    jobs = load_jobs_jsonl(jobs_file)
    cv_text = cv_file.read_text(encoding="utf-8")

    embedder = OpenAIEmbedder(settings)
    ranker = Ranker(embedder, QdrantJobIndex(settings, embedder.dim))
    ranker.index_jobs(jobs)
    results = ranker.rank(cv_text, top_k=top_k, countries=countries, require_visa=require_visa)

    for r in results:
        typer.echo(
            f"{r.match_score:.2f}  {r.job.title} @ {r.job.company} "
            f"({r.job.country or '?'})  [{', '.join(r.match_reasons)}]"
        )
    typer.echo(f"\nindexed {len(jobs)} jobs, cost: {json.dumps(embedder.cost.summary())}", err=True)


@app.command()
def ingest(
    sources_file: Annotated[
        Path, typer.Option(help="JSON list of sources (id, seed_urls, anti_bot)")
    ],
    out: Annotated[Path, typer.Option(help="Write indexed jobs as JSONL (feeds `rank`)")],
) -> None:
    """Run the LangGraph pipeline over the configured sources and index the jobs."""
    settings = load_settings()
    sources = load_sources(sources_file)

    provider = get_provider(settings)
    extractor = Extractor(provider, ExtractionCache(settings.cache_dir), settings)
    embedder = OpenAIEmbedder(settings)
    pipeline = IngestPipeline(
        fetch_fn=make_fetch_fn(sources, settings),
        extractor=extractor,
        embedder=embedder,
        index=QdrantJobIndex(settings, embedder.dim),
    )

    result = pipeline.run(sources)
    save_jobs_jsonl(out, result.indexed)

    typer.echo(f"ingest: {json.dumps(result.summary())} -> {out}")
    for src, url, msg in result.errors:
        typer.echo(f"  error [{src}] {url}: {msg}", err=True)
    typer.echo(
        f"cost: extract={json.dumps(extractor.cost.summary())} "
        f"embed={json.dumps(embedder.cost.summary())}",
        err=True,
    )


@app.command()
def export(
    jobs_file: Annotated[Path, typer.Option(help="JSONL corpus of extracted Jobs")],
    cv_file: Annotated[Path, typer.Option(help="CV as markdown/plain text")],
    out: Annotated[Path, typer.Option(help="CSV output (aplicacoes-tracker format)")],
    top_k: Annotated[int, typer.Option()] = 20,
    country: Annotated[list[str] | None, typer.Option()] = None,
    require_visa: Annotated[bool, typer.Option()] = True,
) -> None:
    """Rank a corpus against a CV and write the top matches as a tracker-ready CSV."""
    settings = load_settings()
    countries = country or ["IE", "DE", "NL", "CA"]

    embedder = OpenAIEmbedder(settings)
    ranker = Ranker(embedder, QdrantJobIndex(settings, embedder.dim))
    ranker.index_jobs(load_jobs_jsonl(jobs_file))
    ranked = ranker.rank(
        cv_file.read_text(encoding="utf-8"),
        top_k=top_k,
        countries=countries,
        require_visa=require_visa,
    )
    out.write_text(ranked_to_csv(ranked), encoding="utf-8")
    typer.echo(f"wrote {len(ranked)} rows -> {out}")


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("jobradar.api:app", host=host, port=port)


@app.command()
def version() -> None:
    """Print the JobRadar AI version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("jobradar-ai"))


if __name__ == "__main__":
    app()
