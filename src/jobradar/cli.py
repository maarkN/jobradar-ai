"""JobRadar AI command line. M1 exposes `extract`; more commands land per milestone."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from jobradar.cache import ExtractionCache, NullCache
from jobradar.config import load_settings
from jobradar.extract import Extractor, html_to_text
from jobradar.fetch import AntiBotFetcher, PlainFetcher
from jobradar.llm import get_provider

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
def version() -> None:
    """Print the JobRadar AI version."""
    from importlib.metadata import version as pkg_version

    typer.echo(pkg_version("jobradar-ai"))


if __name__ == "__main__":
    app()
