"""Source adapters. A source yields job-posting URLs and knows how to fetch them.

M3 uses seed URLs directly; discovery from listing pages is a thin per-source
addition later. Anti-bot sources are fetched through the cloudscraper-go server.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from jobradar.config import Settings
from jobradar.fetch import AntiBotFetcher, Fetcher, PlainFetcher

FetchFn = Callable[[str, str], str]


@dataclass(frozen=True)
class Source:
    id: str
    seed_urls: list[str] = field(default_factory=list)
    anti_bot: bool = False


def load_sources(path: Path) -> list[Source]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Source(
            id=item["id"], seed_urls=item.get("seed_urls", []), anti_bot=item.get("anti_bot", False)
        )
        for item in raw
    ]


def make_fetch_fn(sources: list[Source], settings: Settings) -> FetchFn:
    """Build a `(source_id, url) -> html` function backed by the right fetcher per source."""
    fetchers: dict[str, Fetcher] = {
        src.id: (AntiBotFetcher(settings) if src.anti_bot else PlainFetcher()) for src in sources
    }

    def fetch(source_id: str, url: str) -> str:
        return fetchers[source_id].fetch(url)

    return fetch
