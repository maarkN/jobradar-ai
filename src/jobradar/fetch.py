"""Fetchers behind one small interface.

``AntiBotFetcher`` reuses the cloudscraper-go server (its /fetch endpoint), which
ties this project to the Go binary from the sibling repo. ``PlainFetcher`` is a
direct httpx GET for sources without anti-bot protection.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from jobradar.config import Settings


class Fetcher(Protocol):
    def fetch(self, url: str, *, session: str = "default") -> str:
        """Return the HTML for url, or raise on failure."""
        ...


class PlainFetcher:
    """Direct GET with a browser-ish User-Agent, for open sources."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    def fetch(self, url: str, *, session: str = "default") -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; JobRadarAI/0.1)"}
        resp = httpx.get(url, headers=headers, timeout=self._timeout, follow_redirects=True)
        resp.raise_for_status()
        return resp.text


class AntiBotFetcher:
    """Fetch through the cloudscraper-go server for anti-bot protected sources."""

    def __init__(self, settings: Settings, timeout: float = 60.0) -> None:
        self._base = settings.cloudscraper_server_url.rstrip("/")
        self._timeout = timeout

    def fetch(self, url: str, *, session: str = "default") -> str:
        resp = httpx.get(
            f"{self._base}/fetch",
            params={"url": url, "session": session},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.text
