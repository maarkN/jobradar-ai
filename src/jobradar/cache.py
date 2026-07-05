"""Content-addressed extraction cache: identical input text is never re-extracted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def text_hash(text: str) -> str:
    """Extraction-cache key over the *input* text sent to the LLM."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExtractionCache:
    """A tiny on-disk JSON cache keyed by :func:`text_hash`."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8"
        )


class NullCache(ExtractionCache):
    """Cache that never stores or returns anything (for tests / --no-cache)."""

    def __init__(self) -> None:
        pass

    def get(self, key: str) -> dict[str, Any] | None:
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        return None
