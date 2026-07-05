"""LangGraph state and the aggregate result of an ingest run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from jobradar.models import Job


class GraphState(TypedDict, total=False):
    source: str
    url: str
    raw_html: str
    clean_text: str
    # Job stored as a plain dict so the checkpointer can serialize the state.
    job: dict[str, Any]
    is_duplicate: bool
    error: str


@dataclass
class IngestResult:
    indexed: list[Job] = field(default_factory=list)
    duplicates: int = 0
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # (source, url, message)

    def summary(self) -> dict[str, int]:
        return {
            "indexed": len(self.indexed),
            "duplicates": self.duplicates,
            "errors": len(self.errors),
        }
