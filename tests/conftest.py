"""Shared test fixtures, including a fake LLM provider so tests never hit the network."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from jobradar.cost import Usage
from jobradar.llm.base import LLMResponse, Tier

FIXTURES = Path(__file__).parent / "fixtures"


class FakeProvider:
    """Returns queued responses in order, recording the tiers it was called with.

    Pass one string to always return it, or a list to script a retry/escalation
    sequence (e.g. a bad payload followed by a good one).
    """

    def __init__(self, responses: str | Sequence[str]) -> None:
        self._responses: list[str] = [responses] if isinstance(responses, str) else list(responses)
        self._i = 0
        self.calls: list[Tier] = []

    def complete(
        self, system: str, user: str, tier: Tier, *, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append(tier)
        text = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return LLMResponse(
            text=text, usage=Usage(model=f"fake-{tier}", input_tokens=1000, output_tokens=200)
        )


@pytest.fixture
def sample_html() -> str:
    return (FIXTURES / "sample_job.html").read_text(encoding="utf-8")


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path / "cache"
