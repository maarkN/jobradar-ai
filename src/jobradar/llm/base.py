"""Provider-agnostic LLM interface, so Anthropic and OpenAI are interchangeable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from jobradar.cost import Usage

Tier = Literal["fast", "strong"]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    usage: Usage


class LLMProvider(Protocol):
    """A minimal completion interface. Implementations select the model by tier."""

    def complete(
        self, system: str, user: str, tier: Tier, *, json_mode: bool = False
    ) -> LLMResponse:
        """Return the model's text for the given system/user prompts."""
        ...
