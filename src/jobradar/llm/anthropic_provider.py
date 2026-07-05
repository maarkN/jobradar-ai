"""Anthropic-backed LLM provider (Haiku fast tier, Opus strong tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobradar.config import Settings
from jobradar.cost import Usage
from jobradar.llm.base import LLMResponse, Tier

if TYPE_CHECKING:
    from anthropic import Anthropic


class AnthropicProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic

        self._client: Anthropic = Anthropic(api_key=settings.anthropic_api_key)
        self._settings = settings

    def complete(
        self, system: str, user: str, tier: Tier, *, json_mode: bool = False
    ) -> LLMResponse:
        model = self._settings.model_for("anthropic", tier)
        # Anthropic has no strict JSON mode; a prefilled "{" nudges valid JSON.
        messages = [{"role": "user", "content": user}]
        if json_mode:
            messages.append({"role": "assistant", "content": "{"})

        resp = self._client.messages.create(
            model=model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self._settings.extract_max_tokens,
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        if json_mode:
            text = "{" + text
        usage = Usage(
            model=model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        return LLMResponse(text=text, usage=usage)
