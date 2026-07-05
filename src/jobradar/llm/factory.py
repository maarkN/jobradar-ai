"""Select the LLM provider from configuration (env-driven)."""

from __future__ import annotations

from jobradar.config import Settings
from jobradar.llm.base import LLMProvider


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "anthropic":
        from jobradar.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings)
    from jobradar.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(settings)
