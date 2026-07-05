"""OpenAI-backed LLM provider (mini fast tier, full strong tier)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jobradar.config import Settings
from jobradar.cost import Usage
from jobradar.llm.base import LLMResponse, Tier

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIProvider:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        from openai import OpenAI

        self._client: OpenAI = OpenAI(api_key=settings.openai_api_key)
        self._settings = settings

    def complete(
        self, system: str, user: str, tier: Tier, *, json_mode: bool = False
    ) -> LLMResponse:
        model = self._settings.model_for("openai", tier)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._settings.extract_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage_obj = resp.usage
        usage = Usage(
            model=model,
            input_tokens=usage_obj.prompt_tokens if usage_obj else 0,
            output_tokens=usage_obj.completion_tokens if usage_obj else 0,
        )
        return LLMResponse(text=text, usage=usage)
