"""Extract a validated ``Job`` from clean text, with retry, model cascade and cache.

Flow (ADR-4 + ADR-5): try the fast/cheap model, validate against Pydantic; on a
parse or validation failure, retry a few times, then escalate to the strong model
once; if it still fails, raise. Identical input text is served from cache and
never re-sent to the LLM.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import structlog
from pydantic import ValidationError

from jobradar.cache import ExtractionCache, text_hash
from jobradar.config import Settings
from jobradar.cost import CostAccumulator
from jobradar.extract.prompt import SYSTEM, build_user_prompt
from jobradar.llm.base import LLMProvider, Tier
from jobradar.models import ExtractedJob, Job, content_hash
from jobradar.visa import refine_visa

log = structlog.get_logger(__name__)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ExtractionError(RuntimeError):
    """Raised when a posting cannot be extracted into a valid Job."""


def _parse_json_object(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object found in model output")
    parsed: dict[str, object] = json.loads(match.group(0))
    return parsed


class Extractor:
    def __init__(
        self,
        provider: LLMProvider,
        cache: ExtractionCache,
        settings: Settings,
        cost: CostAccumulator | None = None,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._settings = settings
        self.cost = cost or CostAccumulator()

    def extract(self, clean_text: str, *, source: str, url: str, fetched_at: datetime) -> Job:
        key = text_hash(clean_text)
        cached = self._cache.get(key)
        if cached is not None:
            log.debug("extract.cache_hit", source=source, url=url)
            return Job.model_validate(cached)

        extracted = self._extract_with_cascade(clean_text, source=source, url=url)
        job = Job(
            **extracted.model_dump(),
            source=source,
            url=url,
            fetched_at=fetched_at,
            content_hash=content_hash(extracted.company, extracted.title, extracted.location),
        )
        # Refine the critical visa field with the dedicated rule-based classifier.
        job.visa_sponsorship = refine_visa(extracted)
        self._cache.set(key, job.model_dump(mode="json"))
        return job

    def _extract_with_cascade(self, clean_text: str, *, source: str, url: str) -> ExtractedJob:
        user = build_user_prompt(clean_text)
        # Fast model, with a few retries, then one attempt on the strong model.
        fast: Tier = "fast"
        strong: Tier = "strong"
        attempts: list[Tier] = [fast] * (1 + self._settings.extract_fast_retries) + [strong]
        last_error: Exception | None = None
        for tier in attempts:
            try:
                resp = self._provider.complete(SYSTEM, user, tier, json_mode=True)
                self.cost.add(resp.usage)
                return ExtractedJob.model_validate(_parse_json_object(resp.text))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                log.warning("extract.attempt_failed", tier=tier, source=source, error=str(exc))
        raise ExtractionError(f"extraction failed for {url}: {last_error}") from last_error
