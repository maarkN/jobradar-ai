"""Canonical, typed schema for a job posting.

Two hashes are used, kept distinct on purpose (the PRD conflates them):

- ``text_hash`` (see :func:`jobradar.cache.text_hash`) keys the *extraction cache*
  on the input text, so identical HTML is never re-sent to the LLM.
- ``Job.content_hash`` is the *dedup key*, derived from the extracted
  ``company`` + ``title`` + ``location``, so the same role from two sources
  collapses to one record.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl


class Seniority(StrEnum):
    junior = "junior"
    mid = "mid"
    senior = "senior"
    staff = "staff"
    lead = "lead"
    unknown = "unknown"


class RemotePolicy(StrEnum):
    remote = "remote"
    hybrid = "hybrid"
    onsite = "onsite"
    unknown = "unknown"


class VisaSponsorship(StrEnum):
    yes = "yes"
    no = "no"
    unknown = "unknown"


class SalaryRange(BaseModel):
    min: float | None = None
    max: float | None = None
    currency: str | None = Field(default=None, description="ISO 4217, e.g. EUR")
    period: str | None = Field(default=None, description="year | month | day | hour")


class ExtractedJob(BaseModel):
    """The subset a language model is asked to fill from the posting text.

    Provenance fields (source, url, timestamps, hash) are added by the pipeline,
    never by the model, so the LLM can only fill what it actually sees.
    """

    title: str
    company: str
    location: str | None = None
    country: str | None = Field(default=None, description="ISO 3166 alpha-2")
    remote: RemotePolicy = RemotePolicy.unknown
    seniority: Seniority = Seniority.unknown
    stack: list[str] = Field(default_factory=list)
    salary: SalaryRange | None = None
    visa_sponsorship: VisaSponsorship = VisaSponsorship.unknown
    visa_signals: list[str] = Field(
        default_factory=list,
        description="Raw phrases hinting sponsorship, e.g. 'EU Blue Card', 'Critical Skills'",
    )
    description_summary: str | None = Field(default=None, max_length=600)
    posted_at: datetime | None = None


class Job(ExtractedJob):
    """Canonical job posting: the extracted fields plus provenance."""

    source: str = Field(description="Adapter id, e.g. 'irishjobs'")
    url: HttpUrl
    external_id: str | None = None
    fetched_at: datetime
    content_hash: str = Field(description="Dedup key over normalized company+title+location")


class RankedJob(BaseModel):
    job: Job
    match_score: float = Field(ge=0, le=1, description="CV similarity + rule boosts")
    match_reasons: list[str] = Field(default_factory=list)
    visa_eligible: bool


def _normalize(text: str) -> str:
    """Lowercase, strip accents-insensitive punctuation and collapse whitespace."""
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_hash(company: str, title: str, location: str | None) -> str:
    """Stable dedup key over normalized company + title + location."""
    parts = "|".join(_normalize(p) for p in (company, title, location or ""))
    return hashlib.sha256(parts.encode("utf-8")).hexdigest()[:16]
