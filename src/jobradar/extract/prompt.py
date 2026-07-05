"""Extraction prompt: text in, strict JSON for ``ExtractedJob`` out."""

from __future__ import annotations

import json

from jobradar.models import ExtractedJob

SYSTEM = """You are a precise information extraction engine for job postings.
Return ONLY a JSON object matching the provided schema. Do not invent values.
If a field is not explicitly stated, use "unknown" (for enums) or null.

RULES:
- seniority: infer from title/requirements (years of experience). "Senior" => senior;
  "Staff/Principal" => staff; "Lead" => lead.
- visa_sponsorship: "yes" ONLY if the text explicitly offers relocation/visa/sponsorship
  (e.g. "visa sponsorship", "relocation package", "EU Blue Card", "Critical Skills Permit").
  If it says the candidate must already have work authorization, set "no". Otherwise "unknown".
- stack: list concrete technologies mentioned (languages, frameworks, DBs, cloud).
- visa_signals: copy the exact phrases that hint sponsorship.
- country: ISO 3166 alpha-2 (e.g. IE, DE, NL, CA) when the location makes it clear."""


def _schema_json() -> str:
    return json.dumps(ExtractedJob.model_json_schema(), indent=2)


def build_user_prompt(clean_text: str) -> str:
    return f"SCHEMA:\n{_schema_json()}\n\nJOB POSTING TEXT:\n{clean_text}"
