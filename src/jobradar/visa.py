"""A dedicated, explainable visa-sponsorship classifier.

The main extractor already guesses ``visa_sponsorship``, but it is the field that
most changes a decision, so we refine it with conservative rules over the phrases
the model copied (``visa_signals``) plus the summary. An explicit "you must
already have work authorization" wins over any positive hint.
"""

from __future__ import annotations

import re

from jobradar.models import ExtractedJob, VisaSponsorship

_NO_PATTERNS = [
    r"must (already )?have (the )?(right to work|work authoriz)",
    r"(existing|valid) work authoriz",
    r"must be (eligible|authorized) to work",
    r"no (visa )?sponsorship",
    r"(cannot|not able to|unable to) sponsor",
    r"without sponsorship",
]

_YES_PATTERNS = [
    r"visa sponsorship",
    r"sponsor(ship)?( a| your)? visa",
    r"we (will |can )?sponsor",
    r"relocation (package|support|assistance|bonus)",
    r"eu blue card",
    r"blue card",
    r"critical skills",
    r"highly skilled migrant",
    r"\bhsm\b",
    r"\blmia\b",
    r"work permit (provided|support)",
]


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def refine_visa(extracted: ExtractedJob) -> VisaSponsorship:
    """Refine the extracted visa flag. Falls back to the model's value if unsure."""
    text = " ".join([*extracted.visa_signals, extracted.description_summary or ""]).lower()
    if _matches(_NO_PATTERNS, text):
        return VisaSponsorship.no
    if _matches(_YES_PATTERNS, text):
        return VisaSponsorship.yes
    return extracted.visa_sponsorship
