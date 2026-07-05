"""Evaluate extraction quality against a labelled gold set.

Field-level accuracy on the critical fields is the number a reviewer trusts, so
this is a first-class command, not an afterthought. The gold set lives in the
repo as JSON cases; drop real, hand-labelled postings in to raise the bar.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from jobradar.extract.parse import html_to_text
from jobradar.graph import SupportsExtract

CRITICAL_FIELDS = ["title", "company", "seniority", "visa_sponsorship", "remote"]


@dataclass(frozen=True)
class EvalCase:
    id: str
    html: str
    gold: dict[str, str]


def load_evalset(directory: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cases.append(EvalCase(id=path.stem, html=data["html"], gold=data["gold"]))
    return cases


def _norm(value: object) -> str:
    return str(value).strip().lower()


@dataclass
class EvalReport:
    n: int
    correct: dict[str, int]
    mistakes: list[tuple[str, str, str, str]] = field(
        default_factory=list
    )  # case, field, gold, pred

    def field_accuracy(self) -> dict[str, float]:
        return {f: (self.correct[f] / self.n if self.n else 0.0) for f in CRITICAL_FIELDS}

    def overall(self) -> float:
        accuracies = self.field_accuracy()
        return sum(accuracies.values()) / len(CRITICAL_FIELDS)


def evaluate(cases: Sequence[EvalCase], extractor: SupportsExtract) -> EvalReport:
    correct = dict.fromkeys(CRITICAL_FIELDS, 0)
    mistakes: list[tuple[str, str, str, str]] = []
    for case in cases:
        job = extractor.extract(
            html_to_text(case.html),
            source="eval",
            url=f"https://eval/{case.id}",
            fetched_at=datetime.now(UTC),
        )
        for name in CRITICAL_FIELDS:
            gold = _norm(case.gold[name])
            pred = _norm(getattr(job, name))
            if pred == gold:
                correct[name] += 1
            else:
                mistakes.append((case.id, name, gold, pred))
    return EvalReport(n=len(cases), correct=correct, mistakes=mistakes)
