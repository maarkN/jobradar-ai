"""Read/write a JSONL corpus of extracted Jobs (a simple interchange for M2)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from jobradar.models import Job


def load_jobs_jsonl(path: Path) -> list[Job]:
    jobs: list[Job] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            jobs.append(Job.model_validate_json(line))
    return jobs


def save_jobs_jsonl(path: Path, jobs: Iterable[Job]) -> None:
    lines = (job.model_dump_json() for job in jobs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dedup_jobs(jobs: Sequence[Job]) -> list[Job]:
    """Collapse jobs sharing a content_hash (cross-source dedup, exact-key stage)."""
    seen: dict[str, Job] = {}
    for job in jobs:
        seen.setdefault(job.content_hash, job)
    return list(seen.values())
