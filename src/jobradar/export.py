"""CSV export compatible with the applications tracker."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence

from jobradar.models import RankedJob

CSV_COLUMNS = [
    "company",
    "title",
    "country",
    "remote",
    "seniority",
    "visa",
    "match_score",
    "url",
    "source",
    "posted_at",
    "status",
]


def ranked_to_csv(ranked: Sequence[RankedJob], *, status: str = "to_apply") -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for r in ranked:
        job = r.job
        writer.writerow(
            {
                "company": job.company,
                "title": job.title,
                "country": job.country or "",
                "remote": job.remote.value,
                "seniority": job.seniority.value,
                "visa": job.visa_sponsorship.value,
                "match_score": f"{r.match_score:.3f}",
                "url": str(job.url),
                "source": job.source,
                "posted_at": job.posted_at.isoformat() if job.posted_at else "",
                "status": status,
            }
        )
    return buffer.getvalue()
