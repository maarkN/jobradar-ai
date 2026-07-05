from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from conftest import FakeExtractor
from jobradar.eval import evaluate, load_evalset
from jobradar.models import Job, RemotePolicy, Seniority, VisaSponsorship

EVALSET = Path(__file__).parent.parent / "evalset"


def _job_from_gold(case_id: str, gold: dict[str, str]) -> Job:
    return Job(
        title=gold["title"],
        company=gold["company"],
        seniority=Seniority(gold["seniority"]),
        remote=RemotePolicy(gold["remote"]),
        visa_sponsorship=VisaSponsorship(gold["visa_sponsorship"]),
        source="eval",
        url=f"https://eval/{case_id}",
        fetched_at=datetime.now(UTC),
        content_hash="0" * 16,
    )


def test_perfect_extractor_scores_100() -> None:
    cases = load_evalset(EVALSET)
    assert len(cases) >= 3

    by_url = {f"https://eval/{c.id}": _job_from_gold(c.id, c.gold) for c in cases}
    report = evaluate(cases, FakeExtractor(by_url))

    assert report.overall() == 1.0
    assert all(acc == 1.0 for acc in report.field_accuracy().values())
    assert report.mistakes == []


def test_sloppy_extractor_records_mistakes() -> None:
    cases = load_evalset(EVALSET)

    def wrong(case_id: str, gold: dict[str, str]) -> Job:
        job = _job_from_gold(case_id, gold)
        job.seniority = Seniority.unknown
        job.visa_sponsorship = VisaSponsorship.unknown
        return job

    by_url = {f"https://eval/{c.id}": wrong(c.id, c.gold) for c in cases}
    report = evaluate(cases, FakeExtractor(by_url))

    accuracy = report.field_accuracy()
    assert accuracy["title"] == 1.0
    assert accuracy["seniority"] < 1.0
    assert len(report.mistakes) > 0
