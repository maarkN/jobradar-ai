from __future__ import annotations

from datetime import UTC, datetime

from jobradar.models import Job, Seniority, VisaSponsorship, content_hash


def test_content_hash_is_normalized_and_stable() -> None:
    a = content_hash("Acme  Cloud", "Senior Go Engineer", "Dublin, Ireland")
    b = content_hash("acme cloud", "senior go engineer!", "dublin ireland")
    assert a == b
    assert len(a) == 16


def test_content_hash_differs_on_different_role() -> None:
    a = content_hash("Acme", "Senior Go Engineer", "Dublin")
    b = content_hash("Acme", "Staff Go Engineer", "Dublin")
    assert a != b


def test_job_requires_provenance() -> None:
    job = Job(
        title="Senior Go Engineer",
        company="Acme",
        seniority=Seniority.senior,
        visa_sponsorship=VisaSponsorship.yes,
        source="irishjobs",
        url="https://irishjobs.ie/job/123",
        fetched_at=datetime.now(UTC),
        content_hash="deadbeefdeadbeef",
    )
    assert job.seniority is Seniority.senior
    assert str(job.url).startswith("https://")
