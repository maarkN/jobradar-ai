from __future__ import annotations

from jobradar.models import ExtractedJob, VisaSponsorship
from jobradar.visa import refine_visa


def _ex(**kwargs: object) -> ExtractedJob:
    return ExtractedJob(title="Engineer", company="Acme", **kwargs)


def test_positive_signal_yields_yes() -> None:
    assert refine_visa(_ex(visa_signals=["EU Blue Card"])) is VisaSponsorship.yes
    assert (
        refine_visa(_ex(description_summary="We offer visa sponsorship and relocation."))
        is VisaSponsorship.yes
    )


def test_explicit_no_overrides_positive_hint() -> None:
    ex = _ex(
        visa_signals=["relocation package"],
        description_summary="You must already have the right to work; no visa sponsorship.",
    )
    assert refine_visa(ex) is VisaSponsorship.no


def test_falls_back_to_extracted_value_without_signals() -> None:
    assert refine_visa(_ex(visa_sponsorship=VisaSponsorship.unknown)) is VisaSponsorship.unknown
    assert refine_visa(_ex(visa_sponsorship=VisaSponsorship.yes)) is VisaSponsorship.yes
