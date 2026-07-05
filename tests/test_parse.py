from __future__ import annotations

from jobradar.extract.parse import html_to_text, quick_title


def test_html_to_text_strips_boilerplate_and_keeps_content(sample_html: str) -> None:
    text = html_to_text(sample_html)
    assert "Senior Go Engineer" in text
    assert "Critical Skills Permit" in text
    # script/nav/footer noise is removed
    assert "tracking noise" not in text
    assert "All rights reserved" not in text


def test_html_to_text_truncates(sample_html: str) -> None:
    text = html_to_text(sample_html, max_chars=20)
    assert len(text) == 20


def test_quick_title(sample_html: str) -> None:
    assert quick_title(sample_html) == "Senior Go Engineer"
