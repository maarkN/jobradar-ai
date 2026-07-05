"""HTML to clean text. Cheap, deterministic, and the input to the LLM extractor."""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser

_DROP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "form")


def html_to_text(html: str, *, max_chars: int = 12_000) -> str:
    """Strip boilerplate tags and collapse whitespace into readable text.

    Truncated to ``max_chars`` to bound token cost; job postings almost always
    fit, and the important fields sit near the top.
    """
    tree = HTMLParser(html)
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()

    body = tree.body or tree.root
    text = body.text(separator="\n", strip=True) if body is not None else ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text[:max_chars]


def quick_title(html: str) -> str | None:
    """Best-effort title from <h1> or <title> (the cheap 'easy scope' fields)."""
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    if h1 is not None and h1.text(strip=True):
        return h1.text(strip=True)
    title = tree.css_first("title")
    if title is not None and title.text(strip=True):
        return title.text(strip=True)
    return None
