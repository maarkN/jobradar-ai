"""Token usage and USD cost accounting.

Prices are approximate USD per 1M tokens and are meant to be edited as pricing
changes. Unknown models fall back to zero so the machinery never crashes a run;
the point is to make cost *visible*, which is the portfolio signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# (input_per_mtok, output_per_mtok) in USD. Approximate; update as needed.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-8": (5.0, 25.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5": (1.25, 10.0),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def price_for(model: str) -> tuple[float, float]:
    return _PRICES.get(model, (0.0, 0.0))


@dataclass(frozen=True)
class Usage:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        in_price, out_price = price_for(self.model)
        return self.input_tokens / 1_000_000 * in_price + self.output_tokens / 1_000_000 * out_price


@dataclass
class CostAccumulator:
    """Sums usage across a run for reporting."""

    entries: list[Usage] = field(default_factory=list)

    def add(self, usage: Usage) -> None:
        self.entries.append(usage)

    @property
    def total_usd(self) -> float:
        return sum(u.cost_usd for u in self.entries)

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.entries)

    def summary(self) -> dict[str, float | int]:
        return {
            "calls": len(self.entries),
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_usd, 6),
        }
