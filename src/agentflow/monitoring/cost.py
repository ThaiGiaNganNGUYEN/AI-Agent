"""Published list prices used for cost tracking.

Prices are USD per token. They are a planning estimate, not an invoice.
"""

from __future__ import annotations

PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("local", "*"): (0.0, 0.0),
    ("openai", "gpt-4o-mini"): (0.15 / 1_000_000, 0.60 / 1_000_000),
    ("openai", "*"): (2.50 / 1_000_000, 10.0 / 1_000_000),
    ("anthropic", "claude-3-5-haiku-latest"): (0.80 / 1_000_000, 4.0 / 1_000_000),
    ("anthropic", "*"): (3.0 / 1_000_000, 15.0 / 1_000_000),
}


def price(provider: str, model: str, tokens_in: int, tokens_out: int) -> float:
    rates = PRICES.get((provider, model), PRICES.get((provider, "*"), (0.0, 0.0)))
    return rates[0] * tokens_in + rates[1] * tokens_out
