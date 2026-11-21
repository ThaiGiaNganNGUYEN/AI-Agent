"""Least-squares forecast for the next workflow latency.

This is a small linear model over recent successful runs. It is there to
surface a trend, not to page anyone by itself.
"""


def linear_forecast(values: list[float]) -> float | None:
    count = len(values)
    if count < 2:
        return None
    xs = list(range(count))
    mean_x = sum(xs) / count
    mean_y = sum(values) / count
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs) or 1.0
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return intercept + slope * count
