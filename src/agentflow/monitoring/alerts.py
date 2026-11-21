"""Threshold alerts derived from recent run statistics."""

from dataclasses import dataclass, field


@dataclass
class AlertEngine:
    error_rate: float
    latency_ms: int
    current: list[dict] = field(default_factory=list)

    def evaluate(self, stats: dict) -> list[dict]:
        alerts: list[dict] = []
        samples = int(stats.get("samples") or 0)
        if samples >= 3 and float(stats.get("error_rate") or 0) >= self.error_rate:
            alerts.append(
                {
                    "code": "error_rate",
                    "severity": "warning",
                    "message": (
                        f"Workflow error rate is {float(stats['error_rate']):.0%}, "
                        f"above the {self.error_rate:.0%} threshold."
                    ),
                }
            )
        latency = float(stats.get("avg_latency_ms") or 0)
        if samples >= 3 and latency >= self.latency_ms:
            alerts.append(
                {
                    "code": "latency",
                    "severity": "warning",
                    "message": f"Average latency is {latency:.0f} ms, above {self.latency_ms} ms.",
                }
            )
        spend = float(stats.get("spend_usd") or 0)
        budget = float(stats.get("budget_usd") or 0)
        if budget and spend >= budget:
            alerts.append(
                {
                    "code": "budget",
                    "severity": "critical",
                    "message": f"Model spend ${spend:.4f} has reached the ${budget:.2f} budget.",
                }
            )
        self.current = alerts
        return alerts
