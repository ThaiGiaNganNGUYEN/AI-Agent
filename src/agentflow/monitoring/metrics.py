"""Prometheus metrics plus a small in-process cache counter."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

WORKFLOW_RUNS = Counter(
    "agentflow_workflow_runs_total",
    "Workflow runs by name and terminal status.",
    ["workflow", "status"],
)
WORKFLOW_LATENCY = Histogram(
    "agentflow_workflow_latency_seconds",
    "End-to-end workflow latency.",
    ["workflow"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
TOOL_CALLS = Counter(
    "agentflow_tool_calls_total",
    "External tool calls.",
    ["tool", "status"],
)
CACHE_EVENTS = Counter(
    "agentflow_cache_events_total",
    "Model response cache lookups.",
    ["result"],
)
SAFETY_DECISIONS = Counter(
    "agentflow_safety_decisions_total",
    "Pre-execution safety decisions.",
    ["decision"],
)
HTTP_REQUESTS = Counter(
    "agentflow_http_requests_total",
    "HTTP requests served by the API.",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "agentflow_http_latency_seconds",
    "HTTP request latency.",
    ["path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)


class CacheStats:
    def __init__(self) -> None:
        self.hits = 0
        self.misses = 0

    def hit_counter(self) -> None:
        CACHE_EVENTS.labels(result="hit").inc()

    def miss_counter(self) -> None:
        CACHE_EVENTS.labels(result="miss").inc()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
