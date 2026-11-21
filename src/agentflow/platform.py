"""Composition root that wires the engine, providers, and storage."""

from __future__ import annotations

import json
import logging
import time
import uuid

import httpx

from agentflow.agents.builtins import build_agents
from agentflow.cache.store import build_cache
from agentflow.config import Settings
from agentflow.context.builder import ContextBuilder
from agentflow.errors import AgentFlowError, NotFoundError, WorkflowError
from agentflow.evaluation.harness import EvaluationHarness
from agentflow.events import Event, EventBus
from agentflow.loading import load_yaml
from agentflow.monitoring.alerts import AlertEngine
from agentflow.monitoring.forecast import linear_forecast
from agentflow.monitoring.metrics import SAFETY_DECISIONS, WORKFLOW_LATENCY, WORKFLOW_RUNS, CacheStats
from agentflow.monitoring.safety import SafetyMonitor
from agentflow.persistence.repository import Repository
from agentflow.prompts.manager import PromptManager
from agentflow.providers.factory import build_provider
from agentflow.rag.seed import packaged_documents
from agentflow.rag.store import build_store
from agentflow.routing.policy import PolicyRouter
from agentflow.schemas import Document, RunContext, StepResult, WorkflowSpec
from agentflow.tools.base import ToolFactory
from agentflow.orchestration.engine import Orchestrator

logger = logging.getLogger("agentflow.platform")

EDITABLE = {
    "provider",
    "openai_model",
    "anthropic_model",
    "cache_ttl_seconds",
    "log_level",
    "monthly_budget_usd",
    "alert_webhook",
}
PROVIDERS = {"local", "openai", "anthropic"}


class Platform:
    def __init__(self, settings: Settings, tools: dict | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.runtime: dict = {}
        self.http_client = http_client
        self.bus = EventBus()
        self.bus.subscribe("*", _log_event)
        self.repo = Repository(settings.database_url, settings.password_iterations)
        self.store = build_store(settings.vector_backend, settings.vector_path)
        self.store.add(packaged_documents())
        self.prompts = PromptManager(settings.prompt_seed)
        self.cache = build_cache(settings.redis_url)
        self.cache_stats = CacheStats()
        self.tools = tools if tools is not None else ToolFactory(settings, http_client).create_all()
        self.safety = SafetyMonitor()
        self.context = ContextBuilder(settings.context_char_budget)
        self.provider = build_provider(self)
        self.agents = build_agents(self)
        self.router = PolicyRouter.from_package()
        self.workflows = {
            item["name"]: WorkflowSpec.model_validate(item) for item in load_yaml("workflows.yaml")["workflows"]
        }
        self._validate()
        self.orchestrator = Orchestrator(
            self.agents,
            self.bus,
            retries=max(1, settings.step_retries),
            timeout=settings.step_timeout_seconds,
            retry_delay=settings.step_retry_delay,
        )
        self.alerts = AlertEngine(settings.alert_error_rate, settings.alert_latency_ms)
        self.harness = EvaluationHarness(self)

    def close(self) -> None:
        self.repo.close()

    def effective(self, name: str):
        if name in self.runtime:
            return self.runtime[name]
        return getattr(self.settings, name)

    def model_name(self) -> str:
        provider = str(self.effective("provider"))
        if provider == "openai":
            return str(self.effective("openai_model"))
        if provider == "anthropic":
            return str(self.effective("anthropic_model"))
        return "local-template"

    def update_runtime(self, values: dict, actor: str) -> dict:
        unknown = set(values) - EDITABLE
        if unknown:
            raise AgentFlowError(f"Cannot change {', '.join(sorted(unknown))}")
        if "provider" in values and values["provider"] not in PROVIDERS:
            raise AgentFlowError("provider must be local, openai, or anthropic")
        if "log_level" in values:
            level = str(values["log_level"]).upper()
            if level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
                raise AgentFlowError("log_level must be DEBUG, INFO, WARNING, or ERROR")
            values["log_level"] = level
            logging.getLogger().setLevel(level)
        if "cache_ttl_seconds" in values:
            values["cache_ttl_seconds"] = int(values["cache_ttl_seconds"])
        if "monthly_budget_usd" in values:
            values["monthly_budget_usd"] = float(values["monthly_budget_usd"])
        self.runtime.update(values)
        if "provider" in values:
            self.provider = build_provider(self)
        self.repo.audit(actor, "runtime.update", json.dumps(values, default=str))
        return self.runtime_view()

    def runtime_view(self) -> dict:
        return {
            "provider": self.effective("provider"),
            "model": self.model_name(),
            "cache_ttl_seconds": int(self.effective("cache_ttl_seconds")),
            "log_level": self.effective("log_level"),
            "monthly_budget_usd": float(self.effective("monthly_budget_usd")),
            "alert_webhook_set": bool(self.effective("alert_webhook")),
            "vector_backend": self.settings.vector_backend,
        }

    def list_agents(self) -> list[dict]:
        return [agent.spec.model_dump() for agent in self.agents.values()]

    def list_workflows(self) -> list[dict]:
        return [workflow.model_dump() for workflow in self.workflows.values()]

    def add_document(self, title: str, text: str, actor: str) -> dict:
        document = Document(id=f"user-{uuid.uuid4()}", title=title.strip(), text=text.strip(), source="user")
        self.store.add([document])
        self.repo.audit(actor, "knowledge.add", document.title)
        return document.model_dump()

    def query_knowledge(self, query: str, k: int) -> list[dict]:
        return [
            {
                "id": hit.document.id,
                "title": hit.document.title,
                "source": hit.document.source,
                "score": round(hit.score, 4),
                "text": hit.document.text,
            }
            for hit in self.store.query(query, k=k)
        ]

    def summary(self) -> dict:
        self.refresh_alerts()
        stats = self.repo.stats()
        costs = self.repo.cost_summary()
        forecast = linear_forecast(stats["latencies"])
        return {
            "provider": self.effective("provider"),
            "model": self.model_name(),
            "runs": {
                "total": stats["total"],
                "succeeded": stats["succeeded"],
                "failed": stats["failed"],
                "blocked": stats["blocked"],
            },
            "samples": stats["samples"],
            "error_rate": stats["error_rate"],
            "avg_latency_ms": stats["avg_latency_ms"],
            "forecast_next_latency_ms": forecast,
            "spend_usd": costs["total_usd"],
            "avoided_usd": costs["avoided_usd"],
            "budget_usd": float(self.effective("monthly_budget_usd")),
            "cache": {"hits": self.cache_stats.hits, "misses": self.cache_stats.misses},
            "vectors": self.store.count(),
            "alerts": self.alerts.current,
            "environment": self.settings.environment,
        }

    def report(self) -> str:
        summary = self.summary()
        costs = self.repo.cost_summary()
        lines = [
            f"# {self.settings.app_name}",
            "",
            f"Environment: {summary['environment']}",
            f"Provider: {summary['provider']} ({summary['model']})",
            (
                f"Runs: {summary['runs']['total']} total, {summary['runs']['succeeded']} succeeded, "
                f"{summary['runs']['failed']} failed, {summary['runs']['blocked']} blocked."
            ),
            f"Error rate: {summary['error_rate']:.1%}",
            f"Average latency: {summary['avg_latency_ms']:.0f} ms",
            f"Forecast next latency: {summary['forecast_next_latency_ms']}",
            f"Spend: ${summary['spend_usd']:.6f} of ${summary['budget_usd']:.2f} "
            f"(${summary['avoided_usd']:.6f} avoided by cache)",
            f"Cache: {summary['cache']['hits']} hits, {summary['cache']['misses']} misses",
            f"Knowledge documents: {summary['vectors']}",
            "",
            "## Alerts",
        ]
        if summary["alerts"]:
            lines.extend(f"- {alert['severity']}: {alert['message']}" for alert in summary["alerts"])
        else:
            lines.append("- None")
        lines.extend(["", "## Cost by provider"])
        if costs["by_provider"]:
            for name, bucket in costs["by_provider"].items():
                lines.append(
                    f"- {name}: ${bucket['cost_usd']:.6f} billed, "
                    f"${bucket['avoided_usd']:.6f} avoided across {bucket['calls']} calls"
                )
        else:
            lines.append("- No model calls recorded yet")
        return "\n".join(lines) + "\n"

    def refresh_alerts(self) -> list[dict]:
        stats = self.repo.stats()
        costs = self.repo.cost_summary()
        return self.alerts.evaluate(
            {
                **stats,
                "spend_usd": costs["total_usd"],
                "budget_usd": float(self.effective("monthly_budget_usd")),
            }
        )

    async def execute(self, workflow_name: str, inputs: dict, actor: str, rule: str = "direct") -> dict:
        workflow = self.workflows.get(workflow_name)
        if workflow is None:
            raise NotFoundError(f"Unknown workflow {workflow_name}")
        merged = dict(inputs)
        for field in workflow.inputs:
            merged.setdefault(field.name, field.default)
        question = str(merged.get("question") or merged.get("message") or workflow.description)
        merged["question"] = question
        run_id = str(uuid.uuid4())
        started = time.perf_counter()
        self.repo.create_run(run_id, workflow.name, actor, rule, merged)
        self.repo.audit(actor, "workflow.start", f"{workflow.name} {run_id}")
        if rule != "direct":
            self.bus.publish(Event("route.decided", run_id, {"rule": rule, "workflow": workflow.name}))

        precheck = self.safety.review(question)
        if not precheck.allowed:
            record = self._finish(
                run_id,
                workflow.name,
                status="blocked",
                steps={},
                answer="Request blocked before execution. " + "; ".join(precheck.issues),
                safety=precheck.model_dump(),
                error="",
                started=started,
            )
            await self._after_run()
            return record

        ctx = RunContext(run_id=run_id, goal=question, inputs=merged, actor=actor)
        try:
            steps = await self.orchestrator.run(workflow, ctx)
        except WorkflowError as exc:
            record = self._finish(
                run_id,
                workflow.name,
                status="failed",
                steps=exc.partial,
                answer="",
                safety={},
                error=str(exc),
                started=started,
                workflow=workflow,
            )
            await self._after_run()
            return record

        blocked = any(step.data.get("blocked") or step.data.get("allowed") is False for step in steps.values())
        answer = next((step.output for step in steps.values() if step.agent == "executor"), "")
        safety = next((step.data["safety"] for step in steps.values() if "safety" in step.data), {})
        record = self._finish(
            run_id,
            workflow.name,
            status="blocked" if blocked else "succeeded",
            steps=steps,
            answer=answer,
            safety=safety,
            error="",
            started=started,
            workflow=workflow,
        )
        await self._after_run()
        return record

    async def route_and_execute(self, message: str, domain: str, inputs: dict, actor: str) -> dict:
        decision = self.router.route(message, domain)
        merged = {**inputs, "question": inputs.get("question") or message}
        run = await self.execute(decision.workflow, merged, actor, rule=decision.rule)
        return {"routing": decision.model_dump(), "run": run}

    def _finish(
        self,
        run_id: str,
        workflow_name: str,
        *,
        status: str,
        steps: dict[str, StepResult],
        answer: str,
        safety: dict,
        error: str,
        started: float,
        workflow: WorkflowSpec | None = None,
    ) -> dict:
        ordered = _ordered(steps, workflow)
        for step in ordered:
            if not (step.tokens_in or step.tokens_out or step.cost_usd or step.avoided_usd):
                continue
            self.repo.add_cost(
                run_id=run_id,
                provider=str(step.data.get("provider") or self.effective("provider")),
                model=str(step.data.get("model") or self.model_name()),
                tokens_in=step.tokens_in,
                tokens_out=step.tokens_out,
                cost_usd=step.cost_usd,
                avoided_usd=step.avoided_usd,
                cached=bool(step.data.get("cached")),
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        output = {"answer": answer, "steps": [step.model_dump() for step in ordered], "safety": safety}
        record = self.repo.finish_run(
            run_id,
            status=status,
            output=output,
            error=error,
            cost_usd=sum(step.cost_usd for step in ordered),
            latency_ms=latency_ms,
        )
        WORKFLOW_RUNS.labels(workflow=workflow_name, status=status).inc()
        WORKFLOW_LATENCY.labels(workflow=workflow_name).observe(latency_ms / 1000)
        if status == "blocked":
            SAFETY_DECISIONS.labels(decision="blocked").inc()
        elif status == "succeeded":
            SAFETY_DECISIONS.labels(decision="allowed").inc()
        self.bus.publish(Event("workflow.persisted", run_id, {"status": status, "latency_ms": latency_ms}))
        return record

    async def _after_run(self) -> None:
        previous = {alert["code"] for alert in self.alerts.current}
        alerts = self.refresh_alerts()
        if alerts and {alert["code"] for alert in alerts} != previous:
            await self.notify(alerts)

    async def notify(self, alerts: list[dict]) -> None:
        url = str(self.effective("alert_webhook") or "")
        if not url or not alerts:
            return
        try:
            if self.http_client is not None:
                for alert in alerts:
                    await self.http_client.post(url, json=alert)
                return
            async with httpx.AsyncClient(timeout=5) as client:
                for alert in alerts:
                    await client.post(url, json=alert)
        except httpx.HTTPError:
            logger.warning("alert webhook failed")

    def _validate(self) -> None:
        for workflow in self.workflows.values():
            identifiers = [step.id for step in workflow.steps]
            if len(identifiers) != len(set(identifiers)):
                raise AgentFlowError(f"Duplicate step id in workflow {workflow.name}")
            for step in workflow.steps:
                if step.agent not in self.agents:
                    raise AgentFlowError(f"Workflow {workflow.name} references unknown agent {step.agent}")
                for dependency in step.depends_on:
                    if dependency not in identifiers:
                        raise AgentFlowError(f"Workflow {workflow.name} step {step.id} depends on unknown {dependency}")
                for tool_name in step.tools:
                    if tool_name not in self.tools:
                        raise AgentFlowError(f"Workflow {workflow.name} references unknown tool {tool_name}")
        for rule in self.router.rules:
            if rule.workflow not in self.workflows:
                raise AgentFlowError(f"Routing rule {rule.name} points at unknown workflow {rule.workflow}")


def _ordered(steps: dict[str, StepResult], workflow: WorkflowSpec | None) -> list[StepResult]:
    if workflow is None:
        return list(steps.values())
    return [steps[step.id] for step in workflow.steps if step.id in steps]


def _log_event(event: Event) -> None:
    logging.getLogger("agentflow.audit").info(
        "%s %s",
        event.type,
        json.dumps(event.payload, default=str)[:500],
        extra={"run_id": event.run_id, "event": event.type},
    )
