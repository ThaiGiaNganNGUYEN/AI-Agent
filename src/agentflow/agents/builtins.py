"""The four treasury agents.

Research gathers tools concurrently. Analyst calls the model with retrieved
policy. Compliance is a deterministic policy check. Executor publishes the memo.
"""

import asyncio
import json
import time

from agentflow.errors import AgentFlowError, ToolError
from agentflow.events import Event
from agentflow.monitoring.cost import price
from agentflow.monitoring.metrics import TOOL_CALLS
from agentflow.schemas import AgentSpec, CompletionRequest, RunContext, StepResult, StepSpec, ToolResult


class ResearchAgent:
    def __init__(self, spec: AgentSpec, platform) -> None:
        self.spec = spec
        self.platform = platform

    async def run(self, ctx: RunContext, step: StepSpec) -> StepResult:
        started = time.perf_counter()
        names = step.tools or self.spec.tools
        if not names:
            raise ToolError(f"Research step {step.id} has no tools")
        selected = []
        for name in names:
            tool = self.platform.tools.get(name)
            if tool is None:
                raise ToolError(f"Unknown tool {name}")
            selected.append(tool)

        async def call(tool) -> ToolResult:
            tool_started = time.perf_counter()
            try:
                result = await tool.invoke(ctx.inputs)
            except AgentFlowError as exc:
                result = ToolResult(name=tool.name, ok=False, error=str(exc))
            result.latency_ms = int((time.perf_counter() - tool_started) * 1000)
            TOOL_CALLS.labels(tool=tool.name, status="ok" if result.ok else "error").inc()
            self.platform.bus.publish(
                Event(
                    "tool.invoked",
                    ctx.run_id,
                    {"tool": tool.name, "ok": result.ok, "latency_ms": result.latency_ms, "error": result.error},
                )
            )
            return result

        results = await asyncio.gather(*(call(tool) for tool in selected))
        if not any(result.ok for result in results):
            detail = "; ".join(result.error or result.name for result in results)
            raise ToolError(f"All tools failed: {detail}")
        lines = []
        for result in results:
            if result.ok:
                lines.append(f"{result.name}: {json.dumps(result.data, default=str)}")
            else:
                lines.append(f"{result.name}: failed ({result.error})")
        return StepResult(
            step_id=step.id,
            agent=self.spec.name,
            output="\n".join(lines),
            data={"tools": {result.name: result.model_dump() for result in results}},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class AnalystAgent:
    def __init__(self, spec: AgentSpec, platform) -> None:
        self.spec = spec
        self.platform = platform

    async def run(self, ctx: RunContext, step: StepSpec) -> StepResult:
        started = time.perf_counter()
        question = str(ctx.inputs.get("question") or ctx.goal)
        tool_results: dict = {}
        for prior in ctx.prior.values():
            tool_results.update(prior.data.get("tools") or {})
        docs = self.platform.store.query(question, k=3)
        system, version = self.platform.prompts.render_system(self.spec.prompt or "analyst", question)
        messages = self.platform.context.build(
            system=system,
            goal=ctx.goal,
            question=question,
            docs=docs,
            tool_results=tool_results,
        )
        completion = await self.platform.provider.complete(
            CompletionRequest(
                messages=messages,
                model=self.platform.model_name(),
                temperature=self.spec.temperature,
                purpose="analyst",
            )
        )
        billed, avoided = _bill(completion)
        return StepResult(
            step_id=step.id,
            agent=self.spec.name,
            output=completion.text,
            data={
                "sources": [item.document.title for item in docs],
                "provider": completion.provider,
                "model": completion.model,
                "cached": completion.cached,
            },
            prompt_version=version,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost_usd=billed,
            avoided_usd=avoided,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class ComplianceAgent:
    def __init__(self, spec: AgentSpec, platform) -> None:
        self.spec = spec
        self.platform = platform

    async def run(self, ctx: RunContext, step: StepSpec) -> StepResult:
        started = time.perf_counter()
        text = "\n".join([ctx.goal, *[prior.output for prior in ctx.prior.values()]])
        report = self.platform.safety.review(text)
        self.platform.bus.publish(
            Event(
                "safety.checked",
                ctx.run_id,
                {"allowed": report.allowed, "issues": report.issues, "step": step.id},
            )
        )
        return StepResult(
            step_id=step.id,
            agent=self.spec.name,
            output=report.model_dump_json(),
            data={"safety": report.model_dump(), "allowed": report.allowed},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class ExecutorAgent:
    def __init__(self, spec: AgentSpec, platform) -> None:
        self.spec = spec
        self.platform = platform

    async def run(self, ctx: RunContext, step: StepSpec) -> StepResult:
        started = time.perf_counter()
        safety: dict = {}
        analysis = ""
        for prior in ctx.prior.values():
            if "safety" in prior.data:
                safety = prior.data["safety"]
            if prior.agent == "analyst":
                analysis = prior.output
        if safety and not safety.get("allowed", True):
            issues = "; ".join(safety.get("issues") or [])
            output = f"This request was blocked by compliance policy. {issues}".strip()
            return StepResult(
                step_id=step.id,
                agent=self.spec.name,
                output=output,
                data={"blocked": True, "safety": safety},
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        system, version = self.platform.prompts.render_system(self.spec.prompt or "executor", ctx.goal)
        messages = self.platform.context.build_memo(
            system=system,
            question=ctx.goal,
            analysis=analysis,
            safety=safety,
        )
        completion = await self.platform.provider.complete(
            CompletionRequest(
                messages=messages,
                model=self.platform.model_name(),
                temperature=self.spec.temperature,
                purpose="executor",
            )
        )
        output, redactions = self.platform.safety.redact(completion.text)
        billed, avoided = _bill(completion)
        if redactions:
            self.platform.bus.publish(Event("safety.redacted", ctx.run_id, {"count": redactions}))
        return StepResult(
            step_id=step.id,
            agent=self.spec.name,
            output=output,
            data={
                "provider": completion.provider,
                "model": completion.model,
                "cached": completion.cached,
                "redactions": redactions,
                "blocked": False,
            },
            prompt_version=version,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            cost_usd=billed,
            avoided_usd=avoided,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def _bill(completion) -> tuple[float, float]:
    list_price = price(completion.provider, completion.model, completion.tokens_in, completion.tokens_out)
    if completion.cached:
        return 0.0, list_price
    return completion.cost_usd, 0.0


AGENT_TYPES = {
    "research": ResearchAgent,
    "analyst": AnalystAgent,
    "compliance": ComplianceAgent,
    "executor": ExecutorAgent,
}


def build_agents(platform) -> dict:
    from agentflow.loading import load_yaml

    specs = [AgentSpec.model_validate(item) for item in load_yaml("agents.yaml")["agents"]]
    agents = {}
    for spec in specs:
        try:
            cls = AGENT_TYPES[spec.type]
        except KeyError as exc:
            raise ToolError(f"Unknown agent type {spec.type}") from exc
        agents[spec.name] = cls(spec, platform)
    return agents
