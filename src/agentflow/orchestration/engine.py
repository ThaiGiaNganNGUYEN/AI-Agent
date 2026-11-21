"""Run workflow steps, including independent steps concurrently."""

import asyncio
import time

from agentflow.errors import ToolError, ProviderError, WorkflowError
from agentflow.events import Event
from agentflow.schemas import RunContext, StepResult, StepSpec, WorkflowSpec

RETRYABLE = (ToolError, ProviderError, TimeoutError, asyncio.TimeoutError)


class Orchestrator:
    def __init__(self, agents: dict, bus, *, retries: int, timeout: float, retry_delay: float) -> None:
        self.agents = agents
        self.bus = bus
        self.retries = retries
        self.timeout = timeout
        self.retry_delay = retry_delay

    async def run(self, workflow: WorkflowSpec, ctx: RunContext) -> dict[str, StepResult]:
        self.bus.publish(Event("workflow.started", ctx.run_id, {"workflow": workflow.name}))
        pending = {step.id: step for step in workflow.steps}
        completed: dict[str, StepResult] = {}
        failure: Exception | None = None
        while pending and failure is None:
            ready = [step for step in pending.values() if all(dep in completed for dep in step.depends_on)]
            if not ready:
                failure = WorkflowError("Workflow has a cycle or a missing dependency", partial=completed)
                break
            results = await asyncio.gather(
                *(self._run_step(step, ctx) for step in ready),
                return_exceptions=True,
            )
            for step, result in zip(ready, results):
                del pending[step.id]
                if isinstance(result, Exception):
                    failure = result if isinstance(result, WorkflowError) else WorkflowError(str(result), partial=completed)
                    failure.partial = {**completed}
                else:
                    completed[result.step_id] = result
                    ctx.prior[result.step_id] = result
        if failure is not None:
            self.bus.publish(Event("workflow.failed", ctx.run_id, {"error": str(failure)}))
            if isinstance(failure, WorkflowError):
                failure.partial = completed or failure.partial
                raise failure
            raise WorkflowError(str(failure), partial=completed)
        self.bus.publish(
            Event("workflow.completed", ctx.run_id, {"workflow": workflow.name, "steps": list(completed)})
        )
        return completed

    async def _run_step(self, step: StepSpec, ctx: RunContext) -> StepResult:
        agent = self.agents.get(step.agent)
        if agent is None:
            raise WorkflowError(f"Unknown agent {step.agent}")
        attempts = step.retries or self.retries
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            self.bus.publish(
                Event("agent.started", ctx.run_id, {"agent": step.agent, "step": step.id, "attempt": attempt})
            )
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(agent.run(ctx, step), timeout=self.timeout)
            except RETRYABLE as exc:
                last = exc
                self.bus.publish(
                    Event(
                        "agent.retry",
                        ctx.run_id,
                        {"agent": step.agent, "step": step.id, "attempt": attempt, "error": str(exc)},
                    )
                )
                if attempt == attempts:
                    break
                await asyncio.sleep(self.retry_delay * attempt)
                continue
            elapsed = int((time.perf_counter() - started) * 1000)
            if result.latency_ms == 0:
                result.latency_ms = elapsed
            self.bus.publish(Event("agent.completed", ctx.run_id, {"agent": step.agent, "step": step.id}))
            return result
        self.bus.publish(Event("agent.failed", ctx.run_id, {"agent": step.agent, "step": step.id, "error": str(last)}))
        raise WorkflowError(f"Step {step.id} failed: {last}")
