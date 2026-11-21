import asyncio
import time

import pytest

from agentflow.errors import ToolError, WorkflowError
from agentflow.events import EventBus
from agentflow.orchestration.engine import Orchestrator
from agentflow.schemas import AgentSpec, RunContext, StepResult, StepSpec, ToolResult, WorkflowSpec
from agentflow.agents.builtins import ResearchAgent


def _orchestrator(agent, retries=2, timeout=1):
    return Orchestrator({"analyst": agent}, EventBus(), retries=retries, timeout=timeout, retry_delay=0)


def _workflow(steps):
    return WorkflowSpec(name="demo", description="demo", steps=steps)


def _ctx():
    return RunContext(run_id="run", goal="goal", inputs={}, actor="tester")


@pytest.mark.asyncio
async def test_independent_steps_run_concurrently():
    spans = []

    class Slow:
        async def run(self, ctx, step):
            start = time.perf_counter()
            await asyncio.sleep(0.15)
            spans.append((start, time.perf_counter()))
            return StepResult(step_id=step.id, agent="analyst", output=step.id)

    results = await _orchestrator(Slow()).run(
        _workflow([StepSpec(id="a", agent="analyst"), StepSpec(id="b", agent="analyst")]),
        _ctx(),
    )
    assert set(results) == {"a", "b"}
    assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]


@pytest.mark.asyncio
async def test_retry_then_success_and_hard_failure():
    class Flaky:
        def __init__(self):
            self.calls = 0

        async def run(self, ctx, step):
            self.calls += 1
            if self.calls < 2:
                raise ToolError("temporary")
            return StepResult(step_id=step.id, agent="analyst", output="ok")

    flaky = Flaky()
    results = await _orchestrator(flaky).run(_workflow([StepSpec(id="only", agent="analyst")]), _ctx())
    assert results["only"].output == "ok"
    assert flaky.calls == 2

    class Broken:
        async def run(self, ctx, step):
            raise ToolError("nope")

    with pytest.raises(WorkflowError):
        await _orchestrator(Broken(), retries=1).run(_workflow([StepSpec(id="only", agent="analyst")]), _ctx())


@pytest.mark.asyncio
async def test_timeout_cycle_and_unknown_agent():
    class Sleepy:
        async def run(self, ctx, step):
            await asyncio.sleep(1)
            return StepResult(step_id=step.id, agent="analyst", output="late")

    with pytest.raises(WorkflowError):
        await _orchestrator(Sleepy(), retries=1, timeout=0.05).run(
            _workflow([StepSpec(id="only", agent="analyst")]),
            _ctx(),
        )

    with pytest.raises(WorkflowError, match="cycle"):
        await _orchestrator(Sleepy(), retries=1).run(
            _workflow(
                [
                    StepSpec(id="a", agent="analyst", depends_on=["b"]),
                    StepSpec(id="b", agent="analyst", depends_on=["a"]),
                ]
            ),
            _ctx(),
        )

    with pytest.raises(WorkflowError, match="Unknown agent"):
        await _orchestrator(Sleepy(), retries=1).run(_workflow([StepSpec(id="only", agent="missing")]), _ctx())


@pytest.mark.asyncio
async def test_parallel_failure_keeps_the_successful_sibling():
    class Mixed:
        async def run(self, ctx, step):
            if step.id == "bad":
                raise ToolError("down")
            return StepResult(step_id=step.id, agent="analyst", output="kept")

    with pytest.raises(WorkflowError) as caught:
        await _orchestrator(Mixed(), retries=1).run(
            _workflow([StepSpec(id="bad", agent="analyst"), StepSpec(id="good", agent="analyst")]),
            _ctx(),
        )
    assert "good" in caught.value.partial
    assert caught.value.partial["good"].output == "kept"


@pytest.mark.asyncio
async def test_research_agent_degrades_when_one_tool_fails():
    class Down:
        name = "fx_rates"

        async def invoke(self, inputs):
            raise ToolError("fx down")

    class Up:
        name = "country_info"

        async def invoke(self, inputs):
            return ToolResult(name=self.name, ok=True, data={"code": "DE"})

    class Platform:
        tools = {"fx_rates": Down(), "country_info": Up()}
        bus = EventBus()

    spec = AgentSpec(name="research", type="research", tools=["fx_rates", "country_info"])
    result = await ResearchAgent(spec, Platform()).run(
        _ctx(),
        StepSpec(id="research", agent="research", tools=["fx_rates", "country_info"]),
    )
    assert result.data["tools"]["country_info"]["ok"] is True
    assert result.data["tools"]["fx_rates"]["ok"] is False

    class NoneUp:
        name = "country_info"

        async def invoke(self, inputs):
            raise ToolError("also down")

    Platform.tools["country_info"] = NoneUp()
    with pytest.raises(ToolError, match="All tools failed"):
        await ResearchAgent(spec, Platform()).run(
            _ctx(),
            StepSpec(id="research", agent="research", tools=["fx_rates", "country_info"]),
        )
