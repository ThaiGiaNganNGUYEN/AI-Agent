"""Shared models for agents, workflows, and model calls."""

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str = "local-template"
    temperature: float = 0.2
    max_tokens: int = 800
    purpose: str = "analyst"


class Completion(BaseModel):
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0
    cached: bool = False


class ToolResult(BaseModel):
    name: str
    ok: bool
    data: dict = Field(default_factory=dict)
    error: str | None = None
    latency_ms: int = 0


class StepResult(BaseModel):
    step_id: str
    agent: str
    output: str
    data: dict = Field(default_factory=dict)
    prompt_version: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0
    avoided_usd: float = 0
    latency_ms: int = 0


class AgentSpec(BaseModel):
    name: str
    type: str
    description: str = ""
    tools: list[str] = Field(default_factory=list)
    prompt: str = ""
    temperature: float = 0.2


class InputField(BaseModel):
    name: str
    label: str
    type: str = "text"
    required: bool = False
    default: str = ""


class StepSpec(BaseModel):
    id: str
    agent: str
    depends_on: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    retries: int | None = None


class WorkflowSpec(BaseModel):
    name: str
    description: str
    domain: str = "finance"
    inputs: list[InputField] = Field(default_factory=list)
    steps: list[StepSpec]


class RoutingDecision(BaseModel):
    workflow: str
    rule: str
    reason: str


class Document(BaseModel):
    id: str
    title: str
    text: str
    source: str = ""


class ScoredDocument(BaseModel):
    document: Document
    score: float


class SafetyReport(BaseModel):
    allowed: bool
    issues: list[str] = Field(default_factory=list)


@dataclass
class RunContext:
    run_id: str
    goal: str
    inputs: dict
    actor: str
    prior: dict[str, StepResult] = field(default_factory=dict)
