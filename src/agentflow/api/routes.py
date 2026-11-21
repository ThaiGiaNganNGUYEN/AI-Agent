"""Versioned HTTP routes for the treasury platform."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from agentflow.auth.security import allows, issue_token, read_token
from agentflow.errors import AuthenticationError, AuthorizationError

bearer = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")
router = APIRouter()


class TaskIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    domain: str = Field(default="finance", min_length=1, max_length=40)
    inputs: dict = Field(default_factory=dict)


class RunIn(BaseModel):
    inputs: dict = Field(default_factory=dict)


class KnowledgeIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=20000)


class QueryIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=8)


class RuntimeIn(BaseModel):
    values: dict = Field(default_factory=dict)


class PromptIn(BaseModel):
    version: str = Field(min_length=1, max_length=40)


def principal(token: Annotated[str, Depends(bearer)], request: Request) -> dict:
    platform = request.app.state.platform
    payload = read_token(token, platform.settings.jwt_secret)
    user = platform.repo.get_user(payload.get("sub", ""))
    if user is None:
        raise AuthenticationError("Unknown user")
    return user


def require(permission: str):
    def checker(user: dict = Depends(principal)) -> dict:
        if not allows(user["role"], permission):
            raise AuthorizationError()
        return user

    return checker


@router.post("/auth/token", summary="Exchange an email and password for a JWT", tags=["auth"])
def issue_access_token(request: Request, form: OAuth2PasswordRequestForm = Depends()) -> dict:
    platform = request.app.state.platform
    user = platform.repo.authenticate(form.username.strip().lower(), form.password)
    if user is None:
        platform.repo.audit(form.username, "auth.failure", "invalid credentials")
        raise AuthenticationError("Email or password is incorrect")
    platform.repo.audit(user["email"], "auth.success", user["role"])
    return {
        "access_token": issue_token(user, platform.settings.jwt_secret, platform.settings.jwt_ttl_minutes),
        "token_type": "bearer",
        "role": user["role"],
        "name": user["name"],
        "email": user["email"],
    }


@router.get("/me", summary="Current user", tags=["auth"])
def me(user: dict = Depends(principal)) -> dict:
    return user


@router.get("/health", summary="Process health", tags=["meta"])
def health(request: Request) -> dict:
    platform = request.app.state.platform
    database = "ok"
    try:
        platform.repo.ping()
    except Exception:
        database = "error"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "provider": platform.effective("provider"),
        "model": platform.model_name(),
        "vectors": platform.store.count(),
        "environment": platform.settings.environment,
    }


@router.get("/agents", summary="Configured agents", tags=["runtime"])
def agents(request: Request, user: dict = Depends(require("runs:read"))) -> dict:
    del user
    return {"agents": request.app.state.platform.list_agents()}


@router.get("/workflows", summary="Workflow catalogue", tags=["runtime"])
def workflows(request: Request, user: dict = Depends(require("runs:read"))) -> dict:
    del user
    return {"workflows": request.app.state.platform.list_workflows()}


@router.post("/tasks", summary="Route a message and run the selected workflow", tags=["runs"])
async def create_task(body: TaskIn, request: Request, user: dict = Depends(require("runs:write"))) -> dict:
    return await request.app.state.platform.route_and_execute(body.message, body.domain, body.inputs, user["email"])


@router.post("/workflows/{name}/runs", summary="Run a workflow by name", tags=["runs"])
async def run_workflow(name: str, body: RunIn, request: Request, user: dict = Depends(require("runs:write"))) -> dict:
    return await request.app.state.platform.execute(name, body.inputs, user["email"])


@router.get("/runs", summary="Recent workflow runs", tags=["runs"])
def list_runs(
    request: Request,
    user: dict = Depends(require("runs:read")),
    workflow: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> dict:
    del user
    return {"runs": request.app.state.platform.repo.list_runs(limit=max(1, min(limit, 100)), workflow=workflow, status=status)}


@router.get("/runs/{run_id}", summary="One workflow run", tags=["runs"])
def get_run(run_id: str, request: Request, user: dict = Depends(require("runs:read"))) -> dict:
    del user
    record = request.app.state.platform.repo.get_run(run_id)
    if record is None:
        from agentflow.errors import NotFoundError

        raise NotFoundError(f"Unknown run {run_id}")
    return record


@router.get("/knowledge", summary="Knowledge documents", tags=["knowledge"])
def list_knowledge(request: Request, user: dict = Depends(require("knowledge:read"))) -> dict:
    del user
    documents = request.app.state.platform.store.list_documents()
    return {
        "documents": [
            {"id": doc.id, "title": doc.title, "source": doc.source, "excerpt": doc.text[:280]} for doc in documents
        ]
    }


@router.post("/knowledge", summary="Add a knowledge document", tags=["knowledge"])
def add_knowledge(body: KnowledgeIn, request: Request, user: dict = Depends(require("knowledge:write"))) -> dict:
    return request.app.state.platform.add_document(body.title, body.text, user["email"])


@router.post("/knowledge/query", summary="Retrieve knowledge", tags=["knowledge"])
def query_knowledge(body: QueryIn, request: Request, user: dict = Depends(require("knowledge:read"))) -> dict:
    del user
    return {"matches": request.app.state.platform.query_knowledge(body.query, body.k)}


@router.get("/prompts", summary="Prompt versions", tags=["prompts"])
def prompts(request: Request, user: dict = Depends(require("runs:read"))) -> dict:
    del user
    return {"prompts": request.app.state.platform.prompts.describe()}


@router.post("/prompts/{name}/activate", summary="Pin a prompt version", tags=["prompts"])
def activate_prompt(name: str, body: PromptIn, request: Request, user: dict = Depends(require("prompts:write"))) -> dict:
    platform = request.app.state.platform
    platform.prompts.activate(name, body.version)
    platform.repo.audit(user["email"], "prompt.activate", f"{name}:{body.version}")
    return {"prompts": platform.prompts.describe()}


@router.get("/costs", summary="Model spend", tags=["monitoring"])
def costs(request: Request, user: dict = Depends(require("metrics:read"))) -> dict:
    del user
    return request.app.state.platform.repo.cost_summary()


@router.get("/metrics/summary", summary="Operational summary", tags=["monitoring"])
def metrics_summary(request: Request, user: dict = Depends(require("metrics:read"))) -> dict:
    del user
    return request.app.state.platform.summary()


@router.get("/alerts", summary="Open alerts", tags=["monitoring"])
def alerts(request: Request, user: dict = Depends(require("metrics:read"))) -> dict:
    del user
    platform = request.app.state.platform
    return {"alerts": platform.refresh_alerts()}


@router.get("/reports/summary", summary="Stakeholder report", tags=["monitoring"])
def report(request: Request, user: dict = Depends(require("reports:read"))) -> dict:
    del user
    return {"markdown": request.app.state.platform.report()}


@router.post("/evaluate", summary="Run the evaluation harness", tags=["monitoring"])
async def evaluate(request: Request, user: dict = Depends(require("evaluate:run"))) -> dict:
    return await request.app.state.platform.harness.run(user["email"])


@router.get("/audit", summary="Audit log", tags=["admin"])
def audit(request: Request, user: dict = Depends(require("runtime:admin"))) -> dict:
    del user
    return {"entries": request.app.state.platform.repo.list_audit()}


@router.get("/runtime", summary="Runtime settings", tags=["admin"])
def get_runtime(request: Request, user: dict = Depends(require("metrics:read"))) -> dict:
    del user
    return request.app.state.platform.runtime_view()


@router.patch("/runtime", summary="Update runtime settings", tags=["admin"])
def patch_runtime(body: RuntimeIn, request: Request, user: dict = Depends(require("runtime:admin"))) -> dict:
    return request.app.state.platform.update_runtime(body.values, user["email"])
