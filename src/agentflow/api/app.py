"""FastAPI application factory."""

from contextlib import asynccontextmanager
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from agentflow.config import Settings
from agentflow.errors import AgentFlowError
from agentflow.logging_config import configure_logging
from agentflow.monitoring.metrics import HTTP_LATENCY, HTTP_REQUESTS, render_metrics
from agentflow.api.limiting import RateLimiter, RateLimitMiddleware
from agentflow.api.routes import router
from agentflow.platform import Platform

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
DESCRIPTION = """
AgentFlow Enterprise routes treasury work across research, analyst, compliance,
and executor agents.

The default model provider is a local template, so the platform runs without an
API key. Set `AGENTFLOW_PROVIDER` to `openai` or `anthropic` to switch models
without changing workflow code. Interactive docs below call the same API as the
dashboard.
"""


class AccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        HTTP_REQUESTS.labels(method=request.method, path=path, status=str(response.status_code)).inc()
        HTTP_LATENCY.labels(path=path).observe(time.perf_counter() - started)
        return response


def create_app(settings: Settings | None = None, tools: dict | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    platform = Platform(settings, tools=tools)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.platform.close()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "auth", "description": "JWT issuance and the current user."},
            {"name": "runs", "description": "Policy routing and workflow execution."},
            {"name": "knowledge", "description": "Retrieval corpus."},
            {"name": "monitoring", "description": "Cost, evaluation, alerts, and reports."},
            {"name": "admin", "description": "Runtime configuration and the audit log."},
        ],
    )
    app.state.platform = platform
    origins = ["*"] if settings.cors_origins.strip() == "*" else [item.strip() for item in settings.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(AccessMiddleware)
    app.add_middleware(RateLimitMiddleware, limiter=RateLimiter(settings.rate_limit_per_minute))
    app.include_router(router, prefix="/api/v1")

    @app.exception_handler(AgentFlowError)
    async def handle_agentflow_error(_request: Request, exc: AgentFlowError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            raise exc
        logging_exception()
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal", "message": "Unexpected server error"}},
        )

    @app.get("/health", tags=["meta"], summary="Liveness and readiness")
    def root_health() -> dict:
        return health_payload(platform)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        body, media_type = render_metrics()
        return Response(content=body, media_type=media_type)

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def health_payload(platform: Platform) -> dict:
    database = "ok"
    try:
        platform.repo.ping()
    except Exception:
        database = "error"
    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "provider": platform.effective("provider"),
        "vectors": platform.store.count(),
    }


def logging_exception() -> None:
    import logging

    logging.getLogger("agentflow.api").exception("unhandled error")
