"""Fixed-window rate limit for API routes."""

import hashlib
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = [stamp for stamp in self.hits.get(key, []) if now - stamp < 60]
        if len(window) >= self.per_minute:
            self.hits[key] = window
            return False
        window.append(now)
        self.hits[key] = window
        return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: RateLimiter) -> None:
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            raw = request.headers.get("authorization") or (request.client.host if request.client else "unknown")
            key = hashlib.sha256(raw.encode()).hexdigest()
            if not self.limiter.allow(key):
                return JSONResponse(
                    {"error": {"code": "rate_limit", "message": "Rate limit exceeded. Try again in a minute."}},
                    status_code=429,
                )
        return await call_next(request)
