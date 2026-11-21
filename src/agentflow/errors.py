"""Errors that the API can explain to a caller."""


class AgentFlowError(Exception):
    """Base error for expected failures."""

    def __init__(self, message: str, code: str = "error", *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class AuthenticationError(AgentFlowError):
    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message, "authentication", status_code=401)


class AuthorizationError(AgentFlowError):
    def __init__(self, message: str = "You do not have permission to do that") -> None:
        super().__init__(message, "authorization", status_code=403)


class NotFoundError(AgentFlowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "not_found", status_code=404)


class RateLimitError(AgentFlowError):
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, "rate_limit", status_code=429)


class ToolError(AgentFlowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "tool_error", status_code=502)


class ProviderError(AgentFlowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "provider_error", status_code=502)


class RoutingError(AgentFlowError):
    def __init__(self, message: str) -> None:
        super().__init__(message, "routing", status_code=422)


class WorkflowError(AgentFlowError):
    def __init__(self, message: str, partial: dict | None = None) -> None:
        super().__init__(message, "workflow", status_code=422)
        self.partial = partial or {}
