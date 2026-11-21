"""Runtime configuration loaded from the environment."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Values that can be changed without editing code.

    Environment variables use the ``AGENTFLOW_`` prefix. A local ``.env`` file
    is read when it exists. Constructor arguments win over both, which keeps
    tests isolated.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTFLOW_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "AgentFlow Enterprise"
    environment: str = "local"
    database_url: str = "sqlite:///./data/agentflow.db"
    jwt_secret: str = "dev-only-change-me"
    jwt_ttl_minutes: int = 480
    password_iterations: int = 120_000
    provider: str = "local"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-haiku-latest"
    anthropic_base_url: str = "https://api.anthropic.com"
    vector_backend: str = "local"
    vector_path: str = "./data/vectors"
    redis_url: str = ""
    cache_ttl_seconds: int = 300
    log_level: str = "INFO"
    cors_origins: str = "*"
    rate_limit_per_minute: int = 120
    http_timeout_seconds: float = 12.0
    http_retries: int = 3
    http_retry_delay: float = 0.2
    step_retries: int = 2
    step_timeout_seconds: float = 30.0
    step_retry_delay: float = 0.05
    context_char_budget: int = 6000
    alert_error_rate: float = 0.3
    alert_latency_ms: int = 8000
    monthly_budget_usd: float = 25.0
    alert_webhook: str = ""
    prompt_seed: int | None = None
