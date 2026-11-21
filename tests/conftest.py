import pytest
from fastapi.testclient import TestClient

from agentflow.api.app import create_app
from agentflow.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{(tmp_path / 'app.db').as_posix()}",
        vector_path=str(tmp_path / "vectors"),
        jwt_secret="test-secret",
        password_iterations=1_000,
        http_retries=2,
        http_retry_delay=0,
        step_retries=2,
        step_retry_delay=0,
        prompt_seed=1,
        rate_limit_per_minute=10_000,
        environment="test",
        log_level="WARNING",
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def login(client: TestClient, email: str, password: str) -> dict:
    response = client.post("/api/v1/auth/token", data={"username": email, "password": password})
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
