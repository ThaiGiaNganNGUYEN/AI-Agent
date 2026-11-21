FROM python:3.12-slim AS build

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

FROM python:3.12-slim

RUN useradd --create-home --uid 10001 agentflow
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AGENTFLOW_DATABASE_URL=sqlite:////app/data/agentflow.db \
    AGENTFLOW_VECTOR_PATH=/app/data/vectors \
    AGENTFLOW_ENVIRONMENT=container
RUN mkdir -p /app/data && chown -R agentflow:agentflow /app
USER agentflow
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "agentflow.main:app", "--host", "0.0.0.0", "--port", "8000"]
