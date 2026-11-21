# AgentFlow Enterprise

Multi-agent workflow platform for a fictional treasury desk. It follows the
AgentFlow Enterprise project template: a core agent engine, a provider layer
with retrieval, container and cloud manifests, an authenticated API and
dashboard, and an evaluation harness.

The industry vertical is **corporate treasury**. Workflows cover an FX brief,
a World Bank indicator review, and policy questions against an internal
knowledge base.

The default model is a local template provider, so the app runs without an API
key. OpenAI and Anthropic are selectable at runtime once their keys are set.

## Run it locally

Python 3.11 or newer is required.

```powershell
cd "\path"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest --cov=agentflow --cov-fail-under=80
.\.venv\Scripts\python.exe -m uvicorn agentflow.main:app --reload
```

Open http://127.0.0.1:8000 for the desk and http://127.0.0.1:8000/docs for the API.

Demo sign-in, also printed on the login screen. Change these before any shared deployment:

| Role | Email | Password |
| --- | --- | --- |
| Admin | admin@agentflow.local | ChangeMe-Admin-1 |
| Operator | operator@agentflow.local | ChangeMe-Operator-1 |
| Viewer | viewer@agentflow.local | ChangeMe-Viewer-1 |

Viewers can read runs, knowledge, and reports. Operators can run workflows and evaluation. Admins can also change the model provider and read the audit log.

## What the desk does

- **Policy Q&A** answers from the packaged treasury notes. This path needs no network.
- **Currency risk brief** calls the Frankfurter exchange-rate API and the REST Countries API, then writes a memo.
- **Economic indicator review** calls the World Bank API and REST Countries.
- Every workflow runs research (when needed), analyst, compliance, and executor. Independent steps run concurrently. Tool and model calls retry. A failed tool does not hide a sibling tool that succeeded.
- Compliance blocks prompt-injection phrasing before a model call, and the executor redacts emails, card numbers, and national-ID patterns.
- Repeating an identical model call is served from cache and billed at zero. Spend, avoided spend, latency, and a linear latency forecast show on the overview.

Copy `.env.example` to `.env` to point at OpenAI or Anthropic:

```
AGENTFLOW_PROVIDER=openai
AGENTFLOW_OPENAI_API_KEY=sk-...
```

An admin can also switch `provider` from the Admin page. Workflow code does not change.

## Layout

| Template section | Where it lives |
| --- | --- |
| Agent framework, factory, and agent types | `src/agentflow/agents` |
| Policy routing | `src/agentflow/routing` and `resources/policies.yaml` |
| Orchestration, retries, concurrency | `src/agentflow/orchestration` |
| Three external APIs | `src/agentflow/tools` (Frankfurter, World Bank, REST Countries) |
| Provider abstraction and cache | `src/agentflow/providers`, `src/agentflow/cache` |
| Retrieval | `src/agentflow/rag` (hybrid BM25 + hashed embeddings, Chroma adapter) |
| Prompt versions and A/B assignment | `src/agentflow/prompts`, `resources/prompts.yaml` |
| Context budget | `src/agentflow/context` |
| JWT, roles, rate limit, audit | `src/agentflow/auth`, `src/agentflow/api` |
| Dashboard | `src/agentflow/static/index.html` |
| Cost, safety, forecast, alerts | `src/agentflow/monitoring` |
| Evaluation harness | `src/agentflow/evaluation` |
| Postgres via Compose, SQLite by default | `src/agentflow/persistence` |
| Image, Compose, Prometheus, Grafana | `Dockerfile`, `docker-compose.yml`, `infra/` |
| Kubernetes probes, limits, HPA | `infra/k8s` |
| Terraform network, registry, cluster | `infra/terraform` |
| CI | `.github/workflows/ci.yml` |

Design patterns used on purpose: factories for agents, tools, providers, and vector stores; strategy objects for providers, tools, stores, and routing rules; an observer bus for agent, tool, and workflow events. Those events are written as JSON logs.

## Docker Compose

This starts the API on Postgres, plus Redis, Prometheus, and Grafana. Grafana is at http://127.0.0.1:3000 (`admin` / `admin`). Prometheus is at http://127.0.0.1:9090.

```powershell
docker compose up --build
```

Set `AGENTFLOW_JWT_SECRET` in the environment before using Compose anywhere but a laptop.

## Kubernetes

```powershell
kubectl apply -f infra/k8s/secret.example.yaml
kubectl apply -k infra/k8s
```

The deployment expects an image named `agentflow:local`. Load the image into the cluster first (`docker build -t agentflow:local .`). Replace the JWT secret before applying it anywhere shared.

The HPA is capped at one replica because the default database file lives on the pod. Point `AGENTFLOW_DATABASE_URL` at Postgres, then raise `maxReplicas` in `infra/k8s/hpa.yaml`.

## Terraform

`infra/terraform` is a small AWS stack: a VPC with two public subnets, an ECR repository with scan-on-push, an ECS cluster with Container Insights, and a 30-day CloudWatch log group. Applying it creates billable resources. Review the variables, then:

```powershell
cd infra/terraform
terraform init
terraform plan
```

Change `availability_zones` if you change `aws_region`. This stack does not create a load balancer or an ECS service; the Kubernetes manifests are the application deployment.

## Backup and restore

Local SQLite and the vector index both live under `data/`.

1. Stop the API.
2. Copy `data/agentflow.db` and `data/vectors/`.
3. Restore those paths and start the API again.

On Compose, dump Postgres and keep the `agentflow-data` volume (it holds the vector index):

```powershell
docker compose exec db pg_dump -U agentflow agentflow > agentflow.sql
```

Restore with `psql` into an empty database, then start the API. Redis only stores the response cache, so losing it increases model spend until the cache warms again. It does not drop runs or audit rows.

Take the database dump on a schedule that matches how much run history you can afford to lose. Restoring is a stop, replace, start: the API creates its schema on startup if the tables are missing.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest --cov=agentflow --cov-report=term-missing --cov-fail-under=80
```

External HTTP calls are mocked. The policy workflow is what the evaluation harness runs end to end.
