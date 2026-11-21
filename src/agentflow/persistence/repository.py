"""SQLAlchemy repository. SQLite locally, PostgreSQL when configured."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path

from sqlalchemy import Float, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from agentflow.auth.security import DEMO_USERS, hash_password, verify_password


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50))
    password_hash: Mapped[str] = mapped_column(String(255))


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workflow: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    actor: Mapped[str] = mapped_column(String(255))
    rule: Mapped[str] = mapped_column(String(100), default="")
    input_json: Mapped[str] = mapped_column(Text, default="{}")
    output_json: Mapped[str] = mapped_column(Text, default="{}")
    error: Mapped[str] = mapped_column(Text, default="")
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[str] = mapped_column(String(40))
    finished_at: Mapped[str] = mapped_column(String(40), default="")


class AuditRow(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[str] = mapped_column(String(40))
    actor: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(100))
    detail: Mapped[str] = mapped_column(Text, default="")


class CostRow(Base):
    __tablename__ = "costs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[str] = mapped_column(String(40))
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0)
    avoided_usd: Mapped[float] = mapped_column(Float, default=0)
    cached: Mapped[int] = mapped_column(Integer, default=0)


class Repository:
    def __init__(self, url: str, password_iterations: int) -> None:
        self.url = url
        _prepare_sqlite(url)
        connect_args = {"check_same_thread": False, "timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self.seed_users(password_iterations)

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def session(self):
        db: Session = self.sessions()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def seed_users(self, iterations: int) -> None:
        with self.session() as db:
            for index, (email, name, role, password) in enumerate(DEMO_USERS, start=1):
                existing = db.scalar(select(UserRow).where(UserRow.email == email))
                if existing is not None:
                    continue
                db.add(
                    UserRow(
                        id=f"user-{index}",
                        email=email,
                        name=name,
                        role=role,
                        password_hash=hash_password(password, iterations),
                    )
                )

    def authenticate(self, email: str, password: str) -> dict | None:
        with self.session() as db:
            row = db.scalar(select(UserRow).where(UserRow.email == email))
            if row is None or not verify_password(password, row.password_hash):
                return None
            return _user_dict(row)

    def get_user(self, email: str) -> dict | None:
        with self.session() as db:
            row = db.scalar(select(UserRow).where(UserRow.email == email))
            return None if row is None else _user_dict(row)

    def create_run(self, run_id: str, workflow: str, actor: str, rule: str, inputs: dict) -> dict:
        with self.session() as db:
            row = RunRow(
                id=run_id,
                workflow=workflow,
                status="running",
                actor=actor,
                rule=rule,
                input_json=json.dumps(inputs, default=str),
                created_at=utcnow(),
            )
            db.add(row)
            db.flush()
            return _run_dict(row)

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        output: dict,
        error: str,
        cost_usd: float,
        latency_ms: int,
    ) -> dict:
        with self.session() as db:
            row = db.get(RunRow, run_id)
            if row is None:
                raise KeyError(run_id)
            row.status = status
            row.output_json = json.dumps(output)
            row.error = error
            row.cost_usd = cost_usd
            row.latency_ms = latency_ms
            row.finished_at = utcnow()
            db.flush()
            return _run_dict(row)

    def get_run(self, run_id: str) -> dict | None:
        with self.session() as db:
            row = db.get(RunRow, run_id)
            return None if row is None else _run_dict(row)

    def list_runs(self, limit: int = 20, workflow: str | None = None, status: str | None = None) -> list[dict]:
        with self.session() as db:
            query = select(RunRow)
            if workflow:
                query = query.where(RunRow.workflow == workflow)
            if status:
                query = query.where(RunRow.status == status)
            query = query.order_by(RunRow.created_at.desc()).limit(limit)
            return [_run_dict(row) for row in db.scalars(query)]

    def stats(self) -> dict:
        with self.session() as db:
            rows = list(db.scalars(select(RunRow)))
            total = len(rows)
            failed = sum(1 for row in rows if row.status == "failed")
            blocked = sum(1 for row in rows if row.status == "blocked")
            succeeded = sum(1 for row in rows if row.status == "succeeded")
            latencies = [float(row.latency_ms) for row in rows if row.status == "succeeded"]
            average = sum(latencies) / len(latencies) if latencies else 0
            return {
                "total": total,
                "failed": failed,
                "blocked": blocked,
                "succeeded": succeeded,
                "samples": total,
                "error_rate": failed / total if total else 0,
                "avg_latency_ms": average,
                "latencies": latencies[-20:],
            }

    def audit(self, actor: str, action: str, detail: str = "") -> None:
        with self.session() as db:
            db.add(AuditRow(at=utcnow(), actor=actor, action=action, detail=detail[:2000]))

    def list_audit(self, limit: int = 50) -> list[dict]:
        with self.session() as db:
            rows = db.scalars(select(AuditRow).order_by(AuditRow.id.desc()).limit(limit))
            return [
                {"id": row.id, "at": row.at, "actor": row.actor, "action": row.action, "detail": row.detail}
                for row in rows
            ]

    def add_cost(
        self,
        *,
        run_id: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        avoided_usd: float,
        cached: bool,
    ) -> None:
        with self.session() as db:
            db.add(
                CostRow(
                    at=utcnow(),
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost_usd,
                    avoided_usd=avoided_usd,
                    cached=1 if cached else 0,
                )
            )

    def cost_summary(self) -> dict:
        with self.session() as db:
            rows = list(db.scalars(select(CostRow).order_by(CostRow.id.desc()).limit(500)))
            by_provider: dict[str, dict] = {}
            for row in rows:
                bucket = by_provider.setdefault(
                    row.provider,
                    {"cost_usd": 0.0, "avoided_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "calls": 0, "cached": 0},
                )
                bucket["cost_usd"] += row.cost_usd
                bucket["avoided_usd"] += row.avoided_usd
                bucket["tokens_in"] += row.tokens_in
                bucket["tokens_out"] += row.tokens_out
                bucket["calls"] += 1
                bucket["cached"] += row.cached
            entries = [
                {
                    "at": row.at,
                    "run_id": row.run_id,
                    "provider": row.provider,
                    "model": row.model,
                    "tokens_in": row.tokens_in,
                    "tokens_out": row.tokens_out,
                    "cost_usd": row.cost_usd,
                    "avoided_usd": row.avoided_usd,
                    "cached": bool(row.cached),
                }
                for row in rows[:50]
            ]
            return {
                "total_usd": sum(row.cost_usd for row in rows),
                "avoided_usd": sum(row.avoided_usd for row in rows),
                "by_provider": by_provider,
                "entries": entries,
            }

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")


def _prepare_sqlite(url: str) -> None:
    if not url.startswith("sqlite"):
        return
    raw = url.removeprefix("sqlite:///")
    if raw in {":memory:", ""} or raw.startswith("file:"):
        return
    path = Path(raw)
    if path.parent and str(path.parent) not in {"", "."}:
        path.parent.mkdir(parents=True, exist_ok=True)


def _user_dict(row: UserRow) -> dict:
    return {"id": row.id, "email": row.email, "name": row.name, "role": row.role}


def _run_dict(row: RunRow) -> dict:
    return {
        "id": row.id,
        "workflow": row.workflow,
        "status": row.status,
        "actor": row.actor,
        "rule": row.rule,
        "inputs": json.loads(row.input_json or "{}"),
        "output": json.loads(row.output_json or "{}"),
        "error": row.error,
        "cost_usd": row.cost_usd,
        "latency_ms": row.latency_ms,
        "created_at": row.created_at,
        "finished_at": row.finished_at,
    }
