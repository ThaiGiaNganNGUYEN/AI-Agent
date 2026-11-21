"""Password hashing, JWT, and role checks. Standard library only."""

import base64
import hashlib
import hmac
import json
import secrets
import time

from agentflow.errors import AuthenticationError

ROLES: dict[str, set[str]] = {
    "viewer": {"runs:read", "knowledge:read", "metrics:read", "reports:read"},
    "operator": {
        "runs:read",
        "runs:write",
        "knowledge:read",
        "knowledge:write",
        "metrics:read",
        "reports:read",
        "evaluate:run",
        "prompts:write",
    },
    "admin": {"*"},
}

DEMO_USERS = (
    ("admin@agentflow.local", "Platform Admin", "admin", "ChangeMe-Admin-1"),
    ("operator@agentflow.local", "Treasury Operator", "operator", "ChangeMe-Operator-1"),
    ("viewer@agentflow.local", "Read Only", "viewer", "ChangeMe-Viewer-1"),
)


def allows(role: str, permission: str) -> bool:
    granted = ROLES.get(role, set())
    return "*" in granted or permission in granted


def hash_password(password: str, iterations: int) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"{iterations}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        iterations, salt_hex, derived_hex = stored.split("$")
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode(),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived.hex(), derived_hex)


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def issue_token(user: dict, secret: str, ttl_minutes: int) -> str:
    header = _encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = {
        "sub": user["email"],
        "name": user["name"],
        "role": user["role"],
        "exp": int(time.time()) + ttl_minutes * 60,
    }
    body = _encode(json.dumps(payload, separators=(",", ":")).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_encode(signature)}"


def read_token(token: str, secret: str) -> dict:
    try:
        header, body, signature = token.split(".")
    except ValueError as exc:
        raise AuthenticationError("Malformed token") from exc
    expected = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(_encode(expected), signature):
        raise AuthenticationError("Invalid token")
    try:
        payload = json.loads(_decode(body))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthenticationError("Malformed token") from exc
    if int(payload.get("exp", 0)) <= time.time():
        raise AuthenticationError("Token expired")
    return payload
