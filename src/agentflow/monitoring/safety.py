"""Rule-based safety and privacy checks.

The compliance agent treats this monitor as the source of truth. Patterns are
intentionally narrow so ordinary treasury language is not blocked.
"""

import re

from agentflow.schemas import SafetyReport

INJECTION = re.compile(
    r"(ignore|disregard|bypass).{0,40}(instruction|policy|rule)"
    r"|reveal.{0,30}(secret|password|jwt)"
    r"|system prompt",
    re.IGNORECASE,
)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
CARD = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


class SafetyMonitor:
    def review(self, text: str) -> SafetyReport:
        issues: list[str] = []
        if INJECTION.search(text or ""):
            issues.append("Prompt injection or secret-exfiltration pattern detected")
        lowered = (text or "").lower()
        if "jwt_secret" in lowered or "api_key=" in lowered or "begin private key" in lowered:
            issues.append("Secret material referenced")
        return SafetyReport(allowed=not issues, issues=issues)

    def redact(self, text: str) -> tuple[str, int]:
        redactions = 0

        def replace(pattern: re.Pattern[str], repl: str, value: str) -> str:
            nonlocal redactions
            updated, count = pattern.subn(repl, value)
            redactions += count
            return updated

        text = replace(EMAIL, "[redacted-email]", text)
        text = replace(CARD, "[redacted-number]", text)
        text = replace(SSN, "[redacted-id]", text)
        return text, redactions
