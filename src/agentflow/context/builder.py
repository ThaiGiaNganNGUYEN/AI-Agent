"""Assemble a bounded prompt from policy excerpts and tool results."""

import json
import re

from agentflow.schemas import Message, ScoredDocument

_SECTION = re.compile(r"<<<([A-Z_]+)>>>\n(.*?)(?=\n<<<[A-Z_]+>>>|\Z)", re.DOTALL)


def extract_section(text: str, name: str) -> str:
    for label, body in _SECTION.findall(text or ""):
        if label == name:
            return body.strip()
    return ""


class ContextBuilder:
    """Fits retrieved text into a character budget, highest score first."""

    def __init__(self, max_chars: int = 6000) -> None:
        self.max_chars = max_chars

    def build(
        self,
        *,
        system: str,
        goal: str,
        question: str,
        docs: list[ScoredDocument],
        tool_results: dict,
    ) -> list[Message]:
        excerpts: list[str] = []
        used = 0
        for item in docs:
            chunk = f"[{item.document.title}] {item.document.text.strip()}"
            remaining = self.max_chars - used
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining].rstrip()
            excerpts.append(chunk)
            used += len(chunk)
        tools = json.dumps(tool_results, indent=2, default=str)
        if len(tools) > 4000:
            tools = tools[:4000] + "\n…"
        user = (
            f"<<<GOAL>>>\n{goal}\n"
            f"<<<QUESTION>>>\n{question}\n"
            f"<<<CONTEXT>>>\n{chr(10).join(excerpts)}\n"
            f"<<<TOOL_RESULTS>>>\n{tools}\n"
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def build_memo(self, *, system: str, question: str, analysis: str, safety: dict) -> list[Message]:
        user = (
            f"<<<QUESTION>>>\n{question}\n"
            f"<<<ANALYSIS>>>\n{analysis}\n"
            f"<<<COMPLIANCE>>>\n{json.dumps(safety)}\n"
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]
