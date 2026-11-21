"""Provider interface, local template model, and response cache wrapper."""

from abc import ABC, abstractmethod
import hashlib
import re

from agentflow.context.builder import extract_section
from agentflow.schemas import Completion, CompletionRequest


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMProvider(ABC):
    name = "provider"

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> Completion:
        raise NotImplementedError


class LocalProvider(LLMProvider):
    """Deterministic writer used when no external model key is configured.

    It only restates evidence already placed in the prompt, so workflows stay
    runnable and testable offline. It does not invent market figures.
    """

    name = "local"

    async def complete(self, request: CompletionRequest) -> Completion:
        user = "\n\n".join(message.content for message in request.messages if message.role == "user")
        if request.purpose == "executor":
            text = self._memo(user)
        else:
            text = self._analysis(user)
        prompt = "\n".join(message.content for message in request.messages)
        return Completion(
            text=text,
            model=request.model or "local-template",
            provider=self.name,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(text),
            cost_usd=0,
        )

    def _analysis(self, user: str) -> str:
        question = extract_section(user, "QUESTION") or extract_section(user, "GOAL") or "the request"
        context = extract_section(user, "CONTEXT")
        tools = extract_section(user, "TOOL_RESULTS")
        parts = [f"Assessment for: {question}"]
        if context:
            parts.append("Evidence from company knowledge:")
            parts.append(context[:2000])
        if tools and tools not in {"{}", "[]"}:
            parts.append("External data:")
            parts.append(tools[:2000])
        if not context and (not tools or tools in {"{}", "[]"}):
            parts.append("The knowledge base and tools did not return supporting evidence.")
        parts.append(
            "Recommendation: follow the cited policy, and confirm live figures in the source system before acting."
        )
        return "\n\n".join(parts)

    def _memo(self, user: str) -> str:
        question = extract_section(user, "QUESTION") or "the request"
        analysis = extract_section(user, "ANALYSIS") or "No analysis was produced."
        analysis = re.sub(r"\n{3,}", "\n\n", analysis).strip()
        return (
            "Treasury memo\n\n"
            f"Question: {question}\n\n"
            f"{analysis}\n\n"
            "Compliance review is recorded in the workflow trace. "
            "Confirm any live figures in the source system before booking a trade."
        )


class CachingProvider(LLMProvider):
    """Caches identical completions and bills a cache hit at zero."""

    def __init__(self, inner: LLMProvider, cache, ttl_getter, stats) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache = cache
        self.ttl_getter = ttl_getter
        self.stats = stats

    async def complete(self, request: CompletionRequest) -> Completion:
        key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.hits += 1
            self.stats.hit_counter()
            cached.cached = True
            cached.cost_usd = 0
            return cached
        self.stats.misses += 1
        self.stats.miss_counter()
        result = await self.inner.complete(request)
        self.cache.set(key, result.model_dump(), int(self.ttl_getter()))
        return result
