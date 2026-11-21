"""HTTP adapters for OpenAI and Anthropic chat APIs."""

import asyncio

import httpx

from agentflow.config import Settings
from agentflow.errors import ProviderError
from agentflow.providers.base import LLMProvider, estimate_tokens
from agentflow.monitoring.cost import price
from agentflow.schemas import Completion, CompletionRequest, Message

RETRYABLE = {429, 500, 502, 503, 504}


class _HttpProvider(LLMProvider):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.client = client

    async def _post(self, url: str, headers: dict, body: dict) -> dict:
        attempts = max(1, self.settings.http_retries)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                if self.client is not None:
                    response = await self.client.post(url, headers=headers, json=body)
                else:
                    async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                        response = await client.post(url, headers=headers, json=body)
                if response.status_code in RETRYABLE and attempt < attempts:
                    last_error = ProviderError(f"{self.name} returned {response.status_code}")
                    await asyncio.sleep(self.settings.http_retry_delay * attempt)
                    continue
                if response.status_code >= 400:
                    raise ProviderError(f"{self.name} returned {response.status_code}: {response.text[:300]}")
                return response.json()
            except ProviderError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt == attempts:
                    break
                await asyncio.sleep(self.settings.http_retry_delay * attempt)
        raise ProviderError(f"{self.name} failed after {attempts} attempts: {last_error}")


class OpenAIProvider(_HttpProvider):
    name = "openai"

    async def complete(self, request: CompletionRequest) -> Completion:
        if not self.settings.openai_api_key:
            raise ProviderError(
                "OpenAI API key is not configured. Set AGENTFLOW_OPENAI_API_KEY or switch the provider to local."
            )
        payload = await self._post(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"},
            {
                "model": request.model or self.settings.openai_model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "messages": [message.model_dump() for message in request.messages],
            },
        )
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI returned an unexpected payload") from exc
        usage = payload.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or estimate_tokens(_join(request.messages)))
        tokens_out = int(usage.get("completion_tokens") or estimate_tokens(text))
        model = request.model or self.settings.openai_model
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=price(self.name, model, tokens_in, tokens_out),
        )


class AnthropicProvider(_HttpProvider):
    name = "anthropic"

    async def complete(self, request: CompletionRequest) -> Completion:
        if not self.settings.anthropic_api_key:
            raise ProviderError(
                "Anthropic API key is not configured. Set AGENTFLOW_ANTHROPIC_API_KEY or switch the provider to local."
            )
        system = "\n\n".join(message.content for message in request.messages if message.role == "system")
        messages = [
            {"role": message.role, "content": message.content}
            for message in request.messages
            if message.role in {"user", "assistant"}
        ]
        if not messages:
            messages = [{"role": "user", "content": system or "Respond briefly."}]
            system = ""
        model = request.model or self.settings.anthropic_model
        payload = await self._post(
            f"{self.settings.anthropic_base_url.rstrip('/')}/v1/messages",
            {
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            {
                "model": model,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "system": system,
                "messages": messages,
            },
        )
        blocks = payload.get("content") if isinstance(payload, dict) else None
        if not isinstance(blocks, list):
            raise ProviderError("Anthropic returned an unexpected payload")
        text = "".join(block.get("text", "") for block in blocks if isinstance(block, dict))
        usage = payload.get("usage") or {}
        tokens_in = int(usage.get("input_tokens") or estimate_tokens(_join(request.messages)))
        tokens_out = int(usage.get("output_tokens") or estimate_tokens(text))
        return Completion(
            text=text,
            model=model,
            provider=self.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=price(self.name, model, tokens_in, tokens_out),
        )


def _join(messages: list[Message]) -> str:
    return "\n".join(message.content for message in messages)
