"""Select a model provider from configuration."""

from agentflow.errors import AgentFlowError
from agentflow.providers.base import CachingProvider, LocalProvider
from agentflow.providers.remote import AnthropicProvider, OpenAIProvider


def build_provider(platform) -> CachingProvider:
    """Factory: the active name comes from settings or a runtime override."""

    name = str(platform.effective("provider"))
    settings = platform.settings
    if name == "local":
        inner = LocalProvider()
    elif name == "openai":
        inner = OpenAIProvider(settings, client=platform.http_client)
    elif name == "anthropic":
        inner = AnthropicProvider(settings, client=platform.http_client)
    else:
        raise AgentFlowError("provider must be local, openai, or anthropic")
    return CachingProvider(
        inner,
        platform.cache,
        lambda: platform.effective("cache_ttl_seconds"),
        platform.cache_stats,
    )
