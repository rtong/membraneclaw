"""Explicit registration; gateway code has no provider-name branches."""
from .types import ConfigurationError


class ProviderRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, provider, factory):
        if provider in self._factories:
            raise ConfigurationError("provider already registered")
        self._factories[provider] = factory

    def create_for_model(self, provider, model):
        from .types import ModelRoute
        factory = self._factories.get(provider)
        if factory is None:
            raise ConfigurationError("provider is not registered")
        adapter = factory()
        adapter.capabilities(model)
        return adapter, ModelRoute(provider, model, adapter.api_family)

    def create(self, route):
        factory = self._factories.get(route.provider)
        if factory is None:
            raise ConfigurationError("provider is not registered")
        adapter = factory()
        if adapter.api_family != route.api_family:
            raise ConfigurationError("API family does not match provider")
        adapter.capabilities(route.model)
        return adapter


def default_registry():
    from .gemini import GeminiAdapter
    from .openai import OpenAIAdapter
    from .anthropic import AnthropicAdapter
    registry = ProviderRegistry()
    registry.register("gemini", GeminiAdapter)
    registry.register("openai", OpenAIAdapter)
    registry.register("anthropic", AnthropicAdapter)
    return registry
