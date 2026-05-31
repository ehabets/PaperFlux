"""LLM provider registry.

Adding a backend is additive: implement ``analyze_pdf`` in a new module and add
one entry below. Providers are imported lazily so selecting one backend never
imports another backend's SDK.
"""

import importlib

from .base import LLMProvider, ProgressCallback

# provider name -> "module.path:ClassName"
_PROVIDERS = {
    "openai": "paperflux.providers.openai_provider:OpenAIProvider",
    "anthropic": "paperflux.providers.anthropic_provider:AnthropicProvider",
}


def available_providers() -> list:
    """Return the list of registered provider names."""
    return sorted(_PROVIDERS)


def get_provider(name: str) -> LLMProvider:
    """Instantiate the provider registered under ``name``."""
    try:
        spec = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(available_providers())}"
        )
    module_path, class_name = spec.split(":")
    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name)
    return provider_cls()


__all__ = ["LLMProvider", "ProgressCallback", "get_provider", "available_providers"]
