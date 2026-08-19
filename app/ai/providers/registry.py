from __future__ import annotations

from app.ai.providers.base import LLMProvider
from app.ai.providers.config import (
    LLMProviderConfig,
    llm_provider_config_from_settings,
    validate_llm_provider_config,
)
from app.ai.providers.openai_compatible import OpenAICompatibleProvider


def create_llm_provider(
    config: LLMProviderConfig | None = None,
) -> LLMProvider:
    """Select and initialize the configured LLM provider."""
    resolved = config or llm_provider_config_from_settings()
    validate_llm_provider_config(resolved)
    return OpenAICompatibleProvider(resolved)


def get_llm_provider() -> LLMProvider:
    """Build the configured provider. HTTP routes wrap this with a 503 mapping."""
    return create_llm_provider()
