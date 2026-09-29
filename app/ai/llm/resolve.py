"""Shared LLM settings resolution for provider and OpenRouter-compatible clients."""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings

_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class ResolvedLLMSettings:
    """Authoritative LLM connection settings derived from application config."""

    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_output_tokens: int
    max_retries: int
    retry_base_backoff: float
    retry_max_backoff: float


def resolve_llm_settings(*, fast: bool = False) -> ResolvedLLMSettings:
    """Resolve a single LLM config used by all AI layers.

    Precedence:
    - ``LLM_PROVIDER`` selects the provider profile.
    - For ``openrouter``: ``OPENROUTER_API_KEY`` then ``LLM_API_KEY``;
      ``OPENROUTER_BASE_URL`` then ``LLM_BASE_URL`` then OpenRouter default.
    - For ``openai``: ``LLM_API_KEY``; ``LLM_BASE_URL`` then OpenAI default.
      OpenRouter URL defaults are ignored unless ``LLM_PROVIDER=openrouter``.
    - For ``openai_compatible``: ``LLM_API_KEY`` and required ``LLM_BASE_URL``.
    - ``fast=True`` uses ``LLM_FAST_MODEL`` when set, otherwise ``LLM_MODEL``.
    """
    provider = settings.LLM_PROVIDER.strip().lower()
    default_model = settings.LLM_MODEL.strip()
    fast_model = settings.LLM_FAST_MODEL.strip()
    model = (fast_model or default_model) if fast else default_model
    legacy_key = settings.LLM_API_KEY.strip()
    legacy_url = settings.LLM_BASE_URL.strip().rstrip("/")
    openrouter_key = settings.OPENROUTER_API_KEY.strip()
    openrouter_url = settings.OPENROUTER_BASE_URL.strip().rstrip("/")

    if provider == "openrouter":
        api_key = openrouter_key or legacy_key
        base_url = openrouter_url or legacy_url or _DEFAULT_OPENROUTER_BASE_URL
    elif provider == "openai":
        api_key = legacy_key
        base_url = legacy_url or _DEFAULT_OPENAI_BASE_URL
    elif provider == "openai_compatible":
        api_key = legacy_key
        base_url = legacy_url
    else:
        api_key = legacy_key
        base_url = legacy_url

    return ResolvedLLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        max_retries=settings.LLM_MAX_RETRIES,
        retry_base_backoff=settings.LLM_RETRY_BASE_BACKOFF_SECONDS,
        retry_max_backoff=settings.LLM_RETRY_MAX_BACKOFF_SECONDS,
    )
