from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.exceptions import AIConfigurationError
from app.core.config import settings

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout_seconds: float
    temperature: float
    max_output_tokens: int

    def __repr__(self) -> str:
        return (
            "LLMProviderConfig("
            f"provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"temperature={self.temperature}, "
            f"max_output_tokens={self.max_output_tokens})"
        )


def llm_provider_config_from_settings() -> LLMProviderConfig:
    provider = settings.LLM_PROVIDER.strip().lower()
    base_url = settings.LLM_BASE_URL.strip().rstrip("/")
    if not base_url:
        base_url = _PROVIDER_BASE_URLS.get(provider, "")
    return LLMProviderConfig(
        provider=provider,
        api_key=settings.LLM_API_KEY,
        base_url=base_url,
        model=settings.LLM_MODEL.strip(),
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        temperature=settings.LLM_TEMPERATURE,
        max_output_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
    )


def validate_llm_provider_config(config: LLMProviderConfig) -> None:
    if config.provider not in {"openai", "openrouter", "openai_compatible"}:
        raise AIConfigurationError("AI provider is not configured")
    if not config.api_key.strip():
        raise AIConfigurationError("AI provider API key is not configured")
    if not config.model.strip():
        raise AIConfigurationError("AI provider model is not configured")
    if not config.base_url.strip():
        raise AIConfigurationError("AI provider base URL is not configured")
    if config.timeout_seconds <= 0:
        raise AIConfigurationError("AI provider timeout is invalid")
    if config.max_output_tokens < 1:
        raise AIConfigurationError("AI provider max output tokens is invalid")
