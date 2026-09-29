from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.exceptions import AIConfigurationError
from app.ai.llm.resolve import resolve_llm_settings


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
    resolved = resolve_llm_settings()
    return LLMProviderConfig(
        provider=resolved.provider,
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        model=resolved.model,
        timeout_seconds=resolved.timeout_seconds,
        temperature=resolved.temperature,
        max_output_tokens=resolved.max_output_tokens,
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
