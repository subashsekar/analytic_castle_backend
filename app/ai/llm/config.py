"""LLM infrastructure configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.llm.errors import LLMConfigurationError
from app.ai.llm.models import LLMModelConfig
from app.ai.llm.resolve import resolve_llm_settings


@dataclass(frozen=True)
class LLMClientConfig:
    """Runtime configuration for the async OpenRouter-compatible client."""

    api_key: str = field(repr=False)
    base_url: str
    model: str
    timeout: float
    temperature: float
    max_tokens: int
    max_retries: int
    retry_base_backoff: float
    retry_max_backoff: float

    def __repr__(self) -> str:
        return (
            "LLMClientConfig("
            f"base_url={self.base_url!r}, "
            f"model={self.model!r}, "
            f"timeout={self.timeout}, "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, "
            f"max_retries={self.max_retries}, "
            f"retry_base_backoff={self.retry_base_backoff}, "
            f"retry_max_backoff={self.retry_max_backoff})"
        )

    @property
    def model_config(self) -> LLMModelConfig:
        return LLMModelConfig(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            retries=self.max_retries,
            retry_base_backoff=self.retry_base_backoff,
            retry_max_backoff=self.retry_max_backoff,
        )


def llm_client_config_from_settings() -> LLMClientConfig:
    resolved = resolve_llm_settings()
    return LLMClientConfig(
        api_key=resolved.api_key,
        base_url=resolved.base_url,
        model=resolved.model,
        timeout=resolved.timeout_seconds,
        temperature=resolved.temperature,
        max_tokens=resolved.max_output_tokens,
        max_retries=resolved.max_retries,
        retry_base_backoff=resolved.retry_base_backoff,
        retry_max_backoff=resolved.retry_max_backoff,
    )


def validate_llm_client_config(config: LLMClientConfig) -> None:
    if not config.api_key:
        raise LLMConfigurationError("LLM API key is not configured")
    if not config.model:
        raise LLMConfigurationError("LLM model is not configured")
    if not config.base_url:
        raise LLMConfigurationError("LLM base URL is not configured")
    if config.timeout <= 0:
        raise LLMConfigurationError("LLM timeout is invalid")
    if config.max_tokens < 1:
        raise LLMConfigurationError("LLM max tokens is invalid")
    if config.max_retries < 0:
        raise LLMConfigurationError("LLM max retries is invalid")
    if config.retry_base_backoff <= 0:
        raise LLMConfigurationError("LLM retry base backoff is invalid")
    if config.retry_max_backoff <= 0:
        raise LLMConfigurationError("LLM retry max backoff is invalid")
    if config.retry_base_backoff > config.retry_max_backoff:
        raise LLMConfigurationError("LLM retry base backoff cannot exceed max backoff")
