from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIResponseValidationError,
)
from app.ai.llm import (
    AsyncLLMClient,
    LLMClientConfig,
    LLMError,
    LLMInvalidRequestError,
    LLMRequest,
    StructuredOutputConfig,
)
from app.ai.llm import (
    LLMMessage as InfraLLMMessage,
)
from app.ai.providers.base import LLMGeneration, LLMMessage, StructuredGeneration
from app.ai.providers.config import LLMProviderConfig, validate_llm_provider_config
from app.ai.types import TokenUsage
from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_CONTENT_CHARS = 32_000


def _client_config_from_provider(config: LLMProviderConfig) -> LLMClientConfig:
    return LLMClientConfig(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        timeout=config.timeout_seconds,
        temperature=config.temperature,
        max_tokens=config.max_output_tokens,
        max_retries=settings.LLM_MAX_RETRIES,
        retry_base_backoff=settings.LLM_RETRY_BASE_BACKOFF_SECONDS,
        retry_max_backoff=settings.LLM_RETRY_MAX_BACKOFF_SECONDS,
    )


def _map_llm_error(exc: LLMError) -> Exception:
    from app.ai.llm import (
        LLMAuthenticationError,
        LLMInvalidRequestError,
        LLMProviderError,
        LLMRateLimitError,
        LLMResponseValidationError,
        LLMTimeoutError,
    )

    if isinstance(exc, LLMTimeoutError):
        return AIProviderTimeoutError()
    if isinstance(exc, LLMRateLimitError):
        return AIProviderRateLimitError()
    if isinstance(exc, LLMAuthenticationError):
        return AIProviderAuthenticationError()
    if isinstance(exc, LLMResponseValidationError):
        return AIResponseValidationError(raw_content=exc.raw_content)
    if isinstance(exc, LLMInvalidRequestError):
        return AIProviderError()
    if isinstance(exc, LLMProviderError):
        if "unavailable" in str(exc).lower():
            return AIProviderError("AI model is unavailable")
        return AIProviderError()
    return AIProviderError()


def _usage_from_infra(usage: object) -> TokenUsage:
    from app.ai.llm import LLMUsage

    if not isinstance(usage, LLMUsage):
        return TokenUsage()
    return TokenUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


class OpenAICompatibleProvider:
    """OpenAI Chat Completions client. Also used for OpenRouter."""

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        validate_llm_provider_config(config)
        self._config = config
        self._client = AsyncLLMClient(
            _client_config_from_provider(config),
            transport=transport,
        )

    @property
    def name(self) -> str:
        return self._config.provider

    def __repr__(self) -> str:
        return (
            "OpenAICompatibleProvider("
            f"provider={self._config.provider!r}, "
            f"model={self._config.model!r})"
        )

    async def generate(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMGeneration:
        request = self._build_request(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            response = await self._client.complete(request)
        except LLMError as exc:
            raise _map_llm_error(exc) from exc
        content = self._truncate_content(response.content)
        return LLMGeneration(
            content=content,
            model=response.model,
            usage=_usage_from_infra(response.usage),
            finish_reason=response.finish_reason,
        )

    async def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]:
        from app.ai.prompt.structured import structured_output_from_model

        request = self._build_request(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            structured_output=structured_output_from_model(schema),
        )
        try:
            response = await self._client.complete(request)
        except LLMError as exc:
            if (
                not isinstance(exc, LLMInvalidRequestError)
                or not request.structured_output
            ):
                raise _map_llm_error(exc) from exc
            # Client already downgrades json_schema -> json_object -> none.
            # Keep a final unconstrained retry for provider wrappers that still 400.
            request = request.model_copy(update={"structured_output": None})
            try:
                response = await self._client.complete(request)
            except LLMError as retry_exc:
                raise _map_llm_error(retry_exc) from retry_exc
        content = self._truncate_content(response.content)
        parsed = _parse_structured(content, schema)
        return StructuredGeneration(
            result=parsed,
            model=response.model,
            usage=_usage_from_infra(response.usage),
            finish_reason=response.finish_reason,
        )

    def _build_request(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None,
        max_output_tokens: int | None,
        structured_output: StructuredOutputConfig | None = None,
        json_object: bool = False,
    ) -> LLMRequest:
        output = structured_output
        if output is None and json_object:
            output = StructuredOutputConfig()
        return LLMRequest(
            model=self._config.model,
            messages=[
                InfraLLMMessage(role=message.role, content=message.content)  # type: ignore[arg-type]
                for message in messages
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            structured_output=output,
        )

    def _truncate_content(self, content: str) -> str:
        max_output = min(settings.AI_MAX_OUTPUT_CHARS, _MAX_CONTENT_CHARS)
        if len(content) > max_output:
            return content[:max_output]
        return content


def _parse_structured(content: str, schema: type[T]) -> T:
    from app.ai.llm.content import extract_json_object

    try:
        payload = extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Structured LLM JSON extract failed schema=%s error=%s content_chars=%s",
            schema.__name__,
            type(exc).__name__,
            len(content),
        )
        raise AIResponseValidationError(raw_content=content) from exc
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        logger.warning(
            "Structured LLM schema validation failed schema=%s errors=%s "
            "content_chars=%s",
            schema.__name__,
            exc.errors()[:8],
            len(content),
        )
        raise AIResponseValidationError(raw_content=content) from exc
