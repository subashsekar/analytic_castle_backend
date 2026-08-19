from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.ai.exceptions import (
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIResponseValidationError,
)
from app.ai.providers.base import LLMGeneration, LLMMessage, StructuredGeneration
from app.ai.providers.config import LLMProviderConfig, validate_llm_provider_config
from app.ai.types import TokenUsage
from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_MAX_CONTENT_CHARS = 32_000


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
        self._transport = transport

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
        payload = self._completion_payload(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        body = await self._post_completions(payload)
        return self._parse_generation(body)

    async def generate_structured(
        self,
        messages: Sequence[LLMMessage],
        schema: type[T],
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredGeneration[T]:
        payload = self._completion_payload(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_object=True,
        )
        try:
            body = await self._post_completions(payload)
        except AIProviderError as exc:
            if type(exc) is not AIProviderError or "response_format" not in payload:
                raise
            payload.pop("response_format", None)
            body = await self._post_completions(payload)
        generation = self._parse_generation(body)
        parsed = _parse_structured(generation.content, schema)
        return StructuredGeneration(
            result=parsed,
            model=generation.model,
            usage=generation.usage,
            finish_reason=generation.finish_reason,
        )

    def _completion_payload(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None,
        max_output_tokens: int | None,
        json_object: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": (
                self._config.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._config.max_output_tokens
                if max_output_tokens is None
                else max_output_tokens
            ),
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def _post_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        started_model = self._config.model
        try:
            async with httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    _CHAT_COMPLETIONS_PATH,
                    headers=headers,
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            logger.warning(
                "LLM provider timeout provider=%s model=%s",
                self._config.provider,
                started_model,
            )
            raise AIProviderTimeoutError() from exc
        except httpx.RequestError as exc:
            logger.warning(
                "LLM provider request failed provider=%s model=%s error_type=%s",
                self._config.provider,
                started_model,
                type(exc).__name__,
            )
            raise AIProviderError() from exc

        return _decode_provider_response(response, provider=self._config.provider)

    def _parse_generation(self, body: dict[str, Any]) -> LLMGeneration:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIResponseValidationError()
        first = choices[0]
        if not isinstance(first, dict):
            raise AIResponseValidationError()
        message = first.get("message")
        if not isinstance(message, dict):
            raise AIResponseValidationError()
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AIResponseValidationError()
        max_output = min(settings.AI_MAX_OUTPUT_CHARS, _MAX_CONTENT_CHARS)
        if len(content) > max_output:
            content = content[:max_output]
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            model = self._config.model
        finish_reason = first.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = None
        return LLMGeneration(
            content=content,
            model=model,
            usage=_parse_usage(body.get("usage")),
            finish_reason=finish_reason,
        )


def _decode_provider_response(
    response: httpx.Response,
    *,
    provider: str,
) -> dict[str, Any]:
    status = response.status_code
    if status in {401, 403}:
        logger.warning(
            "LLM provider authentication failed provider=%s status=%s",
            provider,
            status,
        )
        raise AIProviderAuthenticationError()
    if status == 429:
        logger.warning(
            "LLM provider rate limited provider=%s status=%s", provider, status
        )
        raise AIProviderRateLimitError()
    if status in {408, 504}:
        logger.warning("LLM provider timeout provider=%s status=%s", provider, status)
        raise AIProviderTimeoutError()
    if status == 404:
        logger.warning("LLM model unavailable provider=%s status=%s", provider, status)
        raise AIProviderError("AI model is unavailable")
    if status >= 500:
        logger.warning("LLM provider failure provider=%s status=%s", provider, status)
        raise AIProviderError()
    if status >= 400:
        logger.warning(
            "LLM provider rejected request provider=%s status=%s", provider, status
        )
        raise AIProviderError()
    try:
        body = response.json()
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM provider returned malformed JSON provider=%s status=%s",
            provider,
            status,
        )
        raise AIResponseValidationError() from exc
    if not isinstance(body, dict):
        raise AIResponseValidationError()
    return body


def _parse_usage(raw: object) -> TokenUsage:
    if not isinstance(raw, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_optional_non_negative_int(raw.get("prompt_tokens")),
        output_tokens=_optional_non_negative_int(raw.get("completion_tokens")),
        total_tokens=_optional_non_negative_int(raw.get("total_tokens")),
    )


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _parse_structured(content: str, schema: type[T]) -> T:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AIResponseValidationError(raw_content=content) from exc
    if not isinstance(payload, dict):
        raise AIResponseValidationError(raw_content=content)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise AIResponseValidationError(raw_content=content) from exc
