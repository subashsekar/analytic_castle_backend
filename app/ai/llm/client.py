"""Async OpenRouter-compatible LLM client with retry, timeout, and observability."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Self, cast

import httpx

from app.ai.llm.config import LLMClientConfig, validate_llm_client_config
from app.ai.llm.content import (
    extract_message_content,
    looks_like_reasoning_truncation,
)
from app.ai.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMInternalError,
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from app.ai.llm.models import LLMRequest, LLMResponse, StructuredOutputConfig
from app.ai.llm.usage import normalize_usage
from app.core.request_id import get_request_id

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
_MAX_TOKEN_BUMP_FACTOR = 4
_MAX_TOKEN_BUMP_CAP = 8_192


class AsyncLLMClient:
    """Reusable async client for OpenRouter-compatible chat completions."""

    def __init__(
        self,
        config: LLMClientConfig,
        *,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        validate_llm_client_config(config)
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def config(self) -> LLMClientConfig:
        return self._config

    def __repr__(self) -> str:
        return f"AsyncLLMClient(model={self._config.model!r})"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None and not self._client.is_closed:
            return self._client
        async with self._client_lock:
            if self._client is not None and not self._client.is_closed:
                return self._client
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=self._config.timeout,
                transport=cast(httpx.AsyncBaseTransport | None, self._transport),
            )
            return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> Self:
        await self._get_client()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._build_payload(request)
        started = time.perf_counter()
        request_id = get_request_id()
        model = request.model or self._config.model
        retries = 0
        format_downgrades = 0
        token_bumps = 0
        last_error: LLMError | None = None

        while True:
            try:
                body = await self._post_chat_completions(payload)
                response = self._parse_response(body, fallback_model=model)
                if (
                    request.structured_output is not None
                    and _is_truncated_finish(response.finish_reason)
                    and token_bumps < 1
                ):
                    payload = _bump_max_tokens(payload)
                    token_bumps += 1
                    logger.warning(
                        "LLM truncated structured output; bumping max_tokens "
                        "request_id=%s model=%s attempt=%s max_tokens=%s "
                        "finish_reason=%s",
                        request_id,
                        response.model,
                        token_bumps,
                        payload.get("max_tokens"),
                        response.finish_reason,
                    )
                    continue
                duration_ms = int((time.perf_counter() - started) * 1000)
                logger.info(
                    "LLM request completed request_id=%s model=%s duration_ms=%s "
                    "retries=%s prompt_tokens=%s completion_tokens=%s "
                    "total_tokens=%s success=true",
                    request_id,
                    response.model,
                    duration_ms,
                    retries,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )
                return response
            except LLMInvalidRequestError as exc:
                last_error = exc
                downgraded = _downgrade_response_format(payload)
                if downgraded is not None and format_downgrades < 2:
                    payload = downgraded
                    format_downgrades += 1
                    logger.warning(
                        "LLM structured-output fallback request_id=%s model=%s "
                        "attempt=%s error_code=%s",
                        request_id,
                        model,
                        format_downgrades,
                        exc.code.value,
                    )
                    continue
                duration_ms = int((time.perf_counter() - started) * 1000)
                logger.warning(
                    "LLM request failed request_id=%s model=%s duration_ms=%s "
                    "retries=%s error_code=%s success=false",
                    request_id,
                    model,
                    duration_ms,
                    retries,
                    exc.code.value,
                )
                raise
            except LLMResponseValidationError as exc:
                last_error = exc
                if (
                    "token budget on reasoning" in str(exc).lower()
                    or "empty content after using the output" in str(exc).lower()
                ) and token_bumps < 1:
                    payload = _bump_max_tokens(payload)
                    token_bumps += 1
                    logger.warning(
                        "LLM max_tokens bump request_id=%s model=%s attempt=%s "
                        "max_tokens=%s error_code=%s",
                        request_id,
                        model,
                        token_bumps,
                        payload.get("max_tokens"),
                        exc.code.value,
                    )
                    continue
                if not self._is_retryable(exc) or retries >= self._config.max_retries:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    logger.warning(
                        "LLM request failed request_id=%s model=%s duration_ms=%s "
                        "retries=%s error_code=%s success=false",
                        request_id,
                        model,
                        duration_ms,
                        retries,
                        exc.code.value,
                    )
                    raise
                backoff = self._backoff_seconds(retries, retry_after=exc.retry_after)
                retries += 1
                await asyncio.sleep(backoff)
            except LLMError as exc:
                last_error = exc
                if not self._is_retryable(exc) or retries >= self._config.max_retries:
                    duration_ms = int((time.perf_counter() - started) * 1000)
                    logger.warning(
                        "LLM request failed request_id=%s model=%s duration_ms=%s "
                        "retries=%s error_code=%s success=false",
                        request_id,
                        model,
                        duration_ms,
                        retries,
                        exc.code.value,
                    )
                    raise
                backoff = self._backoff_seconds(retries, retry_after=exc.retry_after)
                retries += 1
                logger.warning(
                    "LLM request retry request_id=%s model=%s attempt=%s "
                    "backoff_seconds=%s error_code=%s",
                    request_id,
                    model,
                    retries,
                    backoff,
                    exc.code.value,
                )
                await asyncio.sleep(backoff)

        raise last_error or LLMInternalError()

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        temperature = (
            self._config.temperature
            if request.temperature is None
            else request.temperature
        )
        max_tokens = (
            self._config.max_tokens
            if request.max_tokens is None
            else request.max_tokens
        )
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if request.structured_output is not None:
            payload["response_format"] = request.structured_output.response_format
        return payload

    async def _post_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        client = await self._get_client()
        try:
            response = await client.post(
                _CHAT_COMPLETIONS_PATH,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError() from exc
        except httpx.RequestError as exc:
            raise LLMProviderError() from exc
        return self._decode_http_response(response)

    def _decode_http_response(self, response: httpx.Response) -> dict[str, Any]:
        status = response.status_code
        retry_after = _parse_retry_after(response.headers.get("Retry-After"))

        if status in {401, 403}:
            raise LLMAuthenticationError()
        if status == 429:
            raise LLMRateLimitError(retry_after=retry_after)
        if status in {408, 504}:
            raise LLMTimeoutError()
        if status == 404:
            raise LLMProviderError("LLM model is unavailable")
        if status == 400:
            raise LLMInvalidRequestError()
        if status >= 500:
            raise LLMProviderError()
        if status >= 400:
            raise LLMInvalidRequestError()

        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseValidationError() from exc
        if not isinstance(body, dict):
            raise LLMResponseValidationError()
        return body

    def _parse_response(
        self,
        body: dict[str, Any],
        *,
        fallback_model: str,
    ) -> LLMResponse:
        provider_request_id = body.get("id")
        if provider_request_id is not None and not isinstance(provider_request_id, str):
            provider_request_id = None

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMResponseValidationError("LLM provider returned no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMResponseValidationError()
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMResponseValidationError()

        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            model = fallback_model
        finish_reason = first.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = None

        content = extract_message_content(message)
        if content is None:
            if looks_like_reasoning_truncation(first, message, body.get("usage")):
                raise LLMResponseValidationError(
                    "LLM provider returned empty content after using the output "
                    "token budget on reasoning; raise LLM_MAX_OUTPUT_TOKENS or use "
                    "a non-reasoning model"
                )
            raise LLMResponseValidationError("LLM provider returned empty content")

        usage = normalize_usage(
            body.get("usage"),
            model=model,
            request_id=provider_request_id,
        )
        return LLMResponse(
            content=content,
            model=model,
            finish_reason=finish_reason,
            usage=usage,
            request_id=provider_request_id,
        )

    def _is_retryable(self, exc: LLMError) -> bool:
        return isinstance(
            exc,
            (LLMProviderError, LLMRateLimitError, LLMTimeoutError),
        )

    def _backoff_seconds(
        self,
        attempt: int,
        *,
        retry_after: float | None,
    ) -> float:
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self._config.retry_max_backoff)
        delay = self._config.retry_base_backoff * (2**attempt)
        return min(delay, self._config.retry_max_backoff)


def _is_truncated_finish(finish_reason: str | None) -> bool:
    return finish_reason in {"length", "max_tokens", "max_output_tokens"}


def _downgrade_response_format(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Fall back from strict json_schema -> json_object -> unconstrained."""
    current = payload.get("response_format")
    if not isinstance(current, dict):
        return None
    format_type = current.get("type")
    next_payload = dict(payload)
    if format_type == "json_schema":
        next_payload["response_format"] = StructuredOutputConfig().response_format
        return next_payload
    if format_type == "json_object":
        next_payload.pop("response_format", None)
        return next_payload
    return None


def _bump_max_tokens(payload: dict[str, Any]) -> dict[str, Any]:
    current = payload.get("max_tokens")
    if not isinstance(current, int) or current < 1:
        current = 1_024
    bumped = min(max(current * _MAX_TOKEN_BUMP_FACTOR, current + 1_024), _MAX_TOKEN_BUMP_CAP)
    next_payload = dict(payload)
    next_payload["max_tokens"] = bumped
    return next_payload


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        parsed = float(stripped)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed
