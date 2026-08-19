from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from app.ai.exceptions import (
    AIConfigurationError,
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIResponseValidationError,
)
from app.ai.providers import (
    LLMMessage,
    LLMProviderConfig,
    OpenAICompatibleProvider,
    create_llm_provider,
    llm_provider_config_from_settings,
)
from app.ai.types import AIAnalysisResult, TokenUsage
from app.core.config import Settings, settings
from app.core.logging import RedactingFilter, redact_secret
from tests.conftest import run_async

SECRET_KEY = "sk-test-secret-llm-key-do-not-log"


def _config(**overrides: object) -> LLMProviderConfig:
    payload: dict[str, object] = {
        "provider": "openai",
        "api_key": SECRET_KEY,
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "timeout_seconds": 5.0,
        "temperature": 0.2,
        "max_output_tokens": 256,
    }
    payload.update(overrides)
    return LLMProviderConfig(**payload)  # type: ignore[arg-type]


def _success_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "answer": "No live query results are available in context.",
                            "intent": "question",
                            "requires_data_access": True,
                            "metadata_context": ["public.orders"],
                            "warnings": [],
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "total_tokens": 20,
        },
    }
    body.update(overrides)
    return body


def _provider(
    handler: Any,
    **config_overrides: object,
) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(handler)
    return OpenAICompatibleProvider(_config(**config_overrides), transport=transport)


def test_provider_config_from_settings_uses_openai_defaults() -> None:
    configured = llm_provider_config_from_settings()
    assert configured.provider == settings.LLM_PROVIDER
    assert configured.model == settings.LLM_MODEL
    if not settings.LLM_BASE_URL:
        assert configured.base_url == "https://api.openai.com/v1"


def test_provider_config_repr_omits_api_key() -> None:
    rendered = repr(_config())
    assert SECRET_KEY not in rendered
    assert "api_key" not in rendered


def test_settings_repr_omits_llm_api_key() -> None:
    rendered = repr(settings)
    assert "LLM_API_KEY" not in rendered
    if settings.LLM_API_KEY:
        assert settings.LLM_API_KEY not in rendered


def test_create_provider_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_API_KEY", "")
    with pytest.raises(AIConfigurationError, match="API key"):
        create_llm_provider()


def test_create_provider_rejects_unknown_provider() -> None:
    with pytest.raises(AIConfigurationError):
        create_llm_provider(_config(provider="unknown-vendor"))


def test_create_provider_rejects_missing_model() -> None:
    with pytest.raises(AIConfigurationError, match="model"):
        create_llm_provider(_config(model=""))


def test_create_provider_rejects_missing_base_url() -> None:
    with pytest.raises(AIConfigurationError, match="base URL"):
        create_llm_provider(_config(provider="openai_compatible", base_url=""))


def test_openai_compatible_requires_base_url_in_settings() -> None:
    with pytest.raises(Exception):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            LLM_PROVIDER="openai_compatible",
            LLM_BASE_URL="",
        )


def test_successful_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")
        assert SECRET_KEY not in (request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())

    provider = _provider(handler)

    async def _run() -> None:
        result = await provider.generate(
            [LLMMessage(role="user", content="What tables exist?")]
        )
        assert "query results" in result.content
        assert result.model == "gpt-4o-mini"
        assert result.usage == TokenUsage(12, 8, 20)

    run_async(_run())


def test_successful_structured_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json=_success_body())

    provider = _provider(handler)

    async def _run() -> None:
        result = await provider.generate_structured(
            [LLMMessage(role="user", content="Summarize the catalog.")],
            AIAnalysisResult,
        )
        assert result.result.intent == "question"
        assert result.result.requires_data_access is True
        assert result.usage.total_tokens == 20

    run_async(_run())


def test_structured_output_retries_without_response_format() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        calls["count"] += 1
        if "response_format" in payload:
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(200, json=_success_body())

    provider = _provider(handler)

    async def _run() -> None:
        result = await provider.generate_structured(
            [LLMMessage(role="user", content="Hello")],
            AIAnalysisResult,
        )
        assert result.result.answer
        assert calls["count"] == 2

    run_async(_run())


def test_provider_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIProviderTimeoutError, match="timed out"):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_provider_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "upstream"}})

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIProviderError):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_provider_rate_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIProviderRateLimitError):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_provider_authentication_error_hides_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"Incorrect API key provided: {SECRET_KEY}"}},
        )

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIProviderAuthenticationError) as exc_info:
            await provider.generate([LLMMessage(role="user", content="Hello")])
        assert SECRET_KEY not in str(exc_info.value)
        assert SECRET_KEY not in repr(exc_info.value)

    run_async(_run())


def test_unavailable_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model not found"}})

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIProviderError, match="unavailable"):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_malformed_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIResponseValidationError):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_missing_choices_are_invalid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "gpt-4o-mini", "choices": []})

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIResponseValidationError):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    run_async(_run())


def test_invalid_structured_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(
                choices=[
                    {
                        "message": {"role": "assistant", "content": "plain text"},
                        "finish_reason": "stop",
                    }
                ]
            ),
        )

    provider = _provider(handler)

    async def _run() -> None:
        with pytest.raises(AIResponseValidationError) as exc_info:
            await provider.generate_structured(
                [LLMMessage(role="user", content="Hello")],
                AIAnalysisResult,
            )
        assert exc_info.value.raw_content == "plain text"

    run_async(_run())


def test_usage_missing_is_represented_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _success_body()
        body.pop("usage")
        return httpx.Response(200, json=body)

    provider = _provider(handler)

    async def _run() -> None:
        result = await provider.generate([LLMMessage(role="user", content="Hello")])
        assert result.usage.available is False
        assert result.usage.input_tokens is None

    run_async(_run())


def test_provider_logs_do_not_include_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"Incorrect API key provided: {SECRET_KEY}"}},
        )

    provider = _provider(handler)
    logger = logging.getLogger("app.ai.providers.openai_compatible")
    logger.addFilter(RedactingFilter())

    async def _run() -> None:
        with pytest.raises(AIProviderAuthenticationError):
            await provider.generate([LLMMessage(role="user", content="Hello")])

    with caplog.at_level(logging.WARNING, logger=logger.name):
        run_async(_run())

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert SECRET_KEY not in combined
    assert "Bearer " not in combined
    assert redact_secret(f"llm_api_key={SECRET_KEY}") == "llm_api_key=[REDACTED]"
