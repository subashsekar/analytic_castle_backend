from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from app.ai.llm import (
    AsyncLLMClient,
    LLMAuthenticationError,
    LLMClientConfig,
    LLMConfigurationError,
    LLMErrorCode,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseValidationError,
    LLMTimeoutError,
    llm_client_config_from_settings,
    validate_llm_client_config,
)
from app.core.config import Settings, settings
from app.core.logging import RedactingFilter, redact_secret
from tests.conftest import run_async

FAKE_API_KEY = "sk-fake-openrouter-key-not-real"


def _client_config(**overrides: object) -> LLMClientConfig:
    payload: dict[str, object] = {
        "api_key": FAKE_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "timeout": 5.0,
        "temperature": 0.2,
        "max_tokens": 256,
        "max_retries": 0,
        "retry_base_backoff": 0.1,
        "retry_max_backoff": 1.0,
    }
    payload.update(overrides)
    return LLMClientConfig(**payload)  # type: ignore[arg-type]


def _success_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "gen-test-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from the model."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    body.update(overrides)
    return body


def _client(handler: Any, **config_overrides: object) -> AsyncLLMClient:
    transport = httpx.MockTransport(handler)
    return AsyncLLMClient(_client_config(**config_overrides), transport=transport)


def _request(**overrides: object) -> LLMRequest:
    payload: dict[str, object] = {
        "model": "openai/gpt-4o-mini",
        "messages": [LLMMessage(role="user", content="Hello")],
    }
    payload.update(overrides)
    return LLMRequest(**payload)  # type: ignore[arg-type]


# --- Configuration ---


def test_valid_configuration() -> None:
    config = _client_config()
    validate_llm_client_config(config)
    assert config.model == "openai/gpt-4o-mini"


def test_missing_api_key_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="API key"):
        validate_llm_client_config(_client_config(api_key=""))


def test_invalid_timeout_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="timeout"):
        validate_llm_client_config(_client_config(timeout=0))


def test_invalid_max_tokens_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="max tokens"):
        validate_llm_client_config(_client_config(max_tokens=0))


def test_invalid_retry_count_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="retries"):
        validate_llm_client_config(_client_config(max_retries=-1))


def test_invalid_backoff_raises_configuration_error() -> None:
    with pytest.raises(LLMConfigurationError, match="backoff"):
        validate_llm_client_config(
            _client_config(retry_base_backoff=2.0, retry_max_backoff=1.0)
        )


def test_config_repr_omits_api_key() -> None:
    rendered = repr(_client_config())
    assert FAKE_API_KEY not in rendered
    assert "api_key" not in rendered


def test_config_from_settings_uses_provider_aware_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "")
    configured = llm_client_config_from_settings()
    assert configured.base_url == "https://openrouter.ai/api/v1"


def test_openai_provider_ignores_openrouter_base_url_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-openai")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    configured = llm_client_config_from_settings()
    assert configured.base_url == "https://api.openai.com/v1"
    assert configured.api_key == "sk-openai"


def test_openrouter_api_key_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-openrouter-only")
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-legacy")
    configured = llm_client_config_from_settings()
    assert configured.api_key == "sk-openrouter-only"


def test_settings_repr_omits_openrouter_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", FAKE_API_KEY)
    rendered = repr(settings)
    assert "OPENROUTER_API_KEY" not in rendered
    assert FAKE_API_KEY not in rendered


def test_settings_validate_retry_backoff_order() -> None:
    with pytest.raises(ValidationError):
        Settings(
            APP_ENV="local",
            DATABASE_URL=settings.DATABASE_URL,
            JWT_SECRET_KEY="change-me-to-a-long-random-secret-key",
            LLM_RETRY_BASE_BACKOFF_SECONDS=10,
            LLM_RETRY_MAX_BACKOFF_SECONDS=1,
        )


# --- Client ---


def test_successful_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_success_body())

    client = _client(handler)

    async def _run() -> None:
        response = await client.complete(_request())
        assert response.content == "Hello from the model."
        assert response.model == "openai/gpt-4o-mini"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15
        assert response.request_id == "gen-test-1"
        await client.aclose()

    run_async(_run())
    assert captured["method"] == "POST"
    assert str(captured["path"]).endswith("/chat/completions")
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers.get("authorization", "").startswith("Bearer ")
    assert FAKE_API_KEY in headers.get("authorization", "")
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["messages"][0]["role"] == "user"


def test_request_validation_rejects_empty_messages() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(model="openai/gpt-4o-mini", messages=[])


def test_missing_usage_is_optional() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _success_body()
        body.pop("usage")
        return httpx.Response(200, json=body)

    client = _client(handler)

    async def _run() -> None:
        response = await client.complete(_request())
        assert response.usage.available is False
        await client.aclose()

    run_async(_run())


def test_empty_choices_raise_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "x", "choices": []})

    client = _client(handler)

    async def _run() -> None:
        with pytest.raises(LLMResponseValidationError):
            await client.complete(_request())
        await client.aclose()

    run_async(_run())


def test_malformed_json_raises_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = _client(handler)

    async def _run() -> None:
        with pytest.raises(LLMResponseValidationError):
            await client.complete(_request())
        await client.aclose()

    run_async(_run())


# --- Errors ---


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, LLMAuthenticationError),
        (400, LLMInvalidRequestError),
        (429, LLMRateLimitError),
        (500, LLMProviderError),
    ],
)
def test_error_mapping(status: int, expected: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "failed"}})

    client = _client(handler)

    async def _run() -> None:
        with pytest.raises(expected) as exc_info:
            await client.complete(_request())
        assert isinstance(exc_info.value, expected)
        assert exc_info.value.code is not None
        await client.aclose()

    run_async(_run())


def test_authentication_error_has_stable_code() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    client = _client(handler)

    async def _run() -> None:
        with pytest.raises(LLMAuthenticationError) as exc_info:
            await client.complete(_request())
        assert exc_info.value.code == LLMErrorCode.LLM_AUTHENTICATION_ERROR
        await client.aclose()

    run_async(_run())


# --- Timeout ---


def test_timeout_maps_to_llm_timeout_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = _client(handler)

    async def _run() -> None:
        with pytest.raises(LLMTimeoutError) as exc_info:
            await client.complete(_request())
        assert exc_info.value.code == LLMErrorCode.LLM_TIMEOUT
        await client.aclose()

    run_async(_run())


def test_client_cleanup_releases_connection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body())

    client = _client(handler)

    async def _run() -> None:
        await client.complete(_request())
        inner = await client._get_client()
        assert not inner.is_closed
        await client.aclose()
        assert client._client is None

    run_async(_run())


# --- Security ---


def test_api_key_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": f"Incorrect API key: {FAKE_API_KEY}"}},
        )

    client = _client(handler)
    logger = logging.getLogger("app.ai.llm.client")
    logger.addFilter(RedactingFilter())

    async def _run() -> None:
        with pytest.raises(LLMAuthenticationError):
            await client.complete(_request())
        await client.aclose()

    with caplog.at_level(logging.WARNING, logger=logger.name):
        run_async(_run())

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert FAKE_API_KEY not in combined
    assert "Bearer " not in combined
    assert redact_secret(f"openrouter_api_key={FAKE_API_KEY}") == (
        "openrouter_api_key=[REDACTED]"
    )


def test_prompt_content_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    secret_prompt = "customer-ssn-123-456-789"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body())

    client = _client(handler)
    logger = logging.getLogger("app.ai.llm.client")

    async def _run() -> None:
        await client.complete(
            _request(messages=[LLMMessage(role="user", content=secret_prompt)])
        )
        await client.aclose()

    with caplog.at_level(logging.INFO, logger=logger.name):
        run_async(_run())

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_prompt not in combined


# --- Concurrency ---


def test_concurrent_requests_are_isolated() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        body = _success_body()
        body["choices"][0]["message"]["content"] = f"reply-{calls['count']}"
        return httpx.Response(200, json=body)

    client = _client(handler)

    async def _run() -> None:
        results = await asyncio.gather(
            client.complete(_request()),
            client.complete(_request()),
            client.complete(_request()),
        )
        contents = {result.content for result in results}
        assert len(contents) == 3
        await client.aclose()

    run_async(_run())
