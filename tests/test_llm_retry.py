from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.ai.llm import (
    AsyncLLMClient,
    LLMAuthenticationError,
    LLMClientConfig,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMTimeoutError,
)
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
        "max_retries": 3,
        "retry_base_backoff": 0.01,
        "retry_max_backoff": 0.05,
    }
    payload.update(overrides)
    return LLMClientConfig(**payload)  # type: ignore[arg-type]


def _success_body() -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _request() -> LLMRequest:
    return LLMRequest(
        model="openai/gpt-4o-mini",
        messages=[LLMMessage(role="user", content="Hello")],
    )


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retries_transient_http_status(status: int) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            return httpx.Response(status, json={"error": {"message": "retry"}})
        return httpx.Response(200, json=_success_body())

    client = AsyncLLMClient(
        _client_config(max_retries=3),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        response = await client.complete(_request())
        assert response.content == "ok"
        assert calls["count"] == 3
        await client.aclose()

    run_async(_run())


def test_retries_transient_network_failure() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=_success_body())

    client = AsyncLLMClient(
        _client_config(max_retries=2),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        response = await client.complete(_request())
        assert response.content == "ok"
        assert calls["count"] == 2
        await client.aclose()

    run_async(_run())


def test_exponential_backoff_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.ai.llm.client.asyncio.sleep", fake_sleep)

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            return httpx.Response(503, json={"error": {"message": "busy"}})
        return httpx.Response(200, json=_success_body())

    client = AsyncLLMClient(
        _client_config(
            max_retries=3,
            retry_base_backoff=0.5,
            retry_max_backoff=30.0,
        ),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        await client.complete(_request())
        await client.aclose()

    run_async(_run())
    assert sleeps == [0.5, 1.0]


def test_respects_retry_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.ai.llm.client.asyncio.sleep", fake_sleep)

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limited"}},
                headers={"Retry-After": "2"},
            )
        return httpx.Response(200, json=_success_body())

    client = AsyncLLMClient(
        _client_config(max_retries=1, retry_max_backoff=30.0),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        await client.complete(_request())
        await client.aclose()

    run_async(_run())
    assert sleeps == [2.0]


def test_maximum_retry_limit_enforced() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(500, json={"error": {"message": "fail"}})

    client = AsyncLLMClient(
        _client_config(max_retries=2),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        with pytest.raises(LLMProviderError):
            await client.complete(_request())
        assert calls["count"] == 3
        await client.aclose()

    run_async(_run())


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, LLMInvalidRequestError),
        (401, LLMAuthenticationError),
    ],
)
def test_non_retryable_errors_fail_immediately(
    status: int,
    expected: type[Exception],
) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(status, json={"error": {"message": "no retry"}})

    client = AsyncLLMClient(
        _client_config(max_retries=3),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        with pytest.raises(expected):
            await client.complete(_request())
        assert calls["count"] == 1
        await client.aclose()

    run_async(_run())


def test_timeout_is_not_retried_when_max_retries_zero() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.TimeoutException("timed out")

    client = AsyncLLMClient(
        _client_config(max_retries=0),
        transport=httpx.MockTransport(handler),
    )

    async def _run() -> None:
        with pytest.raises(LLMTimeoutError):
            await client.complete(_request())
        assert calls["count"] == 1
        await client.aclose()

    run_async(_run())
