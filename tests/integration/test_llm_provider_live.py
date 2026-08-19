from __future__ import annotations

import os

import pytest

from app.ai.providers import (
    LLMMessage,
    LLMProviderConfig,
    OpenAICompatibleProvider,
)
from app.ai.types import AIAnalysisResult
from tests.conftest import run_async

_SKIP_REASON = "LLM integration tests: SKIPPED — test provider not configured"


def _integration_config() -> LLMProviderConfig | None:
    api_key = os.environ.get("TEST_LLM_API_KEY")
    if not api_key:
        return None
    provider = os.environ.get("TEST_LLM_PROVIDER", "openai").strip().lower()
    base_url = os.environ.get("TEST_LLM_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        base_url = (
            "https://openrouter.ai/api/v1"
            if provider == "openrouter"
            else "https://api.openai.com/v1"
        )
    model = os.environ.get("TEST_LLM_MODEL", "gpt-4o-mini").strip()
    return LLMProviderConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=30.0,
        temperature=0.0,
        max_output_tokens=64,
    )


def _require_config() -> LLMProviderConfig:
    config = _integration_config()
    if config is None:
        pytest.skip(_SKIP_REASON)
    return config


def test_live_provider_generate_structured() -> None:
    config = _require_config()
    provider = OpenAICompatibleProvider(config)

    async def _run() -> None:
        result = await provider.generate_structured(
            [
                LLMMessage(
                    role="system",
                    content="Return JSON with keys answer, intent, requires_data_access, metadata_context, warnings.",
                ),
                LLMMessage(
                    role="user",
                    content="Reply that required database facts are unavailable.",
                ),
            ],
            AIAnalysisResult,
        )
        assert result.result.answer.strip()
        assert config.api_key not in result.result.answer
        assert config.api_key not in repr(provider)

    run_async(_run())


def test_live_intent_detection_when_configured() -> None:
    from app.ai.intent_types import LLMIntentDetection

    config = _require_config()
    provider = OpenAICompatibleProvider(config)

    async def _run() -> None:
        result = await provider.generate_structured(
            [
                LLMMessage(
                    role="system",
                    content=(
                        "Return JSON with keys intent, operation, subject, metrics, "
                        "dimensions, filters, time_range, sort, requested_limit, "
                        "requires_data_access, requires_metadata, requires_relationships, "
                        "confidence, requires_clarification, clarification_question, "
                        "unsupported_reason. intent must be SCHEMA_QUESTION. "
                        "Do not generate SQL."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content="What tables do we have? Do not invent schema names.",
                ),
            ],
            LLMIntentDetection,
        )
        assert result.result.intent.value
        assert config.api_key not in repr(result.result)

    run_async(_run())


def test_live_provider_skipped_without_explicit_key() -> None:
    if os.environ.get("TEST_LLM_API_KEY"):
        return
    pytest.skip(_SKIP_REASON)
