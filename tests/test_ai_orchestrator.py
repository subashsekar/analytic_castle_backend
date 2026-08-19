from __future__ import annotations

import logging
import uuid

import pytest

from app.ai.exceptions import (
    AIContextError,
    AIProviderError,
    AIRequestValidationError,
    AIResponseValidationError,
)
from app.ai.orchestrator import AIAnalystOrchestrator, build_prompt_messages
from app.ai.types import (
    AI_CAPABILITY_CHAT,
    AIContext,
    AIRequest,
    MetadataSnippet,
)
from app.core.config import settings
from app.core.logging import RedactingFilter
from tests.ai_fakes import FakeLLMProvider
from tests.conftest import run_async

SECRET_PASSWORD = "CustomerDbPassword!@# 42"
API_KEY = "sk-test-secret-llm-key-do-not-log"


def _context(**overrides: object) -> AIContext:
    payload: dict[str, object] = {
        "user_id": uuid.uuid4(),
        "workspace_id": uuid.uuid4(),
        "organization_id": uuid.uuid4(),
        "data_source_id": uuid.uuid4(),
        "data_source_name": "Analytics Warehouse",
        "data_source_type": "POSTGRESQL",
        "workspace_name": "Workspace A",
        "workspace_role": "MEMBER",
    }
    payload.update(overrides)
    return AIContext(**payload)  # type: ignore[arg-type]


class _FakeMetadata:
    def __init__(self, snippets: tuple[MetadataSnippet, ...] = ()) -> None:
        self.snippets = snippets
        self.calls: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        self.error: Exception | None = None

    def get_relevant_metadata(
        self,
        *,
        data_source_id: uuid.UUID,
        workspace_id: uuid.UUID,
        query: str,
        limit: int | None = None,
    ) -> tuple[MetadataSnippet, ...]:
        del limit
        self.calls.append((data_source_id, workspace_id, query))
        if self.error is not None:
            raise self.error
        return self.snippets


def test_orchestrator_validates_context_data_source() -> None:
    provider = FakeLLMProvider()
    context = _context()
    request = AIRequest(message="What tables exist?", data_source_id=uuid.uuid4())

    async def _run() -> None:
        with pytest.raises(AIContextError):
            await AIAnalystOrchestrator(provider).chat(request, context)

    run_async(_run())
    assert provider.structured_calls == 0


def test_orchestrator_rejects_oversized_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "AI_MAX_MESSAGE_CHARS", 8)
    provider = FakeLLMProvider()
    context = _context()
    request = AIRequest(
        message="this is too long", data_source_id=context.data_source_id
    )

    async def _run() -> None:
        with pytest.raises(AIRequestValidationError, match="maximum length"):
            await AIAnalystOrchestrator(provider).chat(request, context)

    run_async(_run())


def test_orchestrator_selects_injected_provider() -> None:
    provider = FakeLLMProvider()
    context = _context()
    request = AIRequest(
        message="Describe the catalog.", data_source_id=context.data_source_id
    )

    async def _run() -> None:
        result = await AIAnalystOrchestrator(provider).chat(request, context)
        assert result.intent is not None
        assert result.plan is not None
        assert result.intent.intent.value == "SCHEMA_QUESTION"
        assert result.model == "fake-model"
        assert result.usage.total_tokens == 18
        assert "SELECT" not in result.response
        assert "sql" not in result.response.lower()

    run_async(_run())
    assert provider.structured_calls == 1
    assert provider.name == "fake"


def test_orchestrator_successful_response_includes_metadata() -> None:
    provider = FakeLLMProvider()
    context = _context()
    metadata = _FakeMetadata(
        (
            MetadataSnippet(
                schema_name="public",
                table_name="orders",
                matched_name="orders",
            ),
        )
    )
    request = AIRequest(message="orders", data_source_id=context.data_source_id)

    async def _run() -> None:
        result = await AIAnalystOrchestrator(provider, metadata).chat(request, context)
        assert result.analysis is not None
        assert result.analysis.intent == "SCHEMA_QUESTION"
        assert result.intent is not None
        assert result.plan is not None

    run_async(_run())
    assert metadata.calls
    assert "public.orders" in provider.messages[0][1].content


def test_orchestrator_maps_provider_error() -> None:
    provider = FakeLLMProvider()
    provider.error = AIProviderError()
    context = _context()
    request = AIRequest(message="Hello", data_source_id=context.data_source_id)

    async def _run() -> None:
        with pytest.raises(AIProviderError):
            await AIAnalystOrchestrator(provider).chat(request, context)

    run_async(_run())


def test_orchestrator_rejects_invalid_llm_response() -> None:
    provider = FakeLLMProvider()
    provider.invalid_content = "I am not JSON, just a helpful sentence."
    context = _context()
    request = AIRequest(message="Hello", data_source_id=context.data_source_id)

    async def _run() -> None:
        with pytest.raises(AIResponseValidationError):
            await AIAnalystOrchestrator(provider).chat(request, context)

    run_async(_run())


def test_prompt_omits_secrets_and_sample_data() -> None:
    context = _context()
    messages = build_prompt_messages("How many orders are there?", context)
    combined = "\n".join(message.content for message in messages)
    assert SECRET_PASSWORD not in combined
    assert API_KEY not in combined
    assert "password" not in combined.lower() or "passwords" in combined.lower()
    assert "Do not generate or execute SQL" in combined
    assert "untrusted" in combined.lower()
    assert context.data_source_name in combined
    assert "john.doe@example.com" not in combined


def test_prompt_does_not_load_full_schema() -> None:
    snippets = tuple(
        MetadataSnippet(
            schema_name="public",
            table_name=f"table_{index}",
            matched_name=f"table_{index}",
        )
        for index in range(3)
    )
    context = _context(metadata=snippets)
    messages = build_prompt_messages("tables", context)
    user_prompt = messages[1].content
    assert "table_0" in user_prompt
    assert "CREATE TABLE" not in user_prompt


def test_metadata_failure_does_not_fail_chat() -> None:
    provider = FakeLLMProvider()
    metadata = _FakeMetadata()
    metadata.error = RuntimeError("search failed")
    context = _context()
    request = AIRequest(message="Hello", data_source_id=context.data_source_id)

    async def _run() -> None:
        result = await AIAnalystOrchestrator(provider, metadata).chat(request, context)
        assert result.response
        assert result.intent is not None

    run_async(_run())


def test_orchestrator_logs_do_not_include_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = FakeLLMProvider()
    context = _context()
    request = AIRequest(
        message="Customer email is john.doe@example.com",
        data_source_id=context.data_source_id,
    )
    logger = logging.getLogger("app.ai.orchestrator")
    logger.addFilter(RedactingFilter())

    async def _run() -> None:
        await AIAnalystOrchestrator(provider).chat(request, context)

    with caplog.at_level(logging.INFO, logger=logger.name):
        run_async(_run())

    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "john.doe@example.com" not in combined
    assert API_KEY not in combined
    assert str(context.data_source_id) in combined
    assert "fake-model" in combined
    assert AI_CAPABILITY_CHAT in context.allowed_capabilities
