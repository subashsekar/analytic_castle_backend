"""Focused tests for Step 6 speed/token optimizations."""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from app.ai.llm.resolve import resolve_llm_settings
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataTableCandidate,
    ResolvedMetadataContext,
)
from app.ai.sql_generation.schema_cache import (
    cache_size,
    cached_schema_prompt_context,
    invalidate_schema_context_cache,
    schema_fingerprint,
)
from app.core.config import settings
from tests.conftest import run_async


def _metadata(*, amount_desc: str = "revenue") -> ResolvedMetadataContext:
    ds = uuid.uuid4()
    table_id = uuid.uuid4()
    return ResolvedMetadataContext(
        data_source_id=ds,
        tables=[
            MetadataTableCandidate(
                table_id=table_id,
                schema_name="public",
                table_name="sales",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=10,
            )
        ],
        columns=[
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=table_id,
                schema_name="public",
                table_name="sales",
                column_name="amount",
                data_type="numeric",
                description=amount_desc,
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=10,
            )
        ],
    )


def test_fast_model_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LLM_MODEL", "gpt-main")
    monkeypatch.setattr(settings, "LLM_FAST_MODEL", "")
    assert resolve_llm_settings(fast=True).model == "gpt-main"
    monkeypatch.setattr(settings, "LLM_FAST_MODEL", "gpt-fast")
    assert resolve_llm_settings(fast=True).model == "gpt-fast"
    assert resolve_llm_settings(fast=False).model == "gpt-main"


def test_schema_context_cache_hits_and_invalidates() -> None:
    invalidate_schema_context_cache()
    meta = _metadata()
    first = cached_schema_prompt_context(meta)
    second = cached_schema_prompt_context(meta)
    assert first is second
    assert cache_size() >= 1

    changed = _metadata(amount_desc="gross sales")
    changed = changed.model_copy(update={"data_source_id": meta.data_source_id})
    # Same data source id but different fingerprint → new entry.
    assert schema_fingerprint(meta) != schema_fingerprint(changed)
    third = cached_schema_prompt_context(changed)
    assert third is not first

    before = cache_size()
    invalidate_schema_context_cache(meta.data_source_id)
    assert cache_size() < before or cache_size() == 0
    rebuilt = cached_schema_prompt_context(meta)
    assert rebuilt.text == first.text
    assert rebuilt is not first


def test_pipeline_emits_partial_facts_and_performance(
    db_session, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.ai import analysis_pipeline as pipeline_mod
    from app.ai.analysis_pipeline import AnalysisPipelineParams
    from app.ai.intent_types import (
        AIConfidence,
        AIIntent,
        AIIntentType,
        AIMetric,
        AIRequestPlan,
        AggregationType,
    )
    from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
    from app.ai.sql_generation.models import (
        GeneratedSQL,
        SQLGenerationConfidence,
        SQLGenerationOutcome,
        SQLDialect,
    )
    from app.ai.sql_validation.models import ValidatedSQL, SQLValidationResult

    events: list[dict] = []

    async def on_partial(event: dict) -> None:
        events.append(event)

    class _Gen:
        async def generate(self, params):  # noqa: ANN001
            return type(
                "R",
                (),
                {
                    "outcome": SQLGenerationOutcome(
                        generated=GeneratedSQL(
                            sql="SELECT SUM(amount) AS total FROM public.sales",
                            dialect=SQLDialect.POSTGRESQL,
                            referenced_tables=["public.sales"],
                            referenced_columns=["public.sales.amount"],
                            confidence=SQLGenerationConfidence.HIGH,
                            generation_version="v1",
                        )
                    )
                },
            )()

    class _Val:
        def validate(self, params):  # noqa: ANN001
            return type(
                "R",
                (),
                {
                    "result": SQLValidationResult(
                        is_valid=True,
                        validated=ValidatedSQL(
                            sql=params.sql,
                            referenced_tables=["public.sales"],
                            referenced_columns=["public.sales.amount"],
                        ),
                    )
                },
            )()

    class _Exec:
        async def execute(self, params):  # noqa: ANN001
            return type(
                "R",
                (),
                {
                    "result": SQLExecutionResult(
                        status=SQLExecutionStatus.SUCCEEDED,
                        columns=["total"],
                        rows=[[42]],
                        row_count=1,
                    )
                },
            )()

    monkeypatch.setattr(pipeline_mod, "SQLGenerationService", lambda *_a, **_k: _Gen())
    monkeypatch.setattr(pipeline_mod, "SQLValidationService", lambda *_a, **_k: _Val())
    monkeypatch.setattr(pipeline_mod, "SQLExecutionService", lambda *_a, **_k: _Exec())
    monkeypatch.setattr(
        pipeline_mod, "SQLCorrectionService", lambda *_a, **_k: object()
    )
    monkeypatch.setattr(
        pipeline_mod, "build_postgres_mcp", lambda *_a, **_k: (object(), object())
    )
    monkeypatch.setattr(
        pipeline_mod,
        "QueryHistoryService",
        lambda *_a, **_k: type(
            "H",
            (),
            {
                "record_from_validation_result": lambda *a, **k: None,
                "record_from_execution_result": lambda *a, **k: None,
                "record_from_correction_outcome": lambda *a, **k: None,
            },
        )(),
    )

    ds = uuid.uuid4()
    meta = _metadata()
    meta = meta.model_copy(update={"data_source_id": ds})
    intent = AIIntent(
        intent=AIIntentType.AGGREGATION,
        metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_metadata=True,
    )
    plan = AIRequestPlan(
        intent=AIIntentType.AGGREGATION,
        requires_database=True,
        requires_metadata=True,
        requires_aggregation=True,
    )
    result = run_async(
        pipeline_mod.run_analysis_pipeline(
            AnalysisPipelineParams(
                db=db_session,
                session_id=uuid.uuid4(),
                workspace_id=uuid.uuid4(),
                organization_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                data_source_id=ds,
                message="What is total revenue?",
                metadata=meta,
                intent=intent,
                plan=plan,
                run_evaluation=False,
                on_partial=on_partial,
            )
        )
    )
    assert events, "expected a partial facts event before phase8"
    assert events[0]["stage"] == "facts"
    assert "42" in events[0]["answer"]
    assert result.payload is not None
    assert result.payload.performance is not None
    stages = {item["stage"] for item in result.payload.performance["stages"]}
    assert {"sql", "phase8", "total"} <= stages
    assert "42" in result.answer
