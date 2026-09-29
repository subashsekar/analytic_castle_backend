"""Focused tests for conversational follow-up resolution (Step 5)."""

from __future__ import annotations

import uuid
from datetime import date

from app.ai.follow_up import (
    LAST_INTENT_KEY,
    LAST_SPEC_KEY,
    LAST_SQL_KEY,
    apply_follow_up_intent,
    enrich_conversation_context,
    looks_like_follow_up,
    pack_analysis_context,
    prior_analysis_spec,
    prior_sql,
    unpack_prior_intent,
)
from app.ai.intent_types import (
    AIConfidence,
    AIDimension,
    AIIntent,
    AIIntentType,
    AIMetric,
    AITimeRange,
    AggregationType,
    TimeRangePreset,
)
from app.ai.metadata_types import (
    MetadataColumnCandidate,
    MetadataMatchReason,
    MetadataTableCandidate,
    ResolvedMetadataContext,
)


def _prior(**overrides: object) -> AIIntent:
    base = dict(
        intent=AIIntentType.AGGREGATION,
        subject="revenue",
        metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
        time_range=AITimeRange(
            preset=TimeRangePreset.CUSTOM_RANGE,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 3, 31),
        ),
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
        requires_metadata=True,
    )
    base.update(overrides)
    return AIIntent(**base)  # type: ignore[arg-type]


def _metadata_with_region() -> ResolvedMetadataContext:
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
                column_name="region",
                data_type="text",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=10,
            ),
            MetadataColumnCandidate(
                column_id=uuid.uuid4(),
                table_id=table_id,
                schema_name="public",
                table_name="sales",
                column_name="amount",
                data_type="numeric",
                match_reason=MetadataMatchReason.EXACT,
                relevance_score=8,
            ),
        ],
    )


def test_pack_unpack_roundtrip_and_prior_accessors() -> None:
    prior = _prior()
    packed = pack_analysis_context(
        sql="SELECT SUM(amount) FROM sales",
        analysis_spec="Metrics: revenue (SUM)",
        intent=prior,
    )
    assert packed[LAST_SQL_KEY].startswith("SELECT")
    assert "revenue" in packed[LAST_SPEC_KEY]
    assert LAST_INTENT_KEY in packed
    restored = unpack_prior_intent(packed)
    assert restored is not None
    assert restored.metrics[0].name == "revenue"
    assert prior_sql(packed) == "SELECT SUM(amount) FROM sales"
    assert prior_analysis_spec(packed) == "Metrics: revenue (SUM)"


def test_looks_like_follow_up_cues() -> None:
    empty = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    assert looks_like_follow_up("break that down by region", empty)
    assert looks_like_follow_up("only March", empty)
    assert looks_like_follow_up("compare with last year", empty)
    assert not looks_like_follow_up(
        "What is total revenue by product in 2024?",
        AIIntent(
            intent=AIIntentType.AGGREGATION,
            metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
            confidence=AIConfidence.HIGH,
        ),
    )


def test_break_down_by_region_preserves_metrics_and_range() -> None:
    prior = _prior()
    current = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    merged = apply_follow_up_intent(
        prior,
        "break that down by region",
        current,
        metadata=_metadata_with_region(),
    )
    assert [m.name for m in merged.metrics] == ["revenue"]
    assert merged.time_range == prior.time_range
    assert any(d.name == "region" for d in merged.dimensions)
    assert not merged.requires_clarification


def test_only_march_updates_date_range_keeps_metrics() -> None:
    prior = _prior()
    current = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    merged = apply_follow_up_intent(prior, "only March", current, today=date(2026, 9, 28))
    assert [m.name for m in merged.metrics] == ["revenue"]
    assert merged.time_range is not None
    assert merged.time_range.start_date == date(2025, 3, 1)
    assert merged.time_range.end_date == date(2025, 3, 31)


def test_compare_last_year_expands_window() -> None:
    prior = _prior()
    current = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    merged = apply_follow_up_intent(prior, "compare with last year", current)
    assert merged.intent == AIIntentType.COMPARISON
    assert merged.time_range is not None
    assert merged.time_range.start_date == date(2024, 1, 1)
    assert merged.time_range.end_date == date(2025, 3, 31)
    assert [m.name for m in merged.metrics] == ["revenue"]


def test_unknown_dimension_asks_clarification() -> None:
    prior = _prior()
    current = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    merged = apply_follow_up_intent(
        prior,
        "break that down by spaceship",
        current,
        metadata=_metadata_with_region(),
    )
    assert merged.requires_clarification
    assert merged.clarification_question is not None
    assert "spaceship" in merged.clarification_question
    assert [m.name for m in merged.metrics] == ["revenue"]


def test_enrich_conversation_context_includes_prior_sql_and_spec() -> None:
    text = enrich_conversation_context(
        "user: Total revenue in Q1\nassistant: 12,000",
        last_sql="SELECT SUM(amount) AS revenue FROM sales WHERE sale_date >= '2025-01-01'",
        last_analysis_spec="Metrics: revenue (SUM)\nTime range: 2025-01-01 to 2025-03-31",
    )
    assert text is not None
    assert "Previous analysis specification:" in text
    assert "Previous analysis SQL:" in text
    assert "SUM(amount)" in text
    assert "modify only the filters" in text


def test_follow_up_keeps_existing_dimension_when_filtering_month() -> None:
    prior = _prior(dimensions=[AIDimension(name="region")])
    current = AIIntent(intent=AIIntentType.UNKNOWN, confidence=AIConfidence.LOW)
    merged = apply_follow_up_intent(prior, "only March 2025", current)
    assert [d.name for d in merged.dimensions] == ["region"]
    assert merged.time_range is not None
    assert merged.time_range.start_date == date(2025, 3, 1)
