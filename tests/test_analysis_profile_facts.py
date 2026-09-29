"""Question profiling, analysis spec, and deterministic result facts."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest

from app.ai.analysis_profile import (
    QuestionKind,
    build_analysis_profile,
    build_analysis_spec,
    build_conversation_context,
)
from app.ai.intent_types import (
    AggregationType,
    AIDimension,
    AIIntent,
    AIIntentType,
    AIMetric,
    AITimeRange,
    TimeRangePreset,
)
from app.ai.metadata_types import (
    ConceptColumnResolution,
    MetadataColumnCandidate,
    MetadataMatchReason,
    ResolvedMetadataContext,
)
from app.ai.result_facts import build_result_facts
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.state.models import ConversationContextData, ConversationMessage


def _result(columns: list[str], rows: list[list[object]]) -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


@pytest.mark.parametrize(
    ("message", "expected", "agents"),
    [
        ("What is the total number of orders?", {QuestionKind.AGGREGATE}, set()),
        ("Top 5 customers by spend", {QuestionKind.RANKING}, {"data_analyst"}),
        ("Show monthly signups over time", {QuestionKind.TREND}, {"trend", "anomaly"}),
        ("Were there any unusual spikes in tickets?", {QuestionKind.ANOMALY}, {"trend", "anomaly"}),
        (
            "Why did bookings drop last quarter?",
            {QuestionKind.DIAGNOSTIC},
            {"trend", "anomaly", "root_cause", "insight"},
        ),
        (
            "What should we do to improve retention?",
            {QuestionKind.RECOMMENDATION},
            {"insight", "recommendation"},
        ),
        ("Compare Q1 vs Q2 costs", {QuestionKind.COMPARISON}, {"data_analyst"}),
    ],
)
def test_profile_selects_only_relevant_agents(message, expected, agents) -> None:
    profile = build_analysis_profile(message)
    assert expected <= profile.kinds
    selected = {
        name
        for name, flag in (
            ("data_analyst", profile.run_data_analyst),
            ("trend", profile.run_trend),
            ("anomaly", profile.run_anomaly),
            ("root_cause", profile.run_root_cause),
            ("insight", profile.run_insight),
            ("recommendation", profile.run_recommendation),
        )
        if flag
    }
    assert agents <= selected
    if not agents:
        assert not profile.uses_llm_agents


def test_intent_type_contributes_to_profile() -> None:
    intent = AIIntent(intent=AIIntentType.TREND_ANALYSIS)
    assert QuestionKind.TREND in build_analysis_profile("show it", intent=intent).kinds


def test_analysis_spec_is_schema_grounded() -> None:
    ds = uuid.uuid4()
    table_id = uuid.uuid4()
    column = MetadataColumnCandidate(
        column_id=uuid.uuid4(),
        table_id=table_id,
        schema_name="public",
        table_name="orders",
        column_name="amount",
        data_type="numeric",
        match_reason=MetadataMatchReason.EXACT,
        relevance_score=100,
    )
    metadata = ResolvedMetadataContext(
        data_source_id=ds,
        resolved_metrics=[
            ConceptColumnResolution(requested="sales", resolved=True, candidates=[column])
        ],
        unresolved_concepts=["margin"],
    )
    intent = AIIntent(
        intent=AIIntentType.AGGREGATION,
        metrics=[AIMetric(name="sales", aggregation=AggregationType.SUM)],
        dimensions=[AIDimension(name="region")],
        time_range=AITimeRange(preset=TimeRangePreset.LAST_MONTH),
    )
    spec = build_analysis_spec(
        intent=intent,
        plan=None,
        metadata=metadata,
        profile=build_analysis_profile("sales by region last month", intent=intent),
        today=date(2026, 9, 28),
    )
    assert "Metrics: sales (SUM)" in spec
    assert "Dimensions: region" in spec
    assert "last_month" in spec and "Today: 2026-09-28" in spec
    assert "metric 'sales' -> public.orders.amount" in spec
    assert "do not substitute" in spec and "margin" in spec


def test_conversation_context_excludes_current_message() -> None:
    conversation = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="Total sales in 2024"),
            ConversationMessage(role="assistant", content="Total sales in 2024 were 1,000."),
            ConversationMessage(role="user", content="Break that down by region"),
        ]
    )
    text = build_conversation_context(conversation, current_message="Break that down by region")
    assert text is not None
    assert "user: Total sales in 2024" in text
    assert "Break that down" not in text


def test_facts_scalar_and_empty() -> None:
    scalar = build_result_facts(_result(["total_amount"], [[Decimal("1234.5")]]))
    assert scalar.headline == "total amount: 1,234.50"
    empty = build_result_facts(_result(["x"], []))
    assert empty.is_empty and empty.notes


def test_facts_time_series_targets_named_period_and_contributors() -> None:
    rows = [
        [date(2025, 1, 1), "A", 10],
        [date(2025, 1, 1), "B", 10],
        [date(2025, 2, 1), "A", 2],
        [date(2025, 2, 1), "B", 9],
        [date(2025, 3, 1), "A", 12],
        [date(2025, 3, 1), "B", 10],
    ]
    facts = build_result_facts(
        _result(["period", "segment", "value"], rows), message="Why did value dip in February 2025?"
    )
    assert facts.headline == "value for Feb 2025 was 11, down 9 (-45.0%) from Jan 2025 (20)."
    assert any("Largest contributors by segment" in line and "A -8" in line for line in facts.lines)


def test_facts_categorical_ranking_and_non_additive_measures() -> None:
    facts = build_result_facts(_result(["city", "orders"], [["X", 30], ["Y", 70]]))
    assert facts.headline == "Highest orders by city: Y with 70 (70% of total)."
    averages = build_result_facts(_result(["city", "avg_price"], [["X", 3.5], ["Y", 7.25]]))
    assert "of total" not in (averages.headline or "")
    assert not any("total avg" in line for line in averages.lines)


def test_facts_ignore_id_columns_as_measures() -> None:
    facts = build_result_facts(_result(["product_id", "units"], [[3, 500], [1, 420]]))
    assert facts.shape == "categorical"
    assert "units" in (facts.headline or "")
