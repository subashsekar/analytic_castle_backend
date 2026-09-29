"""Multi-step investigation planning and contributor ranking (no LLM, no database)."""

from __future__ import annotations

from datetime import date

from app.ai.analysis_profile import build_analysis_profile
from app.ai.intent_types import AIIntent, AIIntentType, AIMetric, AggregationType
from app.ai.investigation.contributors import format_contributions, rank_contributors
from app.ai.investigation.plan import build_investigation_plan, investigation_plan_summary
from app.ai.investigation.sql_build import build_investigation_sqls
from app.ai.planner_agent.investigation_steps import attach_investigation_plan
from app.ai.planner_agent.validation import plan_from_intent
from app.ai.sql_validation.validation import validate_generated_sql
from evals.analyst.demo_schema import demo_metadata

def _sales_metadata():
    return demo_metadata()


def test_diagnostic_plan_includes_change_breakdown_and_drill() -> None:
    message = "Why did revenue fall in March 2025?"
    meta = _sales_metadata()
    profile = build_analysis_profile(message)
    steps = build_investigation_plan(message, meta, profile=profile)
    kinds = [step.kind.value for step in steps]
    assert "overall_change" in kinds
    assert "breakdown" in kinds
    assert any("Investigation plan" in investigation_plan_summary(steps) for _ in [0])
    for step in steps:
        result = validate_generated_sql(step.sql, meta)
        assert result.is_valid, (step.kind, result.violations)


def test_yoy_and_funnel_steps_when_schema_and_question_match() -> None:
    meta = _sales_metadata()
    yoy = build_investigation_sqls(
        "Compare revenue year over year in 2025",
        meta,
        intent=None,
        profile=build_analysis_profile("Compare revenue year over year in 2025"),
        max_steps=5,
        today=date(2026, 9, 28),
    )
    assert any(kind == "year_over_year" for kind, _, _ in yoy)
    funnel = build_investigation_sqls(
        "What is the conversion funnel performance in March 2025?",
        meta,
        intent=None,
        profile=build_analysis_profile("conversion funnel March 2025"),
        max_steps=5,
        today=date(2026, 9, 28),
    )
    assert any(kind == "funnel" for kind, _, _ in funnel)


def test_cohort_step_skipped_without_customer_column() -> None:
    meta = _sales_metadata()
    steps = build_investigation_sqls(
        "Show cohort retention by month",
        meta,
        intent=None,
        profile=build_analysis_profile("cohort retention"),
        max_steps=5,
    )
    assert not any(kind == "cohort" for kind, _, _ in steps)


def test_contributors_rank_largest_delta_first() -> None:
    rows = [
        [date(2025, 2, 1), "East", 100],
        [date(2025, 2, 1), "West", 50],
        [date(2025, 3, 1), "East", 80],
        [date(2025, 3, 1), "West", 45],
    ]
    ranked = rank_contributors(
        columns=["period", "region", "revenue"],
        rows=rows,
        dimension_column="region",
    )
    assert ranked[0].segment == "East"
    assert ranked[0].delta == -20
    text = format_contributions(ranked)
    assert "correlation" in text.lower() and "not proven causation" in text.lower()


def test_planner_action_steps_extended_for_diagnostic_plan() -> None:
    intent = AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        metrics=[AIMetric(name="revenue", aggregation=AggregationType.SUM)],
        requires_data_access=True,
    )
    base = plan_from_intent(intent)
    enriched = attach_investigation_plan(
        base,
        message="Why did revenue fall in March 2025?",
        metadata=_sales_metadata(),
        intent=intent,
    )
    assert len(enriched.investigation_steps) >= 2
    assert len(enriched.action_steps) > len(base.action_steps)
    assert any("Investigation step" in step.description for step in enriched.action_steps)


def test_simple_aggregate_question_has_no_investigation_steps() -> None:
    message = "What is the total revenue?"
    steps = build_investigation_plan(message, _sales_metadata())
    assert steps == []
