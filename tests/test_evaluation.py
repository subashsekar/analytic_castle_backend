"""Tests for Phase 8 agent evaluation (deterministic checks, no LLM)."""

from __future__ import annotations

from uuid import uuid4

from app.ai.anomaly_detection import scan_for_anomalies
from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.evaluation import (
    CheckStatus,
    EvaluatedAgent,
    EvaluationMethod,
    evaluate_agents,
    evidence_numbers,
    ungrounded_numbers,
)
from app.ai.insight.evidence import identify_key_metrics
from app.ai.insight.models import (
    BusinessInsight,
    InsightAnalysisResult,
    InsightPriority,
)
from app.ai.recommendation.models import (
    Recommendation,
    RecommendationLevel,
    RecommendationResult,
)
from app.ai.root_cause_analysis.models import (
    RootCauseAnalysisResult,
    RootCauseHypothesis,
)
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.trend_analysis import build_trend_series
from app.ai.trend_analysis.models import TrendAnalysisResult, TrendDirection

_ROWS = [["2024-01", 100], ["2024-02", 105], ["2024-03", 60]]


def _query_result() -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["month", "revenue"],
        rows=_ROWS,
        row_count=len(_ROWS),
        duration_ms=12.5,
    )


def _analysis(*, confidence: AnalysisConfidence = AnalysisConfidence.MEDIUM) -> DataAnalysisResult:
    return DataAnalysisResult(
        interpretation="Revenue peaked then fell in March to 60.",
        summary="Total revenue across three months was 265.",
        comparisons="March revenue of 60 is below February's 105.",
        conclusions=["March revenue fell to 60."],
        confidence_score=confidence,
        confidence_reasoning="Three months is a small sample.",
    )


def _trend(*, direction: TrendDirection | None = None) -> TrendAnalysisResult:
    series = build_trend_series(["month", "revenue"], _ROWS)
    return TrendAnalysisResult(
        direction=direction or series.direction,
        growth_rate_percent=series.growth_rate_percent,
        series=series,
        summary="Revenue fell sharply in March.",
        direction_explanation="Revenue dropped from 105 to 60.",
        period_comparisons="2024-03 fell against 2024-02.",
        significant_changes="2024-03 is a significant decline.",
        conclusions=["Revenue declined in 2024-03."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only three periods are available.",
    )


def _anomalies(*, count_override: int | None = None) -> AnomalyAnalysisResult:
    scan = scan_for_anomalies(["month", "revenue"], _ROWS)
    return AnomalyAnalysisResult(
        anomaly_count=count_override if count_override is not None else len(scan.anomalies),
        highest_severity=scan.highest_severity,
        scan=scan,
        summary="One unexpected decline was detected in revenue.",
        outliers="No statistical outlier was detected.",
        unexpected_changes="2024-03 fell against 2024-02.",
        threshold_breaches="No threshold breach was detected.",
        conclusions=["2024-03 declined unexpectedly in revenue."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Three periods is a small sample.",
    )


def _root_cause(
    *,
    high_open: bool = False,
    primary_mismatch: bool = False,
) -> RootCauseAnalysisResult:
    statement = "The East region stopped ordering."
    hypothesis = RootCauseHypothesis(
        rank=1,
        statement=statement,
        contributing_factors=["regional demand"],
        supporting_evidence="East revenue fell to 60 in 2024-03.",
        confidence_score=AnalysisConfidence.HIGH if high_open else AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only one regional breakdown was available.",
        investigation_question="Break revenue down by region?" if high_open else None,
    )
    return RootCauseAnalysisResult(
        summary="March revenue fell 42.9%.",
        primary_cause="Something else." if primary_mismatch else statement,
        hypotheses=[hypothesis],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only one regional breakdown was available.",
    )


def _insights(*, ungrounded_metric: bool = False) -> InsightAnalysisResult:
    metrics = identify_key_metrics(_query_result())
    item = BusinessInsight(
        rank=1,
        title="Revenue reversed in March",
        insight="Revenue fell from 105 in February to 60 in March.",
        metric="gross_margin" if ungrounded_metric else "revenue",
        business_impact="A 42.9% drop reduces the quarter total to 265.",
        supporting_evidence="Revenue totals 265 with a March value of 60.",
        priority=InsightPriority.HIGH,
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Three months is a small sample.",
    )
    return InsightAnalysisResult(
        summary="Revenue reversed sharply in March.",
        top_insight=item.insight,
        insights=[item],
        key_metrics=metrics,
        data_gaps=["No regional breakdown."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Three months is a small sample.",
    )


def _recommendations(
    *,
    wrong_priority: bool = False,
    guaranteed: bool = False,
) -> RecommendationResult:
    impact = RecommendationLevel.HIGH
    feasibility = RecommendationLevel.HIGH
    priority = (
        RecommendationLevel.LOW
        if wrong_priority
        else RecommendationLevel.HIGH
    )
    item = Recommendation(
        rank=1,
        title="Investigate March drop",
        recommendation="Review cancelled orders for March.",
        evidence_reference="revenue",
        supporting_evidence="Revenue fell from 105 to 60.",
        expected_outcome=(
            "This will increase revenue next quarter."
            if guaranteed
            else "May help identify whether the drop is recoverable."
        ),
        assumptions=["Ops can review cancellations."],
        risks=["Drop may be seasonal."],
        impact=impact,
        feasibility=feasibility,
        priority=priority,
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Cause is still unknown.",
    )
    return RecommendationResult(
        summary="Act on the March revenue reversal.",
        top_recommendation=item.recommendation,
        recommendations=[item],
        data_gaps=["Regional demand data is missing."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Cause is still unknown.",
    )


def test_evaluate_agents_with_no_outputs() -> None:
    report = evaluate_agents()

    assert report.agents == []
    assert set(report.agents_not_evaluated) == set(EvaluatedAgent)
    assert report.verified_correct is False
    assert any("nothing could be evaluated" in note for note in report.notes)
    assert any("No query result" in note for note in report.notes)


def test_evaluate_agents_full_pipeline_passes() -> None:
    session_id = uuid4()
    query = _query_result()
    report = evaluate_agents(
        session_id=session_id,
        query_result=query,
        analysis=_analysis(),
        trend=_trend(),
        anomalies=_anomalies(),
        root_cause=_root_cause(),
        insights=_insights(),
        recommendations=_recommendations(),
    )

    assert report.session_id == session_id
    assert report.metrics.agents_evaluated == 6
    assert report.agents_not_evaluated == []
    assert report.verified_correct is True
    assert report.failures == []
    assert report.metrics.query_duration_ms == 12.5


def test_evaluate_agents_partial_marks_missing() -> None:
    report = evaluate_agents(query_result=_query_result(), analysis=_analysis())

    assert [item.agent for item in report.agents] == [EvaluatedAgent.DATA_ANALYST]
    assert EvaluatedAgent.TREND_ANALYSIS in report.agents_not_evaluated
    assert EvaluatedAgent.RECOMMENDATION in report.agents_not_evaluated


def test_verified_failure_on_trend_direction_mismatch() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        trend=_trend(direction=TrendDirection.INCREASING),
    )

    failures = [check for check in report.failures if check.check_id == "trend.direction_matches_series"]
    assert len(failures) == 1
    assert failures[0].method is EvaluationMethod.VERIFIED
    assert failures[0].status is CheckStatus.FAIL
    assert report.verified_correct is False


def test_verified_failure_on_anomaly_count_mismatch() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        anomalies=_anomalies(count_override=99),
    )

    assert any(check.check_id == "anomaly.count_matches_scan" for check in report.failures)


def test_verified_failure_on_insight_ungrounded_metric() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        insights=_insights(ungrounded_metric=True),
    )

    assert any(check.check_id == "insight.metrics_grounded" for check in report.failures)


def test_verified_failure_on_recommendation_wrong_priority() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        insights=_insights(),
        recommendations=_recommendations(wrong_priority=True),
    )

    assert any(check.check_id == "recommendation.priority_derived" for check in report.failures)


def test_verified_failure_on_root_cause_high_open_question() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        root_cause=_root_cause(high_open=True),
    )

    assert any(
        check.check_id == "root_cause.high_confidence_not_open_question"
        for check in report.failures
    )


def test_verified_failure_on_analyst_high_confidence_empty_rows() -> None:
    empty = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["month", "revenue"],
        rows=[],
        row_count=0,
    )
    report = evaluate_agents(
        query_result=empty,
        analysis=_analysis(confidence=AnalysisConfidence.HIGH),
    )

    assert any(
        check.check_id == "data_analyst.confidence_requires_evidence"
        for check in report.failures
    )


def test_estimated_warning_on_guaranteed_outcome() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        insights=_insights(),
        recommendations=_recommendations(guaranteed=True),
    )

    warnings = [
        check
        for check in report.warnings
        if check.check_id == "recommendation.outcomes_not_guaranteed"
    ]
    assert len(warnings) == 1
    assert warnings[0].method is EvaluationMethod.ESTIMATED
    assert warnings[0].status is CheckStatus.WARN
    # ESTIMATED warnings must not flip verified_correct by themselves when other verified checks pass.
    assert report.metrics.estimated_warnings >= 1


def test_ungrounded_numbers_helper() -> None:
    evidence = evidence_numbers(query_result=_query_result())
    assert 60.0 in evidence
    assert 265.0 in evidence or 100.0 in evidence

    unmatched = ungrounded_numbers(["Revenue jumped to 9999 overnight."], evidence)
    assert 9999.0 in unmatched
    assert ungrounded_numbers(["Revenue fell to 60."], evidence) == []


def test_root_cause_primary_mismatch_fails() -> None:
    report = evaluate_agents(
        query_result=_query_result(),
        root_cause=_root_cause(primary_mismatch=True),
    )

    assert any(
        check.check_id == "root_cause.primary_matches_top_hypothesis"
        for check in report.failures
    )
