"""Tests for the Insight Agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.anomaly_detection import scan_for_anomalies
from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.insight import (
    InsightAgent,
    InsightAuthorizationError,
    InsightLLMError,
    InsightPriority,
    InsightValidationError,
    has_evidence,
    identify_key_metrics,
)
from app.ai.llm import AsyncLLMClient
from app.ai.root_cause_analysis.models import (
    RootCauseAnalysisResult,
    RootCauseHypothesis,
)
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.trend_analysis import build_trend_series
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.db.models import DataSource
from tests.conftest import run_async
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace

_TREND_ROWS = [["2024-01", 100], ["2024-02", 105], ["2024-03", 60]]


def _execution_result(
    columns: list[str] | None = None,
    rows: list[list[Any]] | None = None,
) -> SQLExecutionResult:
    resolved_rows = _TREND_ROWS if rows is None else rows
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=columns or ["month", "revenue"],
        rows=resolved_rows,
        row_count=len(resolved_rows),
    )


def _analysis_result() -> DataAnalysisResult:
    return DataAnalysisResult(
        interpretation="Revenue peaked in February and fell in March.",
        summary="Total revenue was 265 across three months.",
        comparisons="March is 42.9% below February.",
        conclusions=["March revenue fell to 60."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only three months are available.",
    )


def _trend_result() -> TrendAnalysisResult:
    series = build_trend_series(["month", "revenue"], _TREND_ROWS)
    return TrendAnalysisResult(
        direction=series.direction,
        growth_rate_percent=series.growth_rate_percent,
        series=series,
        summary="Revenue fell sharply in March.",
        direction_explanation="Revenue dropped from 105 to 60.",
        period_comparisons="2024-03 fell 42.9% against 2024-02.",
        significant_changes="2024-03 is a significant decline.",
        conclusions=["Revenue declined in 2024-03."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only three periods are available.",
    )


def _anomaly_result() -> AnomalyAnalysisResult:
    scan = scan_for_anomalies(["month", "revenue"], _TREND_ROWS)
    return AnomalyAnalysisResult(
        anomaly_count=len(scan.anomalies),
        highest_severity=scan.highest_severity,
        scan=scan,
        summary="One unexpected decline was detected.",
        outliers="No statistical outlier was detected.",
        unexpected_changes="2024-03 fell 42.9% against 2024-02.",
        threshold_breaches="No threshold breach was detected.",
        conclusions=["2024-03 declined unexpectedly."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Three periods is a small sample.",
    )


def _root_cause_result() -> RootCauseAnalysisResult:
    return RootCauseAnalysisResult(
        summary="March revenue fell 42.9%.",
        primary_cause="The East region stopped ordering.",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                statement="The East region stopped ordering.",
                contributing_factors=["regional demand"],
                supporting_evidence="East revenue fell to 60 in 2024-03.",
                confidence_score=AnalysisConfidence.MEDIUM,
                confidence_reasoning="Only one regional breakdown was available.",
            )
        ],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Only one regional breakdown was available.",
    )


def _insight(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "title": "Revenue reversed in March",
        "insight": "Revenue fell from 105 in February to 60 in March.",
        "metric": "revenue",
        "business_impact": "A 42.9% drop in the latest month reduces the quarter total to 265.",
        "supporting_evidence": "The revenue metric totals 265 with a maximum of 105 and a March value of 60.",
        "priority": "HIGH",
        "confidence_score": "MEDIUM",
        "confidence_reasoning": "Three months is a small sample.",
    }
    body.update(overrides)
    return body


def _output(*insights: dict[str, Any], data_gaps: list[str] | None = None) -> dict[str, Any]:
    return {
        "summary": "Revenue grew into February and then reversed sharply in March.",
        "insights": list(insights) or [_insight()],
        "data_gaps": data_gaps or ["The results do not break revenue down by region."],
    }


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-insight-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _setup(db_session: Session, handler: Any = None) -> tuple[Any, Any, Any, InsightAgent]:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service = AgentStateService(db_session)
    snapshot = state_service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
        )
    )
    agent = InsightAgent(
        db_session,
        llm_client=AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))
        if handler
        else None,
        state_service=state_service,
    )
    return user, workspace, snapshot, agent


# --- Deterministic key metrics and evidence gate (no DB, no LLM) -------------


def test_identify_key_metrics_skips_the_period_column() -> None:
    metrics = identify_key_metrics(_execution_result())

    assert [metric.column for metric in metrics] == ["revenue"]
    assert metrics[0].total == 265.0
    assert metrics[0].maximum == 105.0
    assert metrics[0].minimum == 60.0
    assert metrics[0].average == pytest.approx(88.333333, rel=1e-6)


def test_identify_key_metrics_ignores_non_numeric_columns() -> None:
    result = _execution_result(["region", "revenue"], [["East", 60], ["West", None]])

    assert [metric.column for metric in identify_key_metrics(result)] == ["revenue"]


def test_has_evidence_requires_results_or_analysis() -> None:
    assert has_evidence(_execution_result(), None, None, None, None) is True
    assert has_evidence(None, _analysis_result(), None, None, None) is True
    assert has_evidence(_execution_result([], []), None, None, None, None) is False
    assert has_evidence(None, None, None, None, None) is False


# --- Agent behaviour --------------------------------------------------------


def test_insight_success(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue do this quarter?",
            sql="SELECT month, revenue FROM sales",
            query_result=_execution_result(),
            analysis=_analysis_result(),
            trend=_trend_result(),
            anomalies=_anomaly_result(),
            root_cause=_root_cause_result(),
        )
    )

    assert len(result.insights) == 1
    assert result.insights[0].rank == 1
    assert result.insights[0].priority is InsightPriority.HIGH
    assert result.insights[0].metric == "revenue"
    assert result.top_insight == "Revenue fell from 105 in February to 60 in March."
    assert result.confidence_score is AnalysisConfidence.MEDIUM
    assert [metric.column for metric in result.key_metrics] == ["revenue"]
    assert result.data_gaps == ["The results do not break revenue down by region."]
    assert result.notes == []


def test_insight_prioritizes_by_priority_then_confidence(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _insight(title="Low", priority="LOW", confidence_score="HIGH"),
                    _insight(title="Medium", priority="MEDIUM", confidence_score="HIGH"),
                    _insight(title="High weak", priority="HIGH", confidence_score="LOW"),
                    _insight(title="High strong", priority="HIGH", confidence_score="HIGH"),
                )
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue do?",
            query_result=_execution_result(),
        )
    )

    assert [item.title for item in result.insights] == [
        "High strong",
        "High weak",
        "Medium",
        "Low",
    ]
    assert [item.rank for item in result.insights] == [1, 2, 3, 4]
    assert result.confidence_score is AnalysisConfidence.HIGH


def test_insight_drops_ungrounded_metric(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _insight(title="Invented", metric="gross_margin", priority="HIGH"),
                    _insight(title="Grounded", metric="REVENUE", priority="MEDIUM"),
                )
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue do?",
            query_result=_execution_result(),
        )
    )

    assert [item.title for item in result.insights] == ["Grounded"]
    # The cited name is normalized back to the column spelling in the results.
    assert result.insights[0].metric == "revenue"
    assert any("gross_margin" in note for note in result.notes)


def test_insight_rejects_output_with_no_grounded_insight(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_response(_output(_insight(metric="gross_margin")))
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(InsightValidationError, match="grounded"):
        run_async(
            agent.generate(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue do?",
                query_result=_execution_result(),
            )
        )


def test_insight_skips_llm_without_evidence(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called without analysis evidence")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should I know?",
        )
    )

    assert result.insights == []
    assert result.top_insight is None
    assert result.confidence_score is AnalysisConfidence.LOW
    assert "no business insight could be derived" in result.summary


def test_insight_works_from_analysis_without_query_results(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_response(_output(_insight(metric=None)))
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should I know?",
            trend=_trend_result(),
        )
    )

    assert len(result.insights) == 1
    assert result.insights[0].metric is None
    assert result.key_metrics == []


def test_insight_authorization_error(db_session: Session) -> None:
    user, workspace, snapshot, agent = _setup(db_session)

    _, other_workspace, _ = _seed_workspace(db_session)
    source = db_session.get(DataSource, snapshot.data_source_id)
    assert source is not None
    source.workspace_id = other_workspace.id
    db_session.flush()

    with pytest.raises(InsightAuthorizationError, match="not accessible"):
        run_async(
            agent.generate(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue do?",
                query_result=_execution_result(),
            )
        )


def test_insight_validation_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response({"summary": "x", "insights": []}))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(InsightValidationError, match="failed validation"):
        run_async(
            agent.generate(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue do?",
                query_result=_execution_result(),
            )
        )


def test_insight_llm_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(InsightLLMError):
        run_async(
            agent.generate(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue do?",
                query_result=_execution_result(),
            )
        )


def test_insight_updates_agent_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue do?",
            query_result=_execution_result(),
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    updated = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    custom = updated.agent_state.payload.custom
    assert custom.get("insight_action") == "INSIGHT_COMPLETED"
    assert custom.get("insight_count") == "1"
    assert custom.get("insight_confidence_score") == "MEDIUM"


def test_insight_prompt_carries_upstream_evidence(db_session: Session) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content.decode())
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue do?",
            query_result=_execution_result(),
            analysis=_analysis_result(),
            trend=_trend_result(),
            anomalies=_anomaly_result(),
            root_cause=_root_cause_result(),
        )
    )

    body = captured[0]
    assert "key_metrics" in body
    assert "data_analysis" in body
    assert "East region stopped ordering" in body
    assert "2024-03" in body


def test_insight_records_skipped_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("LLM must not be called without evidence")

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should I know?",
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    updated = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    assert updated.agent_state.payload.custom.get("insight_action") == "INSIGHT_SKIPPED"
