"""Tests for the Trend Analysis Agent."""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import AsyncLLMClient
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.trend_analysis import (
    TrendAnalysisAgent,
    TrendAnalysisAuthorizationError,
    TrendAnalysisLLMError,
    TrendAnalysisResult,
    TrendAnalysisValidationError,
    TrendDirection,
    build_trend_series,
)
from tests.conftest import run_async
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace


def _trend_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": "Revenue rose from 100 in 2024-01 to 160 in 2024-04.",
        "direction_explanation": "The series is increasing, growing 60% overall.",
        "period_comparisons": "Each month gained, with the largest jump in 2024-03.",
        "significant_changes": "2024-03 rose 33.33% over 2024-02.",
        "conclusions": ["Revenue grew every month, from 100 to 160."],
        "confidence_score": "HIGH",
        "confidence_reasoning": "Four consecutive monthly periods with no missing data.",
    }
    body.update(overrides)
    return body


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-trend-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _mock_client(handler: Any) -> AsyncLLMClient:
    return AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))


def _create_session(
    db_session: Session,
    *,
    user: Any,
    workspace: Any,
    organization: Any,
    data_source: Any = None,
) -> tuple[AgentStateService, Any]:
    service = AgentStateService(db_session)
    snapshot = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id if data_source else None,
        )
    )
    return service, snapshot


def _monthly_result(rows: list[list[Any]]) -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["month", "revenue"],
        rows=rows,
        row_count=len(rows),
    )


def _setup(db_session: Session, handler: Any = None) -> tuple[Any, Any, Any, TrendAnalysisAgent]:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    agent = TrendAnalysisAgent(
        db_session,
        llm_client=_mock_client(handler) if handler else None,
        state_service=state_service,
    )
    return user, workspace, snapshot, agent


# --- Deterministic series computation (no DB, no LLM) -----------------------


def test_series_detects_growth_and_significant_changes() -> None:
    series = build_trend_series(
        ["month", "revenue"],
        [["2024-01", 100], ["2024-02", 120], ["2024-03", 160], ["2024-04", 168]],
    )

    assert series.period_column == "month"
    assert series.value_column == "revenue"
    assert series.direction is TrendDirection.INCREASING
    assert series.growth_rate_percent == 68.0
    assert series.point_count == 4
    assert [item.percent_change for item in series.comparisons] == [20.0, 33.3333, 5.0]
    assert [item.period for item in series.significant_changes] == ["2024-02", "2024-03"]


def test_series_detects_decline() -> None:
    series = build_trend_series(
        ["day", "signups"],
        [["2024-01-01", 500], ["2024-01-02", 400], ["2024-01-03", 250]],
    )

    assert series.direction is TrendDirection.DECREASING
    assert series.growth_rate_percent == -50.0
    assert all(item.direction is TrendDirection.DECREASING for item in series.comparisons)


def test_series_sorts_unordered_rows() -> None:
    series = build_trend_series(
        ["period", "total"],
        [["2024-03", 300], ["2024-01", 100], ["2024-02", 200]],
    )

    assert series.reordered is True
    assert series.first_period == "2024-01"
    assert series.last_period == "2024-03"
    assert series.direction is TrendDirection.INCREASING
    assert any("chronological" in note for note in series.notes)


def test_series_flags_stable_and_volatile() -> None:
    stable = build_trend_series(
        ["month", "value"],
        [["2024-01", 100], ["2024-02", 101], ["2024-03", 102]],
    )
    volatile = build_trend_series(
        ["month", "value"],
        [["2024-01", 100], ["2024-02", 180], ["2024-03", 90], ["2024-04", 170], ["2024-05", 105]],
    )

    assert stable.direction is TrendDirection.STABLE
    assert volatile.direction is TrendDirection.VOLATILE
    assert volatile.direction_changes >= 2


def test_series_skips_missing_values_and_sums_duplicates() -> None:
    series = build_trend_series(
        ["month", "revenue"],
        [
            ["2024-01", 100],
            ["2024-01", 50],
            [None, 999],
            ["2024-02", None],
            ["2024-03", 300],
        ],
    )

    assert series.skipped_row_count == 2
    assert series.first_value == 150.0
    assert series.point_count == 2
    assert any("summed" in note for note in series.notes)
    assert any("skipped" in note for note in series.notes)


def test_series_handles_non_time_series_and_short_data() -> None:
    non_series = build_trend_series(["region", "revenue"], [["East", 10], ["West", 20]])
    single_point = build_trend_series(["month", "revenue"], [["2024-01", 10]])
    empty = build_trend_series(["month", "revenue"], [])

    assert non_series.direction is TrendDirection.INSUFFICIENT_DATA
    assert any("not a time series" in note for note in non_series.notes)
    assert single_point.direction is TrendDirection.INSUFFICIENT_DATA
    assert any("at least 2" in note for note in single_point.notes)
    assert empty.direction is TrendDirection.INSUFFICIENT_DATA


def test_series_handles_native_types_and_zero_baseline() -> None:
    series = build_trend_series(
        ["day", "amount"],
        [[date(2024, 1, 1), Decimal("0")], [date(2024, 1, 2), Decimal("25.5")]],
    )

    assert series.point_count == 2
    assert series.growth_rate_percent is None
    assert series.direction is TrendDirection.INCREASING
    assert series.comparisons[0].percent_change is None
    assert series.comparisons[0].significant is True


# --- Agent behaviour --------------------------------------------------------


def test_trend_analysis_success(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_trend_body()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend this year?",
            sql="SELECT month, revenue FROM sales ORDER BY month",
            query_result=_monthly_result(
                [["2024-01", 100], ["2024-02", 120], ["2024-03", 160], ["2024-04", 160]]
            ),
        )
    )

    assert isinstance(result, TrendAnalysisResult)
    assert result.direction is TrendDirection.INCREASING
    assert result.growth_rate_percent == 60.0
    assert result.confidence_score is AnalysisConfidence.HIGH
    assert result.series.point_count == 4
    assert len(result.series.significant_changes) == 2


def test_trend_analysis_skips_llm_for_non_time_series(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called without a usable series")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend?",
            query_result=SQLExecutionResult(
                status=SQLExecutionStatus.SUCCEEDED,
                columns=["region", "revenue"],
                rows=[["East", 42]],
                row_count=1,
            ),
        )
    )

    assert result.direction is TrendDirection.INSUFFICIENT_DATA
    assert result.growth_rate_percent is None
    assert result.confidence_score is AnalysisConfidence.LOW
    assert result.conclusions == []


def test_trend_analysis_missing_query_result(db_session: Session) -> None:
    user, workspace, snapshot, agent = _setup(db_session)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend?",
        )
    )

    assert result.direction is TrendDirection.INSUFFICIENT_DATA
    assert result.confidence_score is AnalysisConfidence.LOW


def test_trend_analysis_authorization_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    agent = TrendAnalysisAgent(db_session, state_service=state_service)

    _, other_workspace, _ = _seed_workspace(db_session)
    data_source.workspace_id = other_workspace.id
    db_session.flush()

    with pytest.raises(TrendAnalysisAuthorizationError, match="not accessible"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue trend?",
            )
        )


def test_trend_analysis_validation_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response({"invalid_field": "value"}))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(TrendAnalysisValidationError, match="failed validation"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue trend?",
                query_result=_monthly_result([["2024-01", 100], ["2024-02", 200]]),
            )
        )


def test_trend_analysis_llm_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(TrendAnalysisLLMError):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="How did revenue trend?",
                query_result=_monthly_result([["2024-01", 100], ["2024-02", 200]]),
            )
        )


def test_trend_analysis_updates_agent_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_trend_body()))

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend?",
            query_result=_monthly_result([["2024-01", 100], ["2024-02", 200]]),
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
    assert custom.get("trend_action") == "TREND_ANALYSIS_COMPLETED"
    assert custom.get("trend_direction") == "INCREASING"
    assert custom.get("trend_confidence_score") == "HIGH"


def test_trend_analysis_records_skipped_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("LLM must not be called when series is insufficient")

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend?",
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    updated = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    assert updated.agent_state.payload.custom.get("trend_action") == "TREND_ANALYSIS_SKIPPED"


def test_trend_analysis_computed_metrics_override_llm_claims(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _trend_body(
                    summary="The model invents an increasing story.",
                    direction_explanation="Claimed increasing despite a decline.",
                    significant_changes="2024-03 rose somehow.",
                    conclusions=["Invented growth."],
                )
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="How did revenue trend?",
            query_result=_monthly_result([["2024-01", 100], ["2024-02", 105], ["2024-03", 60]]),
        )
    )

    assert result.direction is TrendDirection.DECREASING
    assert result.growth_rate_percent == pytest.approx(-40.0)
    assert "Invented growth." in result.conclusions
