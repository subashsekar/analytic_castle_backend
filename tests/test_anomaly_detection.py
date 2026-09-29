"""Tests for the Anomaly Detection Agent."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.anomaly_detection import (
    AnomalyAnalysisResult,
    AnomalyDetectionAgent,
    AnomalyDetectionAuthorizationError,
    AnomalyDetectionLLMError,
    AnomalyDetectionValidationError,
    AnomalySeverity,
    AnomalyType,
    ColumnThreshold,
    scan_for_anomalies,
)
from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import AsyncLLMClient
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.state import AgentStateService, CreateSessionParams
from app.db.models import DataSource
from tests.conftest import run_async
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace

_FLAT = [10.0, 11.0, 10.5, 9.5, 10.2, 10.8]


def _anomaly_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": "One high-severity outlier was detected in revenue.",
        "outliers": "revenue=900 is far outside the expected range around a median of 10.5.",
        "unexpected_changes": "No unexpected period-over-period change was detected.",
        "threshold_breaches": "No threshold breach was detected.",
        "conclusions": ["revenue=900 sits outside the expected range around the median of 10.5."],
        "confidence_score": "MEDIUM",
        "confidence_reasoning": "Seven values is a small sample for outlier scoring.",
    }
    body.update(overrides)
    return body


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-anomaly-1",
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


def _execution_result(columns: list[str], rows: list[list[Any]]) -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


def _outlier_rows() -> list[list[Any]]:
    rows: list[list[Any]] = [[f"2024-01-{index + 1:02d}", value] for index, value in enumerate(_FLAT)]
    rows.append(["2024-01-07", 900.0])
    return rows


def _setup(db_session: Session, handler: Any = None) -> tuple[Any, Any, Any, AnomalyDetectionAgent]:
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
    agent = AnomalyDetectionAgent(
        db_session,
        llm_client=_mock_client(handler) if handler else None,
        state_service=state_service,
    )
    return user, workspace, snapshot, agent


# --- Deterministic detection (no DB, no LLM) --------------------------------


def test_scan_detects_statistical_outlier() -> None:
    scan = scan_for_anomalies(["day", "revenue"], _outlier_rows())

    assert scan.analyzed is True
    assert scan.period_column == "day"
    assert scan.numeric_columns == ["revenue"]
    outliers = [item for item in scan.anomalies if item.anomaly_type is AnomalyType.STATISTICAL_OUTLIER]
    assert len(outliers) == 1
    assert outliers[0].value == 900.0
    assert outliers[0].severity is AnomalySeverity.HIGH
    assert outliers[0].method == "modified_z_score"
    assert outliers[0].period == "2024-01-07"
    assert outliers[0].expected_high is not None and outliers[0].expected_high < 900.0
    assert "expected range" in outliers[0].evidence
    assert scan.highest_severity is AnomalySeverity.HIGH


def test_scan_detects_threshold_breaches_with_severity() -> None:
    scan = scan_for_anomalies(
        ["region", "error_rate"],
        [["East", 0.02], ["West", 0.11], ["North", 0.3]],
        thresholds=[ColumnThreshold(column="error_rate", maximum=0.1)],
    )

    breaches = [item for item in scan.anomalies if item.anomaly_type is AnomalyType.THRESHOLD_BREACH]
    assert [item.value for item in breaches] == [0.3, 0.11]
    assert breaches[0].severity is AnomalySeverity.HIGH
    assert breaches[1].severity is AnomalySeverity.LOW
    assert "above the maximum" in breaches[0].evidence
    # Three values is under the statistical minimum, so only thresholds ran.
    assert any("at least 5" in note for note in scan.notes)


def test_scan_detects_minimum_breach_and_ignores_unknown_threshold_column() -> None:
    scan = scan_for_anomalies(
        ["region", "score"],
        [["East", 80], ["West", 20]],
        thresholds=[
            ColumnThreshold(column="score", minimum=50),
            ColumnThreshold(column="missing_column", maximum=1),
        ],
    )

    breaches = [item for item in scan.anomalies if item.anomaly_type is AnomalyType.THRESHOLD_BREACH]
    assert [item.value for item in breaches] == [20.0]
    assert "below the minimum" in breaches[0].evidence
    assert any("missing_column" in note for note in scan.notes)


def test_scan_detects_unexpected_period_change() -> None:
    scan = scan_for_anomalies(
        ["month", "revenue"],
        [["2024-01", 100], ["2024-02", 105], ["2024-03", 320]],
    )

    changes = [item for item in scan.anomalies if item.anomaly_type is AnomalyType.UNEXPECTED_CHANGE]
    assert len(changes) == 1
    assert changes[0].period == "2024-03"
    assert changes[0].previous_value == 105.0
    assert changes[0].severity is AnomalySeverity.HIGH
    assert changes[0].percent_change is not None and changes[0].percent_change > 100


def test_scan_handles_unsuitable_and_empty_data() -> None:
    non_numeric = scan_for_anomalies(["region", "status"], [["East", "ok"], ["West", "bad"]])
    empty = scan_for_anomalies(["region", "revenue"], [])

    assert non_numeric.analyzed is False
    assert any("No numeric column" in note for note in non_numeric.notes)
    assert non_numeric.anomalies == []
    assert empty.analyzed is False


def test_scan_skips_time_series_without_period_column() -> None:
    scan = scan_for_anomalies(["region", "revenue"], [["r%d" % index, value] for index, value in enumerate(_FLAT)])

    assert scan.analyzed is True
    assert scan.period_column is None
    assert any("time-series anomaly detection was skipped" in note for note in scan.notes)
    assert all(item.anomaly_type is not AnomalyType.UNEXPECTED_CHANGE for item in scan.anomalies)


def test_scan_handles_constant_and_missing_values() -> None:
    scan = scan_for_anomalies(
        ["region", "revenue"],
        [["East", Decimal("5")], ["West", Decimal("5")], ["North", None], ["South", Decimal("5")],
         ["A", Decimal("5")], ["B", Decimal("5")]],
    )

    assert scan.analyzed is True
    assert scan.anomalies == []
    assert scan.column_statistics["revenue"]["median_absolute_deviation"] == 0.0


# --- Agent behaviour --------------------------------------------------------


def test_anomaly_detection_success(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_anomaly_body()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies in daily revenue?",
            sql="SELECT day, revenue FROM sales",
            query_result=_execution_result(["day", "revenue"], _outlier_rows()),
        )
    )

    assert isinstance(result, AnomalyAnalysisResult)
    assert result.anomaly_count >= 1
    assert result.highest_severity is AnomalySeverity.HIGH
    assert result.confidence_score is AnalysisConfidence.MEDIUM
    assert result.scan.analyzed is True


def test_anomaly_detection_skips_llm_when_clean(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called when no anomaly was detected")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies?",
            query_result=_execution_result(
                ["region", "revenue"],
                [["r%d" % index, value] for index, value in enumerate(_FLAT)],
            ),
        )
    )

    assert result.anomaly_count == 0
    assert result.highest_severity is None
    assert result.confidence_score is AnalysisConfidence.MEDIUM
    assert "No anomalies were detected" in result.summary


def test_anomaly_detection_skips_llm_for_unsuitable_data(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called for unscannable data")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies?",
            query_result=_execution_result(["region", "status"], [["East", "ok"]]),
        )
    )

    assert result.scan.analyzed is False
    assert result.anomaly_count == 0
    assert result.confidence_score is AnalysisConfidence.LOW
    assert "No anomaly scan could be run" in result.summary


def test_anomaly_detection_missing_query_result(db_session: Session) -> None:
    user, workspace, snapshot, agent = _setup(db_session)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies?",
        )
    )

    assert result.scan.analyzed is False
    assert result.confidence_score is AnalysisConfidence.LOW


def test_anomaly_detection_applies_thresholds(db_session: Session) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return httpx.Response(200, json=_success_response(_anomaly_body()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Did the error rate breach its limit?",
            query_result=_execution_result(
                ["region", "error_rate"],
                [["East", 0.02], ["West", 0.3]],
            ),
            thresholds=[ColumnThreshold(column="error_rate", maximum=0.1)],
        )
    )

    assert result.anomaly_count == 1
    assert result.scan.anomalies[0].anomaly_type is AnomalyType.THRESHOLD_BREACH
    assert "THRESHOLD_BREACH" in captured["body"]


def test_anomaly_detection_authorization_error(db_session: Session) -> None:
    user, workspace, snapshot, agent = _setup(db_session)

    _, other_workspace, _ = _seed_workspace(db_session)
    source = db_session.get(DataSource, snapshot.data_source_id)
    assert source is not None
    source.workspace_id = other_workspace.id
    db_session.flush()

    with pytest.raises(AnomalyDetectionAuthorizationError, match="not accessible"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Are there anomalies?",
            )
        )


def test_anomaly_detection_validation_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response({"invalid_field": "value"}))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(AnomalyDetectionValidationError, match="failed validation"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Are there anomalies?",
                query_result=_execution_result(["day", "revenue"], _outlier_rows()),
            )
        )


def test_anomaly_detection_llm_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(AnomalyDetectionLLMError):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Are there anomalies?",
                query_result=_execution_result(["day", "revenue"], _outlier_rows()),
            )
        )


def test_anomaly_detection_updates_agent_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_anomaly_body()))

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies?",
            query_result=_execution_result(["day", "revenue"], _outlier_rows()),
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
    assert custom.get("anomaly_action") == "ANOMALY_SCAN_COMPLETED"
    assert int(custom["anomaly_count"]) >= 1
    assert custom.get("anomaly_highest_severity") == "HIGH"


def test_anomaly_detection_records_skipped_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("LLM must not be called when scan cannot run")

    user, workspace, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Are there anomalies?",
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    updated = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    assert updated.agent_state.payload.custom.get("anomaly_action") == "ANOMALY_SCAN_SKIPPED"
