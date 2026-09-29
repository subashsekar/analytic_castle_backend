"""Tests for the Root Cause Analysis Agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.anomaly_detection import scan_for_anomalies
from app.ai.anomaly_detection.models import AnomalyAnalysisResult
from app.ai.data_analyst.models import AnalysisConfidence
from app.ai.llm import AsyncLLMClient
from app.ai.metadata_types import empty_resolved_context
from app.ai.root_cause_analysis import (
    RootCauseAnalysisAgent,
    RootCauseAnalysisAuthorizationError,
    RootCauseAnalysisLLMError,
    RootCauseAnalysisValidationError,
    build_findings_payload,
    has_findings,
)
from app.ai.sql_execution.errors import SQLExecutionValidationError
from app.ai.sql_execution.models import (
    SQLExecuteParams,
    SQLExecutionResult,
    SQLExecutionServiceResult,
    SQLExecutionStatus,
)
from app.ai.sql_generation.models import (
    GeneratedSQL,
    SQLDialect,
    SQLGenerateParams,
    SQLGenerationConfidence,
    SQLGenerationOutcome,
    SQLGenerationResult,
)
from app.ai.sql_validation.models import ValidatedSQL
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.trend_analysis import build_trend_series
from app.ai.trend_analysis.models import TrendAnalysisResult
from app.db.models import DataSource
from tests.conftest import run_async
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace

_TREND_ROWS = [["2024-01", 100], ["2024-02", 105], ["2024-03", 60]]


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


def _hypothesis(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "statement": "A single region stopped ordering in March.",
        "contributing_factors": ["regional demand", "customer churn"],
        "supporting_evidence": "Revenue fell from 105 to 60 in 2024-03.",
        "contradicting_evidence": None,
        "confidence_score": "MEDIUM",
        "confidence_reasoning": "The results are not broken down by region.",
        "investigation_question": None,
    }
    body.update(overrides)
    return body


def _output(*hypotheses: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": "March revenue fell 42.9% and the cause is not visible in these results.",
        "hypotheses": list(hypotheses) or [_hypothesis()],
    }


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-root-cause-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _execution_result(columns: list[str], rows: list[list[Any]]) -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


class _StubGeneration:
    """SQL generation stub standing in for the Phase 7.1 service."""

    def __init__(self, sql: str | None = "SELECT region, SUM(revenue) FROM sales GROUP BY region") -> None:
        self.sql = sql
        self.params: list[SQLGenerateParams] = []

    async def generate(self, params: SQLGenerateParams) -> SQLGenerationResult:
        self.params.append(params)
        generated = (
            GeneratedSQL(
                sql=self.sql,
                dialect=SQLDialect.POSTGRESQL,
                confidence=SQLGenerationConfidence.MEDIUM,
                generation_version="v1",
            )
            if self.sql
            else None
        )
        return SQLGenerationResult(
            outcome=SQLGenerationOutcome(
                generated=generated,
                requires_clarification=generated is None,
                clarification_question=None if generated else "Which region?",
            ),
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )


class _StubExecution:
    """SQL execution stub standing in for the Phase 7.3 MCP-backed service."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.params: list[SQLExecuteParams] = []

    async def execute(self, params: SQLExecuteParams) -> SQLExecutionServiceResult:
        self.params.append(params)
        if self.error is not None:
            raise self.error
        return SQLExecutionServiceResult(
            result=_execution_result(["region", "revenue"], [["East", 60]]),
            validated=ValidatedSQL(sql=params.sql),
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )


def _setup(
    db_session: Session,
    handler: Any = None,
    *,
    generation: Any = None,
    execution: Any = None,
) -> tuple[Any, Any, Any, Any, RootCauseAnalysisAgent]:
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
    agent = RootCauseAnalysisAgent(
        db_session,
        llm_client=AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))
        if handler
        else None,
        state_service=state_service,
        sql_generation_service=generation,
        sql_execution_service=execution,
    )
    return user, workspace, organization, snapshot, agent


# --- Deterministic finding gate (no DB, no LLM) -----------------------------


def test_has_findings_requires_a_detected_change() -> None:
    assert has_findings(_trend_result(), None) is True
    assert has_findings(None, _anomaly_result()) is True
    assert has_findings(None, None) is False


# --- Agent behaviour --------------------------------------------------------


def test_root_cause_analysis_success(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop in March?",
            sql="SELECT month, revenue FROM sales",
            query_result=_execution_result(["month", "revenue"], _TREND_ROWS),
            trend=_trend_result(),
            anomalies=_anomaly_result(),
        )
    )

    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].rank == 1
    assert result.primary_cause == "A single region stopped ordering in March."
    assert result.hypotheses[0].contributing_factors == ["regional demand", "customer churn"]
    assert result.confidence_score is AnalysisConfidence.MEDIUM
    assert result.additional_queries_run == 0
    assert result.evidence == []


def test_root_cause_analysis_ranks_by_confidence(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _hypothesis(statement="Low cause", confidence_score="LOW"),
                    _hypothesis(statement="High cause", confidence_score="HIGH"),
                    _hypothesis(statement="Medium cause", confidence_score="MEDIUM"),
                )
            ),
        )

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop?",
            trend=_trend_result(),
        )
    )

    assert [item.statement for item in result.hypotheses] == [
        "High cause",
        "Medium cause",
        "Low cause",
    ]
    assert [item.rank for item in result.hypotheses] == [1, 2, 3]
    assert result.confidence_score is AnalysisConfidence.HIGH


def test_root_cause_analysis_skips_llm_without_findings(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("LLM must not be called when nothing was detected")

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop?",
        )
    )

    assert result.hypotheses == []
    assert result.primary_cause is None
    assert result.confidence_score is AnalysisConfidence.LOW
    assert "no finding to explain" in result.summary


def test_root_cause_analysis_gathers_additional_evidence(db_session: Session) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.content.decode())
        if len(calls) == 1:
            return httpx.Response(
                200,
                json=_success_response(
                    _output(
                        _hypothesis(
                            investigation_question="How did revenue change by region in March?"
                        )
                    )
                ),
            )
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _hypothesis(
                        statement="East region revenue collapsed in March.",
                        confidence_score="HIGH",
                        supporting_evidence="The regional breakdown shows East at 60.",
                    )
                )
            ),
        )

    generation = _StubGeneration()
    execution = _StubExecution()
    user, workspace, organization, snapshot, agent = _setup(
        db_session, handler, generation=generation, execution=execution
    )

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop in March?",
            trend=_trend_result(),
            organization_id=organization.id,
            metadata=empty_resolved_context(snapshot.data_source_id),
        )
    )

    assert result.additional_queries_run == 1
    assert result.evidence[0].executed is True
    assert result.evidence[0].rows == [["East", 60]]
    assert result.primary_cause == "East region revenue collapsed in March."
    assert result.confidence_score is AnalysisConfidence.HIGH
    # Evidence is re-fed to the model, and every query stays on the session data source.
    assert len(calls) == 2
    assert "East" in calls[1]
    assert execution.params[0].data_source_id == snapshot.data_source_id
    assert execution.params[0].session_id == snapshot.session_id
    assert generation.params[0].data_source_id == snapshot.data_source_id


def test_root_cause_analysis_notes_failed_evidence_query(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(_hypothesis(investigation_question="Revenue by region in March?"))
            ),
        )

    user, workspace, organization, snapshot, agent = _setup(
        db_session,
        handler,
        generation=_StubGeneration(),
        execution=_StubExecution(error=SQLExecutionValidationError("SQL failed validation")),
    )

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop in March?",
            trend=_trend_result(),
            organization_id=organization.id,
            metadata=empty_resolved_context(snapshot.data_source_id),
        )
    )

    assert result.additional_queries_run == 0
    assert result.evidence[0].executed is False
    assert any("failed validation" in note for note in result.notes)
    assert len(result.hypotheses) == 1


def test_root_cause_analysis_notes_missing_metadata(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(_hypothesis(investigation_question="Revenue by region in March?"))
            ),
        )

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop in March?",
            trend=_trend_result(),
        )
    )

    assert result.evidence == []
    assert any("no authorized data source metadata" in note for note in result.notes)


def test_root_cause_analysis_rejects_foreign_metadata(db_session: Session) -> None:
    import uuid

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(_hypothesis(investigation_question="Revenue by region in March?"))
            ),
        )

    user, workspace, organization, snapshot, agent = _setup(
        db_session, handler, generation=_StubGeneration(), execution=_StubExecution()
    )

    with pytest.raises(RootCauseAnalysisAuthorizationError, match="does not match"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Why did revenue drop in March?",
                trend=_trend_result(),
                organization_id=organization.id,
                metadata=empty_resolved_context(uuid.uuid4()),
            )
        )


def test_root_cause_analysis_authorization_error(db_session: Session) -> None:
    user, workspace, _, snapshot, agent = _setup(db_session)

    _, other_workspace, _ = _seed_workspace(db_session)
    source = db_session.get(DataSource, snapshot.data_source_id)
    assert source is not None
    source.workspace_id = other_workspace.id
    db_session.flush()

    with pytest.raises(RootCauseAnalysisAuthorizationError, match="not accessible"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Why did revenue drop?",
                trend=_trend_result(),
            )
        )


def test_root_cause_analysis_validation_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response({"summary": "x", "hypotheses": []}))

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RootCauseAnalysisValidationError, match="failed validation"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Why did revenue drop?",
                trend=_trend_result(),
            )
        )


def test_root_cause_analysis_llm_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    user, workspace, _, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RootCauseAnalysisLLMError):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Why did revenue drop?",
                trend=_trend_result(),
            )
        )


def test_root_cause_analysis_updates_agent_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, _, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop?",
            anomalies=_anomaly_result(),
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
    assert custom.get("root_cause_action") == "ROOT_CAUSE_ANALYSIS_COMPLETED"
    assert custom.get("root_cause_hypothesis_count") == "1"
    assert custom.get("root_cause_queries_run") == "0"


def test_anomaly_only_finding_is_analyzed(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, _, snapshot, agent = _setup(db_session, handler)
    anomalies = _anomaly_result()
    findings = json.loads(build_findings_payload(None, anomalies))
    assert set(findings) == {"anomalies"}

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why is March unusual?",
            anomalies=anomalies,
        )
    )

    assert result.primary_cause is not None


def test_root_cause_analysis_records_skipped_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("LLM must not be called without findings")

    user, workspace, _, snapshot, agent = _setup(db_session, handler)
    state_service = AgentStateService(db_session)

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Why did revenue drop?",
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    updated = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    assert updated.agent_state.payload.custom.get("root_cause_action") == (
        "ROOT_CAUSE_ANALYSIS_SKIPPED"
    )


def test_root_cause_analysis_rejects_organization_mismatch(db_session: Session) -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _hypothesis(
                        investigation_question="Break revenue down by region?",
                    )
                )
            ),
        )

    user, workspace, organization, snapshot, agent = _setup(
        db_session,
        handler,
        generation=_StubGeneration(),
        execution=_StubExecution(),
    )
    _, _, other_org = _seed_workspace(db_session)

    with pytest.raises(RootCauseAnalysisAuthorizationError, match="organization"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Why did revenue drop?",
                query_result=_execution_result(["month", "revenue"], _TREND_ROWS),
                trend=_trend_result(),
                organization_id=other_org.id,
                metadata=empty_resolved_context(snapshot.data_source_id),
            )
        )

    assert organization.id != other_org.id
    assert call_count["n"] == 1
