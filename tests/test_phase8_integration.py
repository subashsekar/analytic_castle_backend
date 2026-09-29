"""Phase 8 audit: agent-to-agent integration, auth isolation, structured outputs."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.anomaly_detection import AnomalyDetectionAgent
from app.ai.data_analyst import DataAnalystAgent
from app.ai.evaluation import EvaluatedAgent, evaluate_agents
from app.ai.insight import InsightAgent
from app.ai.llm import AsyncLLMClient
from app.ai.recommendation import RecommendationAgent, RecommendationAuthorizationError
from app.ai.root_cause_analysis import RootCauseAnalysisAgent
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.state.errors import AnalysisSessionNotFoundError
from app.ai.trend_analysis import TrendAnalysisAgent
from app.db.models import DataSource
from tests.conftest import run_async
from tests.test_anomaly_detection import _anomaly_body
from tests.test_data_analyst import _analyst_body
from tests.test_insight import _insight, _output as _insight_output
from tests.test_recommendation import _output as _recommendation_output, _recommendation
from tests.test_root_cause_analysis import _hypothesis, _output as _rca_output
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace
from tests.test_trend_analysis import _trend_body

_ROWS = [["2024-01", 100], ["2024-02", 105], ["2024-03", 60]]


def _query_result() -> SQLExecutionResult:
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["month", "revenue"],
        rows=_ROWS,
        row_count=len(_ROWS),
    )


def _llm_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-phase8",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _fixed_handler(content: dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_llm_response(content))

    return handler


def _session(db_session: Session) -> tuple[Any, Any, Any, Any, AgentStateService]:
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
    return user, workspace, organization, snapshot, state_service


def test_phase8_agent_pipeline_and_evaluation(db_session: Session) -> None:
    """Analyst → trend → anomaly → RCA → insight → recommendation → evaluate."""
    user, workspace, _organization, snapshot, state_service = _session(db_session)
    query = _query_result()
    message = "Why did revenue fall in March?"

    analysis = run_async(
        DataAnalystAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(
                        _analyst_body(
                            interpretation="Revenue peaked in February at 105 and fell to 60 in March.",
                            summary="Total revenue across three months was 265.",
                            comparisons="March revenue of 60 is below February's 105.",
                            conclusions=["March revenue fell to 60."],
                            confidence_score="MEDIUM",
                            confidence_reasoning="Three months is a small sample.",
                        )
                    )
                ),
            ),
            state_service=state_service,
        ).analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            sql="SELECT month, revenue FROM sales",
            query_result=query,
        )
    )

    trend = run_async(
        TrendAnalysisAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(
                        _trend_body(
                            summary="Revenue fell from 100 in 2024-01 to 60 in 2024-03.",
                            direction_explanation="The series is decreasing after February.",
                            period_comparisons="2024-03 fell against 2024-02.",
                            significant_changes="2024-03 is a significant decline.",
                            conclusions=["Revenue declined in 2024-03."],
                            confidence_score="MEDIUM",
                            confidence_reasoning="Only three periods are available.",
                        )
                    )
                ),
            ),
            state_service=state_service,
        ).analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            query_result=query,
        )
    )

    anomalies = run_async(
        AnomalyDetectionAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(
                        _anomaly_body(
                            summary="One unexpected decline was detected in revenue.",
                            outliers="No statistical outlier was detected.",
                            unexpected_changes="2024-03 fell against 2024-02 in revenue.",
                            threshold_breaches="No threshold breach was detected.",
                            conclusions=["2024-03 declined unexpectedly in revenue."],
                            confidence_score="MEDIUM",
                            confidence_reasoning="Three periods is a small sample.",
                        )
                    )
                ),
            ),
            state_service=state_service,
        ).analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            query_result=query,
        )
    )

    root_cause = run_async(
        RootCauseAnalysisAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(_rca_output(_hypothesis(statement="Demand weakened in March.")))
                ),
            ),
            state_service=state_service,
        ).analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            query_result=query,
            trend=trend,
            anomalies=anomalies,
        )
    )

    insights = run_async(
        InsightAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(_insight_output(_insight(metric="revenue")))
                ),
            ),
            state_service=state_service,
        ).generate(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            query_result=query,
            analysis=analysis,
            trend=trend,
            anomalies=anomalies,
            root_cause=root_cause,
        )
    )

    recommendations = run_async(
        RecommendationAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(),
                transport=httpx.MockTransport(
                    _fixed_handler(
                        _recommendation_output(
                            _recommendation(evidence_reference="Revenue reversed in March")
                        )
                    )
                ),
            ),
            state_service=state_service,
        ).recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message=message,
            query_result=query,
            analysis=analysis,
            trend=trend,
            anomalies=anomalies,
            root_cause=root_cause,
            insights=insights,
        )
    )

    assert analysis.conclusions
    assert trend.series.point_count == 3
    assert anomalies.scan.analyzed is True
    assert root_cause.hypotheses
    assert insights.insights
    assert recommendations.recommendations
    assert insights.insights[0].metric == "revenue"
    assert recommendations.recommendations[0].evidence_reference == (
        "Revenue reversed in March"
    )

    report = evaluate_agents(
        session_id=snapshot.session_id,
        query_result=query,
        analysis=analysis,
        trend=trend,
        anomalies=anomalies,
        root_cause=root_cause,
        insights=insights,
        recommendations=recommendations,
    )

    assert report.metrics.agents_evaluated == 6
    assert report.agents_not_evaluated == []
    assert set(item.agent for item in report.agents) == set(EvaluatedAgent)
    # Grounding and computed-field checks should leave the pipeline verified-correct.
    assert report.verified_correct is True
    assert report.failures == []


def test_phase8_workspace_data_source_isolation(db_session: Session) -> None:
    user, workspace, _organization, snapshot, state_service = _session(db_session)
    _, other_workspace, _ = _seed_workspace(db_session)
    source = db_session.get(DataSource, snapshot.data_source_id)
    assert source is not None
    source.workspace_id = other_workspace.id
    db_session.flush()

    agent = RecommendationAgent(db_session, state_service=state_service)
    with pytest.raises(RecommendationAuthorizationError, match="not accessible"):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_query_result(),
            )
        )


def test_phase8_user_session_isolation(db_session: Session) -> None:
    user, workspace, organization, snapshot, state_service = _session(db_session)
    other_user, _, _ = _seed_workspace(db_session)

    with pytest.raises(AnalysisSessionNotFoundError):
        state_service.get_session(
            snapshot.session_id,
            workspace_id=workspace.id,
            user_id=other_user.id,
        )

    # Confirm the owning user still reaches the session (workspace + org intact).
    owned = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert owned.organization_id == organization.id


def test_phase8_confidence_normalization_and_extra_fields(db_session: Session) -> None:
    """Structured outputs coerce confidence case and ignore unknown LLM fields."""
    user, workspace, _organization, snapshot, state_service = _session(db_session)

    body = _analyst_body(confidence_score="high", invented_field="drop me")

    result = run_async(
        DataAnalystAgent(
            db_session,
            llm_client=AsyncLLMClient(
                _client_config(), transport=httpx.MockTransport(_fixed_handler(body))
            ),
            state_service=state_service,
        ).analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="Summarize revenue",
            query_result=_query_result(),
        )
    )

    assert result.confidence_score.value == "HIGH"


def test_phase8_foreign_session_id_denied(db_session: Session) -> None:
    user, workspace, _organization, _snapshot, state_service = _session(db_session)

    with pytest.raises(AnalysisSessionNotFoundError):
        state_service.get_session(
            uuid4(),
            workspace_id=workspace.id,
            user_id=user.id,
        )
