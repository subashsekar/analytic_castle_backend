"""Tests for the Recommendation Agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.data_analyst.models import AnalysisConfidence, DataAnalysisResult
from app.ai.insight.evidence import identify_key_metrics
from app.ai.insight.models import (
    BusinessInsight,
    InsightAnalysisResult,
    InsightPriority,
)
from app.ai.llm import AsyncLLMClient
from app.ai.recommendation import (
    RecommendationAgent,
    RecommendationAuthorizationError,
    RecommendationLevel,
    RecommendationLLMError,
    RecommendationValidationError,
    derive_priority,
    groundable_names,
    has_evidence,
)
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.state import AgentStateService, CreateSessionParams
from app.db.models import DataSource
from tests.conftest import run_async
from tests.test_insight import (
    _analysis_result,
    _anomaly_result,
    _execution_result,
    _root_cause_result,
    _trend_result,
)
from tests.test_supervisor import _client_config, _seed_data_source, _seed_workspace


def _insights_result() -> InsightAnalysisResult:
    metrics = identify_key_metrics(_execution_result())
    return InsightAnalysisResult(
        summary="Revenue reversed sharply in March.",
        top_insight="Revenue fell from 105 in February to 60 in March.",
        insights=[
            BusinessInsight(
                rank=1,
                title="Revenue reversed in March",
                insight="Revenue fell from 105 in February to 60 in March.",
                metric="revenue",
                business_impact="A 42.9% drop reduces the quarter total to 265.",
                supporting_evidence="Revenue totals 265 with a March value of 60.",
                priority=InsightPriority.HIGH,
                confidence_score=AnalysisConfidence.MEDIUM,
                confidence_reasoning="Three months is a small sample.",
            )
        ],
        key_metrics=metrics,
        data_gaps=["No regional breakdown is available."],
        confidence_score=AnalysisConfidence.MEDIUM,
        confidence_reasoning="Three months is a small sample.",
    )


def _recommendation(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "title": "Investigate the March revenue drop",
        "recommendation": "Review the March sales pipeline for cancelled or delayed orders.",
        "evidence_reference": "Revenue reversed in March",
        "supporting_evidence": "Revenue fell from 105 to 60 in 2024-03.",
        "expected_outcome": "May help identify whether the drop is recoverable next month.",
        "assumptions": ["Operations can review order cancellations."],
        "risks": ["The drop may be seasonal and not actionable."],
        "impact": "HIGH",
        "feasibility": "HIGH",
        "confidence_score": "MEDIUM",
        "confidence_reasoning": "The drop is clear but the cause is unknown.",
    }
    body.update(overrides)
    return body


def _output(
    *recommendations: dict[str, Any],
    data_gaps: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "summary": "Act on the March revenue reversal before the quarter closes.",
        "recommendations": list(recommendations) or [_recommendation()],
        "data_gaps": data_gaps or ["Regional demand data is missing."],
    }


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-recommendation-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _setup(
    db_session: Session, handler: Any = None
) -> tuple[Any, Any, Any, RecommendationAgent]:
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
    agent = RecommendationAgent(
        db_session,
        llm_client=AsyncLLMClient(_client_config(), transport=httpx.MockTransport(handler))
        if handler
        else None,
        state_service=state_service,
    )
    return user, workspace, snapshot, agent


# --- Deterministic helpers --------------------------------------------------


@pytest.mark.parametrize(
    ("impact", "feasibility", "expected"),
    [
        (RecommendationLevel.HIGH, RecommendationLevel.HIGH, RecommendationLevel.HIGH),
        (RecommendationLevel.HIGH, RecommendationLevel.MEDIUM, RecommendationLevel.HIGH),
        (RecommendationLevel.MEDIUM, RecommendationLevel.MEDIUM, RecommendationLevel.MEDIUM),
        (RecommendationLevel.HIGH, RecommendationLevel.LOW, RecommendationLevel.MEDIUM),
        (RecommendationLevel.LOW, RecommendationLevel.LOW, RecommendationLevel.LOW),
        (RecommendationLevel.MEDIUM, RecommendationLevel.LOW, RecommendationLevel.LOW),
    ],
)
def test_derive_priority_matrix(
    impact: RecommendationLevel,
    feasibility: RecommendationLevel,
    expected: RecommendationLevel,
) -> None:
    assert derive_priority(impact, feasibility) is expected


def test_has_evidence_accepts_insights_or_analysis() -> None:
    assert has_evidence(None, None, None, None, None, _insights_result()) is True
    assert has_evidence(_execution_result(), None, None, None, None, None) is True
    assert has_evidence(None, _analysis_result(), None, None, None, None) is True
    assert has_evidence(None, None, None, None, None, None) is False


def test_groundable_names_includes_insight_titles() -> None:
    insights = _insights_result()
    names = groundable_names(_execution_result(), insights.key_metrics, insights)

    assert names["revenue"] == "revenue"
    assert names["revenue reversed in march"] == "Revenue reversed in March"


# --- Agent behaviour --------------------------------------------------------


def test_recommendation_success(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do about revenue?",
            sql="SELECT month, revenue FROM sales",
            query_result=_execution_result(),
            analysis=_analysis_result(),
            trend=_trend_result(),
            anomalies=_anomaly_result(),
            root_cause=_root_cause_result(),
            insights=_insights_result(),
        )
    )

    assert len(result.recommendations) == 1
    assert result.recommendations[0].rank == 1
    assert result.recommendations[0].priority is RecommendationLevel.HIGH
    assert result.recommendations[0].evidence_reference == "Revenue reversed in March"
    assert result.top_recommendation is not None
    assert "pipeline" in result.top_recommendation
    assert result.confidence_score is AnalysisConfidence.MEDIUM
    assert result.data_gaps == ["Regional demand data is missing."]


def test_recommendation_prioritizes_by_impact_feasibility_confidence(
    db_session: Session,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _recommendation(
                        title="Low impact",
                        impact="LOW",
                        feasibility="HIGH",
                        confidence_score="HIGH",
                        evidence_reference="revenue",
                    ),
                    _recommendation(
                        title="High weak",
                        impact="HIGH",
                        feasibility="HIGH",
                        confidence_score="LOW",
                        evidence_reference="revenue",
                    ),
                    _recommendation(
                        title="High strong",
                        impact="HIGH",
                        feasibility="HIGH",
                        confidence_score="HIGH",
                        evidence_reference="revenue",
                    ),
                    _recommendation(
                        title="Medium",
                        impact="MEDIUM",
                        feasibility="HIGH",
                        confidence_score="HIGH",
                        evidence_reference="revenue",
                    ),
                )
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            query_result=_execution_result(),
            insights=_insights_result(),
        )
    )

    assert [item.title for item in result.recommendations] == [
        "High strong",
        "High weak",
        "Medium",
        "Low impact",
    ]
    assert [item.rank for item in result.recommendations] == [1, 2, 3, 4]
    assert result.confidence_score is AnalysisConfidence.HIGH


def test_recommendation_drops_ungrounded_reference(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(
                    _recommendation(
                        title="Invented",
                        evidence_reference="gross_margin",
                    ),
                    _recommendation(
                        title="Grounded",
                        evidence_reference="REVENUE",
                        impact="MEDIUM",
                        feasibility="MEDIUM",
                    ),
                )
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            query_result=_execution_result(),
            insights=_insights_result(),
        )
    )

    assert [item.title for item in result.recommendations] == ["Grounded"]
    assert result.recommendations[0].evidence_reference == "revenue"
    assert any("gross_margin" in note for note in result.notes)


def test_recommendation_rejects_all_ungrounded(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _output(_recommendation(evidence_reference="gross_margin"))
            ),
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RecommendationValidationError, match="grounded"):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_execution_result(),
            )
        )


def test_recommendation_skips_llm_without_evidence(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("LLM must not be called without evidence")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            expected_agent_version=snapshot.agent_state.version
            if snapshot.agent_state
            else 1,
        )
    )

    assert result.recommendations == []
    assert result.top_recommendation is None
    assert result.confidence_score is AnalysisConfidence.LOW
    assert "no action could be recommended" in result.summary.lower()

    state = AgentStateService(db_session).get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert state.agent_state is not None
    assert state.agent_state.payload.custom.get("recommendation_action") == (
        "RECOMMENDATION_SKIPPED"
    )


def test_recommendation_authorization_error(db_session: Session) -> None:
    user, workspace, snapshot, agent = _setup(db_session)

    _, other_workspace, _ = _seed_workspace(db_session)
    source = db_session.get(DataSource, snapshot.data_source_id)
    assert source is not None
    source.workspace_id = other_workspace.id
    db_session.flush()

    with pytest.raises(RecommendationAuthorizationError, match="not accessible"):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_execution_result(),
            )
        )


def test_recommendation_validation_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_response({"summary": "x", "recommendations": []})
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RecommendationValidationError, match="failed validation"):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_execution_result(),
            )
        )


def test_recommendation_llm_error(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RecommendationLLMError):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_execution_result(),
            )
        )


def test_recommendation_empty_llm_content(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-empty",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
            },
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    with pytest.raises(RecommendationLLMError, match="empty"):
        run_async(
            agent.recommend(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What should we do?",
                query_result=_execution_result(),
            )
        )


def test_recommendation_accepts_markdown_wrapped_json(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        wrapped = "```json\n" + json.dumps(_output()) + "\n```"
        return httpx.Response(
            200,
            json={
                "id": "gen-wrapped",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": wrapped},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    user, workspace, snapshot, agent = _setup(db_session, handler)

    result = run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            query_result=_execution_result(),
            insights=_insights_result(),
        )
    )

    assert len(result.recommendations) == 1


def test_recommendation_updates_agent_state(db_session: Session) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            query_result=_execution_result(),
            insights=_insights_result(),
            expected_agent_version=snapshot.agent_state.version
            if snapshot.agent_state
            else 1,
        )
    )

    updated = AgentStateService(db_session).get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated.agent_state is not None
    custom = updated.agent_state.payload.custom
    assert custom.get("recommendation_action") == "RECOMMENDATION_COMPLETED"
    assert custom.get("recommendation_count") == "1"
    assert custom.get("recommendation_confidence_score") == "MEDIUM"


def test_recommendation_prompt_carries_insight_evidence(db_session: Session) -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.content.decode())
        return httpx.Response(200, json=_success_response(_output()))

    user, workspace, snapshot, agent = _setup(db_session, handler)

    run_async(
        agent.recommend(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What should we do?",
            query_result=_execution_result(),
            analysis=_analysis_result(),
            insights=_insights_result(),
        )
    )

    body = captured[0]
    assert "business_insights" in body
    assert "Revenue reversed in March" in body
    assert "data_analysis" in body
