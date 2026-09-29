"""Tests for the Data Analyst Agent."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.data_analyst import (
    DataAnalystAgent,
    DataAnalysisResult,
    AnalysisConfidence,
    DataAnalystAuthorizationError,
    DataAnalystValidationError,
    DataAnalystLLMError,
)
from app.ai.llm import AsyncLLMClient
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.enums import AgentPhase
from tests.conftest import run_async
from tests.test_supervisor import _client_config, _seed_workspace, _seed_data_source


def _analyst_body(**overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "interpretation": "The query results show that the East region has a total revenue of $42.",
        "summary": "Total revenue: $42. Count of regions: 1.",
        "comparisons": "Only East region data is available, so no comparison is possible.",
        "conclusions": ["The East region is the only region with revenue in the dataset."],
        "confidence_score": "HIGH",
        "confidence_reasoning": "The data is complete and sufficient to answer the question.",
    }
    body.update(overrides)
    return body


def _success_response(content: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "gen-analyst-1",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(content),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _mock_client(handler: Any) -> AsyncLLMClient:
    transport = httpx.MockTransport(handler)
    return AsyncLLMClient(_client_config(), transport=transport)


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


def test_data_analyst_success(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_analyst_body()))

    client = _mock_client(handler)
    agent = DataAnalystAgent(db_session, llm_client=client, state_service=state_service)

    query_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["region", "revenue"],
        rows=[["East", 42]],
        row_count=1,
    )

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What is the revenue for the East region?",
            sql="SELECT region, revenue FROM sales WHERE region = 'East'",
            query_result=query_result,
        )
    )

    assert isinstance(result, DataAnalysisResult)
    assert result.interpretation == "The query results show that the East region has a total revenue of $42."
    assert result.confidence_score is AnalysisConfidence.HIGH
    assert len(result.conclusions) == 1
    assert result.conclusions[0] == "The East region is the only region with revenue in the dataset."


def test_data_analyst_empty_data(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_response(
                _analyst_body(
                    interpretation="No records were found matching the criteria.",
                    summary="N/A",
                    comparisons="N/A",
                    conclusions=[],
                    confidence_score="LOW",
                    confidence_reasoning="No data was returned from the database.",
                )
            ),
        )

    client = _mock_client(handler)
    agent = DataAnalystAgent(db_session, llm_client=client, state_service=state_service)

    query_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["region", "revenue"],
        rows=[],
        row_count=0,
    )

    result = run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What is the revenue for the East region?",
            sql="SELECT region, revenue FROM sales WHERE region = 'East'",
            query_result=query_result,
        )
    )

    assert result.confidence_score is AnalysisConfidence.LOW
    assert result.interpretation == "No records were found matching the criteria."
    assert result.summary == "N/A"
    assert len(result.conclusions) == 0


def test_data_analyst_authorization_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    agent = DataAnalystAgent(db_session, state_service=state_service)

    # Seed another workspace and set the data source to belong to it
    _, workspace2, _ = _seed_workspace(db_session)
    data_source.workspace_id = workspace2.id
    db_session.flush()

    with pytest.raises(DataAnalystAuthorizationError, match="not accessible"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What is the revenue for the East region?",
            )
        )


def test_data_analyst_validation_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        # Return invalid fields to trigger validation error
        return httpx.Response(200, json=_success_response({"invalid_field": "value"}))

    client = _mock_client(handler)
    agent = DataAnalystAgent(db_session, llm_client=client, state_service=state_service)

    with pytest.raises(DataAnalystValidationError, match="failed validation"):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What is the revenue for the East region?",
            )
        )


def test_data_analyst_llm_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = _mock_client(handler)
    agent = DataAnalystAgent(db_session, llm_client=client, state_service=state_service)

    with pytest.raises(DataAnalystLLMError):
        run_async(
            agent.analyze(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="What is the revenue for the East region?",
            )
        )


def test_data_analyst_updates_agent_state(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    state_service, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_response(_analyst_body()))

    client = _mock_client(handler)
    agent = DataAnalystAgent(db_session, llm_client=client, state_service=state_service)

    query_result = SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=["region", "revenue"],
        rows=[["East", 42]],
        row_count=1,
    )

    run_async(
        agent.analyze(
            session_id=snapshot.session_id,
            workspace_id=workspace.id,
            user_id=user.id,
            message="What is the revenue for the East region?",
            sql="SELECT region, revenue FROM sales WHERE region = 'East'",
            query_result=query_result,
            expected_agent_version=snapshot.agent_state.version if snapshot.agent_state else 1,
        )
    )

    # Reload session and check custom payload entries
    updated_snapshot = state_service.get_session(
        snapshot.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert updated_snapshot.agent_state is not None
    custom = updated_snapshot.agent_state.payload.custom
    assert custom.get("analyst_action") == "ANALYSIS_COMPLETED"
    assert custom.get("confidence_score") == "HIGH"
