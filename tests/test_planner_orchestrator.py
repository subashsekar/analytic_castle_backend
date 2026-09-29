from __future__ import annotations

import uuid

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.intent_types import AIPlanCapability
from app.ai.orchestrator import AIAnalystOrchestrator
from app.ai.planner_agent import PlannerAgent
from app.ai.supervisor.models import (
    ClassificationConfidence,
    RequestCategory,
    RequestClassification,
)
from app.ai.types import AI_CAPABILITY_CHAT, AIContext, AIRequest
from app.db.models import AnalysisSession, DataSourceType
from app.enums import AgentPhase, AnalysisSessionStatus
from tests.ai_fakes import FakeLLMProvider
from tests.conftest import run_async
from tests.planner_helpers import (
    configure_analytical_provider,
    mock_planner_llm_client,
    persisted_agent_payload,
    plan_body,
    success_response,
)
from tests.test_supervisor import (
    _seed_data_source,
    _seed_workspace,
)


def _context(
    *,
    user_id: uuid.UUID,
    workspace_id: uuid.UUID,
    organization_id: uuid.UUID,
    data_source_id: uuid.UUID,
) -> AIContext:
    return AIContext(
        user_id=user_id,
        workspace_id=workspace_id,
        organization_id=organization_id,
        data_source_id=data_source_id,
        data_source_name="Analytics DB",
        data_source_type=DataSourceType.POSTGRESQL.value,
        workspace_name="Analytics",
        workspace_role="OWNER",
        allowed_capabilities=frozenset({AI_CAPABILITY_CHAT}),
    )


@pytest.fixture(autouse=True)
def mock_supervisor_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _classify(**_kwargs: object) -> RequestClassification:
        return RequestClassification(
            category=RequestCategory.ANALYTICAL_QUERY,
            confidence=ClassificationConfidence.HIGH,
            requires_data_access=True,
        )

    monkeypatch.setattr(
        "app.ai.supervisor.service.classify_request",
        _classify,
    )


def test_orchestrator_supervised_chat_uses_planner_agent_plan(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    context = _context(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    provider = FakeLLMProvider()
    configure_analytical_provider(provider)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_response(plan_body()))

    planner_agent = PlannerAgent(
        db_session, llm_client=mock_planner_llm_client(handler)
    )
    orchestrator = AIAnalystOrchestrator(provider, planner_agent=planner_agent)

    result = run_async(
        orchestrator.chat(
            AIRequest(
                message="Show total revenue by region last month",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    assert result.plan is not None
    assert AIPlanCapability.DATABASE in result.plan.required_capabilities
    assert result.conversation_id is not None
    session = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == result.conversation_id)
        .one()
    )
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING
    payload = persisted_agent_payload(session.agent_state.payload)
    assert payload.plan_version == "v1"
    assert payload.custom.get("planner_action") == "PLAN_CREATED"
