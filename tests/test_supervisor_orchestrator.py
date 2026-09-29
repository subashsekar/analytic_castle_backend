from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.ai.intent_types import AIIntentType
from app.ai.memory import ConversationMemoryService
from app.ai.orchestrator import AIAnalystOrchestrator
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
from tests.planner_helpers import configure_analytical_provider, persisted_agent_payload
from tests.test_supervisor import _seed_data_source, _seed_workspace

pytest_plugins = ["tests.planner_helpers"]


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


@pytest.fixture(autouse=True)
def mock_planner_for_supervised_chat(mock_planner_llm: None) -> None:
    """Ensure supervised orchestrator tests never call a live planner LLM."""


def test_orchestrator_supervised_chat_persists_completed_session(
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
    orchestrator = AIAnalystOrchestrator(provider)

    result = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?", data_source_id=data_source.id
            ),
            context,
            db=db_session,
        )
    )

    assert result.intent is not None
    assert result.intent.intent is AIIntentType.ANALYTICAL_QUERY
    assert result.conversation_id is not None
    sessions = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == result.conversation_id)
        .all()
    )
    assert len(sessions) == 1
    session = sessions[0]
    assert session.user_id == user.id
    assert session.workspace_id == workspace.id
    assert session.data_source_id == data_source.id
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING
    payload = persisted_agent_payload(session.agent_state.payload)
    assert payload.plan_version == "v1"
    assert payload.custom.get("planner_action") == "PLAN_CREATED"


def test_orchestrator_supervised_chat_blocks_unsupported_without_intent_llm(
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
    orchestrator = AIAnalystOrchestrator(provider)

    result = run_async(
        orchestrator.chat(
            AIRequest(
                message="Delete all customer records",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    assert result.intent is not None
    assert result.intent.intent is AIIntentType.UNSUPPORTED
    assert provider.structured_calls == 0
    assert result.conversation_id is not None
    session = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == result.conversation_id)
        .one()
    )
    assert session.status is AnalysisSessionStatus.COMPLETED


def test_orchestrator_supervised_chat_persists_assistant_message(
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
    orchestrator = AIAnalystOrchestrator(provider)

    result = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?", data_source_id=data_source.id
            ),
            context,
            db=db_session,
        )
    )

    assert result.conversation_id is not None
    memory = ConversationMemoryService(db_session)
    messages = memory.get_messages(
        result.conversation_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert messages.total >= 2
    roles = [item.role for item in messages.items]
    assert "user" in roles
    assert "assistant" in roles
