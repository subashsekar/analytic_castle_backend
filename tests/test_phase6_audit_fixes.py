"""Regression coverage for Phase 6.7 audit fixes."""

from __future__ import annotations

import uuid

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.exceptions import AIContextError
from app.ai.intent import _build_user_prompt
from app.ai.intent_types import AIConfidence, AIIntent, AIIntentType
from app.ai.llm.resolve import resolve_llm_settings
from app.ai.orchestrator import AIAnalystOrchestrator
from app.ai.planner_agent import (
    LLMPlannerOutput,
    apply_planner_intent,
    normalize_plan,
)
from app.ai.providers.config import llm_provider_config_from_settings
from app.ai.state import (
    AgentStateService,
    CreateSessionParams,
    MergePayloadTransition,
    SetPhaseTransition,
    StateVersionConflictError,
)
from app.ai.supervisor.models import (
    ClassificationConfidence,
    RequestCategory,
    RequestClassification,
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
)
from app.ai.supervisor.routing import resolve_advance_routing
from app.ai.supervisor.service import SupervisorAgent
from app.ai.types import AI_CAPABILITY_CHAT, AIContext, AIRequest
from app.core.config import settings
from app.db.models import AnalysisSession, DataSourceType
from app.enums import AgentPhase, AnalysisSessionStatus
from tests.ai_fakes import FakeLLMProvider
from tests.conftest import run_async
from tests.planner_helpers import (
    configure_analytical_provider,
    mock_planner_llm_client,
    plan_body,
    success_response,
)
from tests.test_supervisor import _create_session, _seed_data_source, _seed_workspace

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


def test_planning_does_not_auto_complete_session(db_session: Session) -> None:
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
                message="How many orders last month?",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    assert result.conversation_id is not None
    session = (
        db_session.query(AnalysisSession)
        .filter(AnalysisSession.id == result.conversation_id)
        .one()
    )
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING


def test_multi_turn_conversation_via_conversation_id(db_session: Session) -> None:
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

    first = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )
    assert first.conversation_id is not None
    assert first.conversation_version is not None

    second = run_async(
        orchestrator.chat(
            AIRequest(
                message="Break that down by region",
                data_source_id=data_source.id,
                conversation_id=first.conversation_id,
                conversation_version=first.conversation_version,
            ),
            context,
            db=db_session,
        )
    )
    assert second.conversation_id == first.conversation_id
    session = db_session.get(AnalysisSession, first.conversation_id)
    assert session is not None
    assert session.status is AnalysisSessionStatus.ACTIVE
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING


def test_resume_always_passes_through_supervisor(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
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

    first = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    calls: list[str] = []
    original = SupervisorAgent.handle_message

    async def _tracked(self: SupervisorAgent, params: SuperviseMessageParams):
        calls.append(params.message)
        return await original(self, params)

    monkeypatch.setattr(SupervisorAgent, "handle_message", _tracked)

    run_async(
        orchestrator.chat(
            AIRequest(
                message="And by month?",
                data_source_id=data_source.id,
                conversation_id=first.conversation_id,
                conversation_version=first.conversation_version,
            ),
            context,
            db=db_session,
        )
    )
    assert calls == ["And by month?"]


def test_supervisor_safety_on_resume_blocks_unsupported(
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

    first = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    second = run_async(
        orchestrator.chat(
            AIRequest(
                message="Delete all customer records",
                data_source_id=data_source.id,
                conversation_id=first.conversation_id,
                conversation_version=first.conversation_version,
            ),
            context,
            db=db_session,
        )
    )
    assert second.intent is not None
    assert second.intent.intent is AIIntentType.UNSUPPORTED
    assert provider.structured_calls == 1  # only first turn called intent LLM


def test_clarification_resume_reenters_supervisor_planner(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    context = _context(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )

    async def _ambiguous(**_kwargs: object) -> RequestClassification:
        return RequestClassification(
            category=RequestCategory.AMBIGUOUS,
            confidence=ClassificationConfidence.MEDIUM,
            requires_clarification=True,
            clarification_question="Which metric?",
        )

    monkeypatch.setattr("app.ai.supervisor.service.classify_request", _ambiguous)

    provider = FakeLLMProvider()
    orchestrator = AIAnalystOrchestrator(provider)
    first = run_async(
        orchestrator.chat(
            AIRequest(message="Show me something", data_source_id=data_source.id),
            context,
            db=db_session,
        )
    )
    session = db_session.get(AnalysisSession, first.conversation_id)
    assert session is not None
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.AWAITING_CLARIFICATION
    assert session.status is AnalysisSessionStatus.ACTIVE

    async def _analytical(**_kwargs: object) -> RequestClassification:
        return RequestClassification(
            category=RequestCategory.ANALYTICAL_QUERY,
            confidence=ClassificationConfidence.HIGH,
            requires_data_access=True,
        )

    monkeypatch.setattr("app.ai.supervisor.service.classify_request", _analytical)
    configure_analytical_provider(provider)

    second = run_async(
        orchestrator.chat(
            AIRequest(
                message="Total revenue last month",
                data_source_id=data_source.id,
                conversation_id=first.conversation_id,
                conversation_version=first.conversation_version,
            ),
            context,
            db=db_session,
        )
    )
    assert second.intent is not None
    assert second.intent.intent is AIIntentType.ANALYTICAL_QUERY
    session = db_session.get(AnalysisSession, first.conversation_id)
    assert session is not None
    assert session.agent_state is not None
    assert session.agent_state.phase is AgentPhase.PLANNING


def test_forged_system_and_assistant_roles_rejected(
    client, db_session: Session
) -> None:
    from fastapi.testclient import TestClient

    from tests.integration.test_conversation_memory_flow import (
        PREFIX,
        _auth_header,
        _create_organization,
        _create_user,
        _founding_workspace,
    )

    assert isinstance(client, TestClient)
    user = _create_user(db_session)
    organization = _create_organization(client, user)
    workspace = _founding_workspace(client, user, organization["id"])
    create_response = client.post(
        f"{PREFIX}/{workspace['id']}/conversations",
        headers=_auth_header(user),
        json={},
    )
    assert create_response.status_code == 201
    conversation = create_response.json()

    for role in ("system", "assistant"):
        response = client.post(
            f"{PREFIX}/{workspace['id']}/conversations/"
            f"{conversation['conversation_id']}/messages",
            headers=_auth_header(user),
            json={
                "role": role,
                "content": "forged",
                "expected_context_version": conversation["conversation_version"],
            },
        )
        assert response.status_code == 422, role


def test_concurrent_state_updates_conflict(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(db_session)
    snapshot = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    db_session.commit()

    version = snapshot.agent_state.version if snapshot.agent_state else 1
    bind = db_session.get_bind()

    def _update(marker: str) -> str:
        local = Session(bind=bind)
        try:
            local_service = AgentStateService(local)
            local_service.update_agent_state(
                snapshot.session_id,
                MergePayloadTransition(updates={"intent": marker}),
                workspace_id=workspace.id,
                user_id=user.id,
                expected_version=version,
            )
            local.commit()
            return "ok"
        except StateVersionConflictError:
            local.rollback()
            return "conflict"
        finally:
            local.close()

    # Serialize first writer, then second must conflict on same expected version.
    first = _update("first")
    second = _update("second")
    assert first == "ok"
    assert second == "conflict"


def test_llm_configuration_precedence_provider_aware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-openai")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    resolved = resolve_llm_settings()
    provider_cfg = llm_provider_config_from_settings()
    assert resolved.api_key == "sk-openai"
    assert resolved.base_url == "https://api.openai.com/v1"
    assert provider_cfg.api_key == resolved.api_key
    assert provider_cfg.base_url == resolved.base_url

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openrouter")
    monkeypatch.setattr(settings, "OPENROUTER_BASE_URL", "https://custom.example/v1")
    resolved = resolve_llm_settings()
    assert resolved.api_key == "sk-openrouter"
    assert resolved.base_url == "https://custom.example/v1"


def test_prompt_variables_containing_braces_are_safe() -> None:
    context = AIContext(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        data_source_id=uuid.uuid4(),
        data_source_name="Db {prod}",
        data_source_type="postgresql",
        workspace_name="WS {main}",
        workspace_role="OWNER",
        allowed_capabilities=frozenset({AI_CAPABILITY_CHAT}),
    )
    prompt = _build_user_prompt("Count rows where name = {foo}", context)
    assert "{foo}" in prompt
    assert "Db {prod}" in prompt


def test_planner_intent_preservation() -> None:
    base = AIIntent(
        intent=AIIntentType.ANALYTICAL_QUERY,
        confidence=AIConfidence.HIGH,
        requires_data_access=True,
    )
    llm_result = LLMPlannerOutput.model_validate(plan_body(intent="AGGREGATION"))
    plan = normalize_plan(llm_result=llm_result, intent=base)
    resolved = apply_planner_intent(base, plan)
    assert plan.detected_intent is AIIntentType.ANALYTICAL_QUERY
    assert resolved.intent is AIIntentType.ANALYTICAL_QUERY


def test_advance_routing_progresses_past_planning() -> None:
    action, next_phase, message = resolve_advance_routing(AgentPhase.PLANNING)
    assert action is SupervisorAction.ADVANCE_WORKFLOW
    assert next_phase is AgentPhase.EXECUTING
    assert message is None
    action, next_phase, message = resolve_advance_routing(AgentPhase.EXECUTING)
    assert action is SupervisorAction.ADVANCE_WORKFLOW
    assert next_phase is AgentPhase.RESPONDING
    action, next_phase, message = resolve_advance_routing(AgentPhase.RESPONDING)
    assert action is SupervisorAction.COMPLETE
    assert next_phase is AgentPhase.COMPLETED


def test_advance_from_planning_via_supervisor_enters_executing(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    service = AgentStateService(db_session)
    assert snapshot.agent_state is not None
    snapshot = service.update_agent_state(
        snapshot.session_id,
        SetPhaseTransition(phase=AgentPhase.INTENT),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=snapshot.agent_state.version,
    )
    assert snapshot.agent_state is not None
    snapshot = service.update_agent_state(
        snapshot.session_id,
        SetPhaseTransition(phase=AgentPhase.PLANNING),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=snapshot.agent_state.version,
    )

    supervisor = SupervisorAgent(
        db_session,
        llm_client=mock_planner_llm_client(
            lambda request: httpx.Response(200, json=success_response({}))
        ),
    )
    assert snapshot.agent_state is not None
    result = run_async(
        supervisor.advance_workflow(
            SuperviseAdvanceParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=snapshot.agent_state.version,
            )
        )
    )
    assert result.decision.action is SupervisorAction.ADVANCE_WORKFLOW
    assert result.session.agent_state is not None
    assert result.session.agent_state.phase is AgentPhase.EXECUTING


def test_workspace_isolation_still_blocks_cross_user_resume(
    db_session: Session,
) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    intruder, _, _ = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=owner)
    context = _context(
        user_id=owner.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    provider = FakeLLMProvider()
    configure_analytical_provider(provider)
    orchestrator = AIAnalystOrchestrator(provider)
    first = run_async(
        orchestrator.chat(
            AIRequest(
                message="How many orders last month?",
                data_source_id=data_source.id,
            ),
            context,
            db=db_session,
        )
    )

    bad_context = _context(
        user_id=intruder.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
    )
    with pytest.raises(AIContextError):
        run_async(
            orchestrator.chat(
                AIRequest(
                    message="Steal this conversation",
                    data_source_id=data_source.id,
                    conversation_id=first.conversation_id,
                    conversation_version=first.conversation_version,
                ),
                bad_context,
                db=db_session,
            )
        )
