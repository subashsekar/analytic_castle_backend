from __future__ import annotations

import json
import uuid

import httpx
from sqlalchemy.orm import Session

from app.ai.intent_types import AIConfidence, AIIntent, AIIntentType, AIOperationType
from app.ai.llm import AsyncLLMClient
from app.ai.planner_agent import PlanCreateParams, PlannerActionKind, PlannerAgent
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.supervisor import (
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
    SupervisorAgent,
)
from app.db.models import (
    DataSource,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
)
from app.enums import AgentPhase, AnalysisSessionStatus, WorkspaceRole
from tests.conftest import run_async
from tests.planner_helpers import mock_planner_llm_client, plan_body, success_response
from tests.test_supervisor import _classification_body, _client_config


def _create_user(db_session: Session) -> User:
    user = User(
        first_name="Grace",
        last_name="Hopper",
        email=f"grace-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _create_workspace_bundle(
    db_session: Session,
) -> tuple[User, Workspace, Organization]:
    user = _create_user(db_session)
    organization = Organization(
        name="Planner Flow Org",
        slug=f"planner-flow-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Planner Flow Workspace",
        slug=f"planner-flow-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=user.id,
            role=WorkspaceRole.OWNER,
        )
    )
    db_session.flush()
    return user, workspace, organization


def test_planner_end_to_end_at_planning_phase(db_session: Session) -> None:
    user, workspace, organization = _create_workspace_bundle(db_session)
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Orders DB",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()

    state_service = AgentStateService(db_session)
    snapshot = state_service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
        )
    )

    def classification_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "gen-flow-1",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(_classification_body()),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    supervisor = SupervisorAgent(
        db_session,
        llm_client=AsyncLLMClient(
            _client_config(), transport=httpx.MockTransport(classification_handler)
        ),
    )

    routed = run_async(
        supervisor.handle_message(
            SuperviseMessageParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                message="Show total revenue by region",
                expected_agent_version=snapshot.agent_state.version,
                expected_context_version=snapshot.conversation_version or 1,
            )
        )
    )
    assert routed.decision.action is SupervisorAction.RUN_INTENT_AGENT
    assert routed.session.agent_state is not None
    assert routed.session.agent_state.phase is AgentPhase.INTENT

    advanced = run_async(
        supervisor.advance_workflow(
            SuperviseAdvanceParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=routed.session.agent_state.version,
            )
        )
    )
    assert advanced.decision.action is SupervisorAction.RUN_PLANNER
    assert advanced.session.agent_state is not None
    assert advanced.session.agent_state.phase is AgentPhase.PLANNING

    planner = PlannerAgent(
        db_session,
        llm_client=mock_planner_llm_client(
            lambda _: httpx.Response(200, json=success_response(plan_body()))
        ),
    )
    planned = run_async(
        planner.create_plan(
            PlanCreateParams(
                session_id=snapshot.session_id,
                workspace_id=workspace.id,
                user_id=user.id,
                expected_agent_version=advanced.session.agent_state.version,
                intent=AIIntent(
                    intent=AIIntentType.ANALYTICAL_QUERY,
                    operation=AIOperationType.AGGREGATE,
                    subject="orders",
                    confidence=AIConfidence.HIGH,
                    requires_data_access=True,
                    requires_metadata=True,
                ),
                message="Show total revenue by region",
                data_source_name="Orders DB",
            )
        )
    )

    assert planned.plan.action_steps[0].action is PlannerActionKind.RESOLVE_METADATA
    assert planned.session.agent_state is not None
    assert planned.session.agent_state.payload.plan_version == "v1"
    assert planned.session.agent_state.payload.metadata_refs
    assert planned.session.status == AnalysisSessionStatus.ACTIVE.value
