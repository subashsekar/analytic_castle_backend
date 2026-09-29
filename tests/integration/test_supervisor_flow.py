from __future__ import annotations

import json
import uuid

import httpx
from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient
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
from tests.test_supervisor import _classification_body, _client_config


def _create_user(db_session: Session) -> User:
    user = User(
        first_name="Katherine",
        last_name="Johnson",
        email=f"katherine-{uuid.uuid4().hex[:8]}@example.com",
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
        name="Supervisor Flow Org",
        slug=f"supervisor-flow-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Supervisor Flow Workspace",
        slug=f"supervisor-flow-ws-{uuid.uuid4().hex[:8]}",
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


def test_supervisor_end_to_end_workflow(db_session: Session) -> None:
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

    def handler(request: httpx.Request) -> httpx.Response:
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
            _client_config(), transport=httpx.MockTransport(handler)
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
    assert advanced.session.status == AnalysisSessionStatus.ACTIVE.value
