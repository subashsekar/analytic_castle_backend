from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.ai.state import (
    AgentStateService,
    AnalysisSessionNotFoundError,
    ConversationMessage,
    CreateSessionParams,
    SetPhaseTransition,
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
        name="Integration Org",
        slug=f"integration-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Integration Workspace",
        slug=f"integration-ws-{uuid.uuid4().hex[:8]}",
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


def test_agent_state_end_to_end_lifecycle(db_session: Session) -> None:
    user, workspace, organization = _create_workspace_bundle(db_session)
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Sales DB",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()

    service = AgentStateService(db_session)
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            initial_message=ConversationMessage(
                role="user", content="Show revenue trend"
            ),
        )
    )
    assert created.data_source_id == data_source.id
    assert created.agent_state is not None
    assert created.agent_state.phase is AgentPhase.INITIAL

    intent = service.update_agent_state(
        created.session_id,
        SetPhaseTransition(phase=AgentPhase.INTENT),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=created.agent_state.version,
    )
    assert intent.agent_state is not None

    planning = service.update_agent_state(
        created.session_id,
        SetPhaseTransition(phase=AgentPhase.PLANNING),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=intent.agent_state.version,
    )
    assert planning.agent_state is not None
    assert planning.agent_state.phase is AgentPhase.PLANNING

    assert planning.conversation_version == 1
    appended = service.append_message(
        created.session_id,
        ConversationMessage(role="assistant", content="Here is the plan summary."),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_context_version=planning.conversation_version,
    )
    assert appended.conversation_version == 2

    assert planning.agent_state is not None
    completed = service.complete_session(
        created.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
        expected_agent_version=planning.agent_state.version,
    )
    assert completed.status == AnalysisSessionStatus.COMPLETED.value
    assert completed.conversation is not None
    assert completed.conversation.message_count == 2

    reloaded = service.get_session(
        created.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert reloaded.status == AnalysisSessionStatus.COMPLETED.value
    assert reloaded.agent_state is not None
    assert reloaded.agent_state.phase is AgentPhase.COMPLETED


def test_cross_workspace_session_access_is_hidden(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _create_workspace_bundle(db_session)
    user_b, workspace_b, _organization_b = _create_workspace_bundle(db_session)

    service = AgentStateService(db_session)
    session_a = service.create_session(
        CreateSessionParams(
            user_id=user_a.id,
            workspace_id=workspace_a.id,
            organization_id=organization_a.id,
        )
    )

    with pytest.raises(AnalysisSessionNotFoundError):
        service.get_session(
            session_a.session_id,
            workspace_id=workspace_b.id,
            user_id=user_b.id,
        )

    with pytest.raises(AnalysisSessionNotFoundError):
        service.get_session(
            session_a.session_id,
            workspace_id=workspace_a.id,
            user_id=user_b.id,
        )
