from __future__ import annotations

import logging
import uuid

import pytest

from app.ai.llm.models import LLMMessage
from app.ai.state import (
    AgentStateDataSourceNotFoundError,
    AgentStateService,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    ContextLimitExceededError,
    ConversationMessage,
    CreateSessionParams,
    InvalidStateTransitionError,
    MergePayloadTransition,
    MessageLimitExceededError,
    ReplacePayloadTransition,
    SetPhaseTransition,
    StateSerializationError,
    StateVersionConflictError,
    agent_state_log_context,
    apply_state_transition,
    conversation_log_context,
    conversation_message_to_llm,
    conversation_to_llm_messages,
    deserialize_context,
    deserialize_payload,
    initial_agent_state_snapshot,
    llm_message_to_conversation,
    serialize_context,
    serialize_payload,
    session_log_context,
)
from app.ai.state.limits import ContextLimits
from app.ai.state.models import (
    AgentStatePayload,
    AgentStateSnapshot,
    ConversationContextData,
)
from app.core.logging import RedactingFilter, redact_secret
from app.db.models import (
    AgentState,
    AnalysisSession,
    ConversationContext,
    DataSource,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.enums import AgentPhase, AnalysisSessionStatus

SECRET_VALUE = "sk-test-agent-state-secret"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"


def _limits(
    *,
    max_context_chars: int = 500,
    max_message_chars: int = 200,
    max_context_messages: int = 100,
) -> ContextLimits:
    return ContextLimits(
        max_context_chars=max_context_chars,
        max_message_chars=max_message_chars,
        max_context_messages=max_context_messages,
    )


def _user() -> User:
    return User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )


def _seed_workspace(db_session) -> tuple[User, Workspace, Organization]:
    user = _user()
    organization = Organization(
        name="AnalyticCastle",
        slug=f"analyticcastle-{uuid.uuid4().hex[:8]}",
    )
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug=f"analytics-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    return user, workspace, organization


def _seed_data_source(db_session, *, workspace: Workspace, user: User) -> DataSource:
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Production DB",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    return data_source


def test_initial_agent_state_snapshot_defaults() -> None:
    snapshot = initial_agent_state_snapshot()
    assert snapshot.phase is AgentPhase.INITIAL
    assert snapshot.version == 1
    assert snapshot.payload.intent is None


def test_serialize_and_deserialize_payload() -> None:
    payload = AgentStatePayload(intent="lookup", metadata_refs=["public.orders"])
    encoded = serialize_payload(payload)
    decoded = deserialize_payload(encoded)
    assert decoded.intent == "lookup"
    assert decoded.metadata_refs == ["public.orders"]


def test_deserialize_payload_rejects_invalid_data() -> None:
    with pytest.raises(StateSerializationError):
        deserialize_payload({"intent": "x" * 128})


def test_serialize_and_deserialize_context() -> None:
    context = ConversationContextData(
        messages=[ConversationMessage(role="user", content="How many orders?")]
    )
    encoded = serialize_context(context)
    decoded = deserialize_context(encoded)
    assert decoded.message_count == 1
    assert decoded.messages[0].content == "How many orders?"


def test_deserialize_context_rejects_invalid_messages() -> None:
    with pytest.raises(StateSerializationError):
        deserialize_context([{"role": "user", "content": ""}])


def test_apply_set_phase_transition() -> None:
    current = initial_agent_state_snapshot()
    updated = apply_state_transition(
        current, SetPhaseTransition(phase=AgentPhase.INTENT)
    )
    assert updated.phase is AgentPhase.INTENT
    assert updated.version == 2


def test_invalid_phase_jump_is_rejected() -> None:
    current = initial_agent_state_snapshot()
    with pytest.raises(InvalidStateTransitionError):
        apply_state_transition(current, SetPhaseTransition(phase=AgentPhase.PLANNING))


def test_apply_merge_payload_transition() -> None:
    current = AgentStateSnapshot(
        phase=AgentPhase.INTENT,
        payload=AgentStatePayload(intent="lookup"),
        version=2,
    )
    updated = apply_state_transition(
        current,
        MergePayloadTransition(updates={"plan_version": "v1"}),
    )
    assert updated.payload.intent == "lookup"
    assert updated.payload.plan_version == "v1"
    assert updated.version == 3


def test_apply_replace_payload_transition() -> None:
    current = AgentStateSnapshot(
        phase=AgentPhase.INTENT,
        payload=AgentStatePayload(intent="lookup"),
        version=2,
    )
    updated = apply_state_transition(
        current,
        ReplacePayloadTransition(payload=AgentStatePayload(intent="compare")),
    )
    assert updated.payload.intent == "compare"
    assert updated.version == 3


def test_completed_phase_rejects_transition() -> None:
    current = AgentStateSnapshot(
        phase=AgentPhase.COMPLETED,
        payload=AgentStatePayload(),
        version=4,
    )
    with pytest.raises(InvalidStateTransitionError):
        apply_state_transition(current, SetPhaseTransition(phase=AgentPhase.INTENT))


def test_failed_phase_rejects_transition() -> None:
    current = AgentStateSnapshot(
        phase=AgentPhase.FAILED,
        payload=AgentStatePayload(),
        version=4,
    )
    with pytest.raises(InvalidStateTransitionError):
        apply_state_transition(current, SetPhaseTransition(phase=AgentPhase.INTENT))


def test_context_limit_validation() -> None:
    limits = _limits(max_context_chars=10, max_message_chars=20)
    message = ConversationMessage(role="user", content="1234567890")
    context = ConversationContextData(messages=[message, message])
    with pytest.raises(ContextLimitExceededError):
        from app.ai.state.limits import validate_context_size

        validate_context_size(context, limits=limits)


def test_message_count_limit_validation() -> None:
    limits = _limits(max_context_messages=2)
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="a"),
            ConversationMessage(role="assistant", content="b"),
            ConversationMessage(role="user", content="c"),
        ]
    )
    with pytest.raises(ContextLimitExceededError):
        from app.ai.state.limits import validate_context_size

        validate_context_size(context, limits=limits)


def test_message_limit_validation() -> None:
    limits = _limits(max_message_chars=5)
    message = ConversationMessage(role="user", content="123456")
    with pytest.raises(MessageLimitExceededError):
        from app.ai.state.limits import validate_message_size

        validate_message_size(message, limits=limits)


def test_conversation_message_llm_bridge() -> None:
    llm_message = LLMMessage(role="user", content="Hello")
    conversation = ConversationMessage.from_llm(llm_message)
    assert conversation.to_llm() == llm_message
    assert conversation_message_to_llm(conversation) == llm_message
    round_trip = llm_message_to_conversation(llm_message)
    assert round_trip.role == llm_message.role
    assert round_trip.content == llm_message.content


def test_conversation_to_llm_messages() -> None:
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="Hi"),
            ConversationMessage(role="assistant", content="Hello"),
        ]
    )
    assert conversation_to_llm_messages(context) == [
        LLMMessage(role="user", content="Hi"),
        LLMMessage(role="assistant", content="Hello"),
    ]


def test_logging_helpers_do_not_emit_content(caplog: pytest.LogCaptureFixture) -> None:
    snapshot = AgentStateSnapshot(
        phase=AgentPhase.PLANNING,
        payload=AgentStatePayload(intent="lookup", notes=SECRET_VALUE),
        version=2,
    )
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content=f"secret={CUSTOMER_PASSWORD}")
        ]
    )
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("tests.agent_state")
    logger.info(
        "state event",
        extra={
            **session_log_context(
                session_id=uuid.uuid4(),
                workspace_id=uuid.uuid4(),
                user_id=uuid.uuid4(),
                status=AnalysisSessionStatus.ACTIVE,
            ),
            **agent_state_log_context(snapshot),
            **conversation_log_context(context),
        },
    )
    record = caplog.records[-1]
    rendered = RedactingFilter().filter(record)
    assert rendered is True
    assert SECRET_VALUE not in record.getMessage()
    assert CUSTOMER_PASSWORD not in record.getMessage()
    assert "conversation_message_count" in record.__dict__


def test_redact_secret_sanitizes_error_messages() -> None:
    assert SECRET_VALUE not in redact_secret(f"api_key={SECRET_VALUE}")


def test_create_session_persists_state(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=500, max_message_chars=200)
    )
    snapshot = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            initial_message=ConversationMessage(role="user", content="Hello"),
        )
    )
    assert snapshot.status == AnalysisSessionStatus.ACTIVE.value
    assert snapshot.agent_state is not None
    assert snapshot.agent_state.phase is AgentPhase.INITIAL
    assert snapshot.conversation is not None
    assert snapshot.conversation.message_count == 1
    assert snapshot.conversation_version == 1

    row = db_session.get(AnalysisSession, snapshot.session_id)
    assert row is not None
    assert db_session.query(AgentState).filter_by(session_id=row.id).one()
    assert db_session.query(ConversationContext).filter_by(session_id=row.id).one()


def test_create_session_rejects_organization_mismatch(db_session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    other_org = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add(other_org)
    db_session.flush()

    service = AgentStateService(db_session, limits=_limits())
    with pytest.raises(AnalysisSessionNotFoundError):
        service.create_session(
            CreateSessionParams(
                user_id=user.id,
                workspace_id=workspace.id,
                organization_id=other_org.id,
            )
        )


def test_update_agent_state_and_complete(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=500, max_message_chars=200)
    )
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    assert created.agent_state is not None
    updated = service.update_agent_state(
        created.session_id,
        SetPhaseTransition(phase=AgentPhase.INTENT),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=created.agent_state.version,
    )
    assert updated.agent_state is not None
    assert updated.agent_state.phase is AgentPhase.INTENT

    merged = service.update_agent_state(
        created.session_id,
        MergePayloadTransition(updates={"intent": "lookup"}),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_version=updated.agent_state.version,
    )
    assert merged.agent_state is not None
    assert merged.agent_state.payload.intent == "lookup"

    completed = service.complete_session(
        created.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
        expected_agent_version=merged.agent_state.version,
    )
    assert completed.status == AnalysisSessionStatus.COMPLETED.value
    assert completed.agent_state is not None
    assert completed.agent_state.phase is AgentPhase.COMPLETED


def test_agent_state_version_conflict(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(db_session, limits=_limits())
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    assert created.agent_state is not None
    with pytest.raises(StateVersionConflictError):
        service.update_agent_state(
            created.session_id,
            SetPhaseTransition(phase=AgentPhase.INTENT),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_version=created.agent_state.version + 99,
        )


def test_fail_session_marks_failed_and_blocks_updates(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=500, max_message_chars=200)
    )
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    assert created.agent_state is not None
    failed = service.fail_session(
        created.session_id,
        workspace_id=workspace.id,
        user_id=user.id,
        expected_agent_version=created.agent_state.version,
        error_message=f"internal {SECRET_VALUE}",
    )
    assert failed.status == AnalysisSessionStatus.FAILED.value
    assert failed.error_message == "Analysis session failed"
    assert failed.agent_state is not None
    assert failed.agent_state.phase is AgentPhase.FAILED
    assert SECRET_VALUE not in (failed.error_message or "")

    with pytest.raises(AnalysisSessionNotActiveError):
        service.update_agent_state(
            created.session_id,
            SetPhaseTransition(phase=AgentPhase.INTENT),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_version=failed.agent_state.version,
        )


def test_workspace_isolation_hides_other_users_sessions(db_session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    intruder = _user()
    db_session.add(intruder)
    db_session.flush()

    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=500, max_message_chars=200)
    )
    created = service.create_session(
        CreateSessionParams(
            user_id=owner.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )

    with pytest.raises(AnalysisSessionNotFoundError):
        service.get_session(
            created.session_id,
            workspace_id=workspace.id,
            user_id=intruder.id,
        )


def test_data_source_must_belong_to_workspace(db_session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    other_user = _user()
    other_org = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add_all([other_user, other_org])
    db_session.flush()
    other_workspace = Workspace(
        organization_id=other_org.id,
        name="Other WS",
        slug=f"other-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(other_workspace)
    db_session.flush()
    foreign_source = _seed_data_source(
        db_session,
        workspace=other_workspace,
        user=other_user,
    )

    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=500, max_message_chars=200)
    )
    with pytest.raises(AgentStateDataSourceNotFoundError):
        service.create_session(
            CreateSessionParams(
                user_id=owner.id,
                workspace_id=workspace.id,
                organization_id=organization.id,
                data_source_id=foreign_source.id,
            )
        )


def test_append_message_enforces_limits(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(
        db_session, limits=_limits(max_context_chars=15, max_message_chars=10)
    )
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            initial_message=ConversationMessage(role="user", content="123456"),
        )
    )
    assert created.conversation_version == 1
    with pytest.raises(ContextLimitExceededError):
        service.append_message(
            created.session_id,
            ConversationMessage(role="assistant", content="6789012345"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=created.conversation_version,
        )


def test_append_message_version_conflict(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = AgentStateService(db_session, limits=_limits())
    created = service.create_session(
        CreateSessionParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    with pytest.raises(StateVersionConflictError):
        service.append_message(
            created.session_id,
            ConversationMessage(role="user", content="Hello"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=99,
        )
