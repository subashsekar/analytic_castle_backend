from __future__ import annotations

import asyncio
import uuid

import pytest

from app.ai.llm.models import LLMMessage
from app.ai.memory import (
    ConversationMemoryService,
    ConversationNotActiveError,
    ConversationNotFoundError,
    CreateConversationParams,
    InvalidPaginationError,
    PlaceholderSummarizationHook,
    StateVersionConflictError,
    build_llm_context,
    paginate_items,
    select_messages_for_llm,
    trim_context_for_storage,
)
from app.ai.state.limits import ContextLimits
from app.ai.state.models import ConversationContextData, ConversationMessage
from app.db.models import (
    DataSource,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.enums import AnalysisSessionStatus


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


def test_create_and_get_conversation(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(db_session, limits=_limits())
    created = service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            initial_message=ConversationMessage(role="user", content="Hello"),
        )
    )
    fetched = service.get_conversation(
        created.conversation_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert fetched.message_count == 1
    assert fetched.conversation_version == 1
    assert fetched.status == AnalysisSessionStatus.ACTIVE.value


def test_list_conversations_pagination(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(
        db_session,
        limits=_limits(),
        default_page_size=2,
        max_page_size=2,
    )
    for index in range(3):
        service.create_conversation(
            CreateConversationParams(
                user_id=user.id,
                workspace_id=workspace.id,
                organization_id=organization.id,
                initial_message=ConversationMessage(
                    role="user", content=f"Message {index}"
                ),
            )
        )
    page_one = service.list_conversations(
        workspace_id=workspace.id,
        user_id=user.id,
        page=1,
    )
    assert page_one.total == 3
    assert len(page_one.items) == 2
    page_two = service.list_conversations(
        workspace_id=workspace.id,
        user_id=user.id,
        page=2,
    )
    assert len(page_two.items) == 1


def test_get_messages_ordering_and_pagination(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(
        db_session,
        limits=_limits(max_context_chars=2000),
        default_page_size=2,
    )
    created = service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    version = created.conversation_version
    for index in range(3):
        updated = service.append_message(
            created.conversation_id,
            ConversationMessage(role="user", content=f"Turn {index}"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=version,
        )
        version = updated.conversation_version

    page = service.get_messages(
        created.conversation_id,
        workspace_id=workspace.id,
        user_id=user.id,
        page=1,
        page_size=2,
    )
    assert page.total == 3
    assert [item.content for item in page.items] == ["Turn 0", "Turn 1"]


def test_append_message_version_conflict(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(db_session, limits=_limits())
    created = service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    with pytest.raises(StateVersionConflictError):
        service.append_message(
            created.conversation_id,
            ConversationMessage(role="user", content="Late"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=99,
        )


def test_workspace_isolation_hides_other_users(db_session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    intruder = _user()
    db_session.add(intruder)
    db_session.flush()
    service = ConversationMemoryService(db_session, limits=_limits())
    created = service.create_conversation(
        CreateConversationParams(
            user_id=owner.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    with pytest.raises(ConversationNotFoundError):
        service.get_conversation(
            created.conversation_id,
            workspace_id=workspace.id,
            user_id=intruder.id,
        )


def test_cross_workspace_access_hidden(db_session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    other_org = Organization(name="Other", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add(other_org)
    db_session.flush()
    other_workspace = Workspace(
        organization_id=other_org.id,
        name="Other",
        slug=f"other-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(other_workspace)
    db_session.flush()

    service = ConversationMemoryService(db_session, limits=_limits())
    created = service.create_conversation(
        CreateConversationParams(
            user_id=owner.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    with pytest.raises(ConversationNotFoundError):
        service.get_conversation(
            created.conversation_id,
            workspace_id=other_workspace.id,
            user_id=owner.id,
        )


def test_missing_conversation_returns_not_found(db_session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    service = ConversationMemoryService(db_session, limits=_limits())
    with pytest.raises(ConversationNotFoundError):
        service.get_conversation(
            uuid.uuid4(),
            workspace_id=workspace.id,
            user_id=user.id,
        )


def test_select_messages_for_llm_trims_old_messages() -> None:
    limits = _limits(max_context_chars=12, max_context_messages=10)
    messages = [
        ConversationMessage(role="user", content="1234567890"),
        ConversationMessage(role="assistant", content="abcdefghij"),
        ConversationMessage(role="user", content="latest"),
    ]
    selection = select_messages_for_llm(messages, limits=limits)
    assert selection.trimmed is True
    assert selection.dropped_message_count >= 1
    assert selection.messages[-1].content == "latest"


def test_build_llm_context_injects_summary_when_trimmed() -> None:
    limits = _limits(max_context_chars=15, max_context_messages=10)
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="old message"),
            ConversationMessage(role="assistant", content="recent reply"),
        ]
    )
    result = asyncio.run(
        build_llm_context(
            context, limits=limits, summarization_hook=PlaceholderSummarizationHook()
        )
    )
    assert result.trimmed is True
    assert result.summary_injected is True
    assert result.messages[0].role == "system"
    assert "trimmed" in result.messages[0].content.lower()


class _CountingSummarizationHook:
    def __init__(self) -> None:
        self.calls = 0

    async def summarize(self, messages) -> str:
        self.calls += 1
        assert len(messages) >= 1
        return "custom summary"


def test_custom_summarization_hook() -> None:
    hook = _CountingSummarizationHook()
    limits = _limits(max_context_chars=8, max_context_messages=10)
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="first"),
            ConversationMessage(role="user", content="second"),
        ]
    )
    result = asyncio.run(
        build_llm_context(context, limits=limits, summarization_hook=hook)
    )
    assert hook.calls == 1
    assert result.messages[0].content == "custom summary"


def test_trim_context_for_storage_drops_oldest() -> None:
    limits = _limits(max_context_chars=15, max_context_messages=10)
    context = ConversationContextData(
        messages=[ConversationMessage(role="user", content="1234567890")]
    )
    incoming = ConversationMessage(role="assistant", content="6789012345")
    trimmed = trim_context_for_storage(context, incoming, limits=limits)
    assert trimmed.message_count == 1
    assert trimmed.messages[-1].content == "6789012345"


def test_append_with_trim_when_storage_full(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(
        db_session,
        limits=_limits(max_context_chars=15, max_message_chars=10),
    )
    created = service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            initial_message=ConversationMessage(role="user", content="1234567890"),
        )
    )
    updated = service.append_message(
        created.conversation_id,
        ConversationMessage(role="assistant", content="6789012345"),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_context_version=created.conversation_version,
        trim_if_needed=True,
    )
    assert updated.message_count == 1
    assert updated.char_count == 10


def test_invalid_pagination_page_size(db_session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    service = ConversationMemoryService(
        db_session,
        limits=_limits(),
        max_page_size=5,
    )
    with pytest.raises(InvalidPaginationError):
        service.list_conversations(
            workspace_id=workspace.id,
            user_id=user.id,
            page_size=10,
        )


def test_paginate_items_rejects_out_of_range_page() -> None:
    with pytest.raises(InvalidPaginationError):
        paginate_items(["a", "b"], page=3, page_size=1)


def test_build_llm_context_never_returns_full_history_when_over_limit() -> None:
    limits = _limits(max_context_chars=30, max_context_messages=2)
    context = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="one"),
            ConversationMessage(role="assistant", content="two"),
            ConversationMessage(role="user", content="three"),
            ConversationMessage(role="assistant", content="four"),
        ]
    )
    result = asyncio.run(build_llm_context(context, limits=limits))
    assert result.included_message_count <= limits.max_context_messages
    assert result.total_message_count == 4
    assert result.trimmed is True
    assert all(isinstance(message, LLMMessage) for message in result.messages)


def test_sequential_appends_increment_version(db_session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    service = ConversationMemoryService(db_session, limits=_limits())
    created = service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    first = service.append_message(
        created.conversation_id,
        ConversationMessage(role="user", content="Follow-up"),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_context_version=created.conversation_version,
    )
    assert first.conversation_version == created.conversation_version + 1
    second = service.append_message(
        created.conversation_id,
        ConversationMessage(role="assistant", content="Reply"),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_context_version=first.conversation_version,
    )
    assert second.conversation_version == first.conversation_version + 1
    with pytest.raises(StateVersionConflictError):
        service.append_message(
            created.conversation_id,
            ConversationMessage(role="user", content="Stale"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=created.conversation_version,
        )


def test_append_message_allow_inactive_on_completed_session(db_session) -> None:
    from app.ai.state import AgentStateService
    from app.enums import AgentPhase

    user, workspace, organization = _seed_workspace(db_session)
    state_service = AgentStateService(db_session)
    memory_service = ConversationMemoryService(db_session, limits=_limits())
    created = memory_service.create_conversation(
        CreateConversationParams(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
        )
    )
    snapshot = state_service.get_session(
        created.conversation_id,
        workspace_id=workspace.id,
        user_id=user.id,
    )
    assert snapshot.agent_state is not None
    state_service.complete_session(
        created.conversation_id,
        workspace_id=workspace.id,
        user_id=user.id,
        expected_agent_version=snapshot.agent_state.version,
        final_phase=AgentPhase.COMPLETED,
    )
    with pytest.raises(ConversationNotActiveError):
        memory_service.append_message(
            created.conversation_id,
            ConversationMessage(role="assistant", content="Blocked"),
            workspace_id=workspace.id,
            user_id=user.id,
            expected_context_version=created.conversation_version,
        )
    updated = memory_service.append_message(
        created.conversation_id,
        ConversationMessage(role="assistant", content="Persisted"),
        workspace_id=workspace.id,
        user_id=user.id,
        expected_context_version=created.conversation_version,
        allow_inactive=True,
    )
    assert updated.message_count == 1


def test_build_intent_prompt_messages_with_memory_includes_prior_turns() -> None:
    from app.ai.intent import build_intent_prompt_messages_with_memory
    from app.ai.types import AIContext

    context = AIContext(
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        data_source_id=uuid.uuid4(),
        data_source_name="Analytics DB",
        data_source_type="POSTGRESQL",
        workspace_name="Workspace",
        workspace_role="OWNER",
    )
    conversation = ConversationContextData(
        messages=[
            ConversationMessage(role="user", content="Earlier question"),
            ConversationMessage(role="assistant", content="Earlier answer"),
            ConversationMessage(role="user", content="Follow-up question"),
        ]
    )
    messages = asyncio.run(
        build_intent_prompt_messages_with_memory(
            "Follow-up question",
            context,
            conversation=conversation,
        )
    )
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert "Follow-up question" in messages[-1].content
    assert any("Earlier question" in message.content for message in messages[:-1])
