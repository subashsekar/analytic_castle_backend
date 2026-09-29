"""Conversation memory persistence and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.ai.memory.errors import (
    AgentStateDataSourceNotFoundError,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    ContextLimitExceededError,
    ConversationNotActiveError,
    ConversationNotFoundError,
    InvalidPaginationError,
    MessageLimitExceededError,
    StateVersionConflictError,
)
from app.ai.memory.models import ConversationRecord, MessageRecord, Page
from app.ai.memory.pagination import normalize_page_params, paginate_items
from app.ai.state import AgentStateService, CreateSessionParams
from app.ai.state.limits import ContextLimits
from app.ai.state.models import ConversationContextData, ConversationMessage
from app.db.models import AnalysisSession, Workspace
from app.enums import AnalysisSessionStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreateConversationParams:
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None = None
    initial_message: ConversationMessage | None = None


@dataclass(frozen=True)
class UpdateConversationParams:
    status: AnalysisSessionStatus | None = None


def _map_not_found(exc: AnalysisSessionNotFoundError) -> ConversationNotFoundError:
    return ConversationNotFoundError(str(exc))


def _map_not_active(exc: AnalysisSessionNotActiveError) -> ConversationNotActiveError:
    return ConversationNotActiveError(str(exc))


class ConversationMemoryService:
    """Manage conversation history separate from MCP/query execution.

    Wraps ``AgentStateService`` for session lifecycle while owning message
    pagination, storage trimming, and LLM context selection.
    """

    def __init__(
        self,
        session: Session,
        *,
        limits: ContextLimits | None = None,
        default_page_size: int | None = None,
        max_page_size: int | None = None,
    ) -> None:
        from app.core.config import settings

        self._session = session
        self._state_service = AgentStateService(session, limits=limits)
        self._limits = limits or ContextLimits.from_settings()
        self._default_page_size = (
            default_page_size or settings.AI_CONVERSATION_DEFAULT_PAGE_SIZE
        )
        self._max_page_size = max_page_size or settings.AI_CONVERSATION_MAX_PAGE_SIZE

    def create_conversation(
        self, params: CreateConversationParams
    ) -> ConversationRecord:
        try:
            snapshot = self._state_service.create_session(
                CreateSessionParams(
                    user_id=params.user_id,
                    workspace_id=params.workspace_id,
                    organization_id=params.organization_id,
                    data_source_id=params.data_source_id,
                    initial_message=params.initial_message,
                )
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        except AgentStateDataSourceNotFoundError:
            raise
        return self._to_record(snapshot)

    def get_conversation(
        self,
        conversation_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> ConversationRecord:
        try:
            snapshot = self._state_service.get_session(
                conversation_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        return self._to_record(snapshot)

    def list_conversations(
        self,
        *,
        workspace_id: UUID,
        user_id: UUID,
        page: int | None = None,
        page_size: int | None = None,
        status: AnalysisSessionStatus | None = None,
    ) -> Page[ConversationRecord]:
        page_num = page or 1
        size = page_size or self._default_page_size
        page_num, size = normalize_page_params(
            page=page_num,
            page_size=size,
            max_page_size=self._max_page_size,
        )
        self._ensure_workspace_exists(workspace_id)

        filters = [
            AnalysisSession.workspace_id == workspace_id,
            AnalysisSession.user_id == user_id,
        ]
        if status is not None:
            filters.append(AnalysisSession.status == status)

        total = self._session.scalar(
            select(func.count()).select_from(AnalysisSession).where(*filters)
        )
        total_count = int(total or 0)
        if total_count == 0:
            return Page(items=(), page=page_num, page_size=size, total=0)

        start = (page_num - 1) * size
        if start >= total_count:
            raise InvalidPaginationError("Page exceeds available results")

        rows = self._session.scalars(
            select(AnalysisSession)
            .where(*filters)
            .order_by(AnalysisSession.updated_at.desc())
            .offset(start)
            .limit(size)
        ).all()
        # Touch relationships used by _record_from_row (agent_version).
        for row in rows:
            _ = row.agent_state
            _ = row.conversation_context

        items = tuple(self._record_from_row(row) for row in rows)
        return Page(items=items, page=page_num, page_size=size, total=total_count)

    def update_conversation(
        self,
        conversation_id: UUID,
        params: UpdateConversationParams,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_agent_version: int,
    ) -> ConversationRecord:
        if params.status is AnalysisSessionStatus.COMPLETED:
            try:
                snapshot = self._state_service.complete_session(
                    conversation_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    expected_agent_version=expected_agent_version,
                )
            except AnalysisSessionNotFoundError as exc:
                raise _map_not_found(exc) from exc
            except AnalysisSessionNotActiveError as exc:
                raise _map_not_active(exc) from exc
            except StateVersionConflictError:
                raise
            return self._to_record(snapshot)
        raise InvalidPaginationError("Unsupported conversation update")

    def append_message(
        self,
        conversation_id: UUID,
        message: ConversationMessage,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_context_version: int,
        trim_if_needed: bool = False,
        allow_inactive: bool = False,
    ) -> ConversationRecord:
        require_active = not allow_inactive
        try:
            if trim_if_needed:
                return self._append_with_trim(
                    conversation_id,
                    message,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    expected_context_version=expected_context_version,
                    require_active=require_active,
                )
            snapshot = self._state_service.append_message(
                conversation_id,
                message,
                workspace_id=workspace_id,
                user_id=user_id,
                expected_context_version=expected_context_version,
                require_active=require_active,
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        except AnalysisSessionNotActiveError as exc:
            raise _map_not_active(exc) from exc
        except (
            StateVersionConflictError,
            MessageLimitExceededError,
            ContextLimitExceededError,
        ):
            raise
        return self._to_record(snapshot)

    def get_messages(
        self,
        conversation_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        page: int | None = None,
        page_size: int | None = None,
    ) -> Page[MessageRecord]:
        page_num = page or 1
        size = page_size or self._default_page_size
        page_num, size = normalize_page_params(
            page=page_num,
            page_size=size,
            max_page_size=self._max_page_size,
        )
        try:
            snapshot = self._state_service.get_session(
                conversation_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        messages = snapshot.conversation.messages if snapshot.conversation else []
        page_result = paginate_items(messages, page=page_num, page_size=size)
        logger.info(
            "conversation messages retrieved",
            extra={
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "page": page_num,
                "page_size": size,
                "total_messages": page_result.total,
            },
        )
        return page_result

    def get_context_data(
        self,
        conversation_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> ConversationContextData:
        try:
            snapshot = self._state_service.get_session(
                conversation_id,
                workspace_id=workspace_id,
                user_id=user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        if snapshot.conversation is None:
            return ConversationContextData()
        return snapshot.conversation

    def _append_with_trim(
        self,
        conversation_id: UUID,
        message: ConversationMessage,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_context_version: int,
        require_active: bool = True,
    ) -> ConversationRecord:
        try:
            snapshot = self._state_service.append_message_with_trim(
                conversation_id,
                message,
                workspace_id=workspace_id,
                user_id=user_id,
                expected_context_version=expected_context_version,
                require_active=require_active,
            )
        except AnalysisSessionNotFoundError as exc:
            raise _map_not_found(exc) from exc
        except AnalysisSessionNotActiveError as exc:
            raise _map_not_active(exc) from exc
        except (
            StateVersionConflictError,
            MessageLimitExceededError,
            ContextLimitExceededError,
        ):
            raise
        return self._to_record(snapshot)

    def _ensure_workspace_exists(self, workspace_id: UUID) -> None:
        workspace = self._session.get(Workspace, workspace_id)
        if workspace is None:
            raise ConversationNotFoundError("Conversation not found")

    def _to_record(self, snapshot) -> ConversationRecord:
        row = self._session.get(AnalysisSession, snapshot.session_id)
        conversation = snapshot.conversation
        return ConversationRecord(
            conversation_id=snapshot.session_id,
            user_id=snapshot.user_id,
            workspace_id=snapshot.workspace_id,
            organization_id=snapshot.organization_id,
            data_source_id=snapshot.data_source_id,
            status=snapshot.status,
            message_count=conversation.message_count if conversation else 0,
            char_count=conversation.char_count if conversation else 0,
            conversation_version=snapshot.conversation_version or 1,
            agent_version=(
                snapshot.agent_state.version
                if snapshot.agent_state is not None
                else 1
            ),
            started_at=row.started_at if row is not None else datetime.now(UTC),
            updated_at=row.updated_at if row is not None else datetime.now(UTC),
            error_message=snapshot.error_message,
        )

    def _record_from_row(self, row: AnalysisSession) -> ConversationRecord:
        context = row.conversation_context
        agent = row.agent_state
        return ConversationRecord(
            conversation_id=row.id,
            user_id=row.user_id,
            workspace_id=row.workspace_id,
            organization_id=row.organization_id,
            data_source_id=row.data_source_id,
            status=row.status.value,
            message_count=context.message_count if context else 0,
            char_count=context.char_count if context else 0,
            conversation_version=context.version if context else 1,
            agent_version=agent.version if agent is not None else 1,
            started_at=row.started_at,
            updated_at=row.updated_at,
            error_message=row.error_message,
        )
