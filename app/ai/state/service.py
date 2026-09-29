"""Persisted agent state and analysis session lifecycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.ai.state.errors import (
    AgentStateDataSourceNotFoundError,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    StateVersionConflictError,
)
from app.ai.state.limits import (
    ContextLimits,
    trim_context_for_storage,
    validate_context_size,
    validate_message_size,
)
from app.ai.state.logging_helpers import (
    agent_state_log_context,
    conversation_log_context,
    session_log_context,
    transition_log_context,
)
from app.ai.state.models import (
    AnalysisSessionSnapshot,
    ConversationContextData,
    ConversationMessage,
    StateTransition,
)
from app.ai.state.serialization import deserialize_context, serialize_context
from app.ai.state.transitions import (
    apply_state_transition,
    initial_agent_state_snapshot,
    payload_to_storage,
    snapshot_from_row,
)
from app.db.models import (
    AgentState,
    AnalysisSession,
    ConversationContext,
    DataSource,
    Workspace,
)
from app.enums import AgentPhase, AnalysisSessionStatus

logger = logging.getLogger(__name__)

_SAFE_FAILURE_MESSAGE = "Analysis session failed"


@dataclass(frozen=True)
class CreateSessionParams:
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None = None
    initial_message: ConversationMessage | None = None


class AgentStateService:
    """Manage analysis sessions, agent state, and conversation context.

    Persistence contract: methods flush changes to the SQLAlchemy session but
    never commit. Callers (typically API route handlers) must call
    ``session.commit()`` after successful operations. ``get_db()`` rolls back on
    errors.

    Mutating methods require optimistic-lock versions. API routes should pass the
    latest ``agent_state.version`` or ``conversation.version`` from the client.
    """

    def __init__(
        self,
        session: Session,
        *,
        limits: ContextLimits | None = None,
    ) -> None:
        self._session = session
        self._limits = limits or ContextLimits.from_settings()

    def create_session(self, params: CreateSessionParams) -> AnalysisSessionSnapshot:
        self._ensure_workspace_organization(
            params.workspace_id,
            params.organization_id,
        )
        if params.data_source_id is not None:
            self._load_workspace_data_source(params.data_source_id, params.workspace_id)

        analysis_session = AnalysisSession(
            user_id=params.user_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
            data_source_id=params.data_source_id,
            status=AnalysisSessionStatus.ACTIVE,
            started_at=datetime.now(UTC),
        )
        self._session.add(analysis_session)
        self._session.flush()

        initial_state = initial_agent_state_snapshot()
        agent_state = AgentState(
            session_id=analysis_session.id,
            phase=initial_state.phase,
            payload=payload_to_storage(initial_state.payload),
            version=initial_state.version,
        )
        context_data = ConversationContextData()
        if params.initial_message is not None:
            validate_message_size(params.initial_message, limits=self._limits)
            context_data = ConversationContextData(messages=[params.initial_message])
            validate_context_size(context_data, limits=self._limits)

        conversation_context = ConversationContext(
            session_id=analysis_session.id,
            messages=serialize_context(context_data),
            char_count=context_data.char_count,
            message_count=context_data.message_count,
            version=1,
        )
        self._session.add_all([agent_state, conversation_context])
        self._session.flush()

        logger.info(
            "analysis session created",
            extra=session_log_context(
                session_id=analysis_session.id,
                workspace_id=analysis_session.workspace_id,
                user_id=analysis_session.user_id,
                status=analysis_session.status,
            ),
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def get_session(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
    ) -> AnalysisSessionSnapshot:
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def update_agent_state(
        self,
        session_id: UUID,
        transition: StateTransition,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_version: int,
    ) -> AnalysisSessionSnapshot:
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            for_update=True,
        )
        self._ensure_active(analysis_session)

        current = snapshot_from_row(
            phase=agent_state.phase,
            payload=agent_state.payload,
            version=agent_state.version,
        )
        if current.version != expected_version:
            raise StateVersionConflictError("Agent state version mismatch")

        updated = apply_state_transition(current, transition)
        result = self._session.execute(
            update(AgentState)
            .where(
                AgentState.id == agent_state.id,
                AgentState.version == expected_version,
            )
            .values(
                phase=updated.phase,
                payload=payload_to_storage(updated.payload),
                version=updated.version,
            )
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise StateVersionConflictError("Agent state version mismatch")
        self._session.refresh(agent_state)
        self._session.flush()

        logger.info(
            "agent state updated",
            extra={
                **session_log_context(
                    session_id=analysis_session.id,
                    workspace_id=analysis_session.workspace_id,
                    user_id=analysis_session.user_id,
                    status=analysis_session.status,
                ),
                **transition_log_context(
                    transition_kind=transition.kind,
                    phase=updated.phase if transition.kind == "set_phase" else None,
                ),
                **agent_state_log_context(updated),
            },
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def append_message(
        self,
        session_id: UUID,
        message: ConversationMessage,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_context_version: int,
        require_active: bool = True,
    ) -> AnalysisSessionSnapshot:
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            for_update=True,
        )
        if require_active:
            self._ensure_active(analysis_session)

        if conversation_context.version != expected_context_version:
            raise StateVersionConflictError("Conversation context version mismatch")

        validate_message_size(message, limits=self._limits)
        context_data = deserialize_context(conversation_context.messages)
        updated_context = ConversationContextData(
            messages=[*context_data.messages, message]
        )
        validate_context_size(updated_context, limits=self._limits)

        result = self._session.execute(
            update(ConversationContext)
            .where(
                ConversationContext.id == conversation_context.id,
                ConversationContext.version == expected_context_version,
            )
            .values(
                messages=serialize_context(updated_context),
                char_count=updated_context.char_count,
                message_count=updated_context.message_count,
                version=expected_context_version + 1,
            )
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise StateVersionConflictError("Conversation context version mismatch")
        self._session.refresh(conversation_context)
        self._session.flush()

        logger.info(
            "conversation message appended",
            extra={
                **session_log_context(
                    session_id=analysis_session.id,
                    workspace_id=analysis_session.workspace_id,
                    user_id=analysis_session.user_id,
                    status=analysis_session.status,
                ),
                **conversation_log_context(updated_context),
            },
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def append_message_with_trim(
        self,
        session_id: UUID,
        message: ConversationMessage,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_context_version: int,
        require_active: bool = True,
    ) -> AnalysisSessionSnapshot:
        """Append a message, dropping oldest turns when storage limits would be exceeded."""
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            for_update=True,
        )
        if require_active:
            self._ensure_active(analysis_session)

        if conversation_context.version != expected_context_version:
            raise StateVersionConflictError("Conversation context version mismatch")

        validate_message_size(message, limits=self._limits)
        context_data = deserialize_context(conversation_context.messages)
        updated_context = trim_context_for_storage(
            context_data,
            message,
            limits=self._limits,
        )
        validate_context_size(updated_context, limits=self._limits)

        result = self._session.execute(
            update(ConversationContext)
            .where(
                ConversationContext.id == conversation_context.id,
                ConversationContext.version == expected_context_version,
            )
            .values(
                messages=serialize_context(updated_context),
                char_count=updated_context.char_count,
                message_count=updated_context.message_count,
                version=expected_context_version + 1,
            )
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise StateVersionConflictError("Conversation context version mismatch")
        self._session.refresh(conversation_context)
        self._session.flush()

        logger.info(
            "conversation message appended with trim",
            extra={
                **session_log_context(
                    session_id=analysis_session.id,
                    workspace_id=analysis_session.workspace_id,
                    user_id=analysis_session.user_id,
                    status=analysis_session.status,
                ),
                **conversation_log_context(updated_context),
            },
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def complete_session(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_agent_version: int,
        final_phase: AgentPhase = AgentPhase.COMPLETED,
    ) -> AnalysisSessionSnapshot:
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            for_update=True,
        )
        self._ensure_active(analysis_session)

        if agent_state.version != expected_agent_version:
            raise StateVersionConflictError("Agent state version mismatch")
        if final_phase != AgentPhase.COMPLETED:
            raise AnalysisSessionNotActiveError("Invalid final phase for completion")

        analysis_session.status = AnalysisSessionStatus.COMPLETED
        analysis_session.completed_at = datetime.now(UTC)
        analysis_session.error_message = None
        result = self._session.execute(
            update(AgentState)
            .where(
                AgentState.id == agent_state.id,
                AgentState.version == expected_agent_version,
            )
            .values(phase=final_phase, version=expected_agent_version + 1)
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise StateVersionConflictError("Agent state version mismatch")
        self._session.refresh(agent_state)
        self._session.flush()

        logger.info(
            "analysis session completed",
            extra=session_log_context(
                session_id=analysis_session.id,
                workspace_id=analysis_session.workspace_id,
                user_id=analysis_session.user_id,
                status=analysis_session.status,
            ),
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def fail_session(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        expected_agent_version: int,
        error_message: str | None = None,
    ) -> AnalysisSessionSnapshot:
        analysis_session, agent_state, conversation_context = self._load_session_bundle(
            session_id,
            workspace_id=workspace_id,
            user_id=user_id,
            for_update=True,
        )
        if analysis_session.status != AnalysisSessionStatus.ACTIVE:
            raise AnalysisSessionNotActiveError("Analysis session is not active")

        if agent_state.version != expected_agent_version:
            raise StateVersionConflictError("Agent state version mismatch")

        analysis_session.status = AnalysisSessionStatus.FAILED
        analysis_session.completed_at = datetime.now(UTC)
        analysis_session.error_message = _SAFE_FAILURE_MESSAGE
        result = self._session.execute(
            update(AgentState)
            .where(
                AgentState.id == agent_state.id,
                AgentState.version == expected_agent_version,
            )
            .values(phase=AgentPhase.FAILED, version=expected_agent_version + 1)
        )
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            raise StateVersionConflictError("Agent state version mismatch")
        self._session.refresh(agent_state)
        self._session.flush()

        logger.warning(
            "analysis session failed",
            extra={
                **session_log_context(
                    session_id=analysis_session.id,
                    workspace_id=analysis_session.workspace_id,
                    user_id=analysis_session.user_id,
                    status=analysis_session.status,
                ),
                "has_error_detail": error_message is not None,
            },
        )
        return self._to_snapshot(analysis_session, agent_state, conversation_context)

    def _load_session_bundle(
        self,
        session_id: UUID,
        *,
        workspace_id: UUID,
        user_id: UUID,
        for_update: bool = False,
    ) -> tuple[AnalysisSession, AgentState, ConversationContext]:
        self._ensure_workspace_organization(workspace_id, organization_id=None)

        session_stmt = select(AnalysisSession).where(AnalysisSession.id == session_id)
        if for_update:
            session_stmt = session_stmt.with_for_update()
        analysis_session = self._session.scalars(session_stmt).first()
        if (
            analysis_session is None
            or analysis_session.workspace_id != workspace_id
            or analysis_session.user_id != user_id
        ):
            raise AnalysisSessionNotFoundError("Analysis session not found")

        workspace = self._session.get(Workspace, workspace_id)
        if (
            workspace is None
            or workspace.organization_id != analysis_session.organization_id
        ):
            raise AnalysisSessionNotFoundError("Analysis session not found")

        agent_stmt = select(AgentState).where(AgentState.session_id == session_id)
        context_stmt = select(ConversationContext).where(
            ConversationContext.session_id == session_id
        )
        if for_update:
            agent_stmt = agent_stmt.with_for_update()
            context_stmt = context_stmt.with_for_update()
        agent_state = self._session.scalars(agent_stmt).first()
        conversation_context = self._session.scalars(context_stmt).first()
        if agent_state is None or conversation_context is None:
            raise AnalysisSessionNotFoundError("Analysis session not found")
        return analysis_session, agent_state, conversation_context

    def _ensure_workspace_organization(
        self,
        workspace_id: UUID,
        organization_id: UUID | None,
    ) -> Workspace:
        workspace = self._session.get(Workspace, workspace_id)
        if workspace is None:
            raise AnalysisSessionNotFoundError("Analysis session not found")
        if organization_id is not None and workspace.organization_id != organization_id:
            raise AnalysisSessionNotFoundError("Analysis session not found")
        return workspace

    def _load_workspace_data_source(
        self,
        data_source_id: UUID,
        workspace_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise AgentStateDataSourceNotFoundError("Data source not found")
        return data_source

    @staticmethod
    def _ensure_active(analysis_session: AnalysisSession) -> None:
        if analysis_session.status != AnalysisSessionStatus.ACTIVE:
            raise AnalysisSessionNotActiveError("Analysis session is not active")

    @staticmethod
    def _to_snapshot(
        analysis_session: AnalysisSession,
        agent_state: AgentState,
        conversation_context: ConversationContext,
    ) -> AnalysisSessionSnapshot:
        state_snapshot = snapshot_from_row(
            phase=agent_state.phase,
            payload=agent_state.payload,
            version=agent_state.version,
        )
        context_data = deserialize_context(conversation_context.messages)
        return AnalysisSessionSnapshot(
            session_id=analysis_session.id,
            user_id=analysis_session.user_id,
            workspace_id=analysis_session.workspace_id,
            organization_id=analysis_session.organization_id,
            data_source_id=analysis_session.data_source_id,
            status=analysis_session.status.value,
            agent_state=state_snapshot,
            conversation=context_data,
            conversation_version=conversation_context.version,
            error_message=analysis_session.error_message,
        )
