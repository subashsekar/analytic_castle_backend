"""Safe observability helpers that never emit conversation content."""

from __future__ import annotations

from uuid import UUID

from app.ai.state.models import AgentStateSnapshot, ConversationContextData
from app.enums import AgentPhase, AnalysisSessionStatus


def session_log_context(
    *,
    session_id: UUID,
    workspace_id: UUID,
    user_id: UUID,
    status: AnalysisSessionStatus,
) -> dict[str, object]:
    return {
        "analysis_session_id": session_id,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "analysis_session_status": status.value,
    }


def agent_state_log_context(snapshot: AgentStateSnapshot) -> dict[str, object]:
    return {
        "agent_phase": snapshot.phase.value,
        "agent_state_version": snapshot.version,
        "has_intent": snapshot.payload.intent is not None,
        "metadata_ref_count": len(snapshot.payload.metadata_refs),
    }


def conversation_log_context(context: ConversationContextData) -> dict[str, object]:
    return {
        "conversation_message_count": context.message_count,
        "conversation_char_count": context.char_count,
    }


def transition_log_context(
    *, transition_kind: str, phase: AgentPhase | None = None
) -> dict[str, object]:
    context: dict[str, object] = {"state_transition_kind": transition_kind}
    if phase is not None:
        context["agent_phase"] = phase.value
    return context
