"""Reusable agent state infrastructure for future AI agents."""

from app.ai.state.errors import (
    AgentStateDataSourceNotFoundError,
    AgentStateError,
    AgentStateErrorCode,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    ContextLimitExceededError,
    InvalidStateTransitionError,
    MessageLimitExceededError,
    StateSerializationError,
    StateVersionConflictError,
)
from app.ai.state.limits import ContextLimits, trim_context_for_storage
from app.ai.state.logging_helpers import (
    agent_state_log_context,
    conversation_log_context,
    session_log_context,
    transition_log_context,
)
from app.ai.state.messages import (
    conversation_message_to_llm,
    conversation_to_llm_messages,
    llm_message_to_conversation,
)
from app.ai.state.models import (
    AgentStatePayload,
    AgentStateSnapshot,
    AnalysisSessionSnapshot,
    ConversationContextData,
    ConversationMessage,
    MergePayloadTransition,
    ReplacePayloadTransition,
    SetPhaseTransition,
    StateTransition,
)
from app.ai.state.serialization import (
    deserialize_context,
    deserialize_payload,
    serialize_context,
    serialize_payload,
)
from app.ai.state.service import AgentStateService, CreateSessionParams
from app.ai.state.transitions import (
    ALLOWED_PHASE_TRANSITIONS,
    RESUMABLE_MESSAGE_PHASES,
    TERMINAL_PHASES,
    apply_state_transition,
    initial_agent_state_snapshot,
)

__all__ = [
    "ALLOWED_PHASE_TRANSITIONS",
    "RESUMABLE_MESSAGE_PHASES",
    "TERMINAL_PHASES",
    "AgentStateDataSourceNotFoundError",
    "AgentStateError",
    "AgentStateErrorCode",
    "AgentStatePayload",
    "AgentStateService",
    "AgentStateSnapshot",
    "AnalysisSessionNotActiveError",
    "AnalysisSessionNotFoundError",
    "AnalysisSessionSnapshot",
    "ContextLimitExceededError",
    "ContextLimits",
    "ConversationContextData",
    "ConversationMessage",
    "CreateSessionParams",
    "InvalidStateTransitionError",
    "MergePayloadTransition",
    "MessageLimitExceededError",
    "ReplacePayloadTransition",
    "SetPhaseTransition",
    "StateSerializationError",
    "StateTransition",
    "StateVersionConflictError",
    "agent_state_log_context",
    "apply_state_transition",
    "conversation_log_context",
    "conversation_message_to_llm",
    "conversation_to_llm_messages",
    "deserialize_context",
    "deserialize_payload",
    "initial_agent_state_snapshot",
    "llm_message_to_conversation",
    "serialize_context",
    "serialize_payload",
    "session_log_context",
    "transition_log_context",
    "trim_context_for_storage",
]

# Re-export for callers that need LLM-specific limits without importing limits twice.
llm_context_limits = ContextLimits.for_llm
