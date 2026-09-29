"""Conversation memory layer for the AI analyst."""

from app.ai.memory.context import (
    PlaceholderSummarizationHook,
    SummarizationHook,
    build_llm_context,
    select_messages_for_llm,
)
from app.ai.memory.errors import (
    AgentStateDataSourceNotFoundError,
    AgentStateError,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    ContextLimitExceededError,
    ConversationMemoryError,
    ConversationMemoryErrorCode,
    ConversationNotActiveError,
    ConversationNotFoundError,
    InvalidPaginationError,
    MessageLimitExceededError,
    StateVersionConflictError,
)
from app.ai.memory.models import (
    ConversationRecord,
    LLMContextBuildResult,
    LLMContextSelection,
    MessageRecord,
    Page,
)
from app.ai.memory.pagination import normalize_page_params, paginate_items
from app.ai.memory.service import (
    ConversationMemoryService,
    CreateConversationParams,
    UpdateConversationParams,
)
from app.ai.state.limits import ContextLimits, trim_context_for_storage

__all__ = [
    "AgentStateDataSourceNotFoundError",
    "AgentStateError",
    "AnalysisSessionNotActiveError",
    "AnalysisSessionNotFoundError",
    "ContextLimitExceededError",
    "ContextLimits",
    "ConversationMemoryError",
    "ConversationMemoryErrorCode",
    "ConversationMemoryService",
    "ConversationNotActiveError",
    "ConversationNotFoundError",
    "ConversationRecord",
    "CreateConversationParams",
    "InvalidPaginationError",
    "LLMContextBuildResult",
    "LLMContextSelection",
    "MessageLimitExceededError",
    "MessageRecord",
    "Page",
    "PlaceholderSummarizationHook",
    "StateVersionConflictError",
    "SummarizationHook",
    "UpdateConversationParams",
    "build_llm_context",
    "normalize_page_params",
    "paginate_items",
    "select_messages_for_llm",
    "trim_context_for_storage",
]
