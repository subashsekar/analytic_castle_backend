"""Conversation memory errors. Reuses agent-state codes where applicable."""

from __future__ import annotations

from enum import Enum

from app.ai.state.errors import (
    AgentStateDataSourceNotFoundError,
    AgentStateError,
    AnalysisSessionNotActiveError,
    AnalysisSessionNotFoundError,
    ContextLimitExceededError,
    MessageLimitExceededError,
    StateVersionConflictError,
)
from app.core.logging import redact_secret


class ConversationMemoryErrorCode(str, Enum):
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    CONVERSATION_NOT_ACTIVE = "CONVERSATION_NOT_ACTIVE"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    MESSAGE_LIMIT_EXCEEDED = "MESSAGE_LIMIT_EXCEEDED"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    DATA_SOURCE_NOT_FOUND = "DATA_SOURCE_NOT_FOUND"
    INVALID_PAGINATION = "INVALID_PAGINATION"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class ConversationMemoryError(Exception):
    code: ConversationMemoryErrorCode = ConversationMemoryErrorCode.INVALID_PAGINATION

    def __init__(
        self,
        message: str = "Conversation memory operation failed",
        *,
        code: ConversationMemoryErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class ConversationNotFoundError(ConversationMemoryError):
    code = ConversationMemoryErrorCode.CONVERSATION_NOT_FOUND

    def __init__(self, message: str = "Conversation not found") -> None:
        super().__init__(message, code=self.code)


class ConversationNotActiveError(ConversationMemoryError):
    code = ConversationMemoryErrorCode.CONVERSATION_NOT_ACTIVE

    def __init__(self, message: str = "Conversation is not active") -> None:
        super().__init__(message, code=self.code)


class InvalidPaginationError(ConversationMemoryError):
    code = ConversationMemoryErrorCode.INVALID_PAGINATION

    def __init__(self, message: str = "Invalid pagination parameters") -> None:
        super().__init__(message, code=self.code)


__all__ = [
    "AgentStateDataSourceNotFoundError",
    "AgentStateError",
    "AnalysisSessionNotActiveError",
    "AnalysisSessionNotFoundError",
    "ContextLimitExceededError",
    "ConversationMemoryError",
    "ConversationMemoryErrorCode",
    "ConversationNotActiveError",
    "ConversationNotFoundError",
    "InvalidPaginationError",
    "MessageLimitExceededError",
    "StateVersionConflictError",
]
