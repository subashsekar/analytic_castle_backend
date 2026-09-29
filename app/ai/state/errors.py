"""Agent state error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class AgentStateErrorCode(str, Enum):
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_NOT_ACTIVE = "SESSION_NOT_ACTIVE"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"
    MESSAGE_LIMIT_EXCEEDED = "MESSAGE_LIMIT_EXCEEDED"
    SERIALIZATION_ERROR = "SERIALIZATION_ERROR"
    DATA_SOURCE_NOT_FOUND = "DATA_SOURCE_NOT_FOUND"
    VERSION_CONFLICT = "VERSION_CONFLICT"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class AgentStateError(Exception):
    """Base error for the agent state layer."""

    code: AgentStateErrorCode = AgentStateErrorCode.INVALID_TRANSITION

    def __init__(
        self,
        message: str = "Agent state operation failed",
        *,
        code: AgentStateErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class AnalysisSessionNotFoundError(AgentStateError):
    code = AgentStateErrorCode.SESSION_NOT_FOUND

    def __init__(self, message: str = "Analysis session not found") -> None:
        super().__init__(message, code=self.code)


class AnalysisSessionNotActiveError(AgentStateError):
    code = AgentStateErrorCode.SESSION_NOT_ACTIVE

    def __init__(self, message: str = "Analysis session is not active") -> None:
        super().__init__(message, code=self.code)


class InvalidStateTransitionError(AgentStateError):
    code = AgentStateErrorCode.INVALID_TRANSITION

    def __init__(self, message: str = "Invalid state transition") -> None:
        super().__init__(message, code=self.code)


class ContextLimitExceededError(AgentStateError):
    code = AgentStateErrorCode.CONTEXT_LIMIT_EXCEEDED

    def __init__(self, message: str = "Conversation context limit exceeded") -> None:
        super().__init__(message, code=self.code)


class MessageLimitExceededError(AgentStateError):
    code = AgentStateErrorCode.MESSAGE_LIMIT_EXCEEDED

    def __init__(self, message: str = "Message size limit exceeded") -> None:
        super().__init__(message, code=self.code)


class StateSerializationError(AgentStateError):
    code = AgentStateErrorCode.SERIALIZATION_ERROR

    def __init__(self, message: str = "State serialization failed") -> None:
        super().__init__(message, code=self.code)


class AgentStateDataSourceNotFoundError(AgentStateError):
    code = AgentStateErrorCode.DATA_SOURCE_NOT_FOUND

    def __init__(self, message: str = "Data source not found") -> None:
        super().__init__(message, code=self.code)


class StateVersionConflictError(AgentStateError):
    code = AgentStateErrorCode.VERSION_CONFLICT

    def __init__(self, message: str = "State version conflict") -> None:
        super().__init__(message, code=self.code)
