"""Supervisor agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class SupervisorErrorCode(str, Enum):
    SUPERVISOR_CONFIGURATION_ERROR = "SUPERVISOR_CONFIGURATION_ERROR"
    SUPERVISOR_AUTHORIZATION_ERROR = "SUPERVISOR_AUTHORIZATION_ERROR"
    SUPERVISOR_CLASSIFICATION_ERROR = "SUPERVISOR_CLASSIFICATION_ERROR"
    SUPERVISOR_ROUTING_ERROR = "SUPERVISOR_ROUTING_ERROR"
    SUPERVISOR_LLM_ERROR = "SUPERVISOR_LLM_ERROR"
    SUPERVISOR_INTERNAL_ERROR = "SUPERVISOR_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class SupervisorError(Exception):
    """Base error for the supervisor agent."""

    code: SupervisorErrorCode = SupervisorErrorCode.SUPERVISOR_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Supervisor request failed",
        *,
        code: SupervisorErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class SupervisorConfigurationError(SupervisorError):
    code = SupervisorErrorCode.SUPERVISOR_CONFIGURATION_ERROR

    def __init__(self, message: str = "Supervisor is not configured") -> None:
        super().__init__(message, code=self.code)


class SupervisorAuthorizationError(SupervisorError):
    code = SupervisorErrorCode.SUPERVISOR_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Supervisor authorization failed") -> None:
        super().__init__(message, code=self.code)


class SupervisorClassificationError(SupervisorError):
    code = SupervisorErrorCode.SUPERVISOR_CLASSIFICATION_ERROR

    def __init__(self, message: str = "Request classification failed") -> None:
        super().__init__(message, code=self.code)


class InvalidRoutingError(SupervisorError):
    code = SupervisorErrorCode.SUPERVISOR_ROUTING_ERROR

    def __init__(self, message: str = "Invalid supervisor routing decision") -> None:
        super().__init__(message, code=self.code)


class SupervisorLLMError(SupervisorError):
    code = SupervisorErrorCode.SUPERVISOR_LLM_ERROR

    def __init__(self, message: str = "Supervisor LLM request failed") -> None:
        super().__init__(message, code=self.code)
