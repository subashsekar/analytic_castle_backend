"""Planner agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class PlannerErrorCode(str, Enum):
    PLANNER_CONFIGURATION_ERROR = "PLANNER_CONFIGURATION_ERROR"
    PLANNER_AUTHORIZATION_ERROR = "PLANNER_AUTHORIZATION_ERROR"
    PLANNER_VALIDATION_ERROR = "PLANNER_VALIDATION_ERROR"
    PLANNER_LLM_ERROR = "PLANNER_LLM_ERROR"
    PLANNER_INTERNAL_ERROR = "PLANNER_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class PlannerError(Exception):
    """Base error for the planner agent."""

    code: PlannerErrorCode = PlannerErrorCode.PLANNER_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Planner request failed",
        *,
        code: PlannerErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class PlannerConfigurationError(PlannerError):
    code = PlannerErrorCode.PLANNER_CONFIGURATION_ERROR

    def __init__(self, message: str = "Planner is not configured") -> None:
        super().__init__(message, code=self.code)


class PlannerAuthorizationError(PlannerError):
    code = PlannerErrorCode.PLANNER_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Planner authorization failed") -> None:
        super().__init__(message, code=self.code)


class PlannerValidationError(PlannerError):
    code = PlannerErrorCode.PLANNER_VALIDATION_ERROR

    def __init__(self, message: str = "Planner output validation failed") -> None:
        super().__init__(message, code=self.code)


class PlannerLLMError(PlannerError):
    code = PlannerErrorCode.PLANNER_LLM_ERROR

    def __init__(self, message: str = "Planner LLM request failed") -> None:
        super().__init__(message, code=self.code)
