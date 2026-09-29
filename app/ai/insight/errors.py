"""Insight agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class InsightErrorCode(str, Enum):
    INSIGHT_CONFIGURATION_ERROR = "INSIGHT_CONFIGURATION_ERROR"
    INSIGHT_AUTHORIZATION_ERROR = "INSIGHT_AUTHORIZATION_ERROR"
    INSIGHT_VALIDATION_ERROR = "INSIGHT_VALIDATION_ERROR"
    INSIGHT_LLM_ERROR = "INSIGHT_LLM_ERROR"
    INSIGHT_INTERNAL_ERROR = "INSIGHT_INTERNAL_ERROR"


class InsightError(Exception):
    """Base error for the insight agent."""

    code: InsightErrorCode = InsightErrorCode.INSIGHT_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Insight request failed",
        *,
        code: InsightErrorCode | None = None,
    ) -> None:
        super().__init__(redact_secret(message))
        if code is not None:
            self.code = code


class InsightConfigurationError(InsightError):
    code = InsightErrorCode.INSIGHT_CONFIGURATION_ERROR

    def __init__(self, message: str = "Insight generation is not configured") -> None:
        super().__init__(message, code=self.code)


class InsightAuthorizationError(InsightError):
    code = InsightErrorCode.INSIGHT_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Insight authorization failed") -> None:
        super().__init__(message, code=self.code)


class InsightValidationError(InsightError):
    code = InsightErrorCode.INSIGHT_VALIDATION_ERROR

    def __init__(self, message: str = "Insight output validation failed") -> None:
        super().__init__(message, code=self.code)


class InsightLLMError(InsightError):
    code = InsightErrorCode.INSIGHT_LLM_ERROR

    def __init__(self, message: str = "Insight LLM request failed") -> None:
        super().__init__(message, code=self.code)
