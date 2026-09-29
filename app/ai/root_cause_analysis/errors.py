"""Root cause analysis agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class RootCauseAnalysisErrorCode(str, Enum):
    ROOT_CAUSE_ANALYSIS_CONFIGURATION_ERROR = "ROOT_CAUSE_ANALYSIS_CONFIGURATION_ERROR"
    ROOT_CAUSE_ANALYSIS_AUTHORIZATION_ERROR = "ROOT_CAUSE_ANALYSIS_AUTHORIZATION_ERROR"
    ROOT_CAUSE_ANALYSIS_VALIDATION_ERROR = "ROOT_CAUSE_ANALYSIS_VALIDATION_ERROR"
    ROOT_CAUSE_ANALYSIS_LLM_ERROR = "ROOT_CAUSE_ANALYSIS_LLM_ERROR"
    ROOT_CAUSE_ANALYSIS_INTERNAL_ERROR = "ROOT_CAUSE_ANALYSIS_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class RootCauseAnalysisError(Exception):
    """Base error for the root cause analysis agent."""

    code: RootCauseAnalysisErrorCode = (
        RootCauseAnalysisErrorCode.ROOT_CAUSE_ANALYSIS_INTERNAL_ERROR
    )

    def __init__(
        self,
        message: str = "Root cause analysis request failed",
        *,
        code: RootCauseAnalysisErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class RootCauseAnalysisConfigurationError(RootCauseAnalysisError):
    code = RootCauseAnalysisErrorCode.ROOT_CAUSE_ANALYSIS_CONFIGURATION_ERROR

    def __init__(self, message: str = "Root cause analysis is not configured") -> None:
        super().__init__(message, code=self.code)


class RootCauseAnalysisAuthorizationError(RootCauseAnalysisError):
    code = RootCauseAnalysisErrorCode.ROOT_CAUSE_ANALYSIS_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Root cause analysis authorization failed") -> None:
        super().__init__(message, code=self.code)


class RootCauseAnalysisValidationError(RootCauseAnalysisError):
    code = RootCauseAnalysisErrorCode.ROOT_CAUSE_ANALYSIS_VALIDATION_ERROR

    def __init__(
        self, message: str = "Root cause analysis output validation failed"
    ) -> None:
        super().__init__(message, code=self.code)


class RootCauseAnalysisLLMError(RootCauseAnalysisError):
    code = RootCauseAnalysisErrorCode.ROOT_CAUSE_ANALYSIS_LLM_ERROR

    def __init__(self, message: str = "Root cause analysis LLM request failed") -> None:
        super().__init__(message, code=self.code)
