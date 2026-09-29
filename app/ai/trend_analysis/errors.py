"""Trend analysis agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class TrendAnalysisErrorCode(str, Enum):
    TREND_ANALYSIS_CONFIGURATION_ERROR = "TREND_ANALYSIS_CONFIGURATION_ERROR"
    TREND_ANALYSIS_AUTHORIZATION_ERROR = "TREND_ANALYSIS_AUTHORIZATION_ERROR"
    TREND_ANALYSIS_VALIDATION_ERROR = "TREND_ANALYSIS_VALIDATION_ERROR"
    TREND_ANALYSIS_LLM_ERROR = "TREND_ANALYSIS_LLM_ERROR"
    TREND_ANALYSIS_INTERNAL_ERROR = "TREND_ANALYSIS_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class TrendAnalysisError(Exception):
    """Base error for the trend analysis agent."""

    code: TrendAnalysisErrorCode = TrendAnalysisErrorCode.TREND_ANALYSIS_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Trend analysis request failed",
        *,
        code: TrendAnalysisErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class TrendAnalysisConfigurationError(TrendAnalysisError):
    code = TrendAnalysisErrorCode.TREND_ANALYSIS_CONFIGURATION_ERROR

    def __init__(self, message: str = "Trend analysis is not configured") -> None:
        super().__init__(message, code=self.code)


class TrendAnalysisAuthorizationError(TrendAnalysisError):
    code = TrendAnalysisErrorCode.TREND_ANALYSIS_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Trend analysis authorization failed") -> None:
        super().__init__(message, code=self.code)


class TrendAnalysisValidationError(TrendAnalysisError):
    code = TrendAnalysisErrorCode.TREND_ANALYSIS_VALIDATION_ERROR

    def __init__(self, message: str = "Trend analysis output validation failed") -> None:
        super().__init__(message, code=self.code)


class TrendAnalysisLLMError(TrendAnalysisError):
    code = TrendAnalysisErrorCode.TREND_ANALYSIS_LLM_ERROR

    def __init__(self, message: str = "Trend analysis LLM request failed") -> None:
        super().__init__(message, code=self.code)
