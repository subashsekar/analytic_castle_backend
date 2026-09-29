"""Data analyst agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class DataAnalystErrorCode(str, Enum):
    DATA_ANALYST_CONFIGURATION_ERROR = "DATA_ANALYST_CONFIGURATION_ERROR"
    DATA_ANALYST_AUTHORIZATION_ERROR = "DATA_ANALYST_AUTHORIZATION_ERROR"
    DATA_ANALYST_VALIDATION_ERROR = "DATA_ANALYST_VALIDATION_ERROR"
    DATA_ANALYST_LLM_ERROR = "DATA_ANALYST_LLM_ERROR"
    DATA_ANALYST_INTERNAL_ERROR = "DATA_ANALYST_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class DataAnalystError(Exception):
    """Base error for the data analyst agent."""

    code: DataAnalystErrorCode = DataAnalystErrorCode.DATA_ANALYST_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Data analyst request failed",
        *,
        code: DataAnalystErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class DataAnalystConfigurationError(DataAnalystError):
    code = DataAnalystErrorCode.DATA_ANALYST_CONFIGURATION_ERROR

    def __init__(self, message: str = "Data analyst is not configured") -> None:
        super().__init__(message, code=self.code)


class DataAnalystAuthorizationError(DataAnalystError):
    code = DataAnalystErrorCode.DATA_ANALYST_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Data analyst authorization failed") -> None:
        super().__init__(message, code=self.code)


class DataAnalystValidationError(DataAnalystError):
    code = DataAnalystErrorCode.DATA_ANALYST_VALIDATION_ERROR

    def __init__(self, message: str = "Data analyst output validation failed") -> None:
        super().__init__(message, code=self.code)


class DataAnalystLLMError(DataAnalystError):
    code = DataAnalystErrorCode.DATA_ANALYST_LLM_ERROR

    def __init__(self, message: str = "Data analyst LLM request failed") -> None:
        super().__init__(message, code=self.code)
