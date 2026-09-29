"""SQL correction error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class SQLCorrectionErrorCode(str, Enum):
    SQL_CORRECTION_CONFIGURATION_ERROR = "SQL_CORRECTION_CONFIGURATION_ERROR"
    SQL_CORRECTION_AUTHORIZATION_ERROR = "SQL_CORRECTION_AUTHORIZATION_ERROR"
    SQL_CORRECTION_SCHEMA_ERROR = "SQL_CORRECTION_SCHEMA_ERROR"
    SQL_CORRECTION_VALIDATION_ERROR = "SQL_CORRECTION_VALIDATION_ERROR"
    SQL_CORRECTION_LLM_ERROR = "SQL_CORRECTION_LLM_ERROR"
    SQL_CORRECTION_INTERNAL_ERROR = "SQL_CORRECTION_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class SQLCorrectionError(Exception):
    """Base error for the SQL correction layer."""

    code: SQLCorrectionErrorCode = SQLCorrectionErrorCode.SQL_CORRECTION_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "SQL correction failed",
        *,
        code: SQLCorrectionErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class SQLCorrectionConfigurationError(SQLCorrectionError):
    code = SQLCorrectionErrorCode.SQL_CORRECTION_CONFIGURATION_ERROR

    def __init__(self, message: str = "SQL correction is not configured") -> None:
        super().__init__(message, code=self.code)


class SQLCorrectionAuthorizationError(SQLCorrectionError):
    code = SQLCorrectionErrorCode.SQL_CORRECTION_AUTHORIZATION_ERROR

    def __init__(self, message: str = "SQL correction authorization failed") -> None:
        super().__init__(message, code=self.code)


class SQLCorrectionSchemaError(SQLCorrectionError):
    code = SQLCorrectionErrorCode.SQL_CORRECTION_SCHEMA_ERROR

    def __init__(
        self, message: str = "Schema context is missing or insufficient"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLCorrectionValidationError(SQLCorrectionError):
    code = SQLCorrectionErrorCode.SQL_CORRECTION_VALIDATION_ERROR

    def __init__(
        self, message: str = "SQL correction output validation failed"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLCorrectionLLMError(SQLCorrectionError):
    code = SQLCorrectionErrorCode.SQL_CORRECTION_LLM_ERROR

    def __init__(self, message: str = "SQL correction LLM request failed") -> None:
        super().__init__(message, code=self.code)
