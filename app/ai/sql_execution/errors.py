"""SQL execution error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.ai.sql_execution.models import SQLExecutionStatus
from app.ai.sql_validation.models import SQLValidationViolation
from app.core.logging import redact_secret


class SQLExecutionErrorCode(str, Enum):
    SQL_EXECUTION_CONFIGURATION_ERROR = "SQL_EXECUTION_CONFIGURATION_ERROR"
    SQL_EXECUTION_AUTHORIZATION_ERROR = "SQL_EXECUTION_AUTHORIZATION_ERROR"
    SQL_EXECUTION_VALIDATION_ERROR = "SQL_EXECUTION_VALIDATION_ERROR"
    SQL_EXECUTION_TIMEOUT_ERROR = "SQL_EXECUTION_TIMEOUT_ERROR"
    SQL_EXECUTION_RESULT_LIMIT_ERROR = "SQL_EXECUTION_RESULT_LIMIT_ERROR"
    SQL_EXECUTION_DATABASE_ERROR = "SQL_EXECUTION_DATABASE_ERROR"
    SQL_EXECUTION_REJECTED_ERROR = "SQL_EXECUTION_REJECTED_ERROR"
    SQL_EXECUTION_CANCELLED_ERROR = "SQL_EXECUTION_CANCELLED_ERROR"
    SQL_EXECUTION_INTERNAL_ERROR = "SQL_EXECUTION_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class SQLExecutionError(Exception):
    """Base error for the SQL execution layer."""

    code: SQLExecutionErrorCode = SQLExecutionErrorCode.SQL_EXECUTION_INTERNAL_ERROR
    status: SQLExecutionStatus = SQLExecutionStatus.FAILED

    def __init__(
        self,
        message: str = "SQL execution failed",
        *,
        code: SQLExecutionErrorCode | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code
        self.duration_ms = duration_ms


class SQLExecutionConfigurationError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_CONFIGURATION_ERROR
    status = SQLExecutionStatus.FAILED

    def __init__(
        self,
        message: str = "SQL execution is not configured",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionAuthorizationError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_AUTHORIZATION_ERROR
    status = SQLExecutionStatus.FAILED

    def __init__(
        self,
        message: str = "SQL execution authorization failed",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionValidationError(SQLExecutionError):
    """Raised when Chapter 7.2 validation rejects SQL before MCP execution."""

    code = SQLExecutionErrorCode.SQL_EXECUTION_VALIDATION_ERROR
    status = SQLExecutionStatus.REJECTED

    def __init__(
        self,
        message: str = "SQL failed validation and was not executed",
        *,
        violations: list[SQLValidationViolation] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)
        self.violations = list(violations or [])


class SQLExecutionTimeoutError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_TIMEOUT_ERROR
    status = SQLExecutionStatus.TIMEOUT

    def __init__(
        self, message: str = "The query timed out", *, duration_ms: float = 0.0
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionResultLimitError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_RESULT_LIMIT_ERROR
    status = SQLExecutionStatus.REJECTED

    def __init__(
        self,
        message: str = "The query result is too large",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionDatabaseError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_DATABASE_ERROR
    status = SQLExecutionStatus.FAILED

    def __init__(
        self,
        message: str = "The database query failed",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionRejectedError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_REJECTED_ERROR
    status = SQLExecutionStatus.REJECTED

    def __init__(
        self,
        message: str = "The query was rejected by the execution boundary",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)


class SQLExecutionCancelledError(SQLExecutionError):
    code = SQLExecutionErrorCode.SQL_EXECUTION_CANCELLED_ERROR
    status = SQLExecutionStatus.CANCELLED

    def __init__(
        self,
        message: str = "SQL execution was cancelled",
        *,
        duration_ms: float = 0.0,
    ) -> None:
        super().__init__(message, code=self.code, duration_ms=duration_ms)
