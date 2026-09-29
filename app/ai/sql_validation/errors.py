"""SQL validation error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class SQLValidationErrorCode(str, Enum):
    SQL_VALIDATION_CONFIGURATION_ERROR = "SQL_VALIDATION_CONFIGURATION_ERROR"
    SQL_VALIDATION_AUTHORIZATION_ERROR = "SQL_VALIDATION_AUTHORIZATION_ERROR"
    SQL_VALIDATION_READONLY_ERROR = "SQL_VALIDATION_READONLY_ERROR"
    SQL_VALIDATION_PARSE_ERROR = "SQL_VALIDATION_PARSE_ERROR"
    SQL_VALIDATION_SCHEMA_ERROR = "SQL_VALIDATION_SCHEMA_ERROR"
    SQL_VALIDATION_TABLE_ERROR = "SQL_VALIDATION_TABLE_ERROR"
    SQL_VALIDATION_COLUMN_ERROR = "SQL_VALIDATION_COLUMN_ERROR"
    SQL_VALIDATION_INTERNAL_ERROR = "SQL_VALIDATION_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class SQLValidationError(Exception):
    """Base error for the SQL validation layer."""

    code: SQLValidationErrorCode = SQLValidationErrorCode.SQL_VALIDATION_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "SQL validation failed",
        *,
        code: SQLValidationErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class SQLValidationConfigurationError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_CONFIGURATION_ERROR

    def __init__(self, message: str = "SQL validation is not configured") -> None:
        super().__init__(message, code=self.code)


class SQLValidationAuthorizationError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_AUTHORIZATION_ERROR

    def __init__(self, message: str = "SQL validation authorization failed") -> None:
        super().__init__(message, code=self.code)


class SQLValidationReadonlyError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_READONLY_ERROR

    def __init__(
        self, message: str = "SQL is not a single read-only SELECT statement"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLValidationParseError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_PARSE_ERROR

    def __init__(self, message: str = "SQL could not be parsed") -> None:
        super().__init__(message, code=self.code)


class SQLValidationSchemaError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_SCHEMA_ERROR

    def __init__(
        self, message: str = "Authorized schema metadata is missing or insufficient"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLValidationTableError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_TABLE_ERROR

    def __init__(self, message: str = "SQL references an unauthorized table") -> None:
        super().__init__(message, code=self.code)


class SQLValidationColumnError(SQLValidationError):
    code = SQLValidationErrorCode.SQL_VALIDATION_COLUMN_ERROR

    def __init__(self, message: str = "SQL references an unauthorized column") -> None:
        super().__init__(message, code=self.code)
