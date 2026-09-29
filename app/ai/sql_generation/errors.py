"""SQL generation error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class SQLGenerationErrorCode(str, Enum):
    SQL_GENERATION_CONFIGURATION_ERROR = "SQL_GENERATION_CONFIGURATION_ERROR"
    SQL_GENERATION_AUTHORIZATION_ERROR = "SQL_GENERATION_AUTHORIZATION_ERROR"
    SQL_GENERATION_SCHEMA_ERROR = "SQL_GENERATION_SCHEMA_ERROR"
    SQL_GENERATION_VALIDATION_ERROR = "SQL_GENERATION_VALIDATION_ERROR"
    SQL_GENERATION_LLM_ERROR = "SQL_GENERATION_LLM_ERROR"
    SQL_GENERATION_INTERNAL_ERROR = "SQL_GENERATION_INTERNAL_ERROR"


def _sanitize(message: str) -> str:
    return redact_secret(message)


class SQLGenerationError(Exception):
    """Base error for the SQL generation layer."""

    code: SQLGenerationErrorCode = SQLGenerationErrorCode.SQL_GENERATION_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "SQL generation failed",
        *,
        code: SQLGenerationErrorCode | None = None,
    ) -> None:
        super().__init__(_sanitize(message))
        if code is not None:
            self.code = code


class SQLGenerationConfigurationError(SQLGenerationError):
    code = SQLGenerationErrorCode.SQL_GENERATION_CONFIGURATION_ERROR

    def __init__(self, message: str = "SQL generation is not configured") -> None:
        super().__init__(message, code=self.code)


class SQLGenerationAuthorizationError(SQLGenerationError):
    code = SQLGenerationErrorCode.SQL_GENERATION_AUTHORIZATION_ERROR

    def __init__(self, message: str = "SQL generation authorization failed") -> None:
        super().__init__(message, code=self.code)


class SQLGenerationSchemaError(SQLGenerationError):
    code = SQLGenerationErrorCode.SQL_GENERATION_SCHEMA_ERROR

    def __init__(
        self, message: str = "Schema context is missing or insufficient"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLGenerationValidationError(SQLGenerationError):
    code = SQLGenerationErrorCode.SQL_GENERATION_VALIDATION_ERROR

    def __init__(
        self, message: str = "SQL generation output validation failed"
    ) -> None:
        super().__init__(message, code=self.code)


class SQLGenerationLLMError(SQLGenerationError):
    code = SQLGenerationErrorCode.SQL_GENERATION_LLM_ERROR

    def __init__(self, message: str = "SQL generation LLM request failed") -> None:
        super().__init__(message, code=self.code)
