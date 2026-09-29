"""Map SQL validation errors to AI analyst exceptions for API handling."""

from __future__ import annotations

from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIRequestValidationError,
)
from app.ai.sql_validation.errors import (
    SQLValidationAuthorizationError,
    SQLValidationColumnError,
    SQLValidationError,
    SQLValidationParseError,
    SQLValidationReadonlyError,
    SQLValidationSchemaError,
    SQLValidationTableError,
)


def map_sql_validation_error(exc: SQLValidationError) -> AIError:
    if isinstance(exc, SQLValidationAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(
        exc,
        (
            SQLValidationReadonlyError,
            SQLValidationParseError,
            SQLValidationSchemaError,
            SQLValidationTableError,
            SQLValidationColumnError,
        ),
    ):
        return AIRequestValidationError(str(exc))
    return AIRequestValidationError(str(exc))
