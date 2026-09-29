"""Map SQL correction errors to AI analyst exceptions for API handling."""

from __future__ import annotations

from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIProviderError,
    AIProviderTimeoutError,
    AIRequestValidationError,
    AIResponseValidationError,
)
from app.ai.llm.errors import LLMResponseValidationError, LLMTimeoutError
from app.ai.sql_correction.errors import (
    SQLCorrectionAuthorizationError,
    SQLCorrectionError,
    SQLCorrectionLLMError,
    SQLCorrectionSchemaError,
    SQLCorrectionValidationError,
)


def map_sql_correction_error(exc: SQLCorrectionError) -> AIError:
    if isinstance(exc, SQLCorrectionAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(exc, SQLCorrectionSchemaError):
        return AIRequestValidationError(str(exc))
    if isinstance(exc, SQLCorrectionValidationError):
        return AIResponseValidationError(str(exc))
    if isinstance(exc, SQLCorrectionLLMError):
        if exc.__cause__ is not None:
            if isinstance(exc.__cause__, LLMTimeoutError):
                return AIProviderTimeoutError(str(exc))
            if isinstance(
                exc.__cause__,
                (LLMResponseValidationError, SQLCorrectionValidationError),
            ):
                return AIResponseValidationError(str(exc))
        return AIProviderError(str(exc))
    return AIProviderError(str(exc))
