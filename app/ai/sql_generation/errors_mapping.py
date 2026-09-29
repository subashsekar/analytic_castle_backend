"""Map SQL generation errors to AI analyst exceptions for API handling."""

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
from app.ai.sql_generation.errors import (
    SQLGenerationAuthorizationError,
    SQLGenerationError,
    SQLGenerationLLMError,
    SQLGenerationSchemaError,
    SQLGenerationValidationError,
)


def map_sql_generation_error(exc: SQLGenerationError) -> AIError:
    if isinstance(exc, SQLGenerationAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(exc, SQLGenerationSchemaError):
        return AIRequestValidationError(str(exc))
    if isinstance(exc, SQLGenerationValidationError):
        return AIResponseValidationError(str(exc))
    if isinstance(exc, SQLGenerationLLMError):
        if exc.__cause__ is not None:
            if isinstance(exc.__cause__, LLMTimeoutError):
                return AIProviderTimeoutError(str(exc))
            if isinstance(
                exc.__cause__,
                (LLMResponseValidationError, SQLGenerationValidationError),
            ):
                return AIResponseValidationError(str(exc))
        return AIProviderError(str(exc))
    return AIProviderError(str(exc))
