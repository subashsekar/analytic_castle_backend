"""Map supervisor errors to AI analyst exceptions for API handling."""

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
from app.ai.supervisor.errors import (
    InvalidRoutingError,
    SupervisorAuthorizationError,
    SupervisorClassificationError,
    SupervisorError,
    SupervisorLLMError,
)


def map_supervisor_error(exc: SupervisorError) -> AIError:
    if isinstance(exc, SupervisorAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(exc, InvalidRoutingError):
        return AIRequestValidationError(str(exc))
    if isinstance(exc, SupervisorClassificationError):
        return AIResponseValidationError(str(exc))
    if isinstance(exc, SupervisorLLMError):
        if exc.__cause__ is not None:
            if isinstance(exc.__cause__, LLMTimeoutError):
                return AIProviderTimeoutError(str(exc))
            if isinstance(
                exc.__cause__,
                (LLMResponseValidationError, SupervisorClassificationError),
            ):
                return AIResponseValidationError(str(exc))
        return AIProviderError(str(exc))
    return AIProviderError(str(exc))
