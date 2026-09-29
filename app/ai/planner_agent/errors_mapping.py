"""Map planner errors to AI analyst exceptions for API handling."""

from __future__ import annotations

from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIProviderError,
    AIProviderTimeoutError,
    AIResponseValidationError,
)
from app.ai.llm.errors import LLMResponseValidationError, LLMTimeoutError
from app.ai.planner_agent.errors import (
    PlannerAuthorizationError,
    PlannerError,
    PlannerLLMError,
    PlannerValidationError,
)


def map_planner_error(exc: PlannerError) -> AIError:
    if isinstance(exc, PlannerAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(exc, PlannerValidationError):
        return AIResponseValidationError(str(exc))
    if isinstance(exc, PlannerLLMError):
        if exc.__cause__ is not None:
            if isinstance(exc.__cause__, LLMTimeoutError):
                return AIProviderTimeoutError(str(exc))
            if isinstance(
                exc.__cause__,
                (LLMResponseValidationError, PlannerValidationError),
            ):
                return AIResponseValidationError(str(exc))
        return AIProviderError(str(exc))
    return AIProviderError(str(exc))
