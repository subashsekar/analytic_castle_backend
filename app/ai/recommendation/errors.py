"""Recommendation agent error hierarchy with stable machine-readable codes."""

from __future__ import annotations

from enum import Enum

from app.core.logging import redact_secret


class RecommendationErrorCode(str, Enum):
    RECOMMENDATION_CONFIGURATION_ERROR = "RECOMMENDATION_CONFIGURATION_ERROR"
    RECOMMENDATION_AUTHORIZATION_ERROR = "RECOMMENDATION_AUTHORIZATION_ERROR"
    RECOMMENDATION_VALIDATION_ERROR = "RECOMMENDATION_VALIDATION_ERROR"
    RECOMMENDATION_LLM_ERROR = "RECOMMENDATION_LLM_ERROR"
    RECOMMENDATION_INTERNAL_ERROR = "RECOMMENDATION_INTERNAL_ERROR"


class RecommendationError(Exception):
    """Base error for the recommendation agent."""

    code: RecommendationErrorCode = RecommendationErrorCode.RECOMMENDATION_INTERNAL_ERROR

    def __init__(
        self,
        message: str = "Recommendation request failed",
        *,
        code: RecommendationErrorCode | None = None,
    ) -> None:
        super().__init__(redact_secret(message))
        if code is not None:
            self.code = code


class RecommendationConfigurationError(RecommendationError):
    code = RecommendationErrorCode.RECOMMENDATION_CONFIGURATION_ERROR

    def __init__(self, message: str = "Recommendation generation is not configured") -> None:
        super().__init__(message, code=self.code)


class RecommendationAuthorizationError(RecommendationError):
    code = RecommendationErrorCode.RECOMMENDATION_AUTHORIZATION_ERROR

    def __init__(self, message: str = "Recommendation authorization failed") -> None:
        super().__init__(message, code=self.code)


class RecommendationValidationError(RecommendationError):
    code = RecommendationErrorCode.RECOMMENDATION_VALIDATION_ERROR

    def __init__(self, message: str = "Recommendation output validation failed") -> None:
        super().__init__(message, code=self.code)


class RecommendationLLMError(RecommendationError):
    code = RecommendationErrorCode.RECOMMENDATION_LLM_ERROR

    def __init__(self, message: str = "Recommendation LLM request failed") -> None:
        super().__init__(message, code=self.code)
