"""Recommendation Agent for AnalyticCastle."""

from app.ai.recommendation.errors import (
    RecommendationAuthorizationError,
    RecommendationConfigurationError,
    RecommendationError,
    RecommendationErrorCode,
    RecommendationLLMError,
    RecommendationValidationError,
)
from app.ai.recommendation.evidence import (
    build_evidence_payload,
    groundable_names,
    has_evidence,
)
from app.ai.recommendation.models import (
    LLMRecommendation,
    LLMRecommendationOutput,
    Recommendation,
    RecommendationLevel,
    RecommendationResult,
    derive_priority,
    level_rank,
)
from app.ai.recommendation.prompts import build_recommendation_prompt_registry
from app.ai.recommendation.service import RecommendationAgent

__all__ = [
    "LLMRecommendation",
    "LLMRecommendationOutput",
    "Recommendation",
    "RecommendationAgent",
    "RecommendationAuthorizationError",
    "RecommendationConfigurationError",
    "RecommendationError",
    "RecommendationErrorCode",
    "RecommendationLLMError",
    "RecommendationLevel",
    "RecommendationResult",
    "RecommendationValidationError",
    "build_evidence_payload",
    "build_recommendation_prompt_registry",
    "derive_priority",
    "groundable_names",
    "has_evidence",
    "level_rank",
]
