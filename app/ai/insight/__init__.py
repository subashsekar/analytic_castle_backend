"""Insight Agent for AnalyticCastle."""

from app.ai.insight.errors import (
    InsightAuthorizationError,
    InsightConfigurationError,
    InsightError,
    InsightErrorCode,
    InsightLLMError,
    InsightValidationError,
)
from app.ai.insight.evidence import (
    build_evidence_payload,
    groundable_names,
    has_evidence,
    identify_key_metrics,
)
from app.ai.insight.models import (
    BusinessInsight,
    InsightAnalysisResult,
    InsightPriority,
    KeyMetric,
    LLMBusinessInsight,
    LLMInsightOutput,
)
from app.ai.insight.prompts import build_insight_prompt_registry
from app.ai.insight.service import InsightAgent

__all__ = [
    "BusinessInsight",
    "InsightAgent",
    "InsightAnalysisResult",
    "InsightAuthorizationError",
    "InsightConfigurationError",
    "InsightError",
    "InsightErrorCode",
    "InsightLLMError",
    "InsightPriority",
    "InsightValidationError",
    "KeyMetric",
    "LLMBusinessInsight",
    "LLMInsightOutput",
    "build_evidence_payload",
    "build_insight_prompt_registry",
    "groundable_names",
    "has_evidence",
    "identify_key_metrics",
]
