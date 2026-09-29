"""Trend Analysis Agent for AnalyticCastle."""

from app.ai.trend_analysis.errors import (
    TrendAnalysisError,
    TrendAnalysisErrorCode,
    TrendAnalysisConfigurationError,
    TrendAnalysisAuthorizationError,
    TrendAnalysisValidationError,
    TrendAnalysisLLMError,
)
from app.ai.trend_analysis.models import (
    LLMTrendAnalysisOutput,
    PeriodComparison,
    TrendAnalysisResult,
    TrendDirection,
    TrendSeries,
)
from app.ai.trend_analysis.prompts import build_trend_analysis_prompt_registry
from app.ai.trend_analysis.series import build_trend_series, series_prompt_payload
from app.ai.trend_analysis.service import TrendAnalysisAgent

__all__ = [
    "TrendAnalysisAgent",
    "TrendAnalysisResult",
    "TrendDirection",
    "TrendSeries",
    "PeriodComparison",
    "LLMTrendAnalysisOutput",
    "TrendAnalysisError",
    "TrendAnalysisErrorCode",
    "TrendAnalysisConfigurationError",
    "TrendAnalysisAuthorizationError",
    "TrendAnalysisValidationError",
    "TrendAnalysisLLMError",
    "build_trend_analysis_prompt_registry",
    "build_trend_series",
    "series_prompt_payload",
]
