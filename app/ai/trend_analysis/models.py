"""Typed models for structured trend analysis."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.data_analyst.models import AnalysisConfidence, _normalize_confidence


class TrendDirection(str, enum.Enum):
    INCREASING = "INCREASING"
    DECREASING = "DECREASING"
    STABLE = "STABLE"
    VOLATILE = "VOLATILE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class PeriodComparison(BaseModel):
    """Deterministic period-over-period comparison between two consecutive periods."""

    model_config = ConfigDict(extra="forbid")

    previous_period: str
    period: str
    previous_value: float
    value: float
    absolute_change: float
    percent_change: float | None = None
    direction: TrendDirection
    significant: bool = False


class TrendSeries(BaseModel):
    """Deterministic time-series evidence computed from the query results."""

    model_config = ConfigDict(extra="forbid")

    period_column: str | None = None
    value_column: str | None = None
    point_count: int = 0
    skipped_row_count: int = 0
    reordered: bool = False
    first_period: str | None = None
    last_period: str | None = None
    first_value: float | None = None
    last_value: float | None = None
    minimum_value: float | None = None
    maximum_value: float | None = None
    direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    total_change: float | None = None
    growth_rate_percent: float | None = None
    average_period_change_percent: float | None = None
    direction_changes: int = 0
    comparisons: list[PeriodComparison] = Field(default_factory=list)
    significant_changes: list[PeriodComparison] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class LLMTrendAnalysisOutput(BaseModel):
    """Structured LLM output for trend analysis."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Narrative summary of how the measure evolved across the observed periods.",
    )
    direction_explanation: str = Field(
        min_length=1,
        max_length=32_000,
        description="Explanation of the overall trend direction and the growth or decline rate.",
    )
    period_comparisons: str = Field(
        min_length=1,
        max_length=32_000,
        description="Period-over-period comparisons described in natural language.",
    )
    significant_changes: str = Field(
        min_length=1,
        max_length=32_000,
        description="Description of the significant trend changes, or a statement that none were detected.",
    )
    conclusions: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Evidence-based trend conclusions, each citing specific periods and values.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence score for the trend analysis: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning for the confidence score, covering series length and data quality issues.",
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence_score(cls, value: Any) -> Any:
        return _normalize_confidence(value)

    @field_validator("conclusions", mode="before")
    @classmethod
    def normalize_conclusions(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = " ".join(item.strip().split())
            if text:
                cleaned.append(text)
        return cleaned


class TrendAnalysisResult(BaseModel):
    """Validated trend analysis combining computed series evidence and narrative."""

    model_config = ConfigDict(extra="forbid")

    direction: TrendDirection
    growth_rate_percent: float | None
    series: TrendSeries
    summary: str
    direction_explanation: str
    period_comparisons: str
    significant_changes: str
    conclusions: list[str]
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
