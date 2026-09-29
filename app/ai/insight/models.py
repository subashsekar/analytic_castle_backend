"""Typed models for structured business insight generation."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.data_analyst.models import AnalysisConfidence, _normalize_confidence
from app.ai.root_cause_analysis.models import _clean_text_list


class InsightPriority(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


_PRIORITY_RANK = {
    InsightPriority.HIGH: 3,
    InsightPriority.MEDIUM: 2,
    InsightPriority.LOW: 1,
}


def priority_rank(priority: InsightPriority) -> int:
    return _PRIORITY_RANK[priority]


def _normalize_priority(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().upper()
    return token if token in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"


class KeyMetric(BaseModel):
    """A numeric measure identified in the query results and summarized deterministically."""

    model_config = ConfigDict(extra="forbid")

    column: str
    value_count: int
    total: float
    average: float
    minimum: float
    maximum: float


class LLMBusinessInsight(BaseModel):
    """A single business insight proposed by the model."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        min_length=1,
        max_length=300,
        description="Short headline for the insight.",
    )
    insight: str = Field(
        min_length=1,
        max_length=4_000,
        description="The business insight, stated as what the analyzed data shows.",
    )
    metric: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "The exact name of the key metric or result column this insight is about, copied "
            "verbatim from the evidence. Null when the insight is not about one specific column."
        ),
    )
    business_impact: str = Field(
        min_length=1,
        max_length=4_000,
        description="What this means for the business, expressed in terms of the observed data.",
    )
    supporting_evidence: str = Field(
        min_length=1,
        max_length=8_000,
        description="The query results or analysis evidence this insight rests on, citing values.",
    )
    priority: InsightPriority = Field(
        description="Priority of this insight: HIGH, MEDIUM, or LOW.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence for this insight: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning for this confidence, covering evidence strength and data gaps.",
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence_score(cls, value: Any) -> Any:
        return _normalize_confidence(value)

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, value: Any) -> Any:
        return _normalize_priority(value)

    @field_validator("metric")
    @classmethod
    def strip_metric(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None


class LLMInsightOutput(BaseModel):
    """Structured LLM output for insight generation."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Narrative summary of what the analyzed data reveals about the business.",
    )
    insights: list[LLMBusinessInsight] = Field(
        min_length=1,
        max_length=15,
        description="Business insights, each with impact, supporting evidence, priority, and confidence.",
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Questions the available data cannot answer, stated as gaps rather than findings.",
    )

    @field_validator("data_gaps", mode="before")
    @classmethod
    def normalize_data_gaps(cls, value: Any) -> Any:
        return _clean_text_list(value)


class BusinessInsight(BaseModel):
    """A prioritized business insight with its evidence and confidence."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    title: str
    insight: str
    metric: str | None = None
    business_impact: str
    supporting_evidence: str
    priority: InsightPriority
    confidence_score: AnalysisConfidence
    confidence_reasoning: str


class InsightAnalysisResult(BaseModel):
    """Validated insights, prioritized, with the metrics they were derived from."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    top_insight: str | None
    insights: list[BusinessInsight]
    key_metrics: list[KeyMetric] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
    notes: list[str] = Field(default_factory=list)
