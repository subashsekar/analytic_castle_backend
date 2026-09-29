"""Typed models for structured business recommendations."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.data_analyst.models import AnalysisConfidence, _normalize_confidence
from app.ai.insight.models import _normalize_priority
from app.ai.root_cause_analysis.models import _clean_text_list


class RecommendationLevel(str, enum.Enum):
    """HIGH/MEDIUM/LOW scale used for impact, feasibility, and derived priority."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


_LEVEL_RANK = {
    RecommendationLevel.HIGH: 3,
    RecommendationLevel.MEDIUM: 2,
    RecommendationLevel.LOW: 1,
}


def level_rank(level: RecommendationLevel) -> int:
    return _LEVEL_RANK[level]


def derive_priority(
    impact: RecommendationLevel,
    feasibility: RecommendationLevel,
) -> RecommendationLevel:
    """Priority is impact and feasibility combined here, never asserted by the model."""
    total = level_rank(impact) + level_rank(feasibility)
    if total >= 5:
        return RecommendationLevel.HIGH
    if total >= 4:
        return RecommendationLevel.MEDIUM
    return RecommendationLevel.LOW


class LLMRecommendation(BaseModel):
    """A single business recommendation proposed by the model."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(
        min_length=1,
        max_length=300,
        description="Short headline for the recommended action.",
    )
    recommendation: str = Field(
        min_length=1,
        max_length=4_000,
        description="The recommended action, stated so a business owner could start it.",
    )
    evidence_reference: str = Field(
        min_length=1,
        max_length=300,
        description=(
            "The exact name of the metric, result column, or insight title this recommendation "
            "rests on, copied verbatim from the evidence."
        ),
    )
    supporting_evidence: str = Field(
        min_length=1,
        max_length=8_000,
        description="The values, periods, metrics, or analysis findings that justify this action.",
    )
    expected_outcome: str = Field(
        min_length=1,
        max_length=4_000,
        description=(
            "The outcome that could plausibly follow, stated as a direction of change that "
            "would need to be measured, never as a guaranteed or quantified result."
        ),
    )
    assumptions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "What this recommendation assumes but the data does not establish, such as "
            "business context, capacity, or cost."
        ),
    )
    risks: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Ways this action could fail or cause harm, including what it might not fix.",
    )
    impact: RecommendationLevel = Field(
        description="Expected business impact if this action works: HIGH, MEDIUM, or LOW.",
    )
    feasibility: RecommendationLevel = Field(
        description="How feasible this action is with no new data or systems: HIGH, MEDIUM, or LOW.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence that the evidence supports this recommendation: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning for this confidence, covering evidence strength and what is assumed.",
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence_score(cls, value: Any) -> Any:
        return _normalize_confidence(value)

    @field_validator("impact", "feasibility", mode="before")
    @classmethod
    def normalize_level(cls, value: Any) -> Any:
        return _normalize_priority(value)

    @field_validator("assumptions", "risks", mode="before")
    @classmethod
    def normalize_text_lists(cls, value: Any) -> Any:
        return _clean_text_list(value)

    @field_validator("evidence_reference")
    @classmethod
    def strip_evidence_reference(cls, value: str) -> str:
        return " ".join(value.strip().split())


class LLMRecommendationOutput(BaseModel):
    """Structured LLM output for recommendation generation."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Narrative summary of what the evidence suggests should be done and why.",
    )
    recommendations: list[LLMRecommendation] = Field(
        min_length=1,
        max_length=15,
        description="Recommended actions, each with evidence, expected outcome, risks, and confidence.",
    )
    data_gaps: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="What would need to be measured or collected before acting with more certainty.",
    )

    @field_validator("data_gaps", mode="before")
    @classmethod
    def normalize_data_gaps(cls, value: Any) -> Any:
        return _clean_text_list(value)


class Recommendation(BaseModel):
    """A prioritized recommendation with its evidence, outcome, risks, and confidence."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    title: str
    recommendation: str
    evidence_reference: str
    supporting_evidence: str
    expected_outcome: str
    assumptions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    impact: RecommendationLevel
    feasibility: RecommendationLevel
    priority: RecommendationLevel
    confidence_score: AnalysisConfidence
    confidence_reasoning: str


class RecommendationResult(BaseModel):
    """Validated recommendations, prioritized by impact and feasibility."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    top_recommendation: str | None
    recommendations: list[Recommendation]
    data_gaps: list[str] = Field(default_factory=list)
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
    notes: list[str] = Field(default_factory=list)
