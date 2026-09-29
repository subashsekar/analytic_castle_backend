"""Typed models for structured data analysis."""

from __future__ import annotations

import enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AnalysisConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def _normalize_confidence(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().upper()
    if token in ("HIGH", "MEDIUM", "LOW"):
        return token
    return "MEDIUM"


class LLMDataAnalystOutput(BaseModel):
    """Structured LLM output for data analysis."""

    model_config = ConfigDict(extra="ignore")

    interpretation: str = Field(
        min_length=1,
        max_length=32_000,
        description="Direct interpretation of the query results, answering the user's question.",
    )
    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Statistical summaries of the data (e.g., counts, averages, sums, min/max, or other key metrics).",
    )
    comparisons: str = Field(
        min_length=1,
        max_length=32_000,
        description="Comparisons between different segments, categories, or time periods in the data.",
    )
    conclusions: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Evidence-based conclusions drawn from the data, with specific data points as evidence.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence score for the analysis: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning explaining the confidence score, including data sufficiency or data quality issues.",
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


class DataAnalysisResult(BaseModel):
    """Validated data analysis with structured sections and confidence assessment."""

    model_config = ConfigDict(extra="forbid")

    interpretation: str
    summary: str
    comparisons: str
    conclusions: list[str]
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
