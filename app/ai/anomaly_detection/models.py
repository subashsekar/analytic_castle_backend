"""Typed models for structured anomaly detection."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.data_analyst.models import AnalysisConfidence, _normalize_confidence


class AnomalyType(str, enum.Enum):
    STATISTICAL_OUTLIER = "STATISTICAL_OUTLIER"
    UNEXPECTED_CHANGE = "UNEXPECTED_CHANGE"
    THRESHOLD_BREACH = "THRESHOLD_BREACH"


class AnomalySeverity(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


_SEVERITY_RANK = {AnomalySeverity.HIGH: 3, AnomalySeverity.MEDIUM: 2, AnomalySeverity.LOW: 1}


def severity_rank(severity: AnomalySeverity) -> int:
    return _SEVERITY_RANK[severity]


class ColumnThreshold(BaseModel):
    """Caller-supplied expected bounds for a numeric column."""

    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None


class Anomaly(BaseModel):
    """A single deterministically detected anomaly with its supporting evidence."""

    model_config = ConfigDict(extra="forbid")

    column: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    method: str
    value: float
    row_index: int | None = None
    period: str | None = None
    previous_value: float | None = None
    percent_change: float | None = None
    expected_low: float | None = None
    expected_high: float | None = None
    score: float | None = None
    evidence: str


class AnomalyScan(BaseModel):
    """Deterministic anomaly scan evidence computed from the query results."""

    model_config = ConfigDict(extra="forbid")

    analyzed: bool = False
    numeric_columns: list[str] = Field(default_factory=list)
    scanned_row_count: int = 0
    period_column: str | None = None
    column_statistics: dict[str, dict[str, float]] = Field(default_factory=dict)
    anomalies: list[Anomaly] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def highest_severity(self) -> AnomalySeverity | None:
        if not self.anomalies:
            return None
        return max((item.severity for item in self.anomalies), key=severity_rank)


class LLMAnomalyDetectionOutput(BaseModel):
    """Structured LLM output for anomaly detection."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Narrative summary of the detected anomalies and the data that was scanned.",
    )
    outliers: str = Field(
        min_length=1,
        max_length=32_000,
        description="Description of the statistical outliers, or a statement that none were detected.",
    )
    unexpected_changes: str = Field(
        min_length=1,
        max_length=32_000,
        description="Description of the unexpected period-over-period changes, or a statement that none were detected.",
    )
    threshold_breaches: str = Field(
        min_length=1,
        max_length=32_000,
        description="Description of the threshold breaches, or a statement that none were detected.",
    )
    conclusions: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Evidence-based conclusions, each citing the anomalous value and its expected range.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence score for the anomaly analysis: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning for the confidence score, covering sample size and data quality issues.",
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


class AnomalyAnalysisResult(BaseModel):
    """Validated anomaly analysis combining computed detections and narrative."""

    model_config = ConfigDict(extra="forbid")

    anomaly_count: int
    highest_severity: AnomalySeverity | None
    scan: AnomalyScan
    summary: str
    outliers: str
    unexpected_changes: str
    threshold_breaches: str
    conclusions: list[str]
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
