"""Typed models for structured root cause analysis."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.data_analyst.models import AnalysisConfidence, _normalize_confidence

_CONFIDENCE_RANK = {
    AnalysisConfidence.HIGH: 3,
    AnalysisConfidence.MEDIUM: 2,
    AnalysisConfidence.LOW: 1,
}


def confidence_rank(confidence: AnalysisConfidence) -> int:
    return _CONFIDENCE_RANK[confidence]


def _clean_text_list(value: Any) -> Any:
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


class RootCauseEvidence(BaseModel):
    """Outcome of one additional read-only query run to test a hypothesis."""

    model_config = ConfigDict(extra="forbid")

    question: str
    sql: str | None = None
    executed: bool = False
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    note: str | None = None


class LLMRootCauseHypothesis(BaseModel):
    """A single candidate cause proposed by the model."""

    model_config = ConfigDict(extra="ignore")

    statement: str = Field(
        min_length=1,
        max_length=4_000,
        description="The candidate root cause, stated as a testable explanation of the finding.",
    )
    contributing_factors: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Possible contributing factors for this cause, named from the available data or schema.",
    )
    supporting_evidence: str = Field(
        min_length=1,
        max_length=8_000,
        description="Evidence from the findings or gathered query results that supports this cause.",
    )
    contradicting_evidence: str | None = Field(
        default=None,
        max_length=8_000,
        description="Evidence that weakens this cause, or null when none was found.",
    )
    confidence_score: AnalysisConfidence = Field(
        description="Confidence for this hypothesis: HIGH, MEDIUM, or LOW.",
    )
    confidence_reasoning: str = Field(
        min_length=1,
        max_length=4_000,
        description="Reasoning for this hypothesis confidence, covering evidence strength and data gaps.",
    )
    investigation_question: str | None = Field(
        default=None,
        max_length=1_000,
        description=(
            "A single natural-language question answerable by a read-only query over the same "
            "data source, when the available data is insufficient to judge this cause. Null otherwise."
        ),
    )

    @field_validator("confidence_score", mode="before")
    @classmethod
    def normalize_confidence_score(cls, value: Any) -> Any:
        return _normalize_confidence(value)

    @field_validator("contributing_factors", mode="before")
    @classmethod
    def normalize_contributing_factors(cls, value: Any) -> Any:
        return _clean_text_list(value)

    @field_validator("contradicting_evidence", "investigation_question")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None


class LLMRootCauseOutput(BaseModel):
    """Structured LLM output for root cause analysis."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        min_length=1,
        max_length=32_000,
        description="Narrative summary of what is being explained and the leading candidate causes.",
    )
    hypotheses: list[LLMRootCauseHypothesis] = Field(
        min_length=1,
        max_length=10,
        description="Candidate root causes, each with contributing factors, evidence, and confidence.",
    )


class RootCauseHypothesis(BaseModel):
    """A ranked candidate cause with its evidence and confidence."""

    model_config = ConfigDict(extra="forbid")

    rank: int
    statement: str
    contributing_factors: list[str]
    supporting_evidence: str
    contradicting_evidence: str | None = None
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
    investigation_question: str | None = None


class RootCauseAnalysisResult(BaseModel):
    """Validated root cause analysis with ranked hypotheses and gathered evidence."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    primary_cause: str | None
    hypotheses: list[RootCauseHypothesis]
    evidence: list[RootCauseEvidence] = Field(default_factory=list)
    additional_queries_run: int = 0
    confidence_score: AnalysisConfidence
    confidence_reasoning: str
    notes: list[str] = Field(default_factory=list)
