"""Investigation plan and run result models."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field

from app.ai.root_cause_analysis.models import RootCauseEvidence


class InvestigationStepKind(str, enum.Enum):
    OVERALL_CHANGE = "overall_change"
    BREAKDOWN = "breakdown"
    DRILL = "drill"
    YEAR_OVER_YEAR = "year_over_year"
    FUNNEL = "funnel"
    COHORT = "cohort"


class InvestigationPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_order: int = Field(ge=1, le=20)
    kind: InvestigationStepKind
    description: str
    question: str
    sql: str


class DimensionContribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: str
    segment: str
    current_value: float
    prior_value: float
    delta: float
    share_of_change: float | None = None


class InvestigationRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence: list[RootCauseEvidence] = Field(default_factory=list)
    contributions: list[DimensionContribution] = Field(default_factory=list)
    summary: str = ""
    notes: list[str] = Field(default_factory=list)
    steps_planned: int = 0
    steps_executed: int = 0

    def prompt_payload(self) -> str:
        """JSON-friendly block for root-cause prompts."""
        return self.summary
