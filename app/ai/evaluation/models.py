"""Typed models for agent evaluation results and reports.

A check is either VERIFIED, meaning it was decided by comparing the agent output
against evidence the pipeline computed itself, or ESTIMATED, meaning it was
decided by a heuristic. Only VERIFIED checks can fail; an ESTIMATED check that
does not hold is a warning, never a statement of incorrectness.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EvaluatedAgent(str, enum.Enum):
    DATA_ANALYST = "DATA_ANALYST"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    ROOT_CAUSE_ANALYSIS = "ROOT_CAUSE_ANALYSIS"
    INSIGHT = "INSIGHT"
    RECOMMENDATION = "RECOMMENDATION"


class EvaluationDimension(str, enum.Enum):
    STRUCTURE = "STRUCTURE"
    COMPLETENESS = "COMPLETENESS"
    ACCURACY = "ACCURACY"
    GROUNDING = "GROUNDING"
    CALIBRATION = "CALIBRATION"
    HALLUCINATION = "HALLUCINATION"


class EvaluationMethod(str, enum.Enum):
    """How a check was decided, so estimates are never read as correctness."""

    VERIFIED = "VERIFIED"
    ESTIMATED = "ESTIMATED"


class CheckStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIPPED = "SKIPPED"


class EvaluationCheck(BaseModel):
    """One evaluation check, traceable to the evidence it was decided against."""

    model_config = ConfigDict(extra="forbid")

    agent: EvaluatedAgent
    check_id: str
    dimension: EvaluationDimension
    status: CheckStatus
    method: EvaluationMethod
    detail: str
    evidence_reference: str | None = None


class AgentEvaluation(BaseModel):
    """Evaluation of a single agent output, with verified and estimated results split."""

    model_config = ConfigDict(extra="forbid")

    agent: EvaluatedAgent
    checks: list[EvaluationCheck]
    verified_passed: int
    verified_failed: int
    estimated_passed: int
    estimated_warnings: int
    skipped: int
    verified_correct: bool
    estimated_quality_score: float

    @classmethod
    def from_checks(
        cls,
        agent: EvaluatedAgent,
        checks: list[EvaluationCheck],
    ) -> AgentEvaluation:
        verified_passed = _count(checks, EvaluationMethod.VERIFIED, CheckStatus.PASS)
        verified_failed = _count(checks, EvaluationMethod.VERIFIED, CheckStatus.FAIL)
        estimated_passed = _count(checks, EvaluationMethod.ESTIMATED, CheckStatus.PASS)
        estimated_warnings = _count(checks, EvaluationMethod.ESTIMATED, CheckStatus.WARN)
        decided = verified_passed + verified_failed + estimated_passed + estimated_warnings
        return cls(
            agent=agent,
            checks=checks,
            verified_passed=verified_passed,
            verified_failed=verified_failed,
            estimated_passed=estimated_passed,
            estimated_warnings=estimated_warnings,
            skipped=sum(1 for check in checks if check.status is CheckStatus.SKIPPED),
            verified_correct=verified_failed == 0 and verified_passed > 0,
            estimated_quality_score=(
                round((verified_passed + estimated_passed) / decided, 4) if decided else 0.0
            ),
        )


class EvaluationMetrics(BaseModel):
    """Aggregate performance metrics across the evaluated agents."""

    model_config = ConfigDict(extra="forbid")

    agents_evaluated: int = 0
    checks_run: int = 0
    verified_checks: int = 0
    verified_failures: int = 0
    estimated_checks: int = 0
    estimated_warnings: int = 0
    skipped_checks: int = 0
    verified_pass_rate: float = 0.0
    estimated_quality_score: float = 0.0
    query_duration_ms: float | None = None


class EvaluationReport(BaseModel):
    """Structured evaluation report for one analysis run."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agents: list[AgentEvaluation] = Field(default_factory=list)
    agents_not_evaluated: list[EvaluatedAgent] = Field(default_factory=list)
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    verified_correct: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def failures(self) -> list[EvaluationCheck]:
        return [
            check
            for agent in self.agents
            for check in agent.checks
            if check.status is CheckStatus.FAIL
        ]

    @property
    def warnings(self) -> list[EvaluationCheck]:
        return [
            check
            for agent in self.agents
            for check in agent.checks
            if check.status is CheckStatus.WARN
        ]


def _count(
    checks: list[EvaluationCheck],
    method: EvaluationMethod,
    status: CheckStatus,
) -> int:
    return sum(1 for check in checks if check.method is method and check.status is status)
