"""Typed models for bounded SQL correction of untrusted drafts."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ai.intent_types import AIIntent
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_generation.models import GeneratedSQL
from app.ai.sql_validation.models import SQLValidationViolation, ValidatedSQL


class SQLCorrectionStatus(str, enum.Enum):
    CORRECTED = "CORRECTED"
    UNCORRECTED = "UNCORRECTED"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    UNCHANGED = "UNCHANGED"


class SQLErrorSource(str, enum.Enum):
    VALIDATION = "VALIDATION"
    EXECUTION = "EXECUTION"


class SQLErrorCorrectability(str, enum.Enum):
    CORRECTABLE = "CORRECTABLE"
    NON_CORRECTABLE = "NON_CORRECTABLE"


class SQLErrorAnalysis(BaseModel):
    """Classification of a validation or execution failure."""

    model_config = ConfigDict(extra="forbid")

    correctability: SQLErrorCorrectability
    source: SQLErrorSource
    codes: list[str] = Field(default_factory=list)
    reason: str


class SQLCorrectionOutcome(BaseModel):
    """Result of attempting to correct untrusted SQL against catalog metadata."""

    model_config = ConfigDict(extra="forbid")

    status: SQLCorrectionStatus
    validated: ValidatedSQL | None = None
    generated: GeneratedSQL | None = None
    violations: list[SQLValidationViolation] = Field(default_factory=list)
    attempt_count: int = 0
    max_attempts: int = 1
    requires_clarification: bool = False
    clarification_question: str | None = None
    schema_truncated: bool = False
    analysis: SQLErrorAnalysis | None = None


@dataclass(frozen=True)
class SQLCorrectParams:
    """Authorized SQL correction request for a single workspace data source."""

    workspace_id: UUID
    organization_id: UUID
    user_id: UUID
    data_source_id: UUID
    sql: str
    metadata: ResolvedMetadataContext
    message: str | None = None
    intent: AIIntent | None = None
    plan_summary: str | None = None
    data_source_name: str | None = None
    session_id: UUID | None = None
    violations: tuple[SQLValidationViolation, ...] = ()
    execution_error_code: str | None = None
    execution_error_message: str | None = None
    max_attempts: int | None = None


@dataclass(frozen=True)
class SQLCorrectionServiceResult:
    outcome: SQLCorrectionOutcome
    data_source_id: UUID
    workspace_id: UUID
    organization_id: UUID
