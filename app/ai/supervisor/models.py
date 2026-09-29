"""Typed models for supervisor classification and routing decisions."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.state.models import AnalysisSessionSnapshot
from app.enums import AgentPhase

_MAX_REASON_CHARS = 256
_MAX_CLARIFICATION_CHARS = 500


class RequestCategory(str, enum.Enum):
    ANALYTICAL_QUERY = "ANALYTICAL_QUERY"
    SCHEMA_QUESTION = "SCHEMA_QUESTION"
    GENERAL = "GENERAL"
    UNSUPPORTED = "UNSUPPORTED"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class ClassificationConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SupervisorAction(str, enum.Enum):
    RUN_INTENT_AGENT = "RUN_INTENT_AGENT"
    RUN_PLANNER = "RUN_PLANNER"
    RESPOND_UNSUPPORTED = "RESPOND_UNSUPPORTED"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    ADVANCE_WORKFLOW = "ADVANCE_WORKFLOW"
    COMPLETE = "COMPLETE"
    FAIL = "FAIL"


def _normalize_enum_token(value: object, *, upper: bool) -> object:
    if not isinstance(value, str):
        return value
    token = value.strip().replace(" ", "_").replace("-", "_")
    if not token:
        return value
    return token.upper() if upper else token.lower()


class LLMRequestClassification(BaseModel):
    """Structured LLM output for request classification."""

    model_config = ConfigDict(extra="ignore")

    category: RequestCategory
    confidence: ClassificationConfidence = ClassificationConfidence.MEDIUM
    requires_data_access: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = Field(
        default=None, max_length=_MAX_CLARIFICATION_CHARS
    )
    reason: str | None = Field(default=None, max_length=_MAX_REASON_CHARS)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: object) -> object:
        return _normalize_enum_token(value, upper=True)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        return _normalize_enum_token(value, upper=True)

    @field_validator("clarification_question", "reason")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None


class RequestClassification(BaseModel):
    """Validated classification used for routing."""

    model_config = ConfigDict(extra="forbid")

    category: RequestCategory
    confidence: ClassificationConfidence
    requires_data_access: bool = False
    requires_clarification: bool = False
    clarification_question: str | None = None
    reason: str | None = None
    source: str = "llm"


class SupervisorDecision(BaseModel):
    """Deterministic routing decision produced by the supervisor."""

    model_config = ConfigDict(extra="forbid")

    action: SupervisorAction
    classification: RequestClassification
    current_phase: AgentPhase
    next_phase: AgentPhase | None = None
    message: str | None = Field(default=None, max_length=_MAX_CLARIFICATION_CHARS)


@dataclass(frozen=True)
class SuperviseMessageParams:
    session_id: UUID
    workspace_id: UUID
    user_id: UUID
    message: str
    expected_agent_version: int
    expected_context_version: int


@dataclass(frozen=True)
class SuperviseAdvanceParams:
    session_id: UUID
    workspace_id: UUID
    user_id: UUID
    expected_agent_version: int


@dataclass(frozen=True)
class SupervisorResult:
    decision: SupervisorDecision
    session: AnalysisSessionSnapshot
