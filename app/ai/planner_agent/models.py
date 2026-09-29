"""Typed models for planner intent detection and analysis planning."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.intent_types import (
    AIIntent,
    AIIntentType,
    AIPlanCapability,
    AIPlanOperation,
    AIRequestPlan,
    validate_concept_name,
)
from app.ai.investigation.models import InvestigationPlanStep
from app.ai.state.models import AnalysisSessionSnapshot

_MAX_LLM_COLLECTION = 50
_MAX_CLARIFICATION_CHARS = 500
_MAX_UNSUPPORTED_REASON_CHARS = 128
_MAX_ACTION_DESCRIPTION_CHARS = 256


def _normalize_enum_token(value: Any, *, upper: bool) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().replace(" ", "_").replace("-", "_")
    if not token:
        return value
    return token.upper() if upper else token.lower()


class PlannerActionKind(str, enum.Enum):
    INSPECT_SCHEMA = "INSPECT_SCHEMA"
    RESOLVE_METADATA = "RESOLVE_METADATA"
    QUERY_DATA = "QUERY_DATA"
    AGGREGATE = "AGGREGATE"
    FILTER = "FILTER"
    COMPARE = "COMPARE"
    RANK = "RANK"
    TREND = "TREND"
    RESPOND = "RESPOND"


class PlannerToolRef(str, enum.Enum):
    """Descriptive tool identifiers. The planner never executes these."""

    METADATA_LOOKUP = "metadata.lookup"
    POSTGRES_QUERY = "postgres.query"
    SAMPLE_DATA = "sample_data.read"


class RequiredDataKind(str, enum.Enum):
    TABLE = "TABLE"
    COLUMN = "COLUMN"
    METRIC = "METRIC"
    DIMENSION = "DIMENSION"
    FILTER = "FILTER"
    TIME_RANGE = "TIME_RANGE"
    RELATIONSHIP = "RELATIONSHIP"


class LLMRequiredDataRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=512)
    kind: RequiredDataKind = RequiredDataKind.TABLE

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return validate_concept_name(value, field_name="required data")

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)


class LLMPlannerActionStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_order: int = Field(ge=1, le=50)
    action: PlannerActionKind
    description: str = Field(min_length=1, max_length=_MAX_ACTION_DESCRIPTION_CHARS)
    tool: PlannerToolRef | None = None

    @field_validator("action", mode="before")
    @classmethod
    def normalize_action(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)

    @field_validator("tool", mode="before")
    @classmethod
    def normalize_tool(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        return _normalize_enum_token(value, upper=False)

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        return " ".join(value.strip().split())


class LLMPlannerOutput(BaseModel):
    """Structured LLM output for analysis planning."""

    model_config = ConfigDict(extra="ignore")

    intent: AIIntentType
    operations: list[AIPlanOperation] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    required_capabilities: list[AIPlanCapability] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    required_data: list[LLMRequiredDataRef] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    action_steps: list[LLMPlannerActionStep] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    missing_information: list[str] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    requires_clarification: bool = False
    clarification_question: str | None = Field(
        default=None, max_length=_MAX_CLARIFICATION_CHARS
    )
    unsupported: bool = False
    unsupported_reason: str | None = Field(
        default=None, max_length=_MAX_UNSUPPORTED_REASON_CHARS
    )

    @field_validator("intent", mode="before")
    @classmethod
    def normalize_intent(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)

    @field_validator("operations", mode="before")
    @classmethod
    def normalize_operations(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [_normalize_enum_token(item, upper=True) for item in value]

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def normalize_capabilities(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [_normalize_enum_token(item, upper=True) for item in value]

    @field_validator("missing_information", mode="before")
    @classmethod
    def normalize_missing_information(cls, value: Any) -> Any:
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

    @field_validator("clarification_question", "unsupported_reason")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None


class RequiredDataRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: RequiredDataKind


class PlannerActionStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_order: int = Field(ge=1, le=50)
    action: PlannerActionKind
    description: str
    tool: PlannerToolRef | None = None


class AnalysisPlan(BaseModel):
    """Validated analysis plan with required data and action sequence."""

    model_config = ConfigDict(extra="forbid")

    request_plan: AIRequestPlan
    detected_intent: AIIntentType
    required_data: list[RequiredDataRef] = Field(default_factory=list)
    action_steps: list[PlannerActionStep] = Field(default_factory=list)
    investigation_steps: list[InvestigationPlanStep] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    plan_version: str


@dataclass(frozen=True)
class PlanCreateParams:
    session_id: UUID
    workspace_id: UUID
    user_id: UUID
    expected_agent_version: int
    intent: AIIntent
    message: str
    data_source_name: str | None = None


@dataclass(frozen=True)
class PlannerResult:
    plan: AnalysisPlan
    session: AnalysisSessionSnapshot
