"""Typed models for schema-aware SQL generation.

Generated SQL is untrusted. This layer structures and bounds LLM output only;
it does not validate SQL safety or execute queries.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.intent_types import AIIntent
from app.ai.metadata_types import ResolvedMetadataContext

_MAX_LLM_COLLECTION = 50
_MAX_CLARIFICATION_CHARS = 500
_MAX_ASSUMPTION_CHARS = 256
_MAX_IDENTIFIER_CHARS = 256


def _normalize_enum_token(value: Any, *, upper: bool) -> Any:
    if not isinstance(value, str):
        return value
    token = value.strip().replace(" ", "_").replace("-", "_")
    if not token:
        return value
    return token.upper() if upper else token.lower()


class SQLDialect(str, enum.Enum):
    POSTGRESQL = "postgresql"


class SQLGenerationConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class LLMSQLOutput(BaseModel):
    """Structured LLM output for SQL generation."""

    model_config = ConfigDict(extra="ignore")

    sql: str | None = Field(default=None, max_length=100_000)
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    referenced_tables: list[str] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    referenced_columns: list[str] = Field(
        default_factory=list, max_length=_MAX_LLM_COLLECTION
    )
    explanation: str | None = Field(default=None, max_length=4_000)
    confidence: SQLGenerationConfidence = SQLGenerationConfidence.MEDIUM
    requires_clarification: bool = False
    clarification_question: str | None = Field(
        default=None, max_length=_MAX_CLARIFICATION_CHARS
    )
    assumptions: list[str] = Field(default_factory=list, max_length=_MAX_LLM_COLLECTION)
    suggested_limit: int | None = Field(default=None, ge=1, le=10_000)

    @field_validator("dialect", mode="before")
    @classmethod
    def normalize_dialect(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=False)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: Any) -> Any:
        return _normalize_enum_token(value, upper=True)

    @field_validator("sql")
    @classmethod
    def strip_sql(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("referenced_tables", "referenced_columns", mode="before")
    @classmethod
    def normalize_identifiers(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = " ".join(item.strip().split())
            if text:
                cleaned.append(text[:_MAX_IDENTIFIER_CHARS])
        return cleaned

    @field_validator("assumptions", mode="before")
    @classmethod
    def normalize_assumptions(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = " ".join(item.strip().split())
            if text:
                cleaned.append(text[:_MAX_ASSUMPTION_CHARS])
        return cleaned

    @field_validator("explanation", "clarification_question")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.strip().split())
        return cleaned or None


class GeneratedSQL(BaseModel):
    """Validated structured SQL generation result. SQL remains untrusted."""

    model_config = ConfigDict(extra="forbid")

    sql: str
    dialect: SQLDialect
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    explanation: str | None = None
    confidence: SQLGenerationConfidence
    assumptions: list[str] = Field(default_factory=list)
    suggested_limit: int | None = None
    generation_version: str
    schema_truncated: bool = False


class SQLGenerationOutcome(BaseModel):
    """Generation result: either structured SQL or a clarification request."""

    model_config = ConfigDict(extra="forbid")

    generated: GeneratedSQL | None = None
    requires_clarification: bool = False
    clarification_question: str | None = None
    schema_truncated: bool = False


@dataclass(frozen=True)
class SQLGenerateParams:
    """Authorized SQL generation request for a single workspace data source."""

    workspace_id: UUID
    organization_id: UUID
    user_id: UUID
    data_source_id: UUID
    message: str
    metadata: ResolvedMetadataContext
    intent: AIIntent | None = None
    plan_summary: str | None = None
    data_source_name: str | None = None
    session_id: UUID | None = None
    conversation_context: str | None = None


@dataclass(frozen=True)
class SQLGenerationResult:
    outcome: SQLGenerationOutcome
    data_source_id: UUID
    workspace_id: UUID
    organization_id: UUID
