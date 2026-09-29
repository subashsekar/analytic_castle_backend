"""Request/response models for the SQL analyst API (Phase 7 chapters 7.1-7.5).

Requests carry the ``metadata_context`` returned by ``POST /api/v1/ai/chat``.
Every catalog id and name in it is re-authorized against the database before
use, so the client never widens its own schema access.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_correction.models import SQLCorrectionOutcome
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.sql_generation.models import SQLGenerationOutcome
from app.ai.sql_validation.models import SQLValidationResult, SQLValidationViolation
from app.core.config import settings


def _strip_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


class _SQLDataSourceRequest(BaseModel):
    """Shared scope for every SQL analyst request."""

    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    metadata_context: ResolvedMetadataContext
    session_id: UUID | None = None


class _SQLTextRequest(_SQLDataSourceRequest):
    """Request carrying untrusted SQL text for validation, execution, or repair."""

    sql: str = Field(min_length=1, max_length=100_000)

    @field_validator("sql", mode="before")
    @classmethod
    def strip_and_limit_sql(cls, value: Any) -> Any:
        value = _strip_text(value)
        if not isinstance(value, str):
            return value
        max_chars = settings.AI_SQL_MAX_SQL_CHARS
        if len(value) > max_chars:
            raise ValueError(f"SQL exceeds maximum length of {max_chars} characters")
        return value


class SQLGenerateRequest(_SQLDataSourceRequest):
    """Ask the analyst to draft read-only SQL for a natural language message."""

    message: str = Field(min_length=1, max_length=32_000)
    plan_summary: str | None = Field(default=None, max_length=8_000)

    @field_validator("message", mode="before")
    @classmethod
    def strip_and_limit_message(cls, value: Any) -> Any:
        value = _strip_text(value)
        if not isinstance(value, str):
            return value
        max_chars = settings.AI_MAX_MESSAGE_CHARS
        if len(value) > max_chars:
            raise ValueError(
                f"Message exceeds maximum length of {max_chars} characters"
            )
        return value


class SQLValidateRequest(_SQLTextRequest):
    """Check untrusted SQL against read-only rules and the authorized catalog."""


class SQLExecuteRequest(_SQLTextRequest):
    """Execute validated SQL through the PostgreSQL MCP query tool."""

    limit: int | None = Field(default=None, ge=1)


class SQLCorrectRequest(_SQLTextRequest):
    """Repair SQL that failed validation or execution, optionally re-running it."""

    message: str | None = Field(default=None, max_length=32_000)
    plan_summary: str | None = Field(default=None, max_length=8_000)
    violations: list[SQLValidationViolation] = Field(default_factory=list)
    execution_error_code: str | None = Field(default=None, max_length=200)
    execution_error_message: str | None = Field(default=None, max_length=2_000)
    max_attempts: int | None = Field(default=None, ge=1)
    execute: bool = False
    limit: int | None = Field(default=None, ge=1)


class SQLGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    data_source_id: UUID
    outcome: SQLGenerationOutcome


class SQLValidateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    data_source_id: UUID
    result: SQLValidationResult
    history_id: UUID | None = None


class SQLExecuteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    data_source_id: UUID
    validated_sql: str
    result: SQLExecutionResult
    history_id: UUID | None = None


class SQLCorrectResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    data_source_id: UUID
    outcome: SQLCorrectionOutcome
    execution: SQLExecutionResult | None = None
    history_id: UUID | None = None
