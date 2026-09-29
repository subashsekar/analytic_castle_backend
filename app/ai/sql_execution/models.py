"""Typed models for authorized SQL query execution via MCP."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_validation.models import ValidatedSQL


class SQLExecutionStatus(str, enum.Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class SQLExecutionResult(BaseModel):
    """Safe, MCP-serialized query result with execution metadata."""

    model_config = ConfigDict(extra="forbid")

    status: SQLExecutionStatus
    columns: list[str] = Field(default_factory=list)
    rows: list[list[object]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: float = 0.0
    applied_row_limit: int | None = None
    sql_char_count: int | None = None
    referenced_table_count: int = 0
    referenced_column_count: int = 0


@dataclass(frozen=True)
class SQLExecuteParams:
    """Authorized SQL execution request for a single workspace data source."""

    workspace_id: UUID
    organization_id: UUID
    user_id: UUID
    data_source_id: UUID
    sql: str
    metadata: ResolvedMetadataContext
    limit: int | None = None
    session_id: UUID | None = None


@dataclass(frozen=True)
class SQLExecutionServiceResult:
    result: SQLExecutionResult
    validated: ValidatedSQL
    data_source_id: UUID
    workspace_id: UUID
    organization_id: UUID
