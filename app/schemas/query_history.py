"""Typed request/response models for query history."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.enums import QueryHistoryStatus


class QueryHistoryFilter(BaseModel):
    """Filter parameters for query history listing."""

    model_config = ConfigDict(extra="forbid")

    status: QueryHistoryStatus | None = None
    data_source_id: UUID | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_duration_ms: float | None = None
    max_duration_ms: float | None = None


class QueryHistoryCreate(BaseModel):
    """Input for creating a query history entry."""

    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None = None
    generated_sql: str = Field(min_length=1, max_length=4000)
    validated_sql: str | None = Field(default=None, max_length=4000)
    corrected_sql: str | None = Field(default=None, max_length=4000)
    status: QueryHistoryStatus
    duration_ms: float = Field(default=0.0, ge=0.0)
    result_metadata: dict[str, Any] | None = None
    error_metadata: dict[str, Any] | None = None


class QueryHistoryRead(BaseModel):
    """Full query history entry for detail responses."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None
    generated_sql: str
    validated_sql: str | None
    corrected_sql: str | None
    status: QueryHistoryStatus
    duration_ms: float
    result_metadata: dict[str, Any] | None
    error_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class QueryHistorySummary(BaseModel):
    """Compact list representation of a query history entry."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    data_source_id: UUID | None
    generated_sql: str
    status: QueryHistoryStatus
    duration_ms: float
    created_at: datetime


class QueryHistoryList(BaseModel):
    """Paginated query history list response."""

    model_config = ConfigDict(extra="forbid")

    items: list[QueryHistorySummary]
    total: int
    page: int
    page_size: int
