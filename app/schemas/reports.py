from __future__ import annotations

from typing import Any
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.charts import ChartSpec


class ReportCreate(BaseModel):
    """Input for generating and persisting a user report (Phase 9.4)."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=32_000)
    summary: str = Field(min_length=1, max_length=32_000)
    key_findings: list[str] = Field(default_factory=list)
    chart_specifications: list[ChartSpec] = Field(default_factory=list)
    supporting_data: dict[str, Any] | None = None


class ReportRead(BaseModel):
    """Full persisted report payload."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID

    question: str
    summary: str
    key_findings: list[str]
    chart_specifications: list[ChartSpec]
    supporting_data: dict[str, Any] | None

    created_at: datetime
    updated_at: datetime


class ReportSummary(BaseModel):
    """Compact list representation of a persisted report."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    id: UUID
    data_source_id: UUID
    question: str
    summary: str
    created_at: datetime


class ReportFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID | None = None


class ReportList(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ReportSummary]
    total: int
    page: int
    page_size: int

