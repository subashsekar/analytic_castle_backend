"""PostgreSQL MCP catalog and sample request models."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.enums import DataSourceTableType


class MCPDataSourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    page: int = Field(default=1, ge=1)
    page_size: int | None = Field(default=None, ge=1)


class MCPListTablesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    schema_id: UUID | None = None
    search: str | None = None
    table_type: DataSourceTableType | None = None
    page: int = Field(default=1, ge=1)
    page_size: int | None = Field(default=None, ge=1)


class MCPTableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    table_id: UUID
    page: int = Field(default=1, ge=1)
    page_size: int | None = Field(default=None, ge=1)


class MCPDescribeTableRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    table_id: UUID


class MCPSampleRowsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    table_id: UUID
    limit: int | None = Field(
        default=None,
        ge=1,
        le=settings.SAMPLE_DATA_MAX_LIMIT,
    )
