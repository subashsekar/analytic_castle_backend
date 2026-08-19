from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.enums import (
    ColumnSensitivity,
    DataSourceRelationshipType,
    DataSourceTableType,
    MetadataSearchType,
    MetadataSyncStatus,
)


class DataSourceSchemaRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    data_source_id: UUID
    name: str
    created_at: datetime
    updated_at: datetime


class DataSourceTableRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    schema_id: UUID
    name: str
    table_type: DataSourceTableType
    description: str | None
    created_at: datetime
    updated_at: datetime


class DataSourceColumnRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    table_id: UUID
    name: str
    ordinal_position: int
    data_type: str
    database_type: str
    is_nullable: bool
    default_value: str | None
    is_primary_key: bool
    is_unique: bool
    description: str | None
    created_at: datetime
    updated_at: datetime


class DataSourceRelationshipRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_table_id: UUID
    source_column_id: UUID
    target_table_id: UUID
    target_column_id: UUID
    relationship_type: DataSourceRelationshipType
    constraint_name: str | None
    created_at: datetime
    updated_at: datetime


class MetadataSchemaResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    data_source_id: UUID
    name: str
    table_count: int
    created_at: datetime
    updated_at: datetime


class MetadataTableResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    data_source_id: UUID
    schema_id: UUID
    schema_name: str
    name: str
    table_type: DataSourceTableType
    description: str | None
    column_count: int
    created_at: datetime
    updated_at: datetime


class MetadataColumnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    table_id: UUID
    name: str
    ordinal_position: int
    data_type: str
    database_type: str
    is_nullable: bool
    is_primary_key: bool
    is_unique: bool
    default_value: str | None
    description: str | None
    created_at: datetime
    updated_at: datetime


class MetadataRelationshipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    source_table_id: UUID
    source_schema_name: str
    source_table_name: str
    source_column_id: UUID
    source_column_name: str
    target_table_id: UUID
    target_schema_name: str
    target_table_name: str
    target_column_id: UUID
    target_column_name: str
    relationship_type: DataSourceRelationshipType
    constraint_name: str | None
    created_at: datetime
    updated_at: datetime


class MetadataPageResponse(BaseModel):
    page: int
    page_size: int
    total: int


class MetadataSchemaListResponse(MetadataPageResponse):
    items: list[MetadataSchemaResponse]


class MetadataTableListResponse(MetadataPageResponse):
    items: list[MetadataTableResponse]


class MetadataColumnListResponse(MetadataPageResponse):
    items: list[MetadataColumnResponse]


class MetadataRelationshipListResponse(MetadataPageResponse):
    items: list[MetadataRelationshipResponse]


class MetadataSearchItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    metadata_type: MetadataSearchType
    schema_name: str
    table_name: str | None
    column_name: str | None
    description: str | None


class MetadataSearchResponse(BaseModel):
    items: list[MetadataSearchItemResponse]
    total: int
    limit: int
    truncated: bool


class MetadataSyncResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: MetadataSyncStatus
    started_at: datetime | None
    completed_at: datetime | None
    schemas: int | None
    tables: int | None
    columns: int | None
    relationships: int | None
    error_message: str | None


class MetadataSyncStatusResponse(MetadataSyncResponse):
    pass


class SampleDataRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(
        default=None,
        ge=1,
        le=settings.SAMPLE_DATA_MAX_LIMIT,
        description=(
            "Maximum sample rows to return. Defaults to "
            f"{settings.SAMPLE_DATA_DEFAULT_LIMIT}. Maximum "
            f"{settings.SAMPLE_DATA_MAX_LIMIT}."
        ),
    )


class SampleColumnResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    data_type: str
    sensitivity: ColumnSensitivity
    masked: bool


class SampleDataResponse(BaseModel):
    data_source_id: UUID
    table_id: UUID
    schema_name: str
    table_name: str
    table_type: DataSourceTableType
    columns: list[SampleColumnResponse]
    rows: list[dict[str, Any]]
    row_count: int
    row_limit: int
    truncated_columns: bool
