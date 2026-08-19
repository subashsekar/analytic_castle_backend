"""In-memory records for metadata catalog reads.

These structures are not ORM models. They are mapped to HTTP schemas by routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from app.enums import DataSourceRelationshipType, DataSourceTableType

T = TypeVar("T")


@dataclass(frozen=True)
class MetadataPage(Generic[T]):
    items: tuple[T, ...]
    page: int
    page_size: int
    total: int


@dataclass(frozen=True)
class SchemaRecord:
    id: UUID
    data_source_id: UUID
    name: str
    table_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class TableRecord:
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


@dataclass(frozen=True)
class ColumnRecord:
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


@dataclass(frozen=True)
class RelationshipRecord:
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
