"""Compact resolved metadata context for the AI analyst.

These structures identify existing Phase 4 catalog records. They are not SQL,
query results, or invented schema.
"""

from __future__ import annotations

import enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.enums import DataSourceRelationshipType


class MetadataMatchReason(str, enum.Enum):
    EXACT = "exact"
    PREFIX = "prefix"
    CONTAINS = "contains"
    DESCRIPTION = "description"


class MetadataPrimaryKey(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_id: UUID
    column_name: str


class MetadataTableCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: UUID
    schema_name: str
    table_name: str
    match_reason: MetadataMatchReason
    relevance_score: int
    primary_key_columns: list[MetadataPrimaryKey] = Field(default_factory=list)


class MetadataColumnCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_id: UUID
    table_id: UUID
    schema_name: str
    table_name: str
    column_name: str
    data_type: str
    is_primary_key: bool = False
    match_reason: MetadataMatchReason
    relevance_score: int


class MetadataRelationshipCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship_id: UUID
    source_schema: str
    source_table: str
    source_column: str
    source_table_id: UUID
    source_column_id: UUID
    target_schema: str
    target_table: str
    target_column: str
    target_table_id: UUID
    target_column_id: UUID
    relationship_type: DataSourceRelationshipType
    constraint_name: str | None = None


class ConceptColumnResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested: str
    resolved: bool
    ambiguous: bool = False
    candidates: list[MetadataColumnCandidate] = Field(default_factory=list)


class ResolvedMetadataContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    tables: list[MetadataTableCandidate] = Field(default_factory=list)
    columns: list[MetadataColumnCandidate] = Field(default_factory=list)
    relationships: list[MetadataRelationshipCandidate] = Field(default_factory=list)
    resolved_metrics: list[ConceptColumnResolution] = Field(default_factory=list)
    resolved_dimensions: list[ConceptColumnResolution] = Field(default_factory=list)
    resolved_filters: list[ConceptColumnResolution] = Field(default_factory=list)
    resolved_time_columns: list[MetadataColumnCandidate] = Field(default_factory=list)
    unresolved_concepts: list[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None


def empty_resolved_context(data_source_id: UUID) -> ResolvedMetadataContext:
    return ResolvedMetadataContext(data_source_id=data_source_id)
