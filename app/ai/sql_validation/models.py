"""Typed models for SQL safety validation of untrusted generated SQL."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ai.metadata_types import ResolvedMetadataContext


class SQLValidationViolationCode(str, enum.Enum):
    EMPTY_SQL = "EMPTY_SQL"
    SQL_TOO_LONG = "SQL_TOO_LONG"
    NOT_READONLY = "NOT_READONLY"
    MULTI_STATEMENT = "MULTI_STATEMENT"
    DANGEROUS_STATEMENT = "DANGEROUS_STATEMENT"
    PARSE_ERROR = "PARSE_ERROR"
    UNKNOWN_TABLE = "UNKNOWN_TABLE"
    UNKNOWN_COLUMN = "UNKNOWN_COLUMN"
    AMBIGUOUS_COLUMN = "AMBIGUOUS_COLUMN"
    STAR_SELECTION = "STAR_SELECTION"
    CROSS_SCHEMA = "CROSS_SCHEMA"
    UNAUTHORIZED_IDENTIFIER = "UNAUTHORIZED_IDENTIFIER"
    INSUFFICIENT_SCHEMA = "INSUFFICIENT_SCHEMA"


class SQLValidationViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: SQLValidationViolationCode
    message: str
    identifier: str | None = None


class ValidatedSQL(BaseModel):
    """Normalized read-only SQL that passed schema allowlist checks."""

    model_config = ConfigDict(extra="forbid")

    sql: str
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)


class SQLValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    validated: ValidatedSQL | None = None
    violations: list[SQLValidationViolation] = Field(default_factory=list)


@dataclass(frozen=True)
class QualifiedTable:
    schema_name: str
    table_name: str

    def key(self) -> str:
        return f"{self.schema_name}.{self.table_name}".lower()


@dataclass(frozen=True)
class QualifiedColumn:
    schema_name: str | None
    table_name: str | None
    column_name: str

    def key(self) -> str | None:
        if self.schema_name is None or self.table_name is None:
            return None
        return f"{self.schema_name}.{self.table_name}.{self.column_name}".lower()


@dataclass(frozen=True)
class ExtractedReferences:
    tables: tuple[QualifiedTable, ...] = ()
    columns: tuple[QualifiedColumn, ...] = ()
    cte_names: frozenset[str] = field(default_factory=frozenset)
    derived_aliases: frozenset[str] = field(default_factory=frozenset)
    has_star_selection: bool = False


@dataclass(frozen=True)
class SchemaAllowlist:
    tables: frozenset[str]
    columns: frozenset[str]
    schemas: frozenset[str]
    columns_by_table: dict[str, frozenset[str]]
    column_names: frozenset[str]


@dataclass(frozen=True)
class SQLValidateParams:
    """Authorized SQL validation request for a single workspace data source."""

    workspace_id: UUID
    organization_id: UUID
    user_id: UUID
    data_source_id: UUID
    sql: str
    metadata: ResolvedMetadataContext
    session_id: UUID | None = None


@dataclass(frozen=True)
class SQLValidationServiceResult:
    result: SQLValidationResult
    data_source_id: UUID
    workspace_id: UUID
    organization_id: UUID
