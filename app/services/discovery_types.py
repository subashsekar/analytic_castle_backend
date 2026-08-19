"""In-memory results of customer-database metadata discovery.

These structures are not ORM models and are not persisted. Phase 4.3 maps
them onto the Phase 4.1 metadata tables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.enums import DataSourceRelationshipType, DataSourceTableType


@dataclass(frozen=True)
class SchemaFilter:
    """Optional allow/deny lists applied after system-schema exclusion."""

    include_schemas: frozenset[str] | None = None
    exclude_schemas: frozenset[str] | None = None


@dataclass(frozen=True)
class DiscoveryLimits:
    max_schemas: int = 100
    max_tables: int = 2_000
    max_columns: int = 20_000
    max_relationships: int = 5_000

    def __post_init__(self) -> None:
        for name in (
            "max_schemas",
            "max_tables",
            "max_columns",
            "max_relationships",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DiscoveredSchema:
    name: str


@dataclass(frozen=True)
class DiscoveredTable:
    schema_name: str
    table_name: str
    table_type: DataSourceTableType


@dataclass(frozen=True)
class DiscoveredColumn:
    schema_name: str
    table_name: str
    column_name: str
    ordinal_position: int
    data_type: str
    database_type: str
    is_nullable: bool
    default_value: str | None
    is_primary_key: bool
    is_unique: bool


@dataclass(frozen=True)
class DiscoveredRelationship:
    """One column pair from a foreign-key constraint.

    Composite foreign keys produce one row per column pair, sharing
    ``constraint_name``. ``ordinal_position`` is the pair order within that
    constraint. Cardinality is inferred from uniqueness metadata, not names.
    """

    source_schema: str
    source_table: str
    source_column: str
    target_schema: str
    target_table: str
    target_column: str
    relationship_type: DataSourceRelationshipType
    constraint_name: str | None
    ordinal_position: int
    constraint_column_count: int


@dataclass(frozen=True)
class DiscoveryResult:
    schemas: tuple[DiscoveredSchema, ...]
    tables: tuple[DiscoveredTable, ...]
    columns: tuple[DiscoveredColumn, ...]
    relationships: tuple[DiscoveredRelationship, ...]


class MetadataQueryExecutor(Protocol):
    """Read-only parameterized query surface used by discovery."""

    async def _fetch_all(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]: ...
