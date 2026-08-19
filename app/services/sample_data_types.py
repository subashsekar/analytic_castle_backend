"""In-memory results of safe sample-data retrieval.

These structures are not ORM models and are not HTTP schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.enums import ColumnSensitivity, DataSourceTableType


@dataclass(frozen=True)
class SampleColumn:
    name: str
    data_type: str
    sensitivity: ColumnSensitivity
    masked: bool


@dataclass(frozen=True)
class SampleDataResult:
    data_source_id: UUID
    table_id: UUID
    schema_name: str
    table_name: str
    table_type: DataSourceTableType
    columns: tuple[SampleColumn, ...]
    rows: tuple[dict[str, object], ...]
    row_limit: int
    truncated_columns: bool

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)
