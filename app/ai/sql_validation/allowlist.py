"""Build schema allowlists from authorized Phase 4 metadata."""

from __future__ import annotations

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_validation.models import SchemaAllowlist


def build_schema_allowlist(metadata: ResolvedMetadataContext) -> SchemaAllowlist:
    """Derive case-insensitive table/column allowlists from resolved metadata.

    Tables come from ``metadata.tables`` plus tables implied by explicit column
    and time-column candidates. Relationships may only add columns to tables
    already authorized — they cannot invent new tables.
    """
    tables: set[str] = set()
    columns: set[str] = set()
    schemas: set[str] = set()
    columns_by_table: dict[str, set[str]] = {}

    for table in metadata.tables:
        schema = table.schema_name.strip().lower()
        name = table.table_name.strip().lower()
        if not schema or not name:
            continue
        key = f"{schema}.{name}"
        tables.add(key)
        schemas.add(schema)
        columns_by_table.setdefault(key, set())

    def _add_column(
        schema_name: str,
        table_name: str,
        column_name: str,
        *,
        allow_new_table: bool,
    ) -> None:
        schema = schema_name.strip().lower()
        table = table_name.strip().lower()
        column = column_name.strip().lower()
        if not schema or not table or not column:
            return
        table_key = f"{schema}.{table}"
        if table_key not in tables:
            if not allow_new_table:
                return
            tables.add(table_key)
            schemas.add(schema)
            columns_by_table.setdefault(table_key, set())
        columns.add(f"{table_key}.{column}")
        columns_by_table.setdefault(table_key, set()).add(column)

    for column in metadata.columns:
        _add_column(
            column.schema_name,
            column.table_name,
            column.column_name,
            allow_new_table=True,
        )

    for column in metadata.resolved_time_columns:
        _add_column(
            column.schema_name,
            column.table_name,
            column.column_name,
            allow_new_table=True,
        )

    for group in (
        metadata.resolved_metrics,
        metadata.resolved_dimensions,
        metadata.resolved_filters,
    ):
        for item in group:
            for candidate in item.candidates:
                _add_column(
                    candidate.schema_name,
                    candidate.table_name,
                    candidate.column_name,
                    allow_new_table=True,
                )

    for relationship in metadata.relationships:
        _add_column(
            relationship.source_schema,
            relationship.source_table,
            relationship.source_column,
            allow_new_table=False,
        )
        _add_column(
            relationship.target_schema,
            relationship.target_table,
            relationship.target_column,
            allow_new_table=False,
        )

    frozen_by_table = {
        key: frozenset(values) for key, values in columns_by_table.items()
    }
    column_names = frozenset(
        name for names in frozen_by_table.values() for name in names
    )
    return SchemaAllowlist(
        tables=frozenset(tables),
        columns=frozenset(columns),
        schemas=frozenset(schemas),
        columns_by_table=frozen_by_table,
        column_names=column_names,
    )


def has_usable_allowlist(allowlist: SchemaAllowlist) -> bool:
    return bool(allowlist.tables) or bool(allowlist.columns)
