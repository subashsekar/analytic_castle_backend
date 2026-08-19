"""Discover PostgreSQL schemas, tables, columns, and relationships.

Reads metadata from a connected customer database through the PostgreSQL
connector. Does not persist metadata and does not use the application
database session.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from app.connectors.exceptions import ConnectorError
from app.enums import DataSourceRelationshipType, DataSourceTableType
from app.services.discovery_exceptions import (
    ColumnDiscoveryError,
    MetadataDiscoveryLimitError,
    RelationshipDiscoveryError,
    SchemaDiscoveryError,
    TableDiscoveryError,
)
from app.services.discovery_types import (
    DiscoveredColumn,
    DiscoveredRelationship,
    DiscoveredSchema,
    DiscoveredTable,
    DiscoveryLimits,
    DiscoveryResult,
    MetadataQueryExecutor,
    SchemaFilter,
)
from app.services.postgresql_types import normalize_postgres_type
from app.services.schema_filters import EXCLUDED_SYSTEM_SCHEMAS, should_include_schema

logger = logging.getLogger(__name__)

_TABLE_TYPE_MAP = {
    "BASE TABLE": DataSourceTableType.TABLE,
    "VIEW": DataSourceTableType.VIEW,
}

# Bound LIKE values. Literal 'pg\_%' / ESCAPE '\' cannot appear in the SQL
# text: psycopg treats '%' as a placeholder marker.
_PG_SYSTEM_SCHEMA_LIKE = r"pg\_%"
_LIKE_ESCAPE = "\\"

_SCHEMAS_SQL = """
SELECT schema_name
FROM information_schema.schemata
WHERE catalog_name = current_database()
  AND schema_name <> ALL(%(excluded)s)
  AND schema_name NOT LIKE %(pg_prefix)s ESCAPE %(like_escape)s
ORDER BY schema_name
LIMIT %(row_limit)s
"""

_SCHEMAS_SQL_INCLUDED = """
SELECT schema_name
FROM information_schema.schemata
WHERE catalog_name = current_database()
  AND schema_name <> ALL(%(excluded)s)
  AND schema_name NOT LIKE %(pg_prefix)s ESCAPE %(like_escape)s
  AND schema_name = ANY(%(included)s)
ORDER BY schema_name
LIMIT %(row_limit)s
"""

_TABLES_SQL = """
SELECT table_schema, table_name, table_type
FROM information_schema.tables
WHERE table_catalog = current_database()
  AND table_schema = ANY(%(schemas)s)
  AND table_type = ANY(%(table_types)s)
ORDER BY table_schema, table_name
LIMIT %(row_limit)s
"""

_COLUMNS_SQL = """
SELECT
    table_schema,
    table_name,
    column_name,
    ordinal_position,
    data_type,
    udt_name,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_catalog = current_database()
  AND table_schema = ANY(%(schemas)s)
  AND (table_schema, table_name) IN (
      SELECT t.table_schema, t.table_name
      FROM unnest(
          CAST(%(table_schemas)s AS text[]),
          CAST(%(table_names)s AS text[])
      ) AS t(table_schema, table_name)
  )
ORDER BY table_schema, table_name, ordinal_position
LIMIT %(row_limit)s
"""

_KEY_CONSTRAINTS_SQL = """
SELECT
    nsp.nspname AS table_schema,
    rel.relname AS table_name,
    con.conname AS constraint_name,
    CASE con.contype
        WHEN 'p' THEN 'PRIMARY KEY'
        WHEN 'u' THEN 'UNIQUE'
    END AS constraint_type,
    att.attname AS column_name,
    ord.ordinality AS ordinal_position
FROM pg_catalog.pg_constraint AS con
JOIN pg_catalog.pg_class AS rel
  ON rel.oid = con.conrelid
JOIN pg_catalog.pg_namespace AS nsp
  ON nsp.oid = rel.relnamespace
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS ord(attnum, ordinality)
  ON TRUE
JOIN pg_catalog.pg_attribute AS att
  ON att.attrelid = con.conrelid
 AND att.attnum = ord.attnum
WHERE con.contype IN ('p', 'u')
  AND NOT att.attisdropped
  AND att.attnum > 0
  AND nsp.nspname = ANY(%(schemas)s)
  AND (nsp.nspname, rel.relname) IN (
      SELECT t.table_schema, t.table_name
      FROM unnest(
          CAST(%(table_schemas)s AS text[]),
          CAST(%(table_names)s AS text[])
      ) AS t(table_schema, table_name)
  )
ORDER BY
    nsp.nspname,
    rel.relname,
    con.conname,
    ord.ordinality
LIMIT %(row_limit)s
"""

_FOREIGN_KEYS_SQL = """
SELECT
    con.conname AS constraint_name,
    src_nsp.nspname AS source_schema,
    src_rel.relname AS source_table,
    src_att.attname AS source_column,
    src_ord.ordinality AS ordinal_position,
    tgt_nsp.nspname AS target_schema,
    tgt_rel.relname AS target_table,
    tgt_att.attname AS target_column
FROM pg_catalog.pg_constraint AS con
JOIN pg_catalog.pg_class AS src_rel
  ON src_rel.oid = con.conrelid
JOIN pg_catalog.pg_namespace AS src_nsp
  ON src_nsp.oid = src_rel.relnamespace
JOIN pg_catalog.pg_class AS tgt_rel
  ON tgt_rel.oid = con.confrelid
JOIN pg_catalog.pg_namespace AS tgt_nsp
  ON tgt_nsp.oid = tgt_rel.relnamespace
JOIN LATERAL unnest(con.conkey) WITH ORDINALITY AS src_ord(attnum, ordinality)
  ON TRUE
JOIN LATERAL unnest(con.confkey) WITH ORDINALITY AS tgt_ord(attnum, ordinality)
  ON tgt_ord.ordinality = src_ord.ordinality
JOIN pg_catalog.pg_attribute AS src_att
  ON src_att.attrelid = con.conrelid
 AND src_att.attnum = src_ord.attnum
JOIN pg_catalog.pg_attribute AS tgt_att
  ON tgt_att.attrelid = con.confrelid
 AND tgt_att.attnum = tgt_ord.attnum
WHERE con.contype = 'f'
  AND con.confrelid <> 0
  AND NOT src_att.attisdropped
  AND NOT tgt_att.attisdropped
  AND src_att.attnum > 0
  AND tgt_att.attnum > 0
  AND src_nsp.nspname = ANY(%(schemas)s)
  AND (src_nsp.nspname, src_rel.relname) IN (
      SELECT t.table_schema, t.table_name
      FROM unnest(
          CAST(%(table_schemas)s AS text[]),
          CAST(%(table_names)s AS text[])
      ) AS t(table_schema, table_name)
  )
  AND (tgt_nsp.nspname, tgt_rel.relname) IN (
      SELECT t.table_schema, t.table_name
      FROM unnest(
          CAST(%(table_schemas)s AS text[]),
          CAST(%(table_names)s AS text[])
      ) AS t(table_schema, table_name)
  )
ORDER BY
    con.conname,
    src_nsp.nspname,
    src_rel.relname,
    src_ord.ordinality
LIMIT %(row_limit)s
"""

_TABLE_TYPES = ("BASE TABLE", "VIEW")


class PostgreSQLMetadataDiscoveryService:
    """Inspect a connected PostgreSQL database and return metadata.

    Requires an already-connected connector. Does not connect, disconnect,
    encrypt credentials, or write to the application database.
    """

    def __init__(
        self,
        connector: MetadataQueryExecutor,
        *,
        limits: DiscoveryLimits | None = None,
        schema_filter: SchemaFilter | None = None,
        data_source_id: UUID | None = None,
    ) -> None:
        self._connector = connector
        self._limits = limits or DiscoveryLimits()
        self._schema_filter = schema_filter or SchemaFilter()
        self._data_source_id = data_source_id

    async def discover(self) -> DiscoveryResult:
        schemas = await self.discover_schemas()
        schema_names = tuple(schema.name for schema in schemas)
        tables = await self.discover_tables(schema_names)
        table_ids = _table_ids(tables)
        constraints = await self._discover_key_constraints(
            schema_names,
            table_ids=table_ids,
        )
        columns = await self.discover_columns(
            schema_names,
            key_constraints=constraints,
            tables=tables,
        )
        relationships = await self.discover_relationships(
            schema_names,
            key_constraints=constraints,
            tables=tables,
        )
        return DiscoveryResult(
            schemas=tuple(schemas),
            tables=tuple(tables),
            columns=tuple(columns),
            relationships=tuple(relationships),
        )

    async def discover_schemas(self) -> list[DiscoveredSchema]:
        excluded = set(EXCLUDED_SYSTEM_SCHEMAS)
        if self._schema_filter.exclude_schemas is not None:
            excluded.update(self._schema_filter.exclude_schemas)
        params: dict[str, object] = {
            "excluded": list(excluded),
            "pg_prefix": _PG_SYSTEM_SCHEMA_LIKE,
            "like_escape": _LIKE_ESCAPE,
            "row_limit": self._limits.max_schemas + 1,
        }
        query = _SCHEMAS_SQL
        if self._schema_filter.include_schemas is not None:
            query = _SCHEMAS_SQL_INCLUDED
            params["included"] = list(self._schema_filter.include_schemas)
        rows = await self._fetch(
            query,
            params,
            error_cls=SchemaDiscoveryError,
            message="Unable to discover database schemas",
        )
        if len(rows) > self._limits.max_schemas:
            raise MetadataDiscoveryLimitError("Schema discovery limit exceeded")
        discovered: list[DiscoveredSchema] = []
        for row in rows:
            name = _as_str(row.get("schema_name"))
            if not should_include_schema(name, self._schema_filter):
                continue
            discovered.append(DiscoveredSchema(name=name))
        discovered.sort(key=lambda item: item.name)
        return discovered

    async def discover_tables(
        self,
        schema_names: Sequence[str] | None = None,
    ) -> list[DiscoveredTable]:
        names = await self._schema_names(schema_names)
        if not names:
            return []
        rows = await self._fetch(
            _TABLES_SQL,
            {
                "schemas": list(names),
                "table_types": list(_TABLE_TYPES),
                "row_limit": self._limits.max_tables + 1,
            },
            error_cls=TableDiscoveryError,
            message="Unable to discover database tables",
        )
        if len(rows) > self._limits.max_tables:
            raise MetadataDiscoveryLimitError("Table discovery limit exceeded")
        discovered: list[DiscoveredTable] = []
        allowed = set(names)
        for row in rows:
            schema_name = _as_str(row.get("table_schema"))
            table_name = _as_str(row.get("table_name"))
            table_type = _TABLE_TYPE_MAP.get(_as_str(row.get("table_type")))
            if schema_name not in allowed or not table_name or table_type is None:
                continue
            discovered.append(
                DiscoveredTable(
                    schema_name=schema_name,
                    table_name=table_name,
                    table_type=table_type,
                )
            )
        discovered.sort(key=lambda item: (item.schema_name, item.table_name))
        return discovered

    async def discover_columns(
        self,
        schema_names: Sequence[str] | None = None,
        *,
        key_constraints: Mapping[tuple[str, str], _KeyIndex] | None = None,
        tables: Sequence[DiscoveredTable] | None = None,
    ) -> list[DiscoveredColumn]:
        names = await self._schema_names(schema_names)
        if not names:
            return []
        table_ids = await self._table_ids_for(names, tables)
        if not table_ids:
            return []
        rows = await self._fetch(
            _COLUMNS_SQL,
            {
                "schemas": list(names),
                **_table_id_params(table_ids),
                "row_limit": self._limits.max_columns + 1,
            },
            error_cls=ColumnDiscoveryError,
            message="Unable to discover database columns",
        )
        if len(rows) > self._limits.max_columns:
            raise MetadataDiscoveryLimitError("Column discovery limit exceeded")
        index = key_constraints
        if index is None:
            index = await self._discover_key_constraints(names, table_ids=table_ids)
        discovered: list[DiscoveredColumn] = []
        for row in rows:
            schema_name = _as_str(row.get("table_schema"))
            table_name = _as_str(row.get("table_name"))
            column_name = _as_str(row.get("column_name"))
            if (schema_name, table_name) not in table_ids or not column_name:
                continue
            ordinal_position = _as_positive_ordinal(row.get("ordinal_position"))
            if ordinal_position is None:
                continue
            database_type = _native_type(row)
            keys = index.get((schema_name, table_name), _KeyIndex())
            discovered.append(
                DiscoveredColumn(
                    schema_name=schema_name,
                    table_name=table_name,
                    column_name=column_name,
                    ordinal_position=ordinal_position,
                    data_type=normalize_postgres_type(database_type),
                    database_type=database_type,
                    is_nullable=_as_yes_no(row.get("is_nullable")),
                    default_value=_as_optional_str(row.get("column_default")),
                    is_primary_key=column_name in keys.primary_key_columns,
                    is_unique=column_name in keys.single_unique_columns,
                )
            )
        discovered.sort(
            key=lambda item: (
                item.schema_name,
                item.table_name,
                item.ordinal_position,
                item.column_name,
            )
        )
        return discovered

    async def discover_relationships(
        self,
        schema_names: Sequence[str] | None = None,
        *,
        key_constraints: Mapping[tuple[str, str], _KeyIndex] | None = None,
        tables: Sequence[DiscoveredTable] | None = None,
    ) -> list[DiscoveredRelationship]:
        names = await self._schema_names(schema_names)
        if not names:
            return []
        known_tables = await self._table_ids_for(names, tables)
        if not known_tables:
            return []
        rows = await self._fetch(
            _FOREIGN_KEYS_SQL,
            {
                "schemas": list(names),
                **_table_id_params(known_tables),
                "row_limit": self._limits.max_relationships + 1,
            },
            error_cls=RelationshipDiscoveryError,
            message="Unable to discover database relationships",
        )
        if len(rows) > self._limits.max_relationships:
            raise MetadataDiscoveryLimitError("Relationship discovery limit exceeded")
        index = key_constraints
        if index is None:
            index = await self._discover_key_constraints(
                names,
                table_ids=known_tables,
            )

        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for row in rows:
            source_schema = _as_str(row.get("source_schema"))
            source_table = _as_str(row.get("source_table"))
            target_schema = _as_str(row.get("target_schema"))
            target_table = _as_str(row.get("target_table"))
            constraint_name = _as_optional_str(row.get("constraint_name")) or ""
            if (source_schema, source_table) not in known_tables:
                continue
            if (target_schema, target_table) not in known_tables:
                continue
            grouped[
                (
                    constraint_name,
                    source_schema,
                    source_table,
                    target_schema,
                    target_table,
                )
            ].append(row)

        discovered: list[DiscoveredRelationship] = []
        for (
            constraint_name,
            source_schema,
            source_table,
            target_schema,
            target_table,
        ), members in grouped.items():
            members.sort(key=lambda item: _as_int(item.get("ordinal_position")))
            source_columns = tuple(
                _as_str(item.get("source_column")) for item in members
            )
            relationship_type = _infer_relationship_type(
                source_columns,
                index.get((source_schema, source_table), _KeyIndex()),
            )
            column_count = len(members)
            for position, member in enumerate(members, start=1):
                discovered.append(
                    DiscoveredRelationship(
                        source_schema=source_schema,
                        source_table=source_table,
                        source_column=_as_str(member.get("source_column")),
                        target_schema=_as_str(member.get("target_schema")),
                        target_table=target_table,
                        target_column=_as_str(member.get("target_column")),
                        relationship_type=relationship_type,
                        constraint_name=constraint_name or None,
                        ordinal_position=position,
                        constraint_column_count=column_count,
                    )
                )
        discovered.sort(
            key=lambda item: (
                item.constraint_name or "",
                item.source_schema,
                item.source_table,
                item.ordinal_position,
                item.source_column,
            )
        )
        return discovered

    async def _discover_key_constraints(
        self,
        schema_names: Sequence[str],
        *,
        table_ids: set[tuple[str, str]] | None = None,
    ) -> dict[tuple[str, str], _KeyIndex]:
        if not schema_names:
            return {}
        resolved_ids = table_ids
        if resolved_ids is None:
            resolved_ids = await self._table_ids_for(schema_names, None)
        if not resolved_ids:
            return {}
        rows = await self._fetch(
            _KEY_CONSTRAINTS_SQL,
            {
                "schemas": list(schema_names),
                **_table_id_params(resolved_ids),
                "row_limit": self._limits.max_columns + 1,
            },
            error_cls=ColumnDiscoveryError,
            message="Unable to discover database columns",
        )
        if len(rows) > self._limits.max_columns:
            raise MetadataDiscoveryLimitError("Column discovery limit exceeded")
        grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for row in rows:
            schema_name = _as_str(row.get("table_schema"))
            table_name = _as_str(row.get("table_name"))
            if (schema_name, table_name) not in resolved_ids:
                continue
            constraint_name = _as_str(row.get("constraint_name"))
            constraint_type = _as_str(row.get("constraint_type"))
            column_name = _as_str(row.get("column_name"))
            if not column_name:
                continue
            grouped[(schema_name, table_name, constraint_name, constraint_type)].append(
                column_name
            )

        result: dict[tuple[str, str], _KeyIndex] = defaultdict(_KeyIndex)
        for (schema_name, table_name, _, constraint_type), columns in grouped.items():
            key = result[(schema_name, table_name)]
            column_set = frozenset(columns)
            if constraint_type == "PRIMARY KEY":
                key.primary_key_columns.update(column_set)
                key.unique_column_sets.append(column_set)
                if len(column_set) == 1:
                    key.single_unique_columns.update(column_set)
            elif constraint_type == "UNIQUE":
                key.unique_column_sets.append(column_set)
                if len(column_set) == 1:
                    key.single_unique_columns.update(column_set)
        return dict(result)

    async def _schema_names(
        self,
        schema_names: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if schema_names is not None:
            return tuple(schema_names)
        return tuple(schema.name for schema in await self.discover_schemas())

    async def _table_ids_for(
        self,
        schema_names: Sequence[str],
        tables: Sequence[DiscoveredTable] | None,
    ) -> set[tuple[str, str]]:
        if tables is not None:
            return _table_ids(tables)
        discovered = await self.discover_tables(schema_names)
        return _table_ids(discovered)

    async def _fetch(
        self,
        query: str,
        params: Mapping[str, object],
        *,
        error_cls: type[Exception],
        message: str,
    ) -> list[dict[str, Any]]:
        try:
            return await self._connector._fetch_all(query, params)
        except ConnectorError as exc:
            logger.warning(
                "Metadata discovery query failed data_source_id=%s error_type=%s",
                self._data_source_id,
                type(exc).__name__,
            )
            raise error_cls(message) from exc
        except (OSError, TimeoutError) as exc:
            logger.warning(
                "Metadata discovery query failed data_source_id=%s error_type=%s",
                self._data_source_id,
                type(exc).__name__,
            )
            raise error_cls(message) from exc


def _table_ids(tables: Sequence[DiscoveredTable]) -> set[tuple[str, str]]:
    return {(table.schema_name, table.table_name) for table in tables}


def _table_id_params(table_ids: set[tuple[str, str]]) -> dict[str, list[str]]:
    pairs = sorted(table_ids)
    return {
        "table_schemas": [schema for schema, _name in pairs],
        "table_names": [name for _schema, name in pairs],
    }


class _KeyIndex:
    def __init__(self) -> None:
        self.primary_key_columns: set[str] = set()
        self.single_unique_columns: set[str] = set()
        self.unique_column_sets: list[frozenset[str]] = []


def _infer_relationship_type(
    source_columns: Sequence[str],
    source_keys: _KeyIndex,
) -> DataSourceRelationshipType:
    """Infer cardinality from uniqueness of the foreign-key source columns.

    MANY_TO_ONE is the safe default when uniqueness cannot be proven.
    MANY_TO_MANY is not inferred from a single foreign key. ONE_TO_MANY is
    the inverse of MANY_TO_ONE and is not emitted as a separate row.
    """
    source_set = frozenset(column for column in source_columns if column)
    if source_set and any(
        unique_set <= source_set for unique_set in source_keys.unique_column_sets
    ):
        return DataSourceRelationshipType.ONE_TO_ONE
    return DataSourceRelationshipType.MANY_TO_ONE


def _native_type(row: Mapping[str, Any]) -> str:
    data_type = _as_str(row.get("data_type"))
    udt_name = _as_str(row.get("udt_name"))
    if data_type in {"USER-DEFINED", "ARRAY"}:
        return udt_name or data_type.lower()
    return data_type or udt_name


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return default


def _as_positive_ordinal(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def _as_yes_no(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    return str(value).strip().upper() in {"YES", "Y", "TRUE", "T", "1"}
