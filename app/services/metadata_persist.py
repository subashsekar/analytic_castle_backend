"""Persist Phase 4.2 discovery results into Phase 4.1 metadata tables.

Identity is (data_source_id, schema name), (schema_id, table name),
(table_id, column name), and the discovered relationship column pair plus
constraint name. Caller owns the transaction.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    DataSourceColumn,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
)
from app.services.discovery_types import (
    DiscoveredColumn,
    DiscoveredRelationship,
    DiscoveredSchema,
    DiscoveredTable,
    DiscoveryResult,
)
from app.services.metadata_sync_exceptions import MetadataSyncValidationError
from app.services.metadata_sync_types import PersistedMetadataCounts

_SchemaKey = str
_TableKey = tuple[str, str]
_ColumnKey = tuple[str, str, str]
_RelationshipKey = tuple[str, str, str, str, str, str, str | None]


def persist_discovered_metadata(
    session: Session,
    data_source_id: UUID,
    result: DiscoveryResult,
) -> PersistedMetadataCounts:
    validate_discovery_result(result)

    existing_schemas = list(
        session.scalars(
            select(DataSourceSchema)
            .where(DataSourceSchema.data_source_id == data_source_id)
            .options(
                selectinload(DataSourceSchema.tables).selectinload(
                    DataSourceTable.columns
                )
            )
        ).all()
    )
    existing_relationships = list(
        session.scalars(
            select(DataSourceRelationship)
            .join(
                DataSourceTable,
                DataSourceRelationship.source_table_id == DataSourceTable.id,
            )
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(DataSourceSchema.data_source_id == data_source_id)
        ).all()
    )

    schemas_by_name = _upsert_schemas(
        session, data_source_id, result.schemas, existing_schemas
    )
    session.flush()

    tables_by_key, existing_tables = _upsert_tables(
        session, result.tables, existing_schemas, schemas_by_name
    )
    session.flush()

    columns_by_key = _upsert_columns(session, result.columns, tables_by_key)
    session.flush()

    _upsert_relationships(
        session,
        result.relationships,
        existing_relationships,
        columns_by_key,
    )
    session.flush()

    _delete_stale(
        session,
        existing_schemas=existing_schemas,
        existing_tables=existing_tables,
        existing_relationships=existing_relationships,
        discovered_schema_names={item.name for item in result.schemas},
        discovered_table_keys={_table_key(item) for item in result.tables},
        discovered_column_keys={_column_key(item) for item in result.columns},
        discovered_relationship_keys={
            _relationship_key(item) for item in result.relationships
        },
        columns_by_key=columns_by_key,
    )
    session.flush()

    return PersistedMetadataCounts(
        schemas=len(result.schemas),
        tables=len(result.tables),
        columns=len(result.columns),
        relationships=len(result.relationships),
    )


def validate_discovery_result(result: DiscoveryResult) -> None:
    schema_names = [_require_name(item.name) for item in result.schemas]
    if len(schema_names) != len(set(schema_names)):
        raise MetadataSyncValidationError("Discovered metadata is invalid")
    schema_set = set(schema_names)

    table_keys: list[_TableKey] = []
    for table in result.tables:
        schema_name = _require_name(table.schema_name)
        table_name = _require_name(table.table_name)
        if schema_name not in schema_set:
            raise MetadataSyncValidationError("Discovered metadata is invalid")
        table_keys.append((schema_name, table_name))
    if len(table_keys) != len(set(table_keys)):
        raise MetadataSyncValidationError("Discovered metadata is invalid")
    table_set = set(table_keys)

    column_keys: list[_ColumnKey] = []
    for column in result.columns:
        schema_name = _require_name(column.schema_name)
        table_name = _require_name(column.table_name)
        column_name = _require_name(column.column_name)
        if (schema_name, table_name) not in table_set:
            raise MetadataSyncValidationError("Discovered metadata is invalid")
        if column.ordinal_position < 1:
            raise MetadataSyncValidationError("Discovered metadata is invalid")
        if not column.data_type.strip() or not column.database_type.strip():
            raise MetadataSyncValidationError("Discovered metadata is invalid")
        column_keys.append((schema_name, table_name, column_name))
    if len(column_keys) != len(set(column_keys)):
        raise MetadataSyncValidationError("Discovered metadata is invalid")
    column_set = set(column_keys)

    relationship_keys: list[_RelationshipKey] = []
    for relation in result.relationships:
        source = (
            _require_name(relation.source_schema),
            _require_name(relation.source_table),
            _require_name(relation.source_column),
        )
        target = (
            _require_name(relation.target_schema),
            _require_name(relation.target_table),
            _require_name(relation.target_column),
        )
        if source not in column_set or target not in column_set:
            raise MetadataSyncValidationError("Discovered metadata is invalid")
        relationship_keys.append(_relationship_key(relation))
    if len(relationship_keys) != len(set(relationship_keys)):
        raise MetadataSyncValidationError("Discovered metadata is invalid")


def _require_name(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise MetadataSyncValidationError("Discovered metadata is invalid")
    return stripped


def _table_key(table: DiscoveredTable) -> _TableKey:
    return (table.schema_name, table.table_name)


def _column_key(column: DiscoveredColumn) -> _ColumnKey:
    return (column.schema_name, column.table_name, column.column_name)


def _relationship_key(relation: DiscoveredRelationship) -> _RelationshipKey:
    return (
        relation.source_schema,
        relation.source_table,
        relation.source_column,
        relation.target_schema,
        relation.target_table,
        relation.target_column,
        relation.constraint_name,
    )


def _existing_relationship_key(
    relation: DataSourceRelationship,
    columns_by_id: dict[UUID, _ColumnKey],
) -> _RelationshipKey | None:
    source = columns_by_id.get(relation.source_column_id)
    target = columns_by_id.get(relation.target_column_id)
    if source is None or target is None:
        return None
    return (
        source[0],
        source[1],
        source[2],
        target[0],
        target[1],
        target[2],
        relation.constraint_name,
    )


def _upsert_schemas(
    session: Session,
    data_source_id: UUID,
    discovered: tuple[DiscoveredSchema, ...],
    existing: list[DataSourceSchema],
) -> dict[_SchemaKey, DataSourceSchema]:
    by_name = {schema.name: schema for schema in existing}
    for item in discovered:
        current = by_name.get(item.name)
        if current is None:
            current = DataSourceSchema(data_source_id=data_source_id, name=item.name)
            session.add(current)
            by_name[item.name] = current
    return by_name


def _upsert_tables(
    session: Session,
    discovered: tuple[DiscoveredTable, ...],
    existing_schemas: list[DataSourceSchema],
    schemas_by_name: dict[_SchemaKey, DataSourceSchema],
) -> tuple[dict[_TableKey, DataSourceTable], list[DataSourceTable]]:
    existing_tables = [table for schema in existing_schemas for table in schema.tables]
    by_key = {
        (schema.name, table.name): table
        for schema in existing_schemas
        for table in schema.tables
    }
    for item in discovered:
        key = _table_key(item)
        current = by_key.get(key)
        if current is None:
            current = DataSourceTable(
                schema_id=schemas_by_name[item.schema_name].id,
                name=item.table_name,
                table_type=item.table_type,
            )
            session.add(current)
            by_key[key] = current
        elif current.table_type is not item.table_type:
            current.table_type = item.table_type
    return by_key, existing_tables


def _upsert_columns(
    session: Session,
    discovered: tuple[DiscoveredColumn, ...],
    tables_by_key: dict[_TableKey, DataSourceTable],
) -> dict[_ColumnKey, DataSourceColumn]:
    by_key: dict[_ColumnKey, DataSourceColumn] = {}
    for (schema_name, _table_name), table in tables_by_key.items():
        for column in table.columns:
            by_key[(schema_name, table.name, column.name)] = column

    for item in discovered:
        key = _column_key(item)
        current = by_key.get(key)
        table = tables_by_key[(item.schema_name, item.table_name)]
        if current is None:
            current = DataSourceColumn(
                table_id=table.id,
                name=item.column_name,
                ordinal_position=item.ordinal_position,
                data_type=item.data_type,
                database_type=item.database_type,
                is_nullable=item.is_nullable,
                default_value=item.default_value,
                is_primary_key=item.is_primary_key,
                is_unique=item.is_unique,
            )
            session.add(current)
            by_key[key] = current
            continue
        _apply_column_changes(current, item)
    return by_key


def _apply_column_changes(current: DataSourceColumn, item: DiscoveredColumn) -> None:
    if current.ordinal_position != item.ordinal_position:
        current.ordinal_position = item.ordinal_position
    if current.data_type != item.data_type:
        current.data_type = item.data_type
    if current.database_type != item.database_type:
        current.database_type = item.database_type
    if current.is_nullable is not item.is_nullable:
        current.is_nullable = item.is_nullable
    if current.default_value != item.default_value:
        current.default_value = item.default_value
    if current.is_primary_key is not item.is_primary_key:
        current.is_primary_key = item.is_primary_key
    if current.is_unique is not item.is_unique:
        current.is_unique = item.is_unique


def _upsert_relationships(
    session: Session,
    discovered: tuple[DiscoveredRelationship, ...],
    existing: list[DataSourceRelationship],
    columns_by_key: dict[_ColumnKey, DataSourceColumn],
) -> None:
    columns_by_id = {column.id: key for key, column in columns_by_key.items()}
    by_key: dict[_RelationshipKey, DataSourceRelationship] = {}
    for relation in existing:
        key = _existing_relationship_key(relation, columns_by_id)
        if key is not None:
            by_key[key] = relation

    for item in discovered:
        key = _relationship_key(item)
        source = columns_by_key[_column_endpoint(item, source=True)]
        target = columns_by_key[_column_endpoint(item, source=False)]
        current = by_key.get(key)
        if current is None:
            session.add(
                DataSourceRelationship(
                    source_table_id=source.table_id,
                    source_column_id=source.id,
                    target_table_id=target.table_id,
                    target_column_id=target.id,
                    relationship_type=item.relationship_type,
                    constraint_name=item.constraint_name,
                )
            )
            continue
        if current.relationship_type is not item.relationship_type:
            current.relationship_type = item.relationship_type


def _column_endpoint(item: DiscoveredRelationship, *, source: bool) -> _ColumnKey:
    if source:
        return (item.source_schema, item.source_table, item.source_column)
    return (item.target_schema, item.target_table, item.target_column)


def _delete_stale(
    session: Session,
    *,
    existing_schemas: list[DataSourceSchema],
    existing_tables: list[DataSourceTable],
    existing_relationships: list[DataSourceRelationship],
    discovered_schema_names: set[str],
    discovered_table_keys: set[_TableKey],
    discovered_column_keys: set[_ColumnKey],
    discovered_relationship_keys: set[_RelationshipKey],
    columns_by_key: dict[_ColumnKey, DataSourceColumn],
) -> None:
    columns_by_id = {column.id: key for key, column in columns_by_key.items()}
    schema_name_by_id = {schema.id: schema.name for schema in existing_schemas}

    for relation in existing_relationships:
        key = _existing_relationship_key(relation, columns_by_id)
        if key is None or key not in discovered_relationship_keys:
            session.delete(relation)

    for table in existing_tables:
        schema_name = schema_name_by_id.get(table.schema_id)
        if schema_name is None:
            continue
        for column in list(table.columns):
            if (schema_name, table.name, column.name) not in discovered_column_keys:
                session.delete(column)

    for table in existing_tables:
        schema_name = schema_name_by_id.get(table.schema_id)
        if schema_name is None:
            continue
        if (schema_name, table.name) not in discovered_table_keys:
            session.delete(table)

    for schema in existing_schemas:
        if schema.name not in discovered_schema_names:
            session.delete(schema)
