"""Verify SQL-generation metadata IDs against the Phase 4 catalog.

Does not invent schema and does not query the customer database. Checks that
resolved metadata references exist for the authorized data source and that
caller-supplied schema/table/column names match the catalog rows for those IDs.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_generation.errors import SQLGenerationAuthorizationError
from app.db.models import (
    DataSourceColumn,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
)


def ensure_metadata_catalog_authorized(
    session: Session,
    *,
    data_source_id: UUID,
    metadata: ResolvedMetadataContext,
) -> None:
    """Reject metadata that invents catalog IDs or forges identifier names."""
    if metadata.data_source_id != data_source_id:
        raise SQLGenerationAuthorizationError(
            "Metadata context does not match the authorized data source"
        )

    table_ids = {table.table_id for table in metadata.tables}
    column_ids = {column.column_id for column in metadata.columns}
    relationship_ids = {
        relationship.relationship_id for relationship in metadata.relationships
    }
    for column in metadata.columns:
        table_ids.add(column.table_id)
    for relationship in metadata.relationships:
        table_ids.add(relationship.source_table_id)
        table_ids.add(relationship.target_table_id)
        column_ids.add(relationship.source_column_id)
        column_ids.add(relationship.target_column_id)
    for time_column in metadata.resolved_time_columns:
        table_ids.add(time_column.table_id)
        column_ids.add(time_column.column_id)

    table_names: dict[UUID, tuple[str, str]] = {}
    if table_ids:
        rows = session.execute(
            select(
                DataSourceTable.id,
                DataSourceSchema.name,
                DataSourceTable.name,
            )
            .join(
                DataSourceSchema,
                DataSourceSchema.id == DataSourceTable.schema_id,
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                DataSourceTable.id.in_(table_ids),
            )
        ).all()
        table_names = {
            table_id: (schema_name.lower(), table_name.lower())
            for table_id, schema_name, table_name in rows
        }
        if set(table_names) != table_ids:
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized table references"
            )

    column_names: dict[UUID, tuple[UUID, str, str, str]] = {}
    if column_ids:
        rows = session.execute(
            select(
                DataSourceColumn.id,
                DataSourceColumn.table_id,
                DataSourceSchema.name,
                DataSourceTable.name,
                DataSourceColumn.name,
            )
            .join(
                DataSourceTable,
                DataSourceTable.id == DataSourceColumn.table_id,
            )
            .join(
                DataSourceSchema,
                DataSourceSchema.id == DataSourceTable.schema_id,
            )
            .where(
                DataSourceSchema.data_source_id == data_source_id,
                DataSourceColumn.id.in_(column_ids),
            )
        ).all()
        column_names = {
            column_id: (
                table_id,
                schema_name.lower(),
                table_name.lower(),
                column_name.lower(),
            )
            for column_id, table_id, schema_name, table_name, column_name in rows
        }
        if set(column_names) != column_ids:
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized column references"
            )

    if relationship_ids:
        valid_relationships = set(
            session.scalars(
                select(DataSourceRelationship.id)
                .join(
                    DataSourceTable,
                    DataSourceTable.id == DataSourceRelationship.source_table_id,
                )
                .join(
                    DataSourceSchema,
                    DataSourceSchema.id == DataSourceTable.schema_id,
                )
                .where(
                    DataSourceSchema.data_source_id == data_source_id,
                    DataSourceRelationship.id.in_(relationship_ids),
                )
            ).all()
        )
        if valid_relationships != relationship_ids:
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized relationship references"
            )

    _assert_table_names_match(metadata, table_names)
    _assert_column_names_match(metadata, column_names)
    _assert_relationship_names_match(metadata, table_names, column_names)


def _norm(value: str) -> str:
    return value.strip().lower()


def _assert_table_names_match(
    metadata: ResolvedMetadataContext,
    table_names: dict[UUID, tuple[str, str]],
) -> None:
    for table in metadata.tables:
        expected = table_names.get(table.table_id)
        if expected is None:
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized table references"
            )
        if (_norm(table.schema_name), _norm(table.table_name)) != expected:
            raise SQLGenerationAuthorizationError(
                "Metadata context table names do not match the authorized catalog"
            )


def _assert_column_names_match(
    metadata: ResolvedMetadataContext,
    column_names: dict[UUID, tuple[UUID, str, str, str]],
) -> None:
    candidates = list(metadata.columns) + list(metadata.resolved_time_columns)
    for column in candidates:
        expected = column_names.get(column.column_id)
        if expected is None:
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized column references"
            )
        table_id, schema_name, table_name, column_name = expected
        if column.table_id != table_id:
            raise SQLGenerationAuthorizationError(
                "Metadata context column table binding does not match the catalog"
            )
        if (
            _norm(column.schema_name),
            _norm(column.table_name),
            _norm(column.column_name),
        ) != (schema_name, table_name, column_name):
            raise SQLGenerationAuthorizationError(
                "Metadata context column names do not match the authorized catalog"
            )


def _assert_relationship_names_match(
    metadata: ResolvedMetadataContext,
    table_names: dict[UUID, tuple[str, str]],
    column_names: dict[UUID, tuple[UUID, str, str, str]],
) -> None:
    for relationship in metadata.relationships:
        source_table = table_names.get(relationship.source_table_id)
        target_table = table_names.get(relationship.target_table_id)
        source_column = column_names.get(relationship.source_column_id)
        target_column = column_names.get(relationship.target_column_id)
        if (
            source_table is None
            or target_table is None
            or source_column is None
            or target_column is None
        ):
            raise SQLGenerationAuthorizationError(
                "Metadata context contains unauthorized relationship references"
            )
        if (
            _norm(relationship.source_schema),
            _norm(relationship.source_table),
        ) != source_table:
            raise SQLGenerationAuthorizationError(
                "Metadata context relationship names do not match the authorized catalog"
            )
        if (
            _norm(relationship.target_schema),
            _norm(relationship.target_table),
        ) != target_table:
            raise SQLGenerationAuthorizationError(
                "Metadata context relationship names do not match the authorized catalog"
            )
        if source_column[0] != relationship.source_table_id or (
            _norm(relationship.source_schema),
            _norm(relationship.source_table),
            _norm(relationship.source_column),
        ) != source_column[1:]:
            raise SQLGenerationAuthorizationError(
                "Metadata context relationship names do not match the authorized catalog"
            )
        if target_column[0] != relationship.target_table_id or (
            _norm(relationship.target_schema),
            _norm(relationship.target_table),
            _norm(relationship.target_column),
        ) != target_column[1:]:
            raise SQLGenerationAuthorizationError(
                "Metadata context relationship names do not match the authorized catalog"
            )
