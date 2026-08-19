"""Read persisted metadata for one workspace-scoped data source.

Queries run against AnalyticCastle metadata tables. This module does not
discover customer databases, synchronize metadata, search across types, or
implement HTTP/RBAC.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.orm import Session, aliased

from app.connectors.exceptions import sanitize_connector_message
from app.core.config import settings
from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
)
from app.enums import DataSourceTableType
from app.services.data_source_connections import DataSourceNotFoundError
from app.services.metadata_catalog_exceptions import (
    MetadataNotFoundError,
    MetadataPageError,
)
from app.services.metadata_catalog_types import (
    ColumnRecord,
    MetadataPage,
    RelationshipRecord,
    SchemaRecord,
    TableRecord,
)
from app.services.metadata_search import escape_like_pattern

logger = logging.getLogger(__name__)

_LIKE_ESCAPE = "\\"
_SelectAny = Select[Any]


class MetadataCatalogService:
    """Load schema, table, column, and relationship metadata for one data source."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_schemas(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[SchemaRecord]:
        self._load_workspace_data_source(data_source_id, workspace_id)
        resolved_page, resolved_size = self._resolve_page(page, page_size)
        started = time.perf_counter()
        table_count = (
            select(func.count(DataSourceTable.id))
            .where(DataSourceTable.schema_id == DataSourceSchema.id)
            .correlate(DataSourceSchema)
            .scalar_subquery()
        )
        stmt = (
            select(DataSourceSchema, table_count.label("table_count"))
            .where(DataSourceSchema.data_source_id == data_source_id)
            .order_by(DataSourceSchema.name, DataSourceSchema.id)
        )
        rows, total = self._fetch_page(stmt, resolved_page, resolved_size)
        items = tuple(
            SchemaRecord(
                id=schema.id,
                data_source_id=schema.data_source_id,
                name=schema.name,
                table_count=int(count or 0),
                created_at=schema.created_at,
                updated_at=schema.updated_at,
            )
            for schema, count in rows
        )
        self._log_completed(
            "schemas listed",
            data_source_id,
            result_count=len(items),
            started=started,
        )
        return MetadataPage(
            items=items,
            page=resolved_page,
            page_size=resolved_size,
            total=total,
        )

    def get_schema(
        self,
        data_source_id: UUID,
        schema_id: UUID,
        *,
        workspace_id: UUID,
    ) -> SchemaRecord:
        self._load_workspace_data_source(data_source_id, workspace_id)
        table_count = (
            select(func.count(DataSourceTable.id))
            .where(DataSourceTable.schema_id == DataSourceSchema.id)
            .correlate(DataSourceSchema)
            .scalar_subquery()
        )
        row = self._session.execute(
            select(DataSourceSchema, table_count.label("table_count")).where(
                DataSourceSchema.id == schema_id,
                DataSourceSchema.data_source_id == data_source_id,
            )
        ).one_or_none()
        if row is None:
            raise MetadataNotFoundError("Schema not found")
        schema, count = row
        return SchemaRecord(
            id=schema.id,
            data_source_id=schema.data_source_id,
            name=schema.name,
            table_count=int(count or 0),
            created_at=schema.created_at,
            updated_at=schema.updated_at,
        )

    def list_tables(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        schema_id: UUID | None = None,
        search: str | None = None,
        table_type: DataSourceTableType | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[TableRecord]:
        self._load_workspace_data_source(data_source_id, workspace_id)
        if schema_id is not None:
            self._require_schema(data_source_id, schema_id)
        resolved_page, resolved_size = self._resolve_page(page, page_size)
        started = time.perf_counter()
        column_count = (
            select(func.count(DataSourceColumn.id))
            .where(DataSourceColumn.table_id == DataSourceTable.id)
            .correlate(DataSourceTable)
            .scalar_subquery()
        )
        stmt = (
            select(
                DataSourceTable,
                DataSourceSchema.name.label("schema_name"),
                DataSourceSchema.data_source_id.label("data_source_id"),
                column_count.label("column_count"),
            )
            .join(
                DataSourceSchema,
                DataSourceTable.schema_id == DataSourceSchema.id,
            )
            .where(DataSourceSchema.data_source_id == data_source_id)
        )
        if schema_id is not None:
            stmt = stmt.where(DataSourceTable.schema_id == schema_id)
        if table_type is not None:
            stmt = stmt.where(DataSourceTable.table_type == table_type)
        search_term = search.strip() if isinstance(search, str) else ""
        if search_term:
            stmt = stmt.where(self._table_search_filter(search_term))
        stmt = stmt.order_by(
            DataSourceSchema.name,
            DataSourceTable.name,
            DataSourceTable.id,
        )
        rows, total = self._fetch_page(stmt, resolved_page, resolved_size)
        items = tuple(
            TableRecord(
                id=table.id,
                data_source_id=source_id,
                schema_id=table.schema_id,
                schema_name=schema_name,
                name=table.name,
                table_type=table.table_type,
                description=table.description,
                column_count=int(count or 0),
                created_at=table.created_at,
                updated_at=table.updated_at,
            )
            for table, schema_name, source_id, count in rows
        )
        self._log_completed(
            "tables listed",
            data_source_id,
            result_count=len(items),
            started=started,
        )
        return MetadataPage(
            items=items,
            page=resolved_page,
            page_size=resolved_size,
            total=total,
        )

    def get_table(
        self,
        data_source_id: UUID,
        table_id: UUID,
        *,
        workspace_id: UUID,
    ) -> TableRecord:
        self._load_workspace_data_source(data_source_id, workspace_id)
        column_count = (
            select(func.count(DataSourceColumn.id))
            .where(DataSourceColumn.table_id == DataSourceTable.id)
            .correlate(DataSourceTable)
            .scalar_subquery()
        )
        row = self._session.execute(
            select(
                DataSourceTable,
                DataSourceSchema.name.label("schema_name"),
                DataSourceSchema.data_source_id.label("data_source_id"),
                column_count.label("column_count"),
            )
            .join(
                DataSourceSchema,
                DataSourceTable.schema_id == DataSourceSchema.id,
            )
            .where(
                DataSourceTable.id == table_id,
                DataSourceSchema.data_source_id == data_source_id,
            )
        ).one_or_none()
        if row is None:
            raise MetadataNotFoundError("Table not found")
        table, schema_name, source_id, count = row
        return TableRecord(
            id=table.id,
            data_source_id=source_id,
            schema_id=table.schema_id,
            schema_name=schema_name,
            name=table.name,
            table_type=table.table_type,
            description=table.description,
            column_count=int(count or 0),
            created_at=table.created_at,
            updated_at=table.updated_at,
        )

    def list_columns(
        self,
        data_source_id: UUID,
        table_id: UUID,
        *,
        workspace_id: UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[ColumnRecord]:
        self._load_workspace_data_source(data_source_id, workspace_id)
        self._require_table(data_source_id, table_id)
        resolved_page, resolved_size = self._resolve_page(page, page_size)
        started = time.perf_counter()
        stmt = (
            select(DataSourceColumn)
            .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(
                DataSourceColumn.table_id == table_id,
                DataSourceSchema.data_source_id == data_source_id,
            )
            .order_by(DataSourceColumn.ordinal_position, DataSourceColumn.id)
        )
        rows, total = self._fetch_page(stmt, resolved_page, resolved_size)
        items = tuple(self._column_record(row[0]) for row in rows)
        self._log_completed(
            "columns listed",
            data_source_id,
            result_count=len(items),
            started=started,
        )
        return MetadataPage(
            items=items,
            page=resolved_page,
            page_size=resolved_size,
            total=total,
        )

    def list_relationships(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[RelationshipRecord]:
        self._load_workspace_data_source(data_source_id, workspace_id)
        resolved_page, resolved_size = self._resolve_page(page, page_size)
        started = time.perf_counter()
        source_table = aliased(DataSourceTable)
        target_table = aliased(DataSourceTable)
        source_schema = aliased(DataSourceSchema)
        target_schema = aliased(DataSourceSchema)
        source_column = aliased(DataSourceColumn)
        target_column = aliased(DataSourceColumn)
        stmt = (
            select(
                DataSourceRelationship,
                source_schema.name.label("source_schema_name"),
                source_table.name.label("source_table_name"),
                source_column.name.label("source_column_name"),
                target_schema.name.label("target_schema_name"),
                target_table.name.label("target_table_name"),
                target_column.name.label("target_column_name"),
            )
            .join(
                source_table,
                DataSourceRelationship.source_table_id == source_table.id,
            )
            .join(source_schema, source_table.schema_id == source_schema.id)
            .join(
                source_column,
                DataSourceRelationship.source_column_id == source_column.id,
            )
            .join(
                target_table,
                DataSourceRelationship.target_table_id == target_table.id,
            )
            .join(target_schema, target_table.schema_id == target_schema.id)
            .join(
                target_column,
                DataSourceRelationship.target_column_id == target_column.id,
            )
            .where(
                source_schema.data_source_id == data_source_id,
                target_schema.data_source_id == data_source_id,
            )
            .order_by(
                DataSourceRelationship.constraint_name.nulls_last(),
                source_schema.name,
                source_table.name,
                source_column.ordinal_position,
                DataSourceRelationship.id,
            )
        )
        rows, total = self._fetch_page(stmt, resolved_page, resolved_size)
        items = tuple(
            RelationshipRecord(
                id=relation.id,
                source_table_id=relation.source_table_id,
                source_schema_name=source_schema_name,
                source_table_name=source_table_name,
                source_column_id=relation.source_column_id,
                source_column_name=source_column_name,
                target_table_id=relation.target_table_id,
                target_schema_name=target_schema_name,
                target_table_name=target_table_name,
                target_column_id=relation.target_column_id,
                target_column_name=target_column_name,
                relationship_type=relation.relationship_type,
                constraint_name=relation.constraint_name,
                created_at=relation.created_at,
                updated_at=relation.updated_at,
            )
            for (
                relation,
                source_schema_name,
                source_table_name,
                source_column_name,
                target_schema_name,
                target_table_name,
                target_column_name,
            ) in rows
        )
        self._log_completed(
            "relationships listed",
            data_source_id,
            result_count=len(items),
            started=started,
        )
        return MetadataPage(
            items=items,
            page=resolved_page,
            page_size=resolved_size,
            total=total,
        )

    def _load_workspace_data_source(
        self,
        data_source_id: UUID,
        workspace_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise DataSourceNotFoundError("Data source not found")
        return data_source

    def _require_schema(
        self, data_source_id: UUID, schema_id: UUID
    ) -> DataSourceSchema:
        schema = self._session.get(DataSourceSchema, schema_id)
        if schema is None or schema.data_source_id != data_source_id:
            raise MetadataNotFoundError("Schema not found")
        return schema

    def _require_table(self, data_source_id: UUID, table_id: UUID) -> DataSourceTable:
        row = self._session.execute(
            select(DataSourceTable)
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(
                DataSourceTable.id == table_id,
                DataSourceSchema.data_source_id == data_source_id,
            )
        ).scalar_one_or_none()
        if row is None:
            raise MetadataNotFoundError("Table not found")
        return row

    def _resolve_page(self, page: int, page_size: int | None) -> tuple[int, int]:
        resolved_size = (
            settings.METADATA_API_DEFAULT_PAGE_SIZE if page_size is None else page_size
        )
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise MetadataPageError("Pagination parameters are invalid")
        if (
            not isinstance(resolved_size, int)
            or isinstance(resolved_size, bool)
            or resolved_size < 1
            or resolved_size > settings.METADATA_API_MAX_PAGE_SIZE
        ):
            raise MetadataPageError(
                f"Page size must be between 1 and {settings.METADATA_API_MAX_PAGE_SIZE}"
            )
        return page, resolved_size

    def _fetch_page(
        self,
        stmt: _SelectAny,
        page: int,
        page_size: int,
    ) -> tuple[list[Any], int]:
        count_stmt = select(func.count()).select_from(stmt.order_by(None).subquery())
        total = int(self._session.scalar(count_stmt) or 0)
        rows = list(
            self._session.execute(
                stmt.offset((page - 1) * page_size).limit(page_size)
            ).all()
        )
        return rows, total

    def _table_search_filter(self, search: str) -> Any:
        escaped = escape_like_pattern(search.lower())
        contains = f"%{escaped}%"
        return or_(
            func.lower(DataSourceTable.name).like(contains, escape=_LIKE_ESCAPE),
            and_(
                DataSourceTable.description.is_not(None),
                func.lower(DataSourceTable.description).like(
                    contains, escape=_LIKE_ESCAPE
                ),
            ),
        )

    def _column_record(self, column: DataSourceColumn) -> ColumnRecord:
        return ColumnRecord(
            id=column.id,
            table_id=column.table_id,
            name=column.name,
            ordinal_position=column.ordinal_position,
            data_type=column.data_type,
            database_type=column.database_type,
            is_nullable=column.is_nullable,
            is_primary_key=column.is_primary_key,
            is_unique=column.is_unique,
            default_value=_safe_default_value(column.default_value),
            description=column.description,
            created_at=column.created_at,
            updated_at=column.updated_at,
        )

    def _log_completed(
        self,
        action: str,
        data_source_id: UUID,
        *,
        result_count: int,
        started: float,
    ) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Metadata %s data_source_id=%s result_count=%s duration_ms=%.0f",
            action,
            data_source_id,
            result_count,
            duration_ms,
        )


def _safe_default_value(value: str | None) -> str | None:
    if value is None:
        return None
    return sanitize_connector_message(value)
