"""Retrieve a small, masked sample of rows from a discovered table.

Validates the table against persisted metadata, queries the customer database
through the existing PostgreSQL connector, then masks sensitive values before
returning a result. This module does not implement HTTP, RBAC, or SQL generation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session, joinedload, selectinload

from app.connectors import (
    ConnectorConfig,
    ConnectorError,
    UnsupportedOperationError,
    connector_lifecycle,
    create_connector,
)
from app.core.config import settings
from app.db.models import DataSourceColumn, DataSourceTable
from app.enums import ColumnSensitivity, DataSourceTableType, DataSourceType
from app.services.credentials import CredentialError, connector_config_from_connection
from app.services.data_masking import mask_value
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionError,
    load_configured_data_source,
)
from app.services.pii_detection import classify_column, detect_value_pii
from app.services.sample_data_exceptions import (
    SampleDataError,
    SampleDataLimitError,
    SampleIdentifierError,
    SampleQueryError,
    SampleSerializationError,
    SampleTableNotFoundError,
)
from app.services.sample_data_types import SampleColumn, SampleDataResult
from app.services.sample_serialization import serialize_sample_value

logger = logging.getLogger(__name__)

_MAX_IDENTIFIER_LENGTH = 255


class SampleQueryExecutor(Protocol):
    async def _fetch_sample_rows(
        self,
        schema_name: str,
        table_name: str,
        column_names: Sequence[str],
        *,
        limit: int,
    ) -> list[dict[str, object]]: ...


class SampleDataService:
    """Load a privacy-safe sample from one workspace-scoped data source table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_sample(
        self,
        data_source_id: UUID,
        table_id: UUID,
        *,
        workspace_id: UUID,
        limit: int | None = None,
    ) -> SampleDataResult:
        data_source, connection = load_configured_data_source(
            self._session,
            data_source_id,
            workspace_id=workspace_id,
        )
        table, selected_columns, truncated_columns = self._load_table(
            data_source_id,
            table_id,
        )
        resolved_limit = self._resolve_limit(limit)
        if data_source.type is not DataSourceType.POSTGRESQL:
            raise UnsupportedOperationError(
                "Sample data is not supported for this connector"
            )
        try:
            config = connector_config_from_connection(connection)
        except CredentialError as exc:
            raise ConnectionConfigurationError(
                "Unable to load data source credentials"
            ) from exc

        self._session.commit()

        schema_name = table.schema.name
        table_name = table.name
        logger.info(
            "Sample data started data_source_id=%s table_id=%s schema=%s "
            "column_count=%s row_limit=%s",
            data_source_id,
            table_id,
            schema_name,
            len(selected_columns),
            resolved_limit,
        )
        started = time.perf_counter()
        try:
            raw_rows = await self._run_sample_query(
                config,
                schema_name=schema_name,
                table_name=table_name,
                columns=selected_columns,
                limit=resolved_limit,
            )
        except SampleDataError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "Sample data failed data_source_id=%s table_id=%s duration_ms=%.0f",
                data_source_id,
                table_id,
                duration_ms,
            )
            raise
        except DataSourceConnectionError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "Sample data failed data_source_id=%s table_id=%s duration_ms=%.0f",
                data_source_id,
                table_id,
                duration_ms,
            )
            raise SampleQueryError("Unable to retrieve sample data") from None

        classifications = tuple(
            classify_column(
                column.name,
                data_type=column.data_type,
                description=column.description,
            )
            for column in selected_columns
        )
        safe_rows = tuple(
            self._safe_row(raw_row, selected_columns, classifications)
            for raw_row in raw_rows
        )
        del raw_rows

        result = SampleDataResult(
            data_source_id=data_source_id,
            table_id=table_id,
            schema_name=schema_name,
            table_name=table_name,
            table_type=table.table_type,
            columns=tuple(
                SampleColumn(
                    name=column.name,
                    data_type=column.data_type,
                    sensitivity=sensitivity,
                    masked=sensitivity is not ColumnSensitivity.PUBLIC,
                )
                for column, sensitivity in zip(
                    selected_columns, classifications, strict=True
                )
            ),
            rows=safe_rows,
            row_limit=resolved_limit,
            truncated_columns=truncated_columns,
        )
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Sample data completed data_source_id=%s table_id=%s schema=%s "
            "row_count=%s column_count=%s duration_ms=%.0f",
            data_source_id,
            table_id,
            schema_name,
            result.row_count,
            result.column_count,
            duration_ms,
        )
        return result

    def _load_table(
        self,
        data_source_id: UUID,
        table_id: UUID,
    ) -> tuple[DataSourceTable, tuple[DataSourceColumn, ...], bool]:
        table = self._session.get(
            DataSourceTable,
            table_id,
            options=(
                joinedload(DataSourceTable.schema),
                selectinload(DataSourceTable.columns),
            ),
        )
        if (
            table is None
            or table.schema is None
            or table.schema.data_source_id != data_source_id
        ):
            raise SampleTableNotFoundError("Table not found")
        if table.table_type not in {
            DataSourceTableType.TABLE,
            DataSourceTableType.VIEW,
        }:
            raise SampleTableNotFoundError("Table not found")

        ordered = tuple(
            sorted(
                table.columns,
                key=lambda column: (column.ordinal_position, column.name),
            )
        )
        if not ordered:
            raise SampleTableNotFoundError("Table not found")
        max_columns = settings.SAMPLE_DATA_MAX_COLUMNS
        truncated = len(ordered) > max_columns
        selected = ordered[:max_columns]
        for column in selected:
            validate_pg_identifier(column.name)
        validate_pg_identifier(table.schema.name)
        validate_pg_identifier(table.name)
        return table, selected, truncated

    def _resolve_limit(self, limit: int | None) -> int:
        if limit is None:
            return settings.SAMPLE_DATA_DEFAULT_LIMIT
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise SampleDataLimitError("Sample row limit is invalid")
        if limit < 1:
            raise SampleDataLimitError("Sample row limit is invalid")
        return min(limit, settings.SAMPLE_DATA_MAX_LIMIT)

    async def _run_sample_query(
        self,
        config: ConnectorConfig,
        *,
        schema_name: str,
        table_name: str,
        columns: Sequence[DataSourceColumn],
        limit: int,
    ) -> list[dict[str, object]]:
        connector = create_connector(DataSourceType.POSTGRESQL)
        if not callable(getattr(connector, "_fetch_sample_rows", None)):
            raise UnsupportedOperationError(
                "Sample data is not supported for this connector"
            )
        column_names = tuple(column.name for column in columns)
        try:
            async with connector_lifecycle(connector, config) as active:
                executor = _as_sample_executor(active)
                return await executor._fetch_sample_rows(
                    schema_name,
                    table_name,
                    column_names,
                    limit=limit,
                )
        except SampleDataError:
            raise
        except ConnectorError as exc:
            raise SampleQueryError("Unable to retrieve sample data") from exc
        except (OSError, TimeoutError) as exc:
            raise DataSourceConnectionError(
                "Unable to connect to the data source"
            ) from exc

    def _safe_row(
        self,
        raw_row: Mapping[str, object],
        columns: Sequence[DataSourceColumn],
        classifications: Sequence[ColumnSensitivity],
    ) -> dict[str, object]:
        safe: dict[str, object] = {}
        for column, sensitivity in zip(columns, classifications, strict=True):
            raw_value = raw_row.get(column.name)
            safe[column.name] = _safe_cell(
                raw_value,
                column_name=column.name,
                sensitivity=sensitivity,
            )
        return safe


def validate_pg_identifier(name: str) -> str:
    """Reject identifiers that cannot be safely quoted for PostgreSQL."""
    if not isinstance(name, str) or not name or "\x00" in name:
        raise SampleIdentifierError("Invalid identifier")
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        raise SampleIdentifierError("Invalid identifier")
    return name


def _safe_cell(
    raw_value: object,
    *,
    column_name: str,
    sensitivity: ColumnSensitivity,
) -> object:
    if raw_value is None:
        return None
    if sensitivity is ColumnSensitivity.SECRET:
        return mask_value(
            raw_value,
            sensitivity=sensitivity,
            column_name=column_name,
        )
    if sensitivity in {ColumnSensitivity.PII, ColumnSensitivity.SENSITIVE}:
        return mask_value(
            raw_value,
            sensitivity=sensitivity,
            column_name=column_name,
        )
    try:
        serialized = serialize_sample_value(raw_value)
    except SampleSerializationError:
        return "[UNAVAILABLE]"
    value_kind = detect_value_pii(serialized)
    if value_kind is None:
        return serialized
    return mask_value(
        serialized,
        sensitivity=ColumnSensitivity.PII,
        column_name=column_name,
        kind=value_kind,
    )


def _as_sample_executor(connector: object) -> SampleQueryExecutor:
    fetch_sample = getattr(connector, "_fetch_sample_rows", None)
    if not callable(fetch_sample):
        raise UnsupportedOperationError(
            "Sample data is not supported for this connector"
        )
    return connector  # type: ignore[return-value]
