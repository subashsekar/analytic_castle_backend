"""Synchronize discovered customer-database metadata into AnalyticCastle.

Discovery runs against the external database before an application-database
transaction persists results. This module does not implement HTTP, RBAC, or
PostgreSQL catalog SQL.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from app.connectors.exceptions import (
    ConnectorError,
    UnsupportedConnectorError,
    sanitize_connector_message,
)
from app.db.models import DataSource, DataSourceMetadataSync
from app.enums import MetadataSyncStatus
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionError,
    DataSourceNotFoundError,
    load_configured_data_source,
)
from app.services.data_source_discovery import DataSourceDiscoveryService
from app.services.discovery_exceptions import MetadataDiscoveryError
from app.services.discovery_types import DiscoveryLimits, SchemaFilter
from app.services.metadata_persist import persist_discovered_metadata
from app.services.metadata_sync_exceptions import (
    ConcurrentMetadataSyncError,
    MetadataSyncError,
    MetadataSyncPersistenceError,
    MetadataSyncValidationError,
)
from app.services.metadata_sync_types import MetadataDiscoverer, MetadataSyncResult

logger = logging.getLogger(__name__)

_SAFE_DISCOVERY_FAILURE = "Unable to discover database metadata"
_SAFE_CONNECTION_FAILURE = "Unable to connect to the data source"
_SAFE_VALIDATION_FAILURE = "Discovered metadata is invalid"
_SAFE_PERSISTENCE_FAILURE = "Unable to save metadata"
_SAFE_TIMEOUT_FAILURE = "Metadata synchronization timed out"
_SAFE_CANCELLED_FAILURE = "Metadata synchronization was cancelled"
_SAFE_GENERIC_FAILURE = "Metadata synchronization failed"

_sync_lock = threading.Lock()
_in_progress: set[UUID] = set()


def reset_metadata_sync_locks() -> None:
    """Clear in-process sync locks. Intended for tests."""
    with _sync_lock:
        _in_progress.clear()


def try_acquire_metadata_sync_lock(data_source_id: UUID) -> bool:
    with _sync_lock:
        if data_source_id in _in_progress:
            return False
        _in_progress.add(data_source_id)
        return True


def release_metadata_sync_lock(data_source_id: UUID) -> None:
    with _sync_lock:
        _in_progress.discard(data_source_id)


class MetadataSyncService:
    """Run metadata discovery and persist the result for one data source."""

    def __init__(
        self,
        session: Session,
        *,
        discovery_service: MetadataDiscoverer | None = None,
    ) -> None:
        self._session = session
        self._discovery: MetadataDiscoverer = (
            discovery_service or DataSourceDiscoveryService(session)
        )

    def get_status(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
    ) -> MetadataSyncResult:
        self._load_workspace_data_source(data_source_id, workspace_id)
        return self._read_status(data_source_id)

    async def synchronize(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        schema_filter: SchemaFilter | None = None,
        limits: DiscoveryLimits | None = None,
    ) -> MetadataSyncResult:
        load_configured_data_source(
            self._session,
            data_source_id,
            workspace_id=workspace_id,
        )
        if not try_acquire_metadata_sync_lock(data_source_id):
            raise ConcurrentMetadataSyncError(
                "Metadata synchronization is already running"
            )

        started = time.perf_counter()
        logger.info("Metadata sync started data_source_id=%s", data_source_id)
        try:
            return await self._run_synchronized(
                data_source_id,
                workspace_id=workspace_id,
                schema_filter=schema_filter,
                limits=limits,
                started=started,
            )
        finally:
            release_metadata_sync_lock(data_source_id)

    async def _run_synchronized(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        schema_filter: SchemaFilter | None,
        limits: DiscoveryLimits | None,
        started: float,
    ) -> MetadataSyncResult:
        try:
            self._mark_running(data_source_id)
            self._session.commit()

            result = await self._discovery.discover(
                data_source_id,
                workspace_id=workspace_id,
                schema_filter=schema_filter,
                limits=limits,
            )
            counts = persist_discovered_metadata(self._session, data_source_id, result)
            self._mark_success(
                data_source_id,
                schema_count=counts.schemas,
                table_count=counts.tables,
                column_count=counts.columns,
                relationship_count=counts.relationships,
            )
            self._session.commit()
            from app.ai.sql_generation.schema_cache import invalidate_schema_context_cache

            invalidate_schema_context_cache(data_source_id)
        except asyncio.CancelledError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_CANCELLED_FAILURE,
                "CancelledError",
            )
            raise
        except MetadataSyncValidationError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_VALIDATION_FAILURE,
                "MetadataSyncValidationError",
            )
            raise
        except MetadataDiscoveryError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_DISCOVERY_FAILURE,
                "MetadataDiscoveryError",
            )
            raise MetadataSyncError(_SAFE_DISCOVERY_FAILURE) from None
        except UnsupportedConnectorError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_CONNECTION_FAILURE,
                "UnsupportedConnectorError",
            )
            raise
        except (DataSourceConnectionError, ConnectionConfigurationError):
            self._record_failure(
                data_source_id,
                started,
                _SAFE_CONNECTION_FAILURE,
                "DataSourceConnectionError",
            )
            raise MetadataSyncError(_SAFE_CONNECTION_FAILURE) from None
        except ConnectorError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_CONNECTION_FAILURE,
                "ConnectorError",
            )
            raise MetadataSyncError(_SAFE_CONNECTION_FAILURE) from None
        except TimeoutError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_TIMEOUT_FAILURE,
                "TimeoutError",
            )
            raise MetadataSyncError(_SAFE_TIMEOUT_FAILURE) from None
        except SQLAlchemyError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_PERSISTENCE_FAILURE,
                "SQLAlchemyError",
            )
            raise MetadataSyncPersistenceError(_SAFE_PERSISTENCE_FAILURE) from None
        except MetadataSyncError:
            self._record_failure(
                data_source_id,
                started,
                _SAFE_GENERIC_FAILURE,
                "MetadataSyncError",
            )
            raise
        except Exception:  # noqa: BLE001 - record safe sync failure for unexpected errors
            self._record_failure(
                data_source_id,
                started,
                _SAFE_GENERIC_FAILURE,
                "Exception",
            )
            raise MetadataSyncError(_SAFE_GENERIC_FAILURE) from None

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Metadata sync completed data_source_id=%s status=%s "
            "schemas=%s tables=%s columns=%s relationships=%s duration_ms=%.0f",
            data_source_id,
            MetadataSyncStatus.SUCCESS.value,
            counts.schemas,
            counts.tables,
            counts.columns,
            counts.relationships,
            duration_ms,
        )
        return self._read_status(data_source_id)

    def _load_workspace_data_source(
        self,
        data_source_id: UUID,
        workspace_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(
            DataSource,
            data_source_id,
            options=(joinedload(DataSource.workspace),),
        )
        if data_source is None or data_source.workspace is None:
            raise DataSourceNotFoundError("Data source not found")
        if data_source.workspace_id != workspace_id:
            raise DataSourceNotFoundError("Data source not found")
        return data_source

    def _get_or_create_sync(self, data_source_id: UUID) -> DataSourceMetadataSync:
        row = self._session.scalar(
            select(DataSourceMetadataSync).where(
                DataSourceMetadataSync.data_source_id == data_source_id
            )
        )
        if row is None:
            row = DataSourceMetadataSync(
                data_source_id=data_source_id,
                status=MetadataSyncStatus.PENDING,
            )
            self._session.add(row)
        return row

    def _mark_running(self, data_source_id: UUID) -> None:
        row = self._get_or_create_sync(data_source_id)
        row.status = MetadataSyncStatus.RUNNING
        row.started_at = datetime.now(UTC)
        row.completed_at = None
        row.error_message = None

    def _mark_success(
        self,
        data_source_id: UUID,
        *,
        schema_count: int,
        table_count: int,
        column_count: int,
        relationship_count: int,
    ) -> None:
        row = self._get_or_create_sync(data_source_id)
        row.status = MetadataSyncStatus.SUCCESS
        row.completed_at = datetime.now(UTC)
        row.error_message = None
        row.schema_count = schema_count
        row.table_count = table_count
        row.column_count = column_count
        row.relationship_count = relationship_count

    def _mark_failed(self, data_source_id: UUID, message: str) -> None:
        row = self._get_or_create_sync(data_source_id)
        row.status = MetadataSyncStatus.FAILED
        row.completed_at = datetime.now(UTC)
        row.error_message = sanitize_connector_message(message)

    def _record_failure(
        self,
        data_source_id: UUID,
        started: float,
        message: str,
        error_type: str,
    ) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        self._fail_safely(data_source_id, message)
        logger.warning(
            "Metadata sync failed data_source_id=%s status=%s "
            "duration_ms=%.0f error_type=%s",
            data_source_id,
            MetadataSyncStatus.FAILED.value,
            duration_ms,
            error_type,
        )

    def _fail_safely(self, data_source_id: UUID, message: str) -> None:
        try:
            self._session.rollback()
        except SQLAlchemyError:
            logger.warning(
                "Unable to roll back metadata sync data_source_id=%s error_type=%s",
                data_source_id,
                "SQLAlchemyError",
            )
        try:
            self._mark_failed(data_source_id, message)
            self._session.commit()
        except SQLAlchemyError:
            try:
                self._session.rollback()
            except SQLAlchemyError:
                pass
            logger.warning(
                "Unable to persist metadata sync failure "
                "data_source_id=%s error_type=%s",
                data_source_id,
                "SQLAlchemyError",
            )

    def _read_status(self, data_source_id: UUID) -> MetadataSyncResult:
        row = self._session.scalar(
            select(DataSourceMetadataSync).where(
                DataSourceMetadataSync.data_source_id == data_source_id
            )
        )
        if row is None:
            return MetadataSyncResult(
                data_source_id=data_source_id,
                status=MetadataSyncStatus.PENDING,
                started_at=None,
                completed_at=None,
                schema_count=None,
                table_count=None,
                column_count=None,
                relationship_count=None,
                error_message=None,
            )
        return MetadataSyncResult(
            data_source_id=data_source_id,
            status=row.status,
            started_at=row.started_at,
            completed_at=row.completed_at,
            schema_count=row.schema_count,
            table_count=row.table_count,
            column_count=row.column_count,
            relationship_count=row.relationship_count,
            error_message=row.error_message,
        )
