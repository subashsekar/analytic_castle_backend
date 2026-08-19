"""Orchestrate data-source connection tests.

Load configuration from the application database, decrypt credentials through
the credential service, resolve a connector from the registry, and test the
customer database. This module does not implement HTTP, JWT, encryption, or
PostgreSQL driver logic.

Status persistence: a completed connection test writes ``ACTIVE`` on success
and ``ERROR`` on failure, and always sets ``last_tested_at``. Precondition
failures (missing source, missing configuration, unsupported connector,
credential errors) do not change status or timestamp.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, joinedload

from app.connectors import (
    ConnectionTestResult,
    ConnectorConfig,
    ConnectorError,
    UnsupportedConnectorError,
    connector_lifecycle,
    create_connector,
    is_implemented,
)
from app.db.models import DataSource, DataSourceConnection
from app.enums import DataSourceStatus, DataSourceType
from app.services.credentials import CredentialError, connector_config_from_connection

logger = logging.getLogger(__name__)


class DataSourceServiceError(Exception):
    """Base error for data-source connection service failures."""


class DataSourceNotFoundError(DataSourceServiceError):
    """The requested data source does not exist or has no workspace."""

    def __init__(self, message: str = "Data source not found") -> None:
        super().__init__(message)


class ConnectionConfigurationError(DataSourceServiceError):
    """Connection settings are missing or cannot be used."""

    def __init__(
        self,
        message: str = "Data source connection configuration is invalid",
    ) -> None:
        super().__init__(message)


class DataSourceConnectionError(DataSourceServiceError):
    """A connection test against the customer data source failed."""

    def __init__(self, message: str = "Unable to connect to the data source") -> None:
        super().__init__(message)


def load_configured_data_source(
    session: Session,
    data_source_id: UUID,
    *,
    workspace_id: UUID | None = None,
) -> tuple[DataSource, DataSourceConnection]:
    """Load a data source and its connection settings.

    When ``workspace_id`` is provided, a data source in another workspace is
    treated as missing so callers cannot inspect isolated tenants.
    """
    data_source = session.get(
        DataSource,
        data_source_id,
        options=(
            joinedload(DataSource.connection),
            joinedload(DataSource.workspace),
        ),
    )
    if data_source is None or data_source.workspace is None:
        raise DataSourceNotFoundError("Data source not found")
    if workspace_id is not None and data_source.workspace_id != workspace_id:
        raise DataSourceNotFoundError("Data source not found")

    connection = data_source.connection
    if connection is None:
        raise ConnectionConfigurationError(
            "Data source connection configuration is missing"
        )

    if not is_implemented(data_source.type):
        raise UnsupportedConnectorError(
            f"Connector type {data_source.type.value} is not implemented"
        )
    return data_source, connection


class DataSourceConnectionService:
    """Internal service for testing customer data-source connectivity."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def test_connection(self, data_source_id: UUID) -> ConnectionTestResult:
        data_source, connection = self._load_configured_data_source(data_source_id)
        source_type = data_source.type
        try:
            config = connector_config_from_connection(connection)
        except CredentialError as exc:
            raise ConnectionConfigurationError(
                "Unable to load data source credentials"
            ) from exc

        self._session.commit()

        logger.info(
            "Data source connection test started data_source_id=%s connector_type=%s",
            data_source_id,
            source_type.value,
        )
        started = time.perf_counter()
        error_type: str | None = None
        try:
            success = await self._run_external_test(source_type, config)
        except DataSourceConnectionError as exc:
            success = False
            cause = exc.__cause__
            error_type = (
                type(cause).__name__ if cause is not None else type(exc).__name__
            )
        duration_ms = (time.perf_counter() - started) * 1000

        if success:
            logger.info(
                "Data source connection test succeeded "
                "data_source_id=%s connector_type=%s duration_ms=%.0f",
                data_source_id,
                source_type.value,
                duration_ms,
            )
        else:
            logger.warning(
                "Data source connection test failed "
                "data_source_id=%s connector_type=%s duration_ms=%.0f error_type=%s",
                data_source_id,
                source_type.value,
                duration_ms,
                error_type or "ConnectionTestResult",
            )

        try:
            self._persist_test_outcome(data_source_id, success=success)
            self._session.commit()
        except SQLAlchemyError:
            self._session.rollback()
            raise
        return ConnectionTestResult(success=success)

    def _load_configured_data_source(
        self,
        data_source_id: UUID,
    ) -> tuple[DataSource, DataSourceConnection]:
        return load_configured_data_source(self._session, data_source_id)

    async def _run_external_test(
        self,
        source_type: DataSourceType,
        config: ConnectorConfig,
    ) -> bool:
        connector = create_connector(source_type)
        try:
            async with connector_lifecycle(connector, config) as active:
                result = await active.test_connection()
        except ConnectorError as exc:
            raise DataSourceConnectionError(
                "Unable to connect to the data source"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise DataSourceConnectionError(
                "Unable to connect to the data source"
            ) from exc
        return result.success

    def _persist_test_outcome(self, data_source_id: UUID, *, success: bool) -> None:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None:
            logger.warning(
                "Data source missing while persisting connection test "
                "data_source_id=%s",
                data_source_id,
            )
            return
        data_source.status = (
            DataSourceStatus.ACTIVE if success else DataSourceStatus.ERROR
        )
        data_source.last_tested_at = datetime.now(UTC)
