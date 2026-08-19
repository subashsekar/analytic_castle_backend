"""Orchestrate PostgreSQL metadata discovery for a data source.

Loads configuration from the application database, decrypts credentials,
connects through the existing PostgreSQL connector, and returns discovery
results. Does not persist metadata.
"""

from __future__ import annotations

import logging
import time
from uuid import UUID

from sqlalchemy.orm import Session

from app.connectors import (
    ConnectorConfig,
    ConnectorError,
    PostgreSQLConnector,
    UnsupportedOperationError,
    connector_lifecycle,
    create_connector,
)
from app.core.config import settings
from app.enums import DataSourceType
from app.services.credentials import CredentialError, connector_config_from_connection
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionError,
    load_configured_data_source,
)
from app.services.discovery_exceptions import MetadataDiscoveryError
from app.services.discovery_types import (
    DiscoveryLimits,
    DiscoveryResult,
    MetadataQueryExecutor,
    SchemaFilter,
)
from app.services.postgresql_discovery import PostgreSQLMetadataDiscoveryService

logger = logging.getLogger(__name__)


class DataSourceDiscoveryService:
    """Run metadata discovery for a workspace-scoped data source."""

    def __init__(self, session: Session) -> None:
        self._session = session

    async def discover(
        self,
        data_source_id: UUID,
        *,
        workspace_id: UUID,
        schema_filter: SchemaFilter | None = None,
        limits: DiscoveryLimits | None = None,
    ) -> DiscoveryResult:
        data_source, connection = load_configured_data_source(
            self._session,
            data_source_id,
            workspace_id=workspace_id,
        )
        if data_source.type is not DataSourceType.POSTGRESQL:
            raise UnsupportedOperationError(
                "Metadata discovery is not supported for this connector"
            )
        try:
            config = connector_config_from_connection(connection)
        except CredentialError as exc:
            raise ConnectionConfigurationError(
                "Unable to load data source credentials"
            ) from exc

        self._session.commit()

        resolved_limits = limits or DiscoveryLimits(
            max_schemas=settings.METADATA_DISCOVERY_MAX_SCHEMAS,
            max_tables=settings.METADATA_DISCOVERY_MAX_TABLES,
            max_columns=settings.METADATA_DISCOVERY_MAX_COLUMNS,
            max_relationships=settings.METADATA_DISCOVERY_MAX_RELATIONSHIPS,
        )
        logger.info(
            "Metadata discovery started data_source_id=%s",
            data_source_id,
        )
        started = time.perf_counter()
        try:
            result = await self._run_discovery(
                config,
                limits=resolved_limits,
                schema_filter=schema_filter,
                data_source_id=data_source_id,
            )
        except MetadataDiscoveryError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "Metadata discovery failed data_source_id=%s duration_ms=%.0f",
                data_source_id,
                duration_ms,
            )
            raise
        except DataSourceConnectionError:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.warning(
                "Metadata discovery failed data_source_id=%s duration_ms=%.0f",
                data_source_id,
                duration_ms,
            )
            raise MetadataDiscoveryError(
                "Unable to discover database metadata"
            ) from None

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Metadata discovery completed data_source_id=%s "
            "schemas_discovered=%s tables_discovered=%s "
            "columns_discovered=%s relationships_discovered=%s duration_ms=%.0f",
            data_source_id,
            len(result.schemas),
            len(result.tables),
            len(result.columns),
            len(result.relationships),
            duration_ms,
        )
        return result

    async def _run_discovery(
        self,
        config: ConnectorConfig,
        *,
        limits: DiscoveryLimits,
        schema_filter: SchemaFilter | None,
        data_source_id: UUID,
    ) -> DiscoveryResult:
        connector = create_connector(DataSourceType.POSTGRESQL)
        if isinstance(connector, PostgreSQLConnector):
            connector = PostgreSQLConnector(
                connect_timeout=settings.POSTGRES_DISCOVERY_TIMEOUT
            )
        if not callable(getattr(connector, "_fetch_all", None)):
            raise UnsupportedOperationError(
                "Metadata discovery is not supported for this connector"
            )
        try:
            async with connector_lifecycle(connector, config) as active:
                service = PostgreSQLMetadataDiscoveryService(
                    _as_query_executor(active),
                    limits=limits,
                    schema_filter=schema_filter,
                    data_source_id=data_source_id,
                )
                return await service.discover()
        except MetadataDiscoveryError:
            raise
        except ConnectorError as exc:
            raise DataSourceConnectionError(
                "Unable to connect to the data source"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise DataSourceConnectionError(
                "Unable to connect to the data source"
            ) from exc


def _as_query_executor(connector: object) -> MetadataQueryExecutor:
    fetch_all = getattr(connector, "_fetch_all", None)
    if not callable(fetch_all):
        raise UnsupportedOperationError(
            "Metadata discovery is not supported for this connector"
        )
    return connector  # type: ignore[return-value]
