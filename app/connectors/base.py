from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Protocol, runtime_checkable

from app.connectors.types import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorConfig,
    QueryResult,
    SchemaInfo,
    TableInfo,
)


@runtime_checkable
class DataConnector(Protocol):
    """Async interface for a customer data source.

    Implementations talk only to the external source identified by
    ``ConnectorConfig``. They must not use the AnalyticCastle application
    database session, engine, or models.

    ``connect`` receives configuration for the duration of the call.
    Implementations must not retain ``credential`` after ``connect`` returns,
    must not encrypt or decrypt credentials, and must release any live
    connection in ``disconnect``.
    """

    async def connect(self, config: ConnectorConfig) -> None: ...

    async def disconnect(self) -> None: ...

    async def test_connection(self) -> ConnectionTestResult: ...

    async def get_schemas(self) -> list[SchemaInfo]: ...

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]: ...

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]: ...

    async def execute_query(self, query: str) -> QueryResult: ...


async def _disconnect_quietly(connector: DataConnector) -> None:
    """Close the connector without hiding connect/operation/cancellation errors."""
    with suppress(Exception):
        await connector.disconnect()


@asynccontextmanager
async def connector_lifecycle(
    connector: DataConnector,
    config: ConnectorConfig,
) -> AsyncIterator[DataConnector]:
    """Connect, yield the connector, then always disconnect.

    ``disconnect`` runs even when ``connect`` fails or is cancelled.
    Disconnect errors do not hide the original connect/operation error, and a
    failed disconnect after a successful operation does not discard the result.
    Does not create a process-wide or application-startup connection.
    """
    try:
        await connector.connect(config)
    except BaseException:
        await _disconnect_quietly(connector)
        raise
    try:
        yield connector
    except BaseException:
        await _disconnect_quietly(connector)
        raise
    await _disconnect_quietly(connector)
