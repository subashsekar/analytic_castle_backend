from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, TypedDict

import psycopg
import psycopg.errors
from psycopg import sql

from app.connectors.exceptions import (
    ConnectorAuthenticationError,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorQueryError,
    UnsupportedOperationError,
    sanitize_connector_message,
)
from app.connectors.readonly_sql import validate_readonly_sql
from app.connectors.registry import register_connector
from app.connectors.types import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorConfig,
    QueryResult,
    SchemaInfo,
    TableInfo,
)
from app.core.config import settings
from app.enums import DataSourceType

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_SSL_MODE = "require"
ALLOWED_SSL_MODES = frozenset(
    {
        "disable",
        "allow",
        "prefer",
        "require",
        "verify-ca",
        "verify-full",
    }
)

_TEST_QUERY = "SELECT 1"
_READ_ONLY_OPTIONS = "-c default_transaction_read_only=on"

_AUTH_MARKERS = (
    "password authentication failed",
    "authentication failed",
    "auth failed",
)
_TIMEOUT_MARKERS = ("timeout expired", "timed out", "timeout")
_DNS_MARKERS = (
    "could not translate host name",
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo failed",
    "no such host",
    "temporary failure in name resolution",
)
_REFUSED_MARKERS = ("connection refused", "actively refused")
_SSL_MARKERS = ("ssl", "certificate")
_UNAVAILABLE_MARKERS = (
    "database is not accepting",
    "too many connections",
    "the database system is starting",
    "the database system is shutting down",
    "does not exist",
)


class PostgresConnectParams(TypedDict):
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str
    connect_timeout: int
    options: str


class PostgreSQLConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    FAILED = "failed"


def build_postgresql_connector() -> PostgreSQLConnector:
    return PostgreSQLConnector(connect_timeout=settings.POSTGRES_CONNECT_TIMEOUT)


def resolve_ssl_mode(ssl_mode: str | None) -> str:
    if ssl_mode is None:
        return DEFAULT_SSL_MODE
    normalized = ssl_mode.strip().lower()
    if normalized not in ALLOWED_SSL_MODES:
        raise ConnectorError("Invalid PostgreSQL SSL mode")
    return normalized


def connect_kwargs(
    config: ConnectorConfig,
    *,
    connect_timeout: float,
    ssl_mode: str,
) -> PostgresConnectParams:
    """Build psycopg connection parameters without concatenating a URL."""
    timeout_seconds = max(1, int(connect_timeout))
    timeout_ms = max(1, int(connect_timeout * 1000))
    return {
        "host": config.host,
        "port": config.port,
        "dbname": config.database_name,
        "user": config.username,
        "password": config.credential,
        "sslmode": ssl_mode,
        "connect_timeout": timeout_seconds,
        "options": f"{_READ_ONLY_OPTIONS} -c statement_timeout={timeout_ms}",
    }


def map_postgres_error(
    exc: BaseException,
    *,
    operation: str = "connect",
) -> ConnectorError:
    if isinstance(exc, ConnectorError):
        return exc

    diagnostic = sanitize_connector_message(str(exc)).lower()

    if operation == "query" and (
        isinstance(
            exc,
            (
                TimeoutError,
                asyncio.TimeoutError,
                psycopg.errors.QueryCanceled,
            ),
        )
        or _contains(diagnostic, _TIMEOUT_MARKERS)
    ):
        return ConnectorQueryError("PostgreSQL query timed out")

    if isinstance(
        exc,
        (
            TimeoutError,
            asyncio.TimeoutError,
            psycopg.errors.ConnectionTimeout,
            psycopg.errors.QueryCanceled,
        ),
    ) or _contains(diagnostic, _TIMEOUT_MARKERS):
        return ConnectorConnectionError("PostgreSQL connection timed out")

    if isinstance(
        exc,
        (
            psycopg.errors.InvalidPassword,
            psycopg.errors.InvalidAuthorizationSpecification,
        ),
    ) or _contains(diagnostic, _AUTH_MARKERS):
        return ConnectorAuthenticationError(
            "Unable to authenticate with the PostgreSQL data source"
        )

    if isinstance(exc, psycopg.errors.CannotConnectNow) or _contains(
        diagnostic, _UNAVAILABLE_MARKERS
    ):
        return ConnectorConnectionError("The PostgreSQL database is unavailable")

    if _contains(diagnostic, _DNS_MARKERS):
        return ConnectorConnectionError("Unable to resolve the PostgreSQL host")

    if _contains(diagnostic, _REFUSED_MARKERS):
        return ConnectorConnectionError("PostgreSQL connection refused")

    if _contains(diagnostic, _SSL_MARKERS):
        return ConnectorConnectionError("PostgreSQL SSL connection failed")

    if operation == "query":
        return ConnectorQueryError("Unable to query the PostgreSQL data source")
    return ConnectorConnectionError("Unable to connect to the PostgreSQL data source")


def _contains(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _log_failure(
    exc: BaseException,
    *,
    host: str,
    port: int,
    operation: str,
    include_detail: bool = True,
) -> None:
    """Log connector failures without leaking customer SQL when applicable.

    Connect/auth failures may include sanitized driver detail (credentials are
    redacted). Query execution failures must not log exception text because
    PostgreSQL diagnostics often embed the submitted statement.
    """

    if include_detail:
        logger.warning(
            "PostgreSQL connector error error_type=%s operation=%s host=%s "
            "port=%s detail=%s",
            type(exc).__name__,
            operation,
            host,
            port,
            sanitize_connector_message(str(exc)),
        )
        return
    logger.warning(
        "PostgreSQL connector error error_type=%s operation=%s host=%s port=%s",
        type(exc).__name__,
        operation,
        host,
        port,
    )


class PostgreSQLConnector:
    """Async read-only PostgreSQL connector for a customer data source.

    Uses a per-instance connection. Does not create a process-wide pool and
    does not use the AnalyticCastle application database.

    Receives already-decrypted credentials via ``ConnectorConfig``. Does not
    encrypt, decrypt, or read the application encryption key.
    """

    def __init__(
        self,
        *,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than 0")
        self._connect_timeout = connect_timeout
        self._connection: psycopg.AsyncConnection[Any] | None = None
        self._state = PostgreSQLConnectionState.DISCONNECTED

    @property
    def connect_timeout(self) -> float:
        return self._connect_timeout

    @property
    def connection_state(self) -> PostgreSQLConnectionState:
        return self._state

    def __repr__(self) -> str:
        return (
            "PostgreSQLConnector("
            f"state={self._state.value!r}, "
            f"connect_timeout={self._connect_timeout!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    async def connect(self, config: ConnectorConfig) -> None:
        await self.disconnect()
        ssl_mode = resolve_ssl_mode(config.ssl_mode)
        params = connect_kwargs(
            config,
            connect_timeout=self._connect_timeout,
            ssl_mode=ssl_mode,
        )
        logger.info(
            "Connecting to PostgreSQL host=%s port=%s database_name=%s ssl_mode=%s",
            config.host,
            config.port,
            config.database_name,
            ssl_mode,
        )
        try:
            async with asyncio.timeout(self._connect_timeout):
                self._connection = await psycopg.AsyncConnection.connect(
                    autocommit=True,
                    **params,
                )
        except (psycopg.Error, OSError, TimeoutError) as exc:
            self._connection = None
            self._state = PostgreSQLConnectionState.FAILED
            _log_failure(
                exc,
                host=config.host,
                port=config.port,
                operation="connect",
            )
            raise map_postgres_error(exc, operation="connect") from None
        self._state = PostgreSQLConnectionState.CONNECTED

    async def disconnect(self) -> None:
        connection = self._connection
        self._connection = None
        if self._state is PostgreSQLConnectionState.CONNECTED:
            self._state = PostgreSQLConnectionState.DISCONNECTED
        if connection is None or connection.closed:
            return
        try:
            async with asyncio.timeout(self._connect_timeout):
                await connection.close()
        except (psycopg.Error, OSError, TimeoutError) as exc:
            logger.warning(
                "PostgreSQL disconnect failed error_type=%s",
                type(exc).__name__,
            )

    async def test_connection(self) -> ConnectionTestResult:
        connection = self._require_connection()
        try:
            async with asyncio.timeout(self._connect_timeout):
                cursor = await connection.execute(_TEST_QUERY)
                try:
                    row = await cursor.fetchone()
                finally:
                    await cursor.close()
        except (psycopg.Error, OSError, TimeoutError) as exc:
            _log_failure(
                exc,
                host="-",
                port=0,
                operation="test_connection",
                include_detail=False,
            )
            raise map_postgres_error(exc, operation="query") from None
        if row is None or row[0] != 1:
            raise ConnectorQueryError("Unable to query the PostgreSQL data source")
        return ConnectionTestResult(success=True)

    async def _fetch_all(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a parameterized read-only SELECT for metadata discovery.

        Not part of the public connector protocol. Discovery uses this method
        with static catalog queries and bound parameters.
        """
        validated = _require_readonly_select(query)
        connection = self._require_connection()
        try:
            async with asyncio.timeout(self._connect_timeout):
                bound = dict(params) if params is not None else None
                cursor = await connection.execute(validated, bound)
                try:
                    rows = await cursor.fetchall()
                    description = cursor.description
                finally:
                    await cursor.close()
        except (psycopg.Error, OSError, TimeoutError) as exc:
            _log_failure(
                exc,
                host="-",
                port=0,
                operation="fetch_all",
                include_detail=False,
            )
            raise map_postgres_error(exc, operation="query") from None
        if description is None:
            return []
        names = [column.name for column in description]
        return [dict(zip(names, row, strict=True)) for row in rows]

    async def _fetch_sample_rows(
        self,
        schema_name: str,
        table_name: str,
        column_names: Sequence[str],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Run a read-only sample SELECT against a metadata-validated table.

        Identifiers are quoted with ``psycopg.sql.Identifier``. ``limit`` is a
        bound parameter. Arbitrary SQL is not accepted.
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ConnectorQueryError("Unable to query the PostgreSQL data source")
        quoted_columns = [_require_pg_identifier(name) for name in column_names]
        if not quoted_columns:
            raise ConnectorQueryError("Unable to query the PostgreSQL data source")
        query = sql.SQL("SELECT {fields} FROM {table} LIMIT %(row_limit)s").format(
            fields=sql.SQL(", ").join(sql.Identifier(name) for name in quoted_columns),
            table=sql.Identifier(
                _require_pg_identifier(schema_name),
                _require_pg_identifier(table_name),
            ),
        )
        connection = self._require_connection()
        try:
            async with asyncio.timeout(self._connect_timeout):
                cursor = await connection.execute(query, {"row_limit": limit})
                try:
                    rows = await cursor.fetchall()
                    description = cursor.description
                finally:
                    await cursor.close()
        except (psycopg.Error, OSError, TimeoutError) as exc:
            _log_failure(
                exc,
                host="-",
                port=0,
                operation="fetch_sample_rows",
                include_detail=False,
            )
            raise map_postgres_error(exc, operation="query") from None
        if description is None:
            return []
        names = [column.name for column in description]
        return [dict(zip(names, row, strict=True)) for row in rows]

    async def get_schemas(self) -> list[SchemaInfo]:
        raise UnsupportedOperationError(
            "Schema discovery is not supported for this connector"
        )

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        raise UnsupportedOperationError(
            "Table discovery is not supported for this connector"
        )

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        raise UnsupportedOperationError(
            "Column discovery is not supported for this connector"
        )

    async def execute_query(
        self, query: str, *, limit: int | None = None
    ) -> QueryResult:
        """Execute one read-only statement and return a bounded result."""
        validated = validate_readonly_sql(query)
        row_limit = _resolve_query_limit(limit)
        connection = self._require_connection()
        fetch_limit = row_limit + 1
        wrapped = _wrap_readonly_query(validated)
        try:
            async with asyncio.timeout(self._connect_timeout):
                cursor = await connection.execute(wrapped, {"row_limit": fetch_limit})
                try:
                    rows = await cursor.fetchmany(fetch_limit)
                    description = cursor.description
                finally:
                    await cursor.close()
        except (psycopg.Error, OSError, TimeoutError) as exc:
            # Never log driver detail here: PostgreSQL errors often embed SQL.
            _log_failure(
                exc,
                host="-",
                port=0,
                operation="execute_query",
                include_detail=False,
            )
            raise map_postgres_error(exc, operation="query") from None
        if description is None:
            return QueryResult(columns=(), rows=(), truncated=False)
        names = tuple(column.name for column in description)
        truncated = len(rows) > row_limit
        bounded = rows[:row_limit]
        return QueryResult(
            columns=names,
            rows=tuple(tuple(row) for row in bounded),
            truncated=truncated,
        )

    def _require_connection(self) -> psycopg.AsyncConnection[Any]:
        connection = self._connection
        if connection is None or connection.closed:
            self._state = PostgreSQLConnectionState.DISCONNECTED
            raise ConnectorConnectionError(
                "Not connected to the PostgreSQL data source"
            )
        return connection


_MAX_IDENTIFIER_LENGTH = 255


def _require_pg_identifier(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    if len(name) > _MAX_IDENTIFIER_LENGTH:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    return name


def _require_readonly_select(query: str) -> str:
    return validate_readonly_sql(query)


def _wrap_readonly_query(validated: str) -> sql.Composed:
    """Bound the result in SQL while preserving an outer ORDER BY when possible.

    PostgreSQL ignores ORDER BY in a subquery unless LIMIT, OFFSET, or a row
    lock is present. Appending OFFSET 0 keeps that order without adding a
    second OFFSET clause when the statement already has one.
    """
    inner = validated
    if not re.search(r"\boffset\b", inner, flags=re.IGNORECASE):
        inner = f"{inner} OFFSET 0"
    return sql.SQL(
        "SELECT * FROM ({inner}) AS _analyticcastle_query LIMIT %(row_limit)s"
    ).format(inner=sql.SQL(inner))


def _resolve_query_limit(limit: int | None) -> int:
    maximum = settings.MCP_QUERY_MAX_LIMIT
    if limit is None:
        return min(settings.MCP_QUERY_DEFAULT_LIMIT, maximum)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ConnectorQueryError("Unable to query the PostgreSQL data source")
    return min(limit, maximum)


register_connector(DataSourceType.POSTGRESQL, build_postgresql_connector)
