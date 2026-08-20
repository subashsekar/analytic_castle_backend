from __future__ import annotations

import asyncio
import logging

import psycopg
import pytest

from app.connectors import (
    ConnectorAuthenticationError,
    ConnectorConfig,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorQueryError,
    DataConnector,
    PostgreSQLConnectionState,
    PostgreSQLConnector,
    QueryResult,
    UnsupportedConnectorError,
    UnsupportedOperationError,
    connector_lifecycle,
    create_connector,
    implemented_types,
    is_implemented,
)
from app.connectors.postgresql import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_SSL_MODE,
    connect_kwargs,
    map_postgres_error,
    resolve_ssl_mode,
)
from app.core.config import settings
from app.enums import DataSourceType
from tests.conftest import run_async

SECRET = "UnitTestPostgresSecret!@#"
SPECIAL_SECRET = "p@ss:w/rd# x"


def _config(
    *,
    host: str = "db.internal.example",
    port: int = 5432,
    database_name: str = "analytics",
    username: str = "readonly",
    credential: str = SECRET,
    ssl_mode: str | None = "require",
) -> ConnectorConfig:
    return ConnectorConfig(
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        credential=credential,
        ssl_mode=ssl_mode,
    )


def test_connector_creation_and_protocol() -> None:
    connector = PostgreSQLConnector()

    assert isinstance(connector, DataConnector)
    assert connector.connection_state is PostgreSQLConnectionState.DISCONNECTED
    assert connector.connect_timeout == DEFAULT_CONNECT_TIMEOUT_SECONDS


def test_timeout_configuration() -> None:
    connector = PostgreSQLConnector(connect_timeout=2.5)

    assert connector.connect_timeout == 2.5
    with pytest.raises(ValueError, match="connect_timeout"):
        PostgreSQLConnector(connect_timeout=0)
    with pytest.raises(ValueError, match="connect_timeout"):
        PostgreSQLConnector(connect_timeout=-1)


def test_registry_resolves_postgresql_connector() -> None:
    connector = create_connector(DataSourceType.POSTGRESQL)

    assert isinstance(connector, PostgreSQLConnector)
    assert is_implemented(DataSourceType.POSTGRESQL) is True
    assert DataSourceType.POSTGRESQL in implemented_types()
    assert connector.connect_timeout == settings.POSTGRES_CONNECT_TIMEOUT
    assert create_connector(DataSourceType.POSTGRESQL) is not connector


def test_unsupported_connectors_still_raise() -> None:
    with pytest.raises(UnsupportedConnectorError, match="MYSQL"):
        create_connector(DataSourceType.MYSQL)
    with pytest.raises(UnsupportedConnectorError, match="CSV"):
        create_connector(DataSourceType.CSV)


def test_ssl_mode_defaults_to_require_and_rejects_unknown() -> None:
    assert resolve_ssl_mode(None) == DEFAULT_SSL_MODE
    assert resolve_ssl_mode(" PREFER ") == "prefer"
    assert resolve_ssl_mode("verify-full") == "verify-full"
    with pytest.raises(ConnectorError, match="Invalid PostgreSQL SSL mode"):
        resolve_ssl_mode("not-a-mode")
    with pytest.raises(ConnectorError, match="Invalid PostgreSQL SSL mode"):
        resolve_ssl_mode("require;options=-cshared_preload_libraries")
    with pytest.raises(ConnectorError, match="Invalid PostgreSQL SSL mode"):
        resolve_ssl_mode("")


def test_connection_kwargs_pass_password_as_parameter() -> None:
    config = _config(credential=SPECIAL_SECRET)
    params = connect_kwargs(config, connect_timeout=10.0, ssl_mode="require")

    assert params["password"] == SPECIAL_SECRET
    assert params["host"] == config.host
    assert params["port"] == config.port
    assert params["dbname"] == config.database_name
    assert params["user"] == config.username
    assert params["sslmode"] == "require"
    assert "default_transaction_read_only=on" in str(params["options"])
    public_values = [value for key, value in params.items() if key != "password"]
    assert all(SPECIAL_SECRET not in str(value) for value in public_values)


def test_repr_and_str_omit_credentials() -> None:
    connector = PostgreSQLConnector(connect_timeout=3)

    assert SECRET not in repr(connector)
    assert SECRET not in str(connector)
    assert "password" not in repr(connector).lower()
    assert connector.connection_state.value in repr(connector)


def test_error_mapping_classifies_failures() -> None:
    timeout = map_postgres_error(TimeoutError())
    auth = map_postgres_error(
        psycopg.OperationalError("password authentication failed for user")
    )
    refused = map_postgres_error(psycopg.OperationalError("Connection refused"))
    dns = map_postgres_error(psycopg.OperationalError("could not translate host name"))
    ssl = map_postgres_error(psycopg.OperationalError("SSL error: certificate"))
    missing = map_postgres_error(
        psycopg.OperationalError('database "missing" does not exist')
    )
    query = map_postgres_error(
        psycopg.ProgrammingError("syntax error"),
        operation="query",
    )
    query_timeout = map_postgres_error(TimeoutError(), operation="query")

    assert isinstance(timeout, ConnectorConnectionError)
    assert "timed out" in str(timeout)
    assert isinstance(query_timeout, ConnectorQueryError)
    assert "timed out" in str(query_timeout)
    assert isinstance(auth, ConnectorAuthenticationError)
    assert isinstance(refused, ConnectorConnectionError)
    assert "refused" in str(refused).lower()
    assert isinstance(dns, ConnectorConnectionError)
    assert "resolve" in str(dns).lower()
    assert isinstance(ssl, ConnectorConnectionError)
    assert "ssl" in str(ssl).lower()
    assert isinstance(missing, ConnectorConnectionError)
    assert "unavailable" in str(missing).lower()
    assert isinstance(query, ConnectorQueryError)


def test_error_mapping_redacts_credentials() -> None:
    uri = f"postgresql://readonly:{SECRET}@db.internal.example:5432/analytics"
    mapped = map_postgres_error(
        psycopg.OperationalError(f"could not connect to {uri} password={SECRET}")
    )

    assert SECRET not in str(mapped)
    assert SECRET not in repr(mapped)
    assert "Unable to connect to the PostgreSQL data source" in str(mapped)


def test_invalid_ssl_mode_fails_before_connecting() -> None:
    async def _run_connect() -> None:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config(ssl_mode="trust-me"))

    with pytest.raises(ConnectorError, match="Invalid PostgreSQL SSL mode"):
        run_async(_run_connect())


def test_test_connection_requires_active_connection() -> None:
    async def _run_test() -> None:
        connector = PostgreSQLConnector()
        await connector.test_connection()

    with pytest.raises(ConnectorConnectionError, match="Not connected"):
        run_async(_run_test())


def test_metadata_methods_are_unsupported_and_writes_are_rejected() -> None:
    async def _run_unsupported() -> None:
        connector = PostgreSQLConnector()
        with pytest.raises(UnsupportedOperationError):
            await connector.get_schemas()
        with pytest.raises(UnsupportedOperationError):
            await connector.get_tables()
        with pytest.raises(UnsupportedOperationError):
            await connector.get_columns("events")
        with pytest.raises(ConnectorConnectionError, match="Not connected"):
            await connector.execute_query("SELECT 1")
        with pytest.raises(ConnectorQueryError, match="Unable to query"):
            await connector.execute_query("INSERT INTO t VALUES (1)")

    run_async(_run_unsupported())


def test_fetch_all_requires_active_connection() -> None:
    async def _run() -> None:
        connector = PostgreSQLConnector()
        await connector._fetch_all("SELECT 1")

    with pytest.raises(ConnectorConnectionError, match="Not connected"):
        run_async(_run())


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET x = 1",
        "DELETE FROM t",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN x int",
        "CREATE TABLE t (id int)",
        "TRUNCATE t",
        "SELECT 1; DROP TABLE users",
        "SELECT 1; INSERT INTO t VALUES (1)",
    ],
)
def test_fetch_all_rejects_write_and_stacked_sql(query: str) -> None:
    async def _run() -> None:
        connector = PostgreSQLConnector()
        await connector._fetch_all(query)

    with pytest.raises(ConnectorQueryError, match="Unable to query"):
        run_async(_run())


def test_fetch_all_uses_bound_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, object]] = []

    class _Cursor:
        def __init__(self) -> None:
            self.description = [type("Col", (), {"name": "schema_name"})()]

        async def fetchall(self) -> list[tuple[str]]:
            return [("public",)]

        async def close(self) -> None:
            return None

    class _Connection:
        closed = False

        async def execute(self, query: str, params: object = None) -> _Cursor:
            executed.append((query, params))
            return _Cursor()

        async def close(self) -> None:
            return None

    connection = _Connection()

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return connection

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> list[dict[str, object]]:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())
        try:
            return await connector._fetch_all(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name = %(schema_name)s",
                {"schema_name": "public'; DROP TABLE users; --"},
            )
        finally:
            await connector.disconnect()

    rows = run_async(_run())
    assert rows == [{"schema_name": "public"}]
    assert executed
    query, params = executed[0]
    assert "DROP TABLE" not in query
    assert params == {"schema_name": "public'; DROP TABLE users; --"}


def test_fetch_all_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        closed = False

        async def execute(self, query: str, params: object = None) -> None:
            await asyncio.sleep(5)

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=0.2)
        await connector.connect(_config())
        try:
            await connector._fetch_all("SELECT 1")
        finally:
            await connector.disconnect()

    with pytest.raises(ConnectorQueryError, match="timed out"):
        run_async(_run())


def test_disconnect_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        closed = False

        async def close(self) -> None:
            await asyncio.sleep(5)

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=0.2)
        await connector.connect(_config())
        await connector.disconnect()

    run_async(_run())


def test_connector_has_no_public_fetch_all() -> None:
    assert not hasattr(PostgreSQLConnector, "fetch_all")
    assert not hasattr(PostgreSQLConnector, "fetch_sample_rows")


def test_fetch_sample_rows_quotes_identifiers_and_binds_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, object]] = []

    class _Cursor:
        def __init__(self) -> None:
            self.description = [
                type("Col", (), {"name": "id"})(),
                type("Col", (), {"name": "email"})(),
            ]

        async def fetchall(self) -> list[tuple[int, str]]:
            return [(1, "ada@example.com")]

        async def close(self) -> None:
            return None

    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> _Cursor:
            sql_text = (
                query.as_string(None) if hasattr(query, "as_string") else str(query)
            )
            executed.append((sql_text, params))
            return _Cursor()

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> list[dict[str, object]]:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())
        try:
            return await connector._fetch_sample_rows(
                'public"; DROP TABLE users; --',
                "customers",
                ["id", "email"],
                limit=10,
            )
        finally:
            await connector.disconnect()

    rows = run_async(_run())
    assert rows == [{"id": 1, "email": "ada@example.com"}]
    assert executed
    query, params = executed[0]
    assert "SELECT" in query.upper()
    assert "*" not in query.split("FROM", 1)[0]
    assert "LIMIT" in query.upper()
    assert "ORDER BY RANDOM" not in query.upper()
    assert params == {"row_limit": 10}
    assert '"customers"' in query
    assert '"id"' in query
    assert '"email"' in query
    assert query.strip().upper().startswith("SELECT")


def test_fetch_sample_rows_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> None:
            await asyncio.sleep(5)

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=0.2)
        await connector.connect(_config())
        try:
            await connector._fetch_sample_rows("public", "customers", ["id"], limit=5)
        finally:
            await connector.disconnect()

    with pytest.raises(ConnectorQueryError, match="timed out"):
        run_async(_run())


def test_connector_has_no_write_api() -> None:
    names = {name for name in dir(PostgreSQLConnector) if not name.startswith("_")}
    assert names.isdisjoint(
        {
            "insert",
            "update",
            "delete",
            "drop",
            "alter",
            "truncate",
            "create",
            "execute_write",
        }
    )


def test_connect_maps_driver_errors_and_sets_failed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_connect(**_kwargs: object) -> None:
        raise psycopg.OperationalError("Connection refused")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run_connect() -> PostgreSQLConnector:
        connector = PostgreSQLConnector(connect_timeout=1)
        with pytest.raises(ConnectorConnectionError, match="refused"):
            await connector.connect(_config())
        return connector

    connector = run_async(_run_connect())
    assert connector.connection_state is PostgreSQLConnectionState.FAILED


def test_connect_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _slow_connect(**_kwargs: object) -> None:
        await asyncio.sleep(5)

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _slow_connect)

    async def _run_connect() -> None:
        connector = PostgreSQLConnector(connect_timeout=0.2)
        await connector.connect(_config())

    with pytest.raises(ConnectorConnectionError, match="timed out"):
        run_async(_run_connect())


def test_lifecycle_disconnects_after_mocked_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cursor:
        async def fetchone(self) -> tuple[int]:
            return (1,)

        async def close(self) -> None:
            return None

    class _Connection:
        closed = False

        async def execute(self, query: str) -> _Cursor:
            assert query == "SELECT 1"
            return _Cursor()

        async def close(self) -> None:
            self.closed = True

    connection = _Connection()

    async def _fake_connect(**kwargs: object) -> _Connection:
        assert kwargs["password"] == SECRET
        assert "sslmode" in kwargs
        return connection

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run_lifecycle() -> PostgreSQLConnector:
        connector = PostgreSQLConnector(connect_timeout=1)
        async with connector_lifecycle(connector, _config()) as active:
            assert active is connector
            assert connector.connection_state is PostgreSQLConnectionState.CONNECTED
            result = await connector.test_connection()
            assert result.success is True
            assert SECRET not in repr(connector)
            assert SECRET not in str(vars(connector))
        return connector

    connector = run_async(_run_lifecycle())
    assert connection.closed is True
    assert connector.connection_state is PostgreSQLConnectionState.DISCONNECTED


def test_connect_does_not_log_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _fake_connect(**_kwargs: object) -> None:
        raise psycopg.OperationalError(
            f"password={SECRET} postgresql://readonly:{SECRET}@host/db"
        )

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)
    logger = logging.getLogger("app.connectors.postgresql")

    async def _run_connect() -> None:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(ConnectorConnectionError),
    ):
        run_async(_run_connect())

    text = caplog.text
    assert SECRET not in text
    assert f"postgresql://readonly:{SECRET}@" not in text
    assert "password=[REDACTED]" in text or "password=" not in text.lower()


def test_invalid_host_fails_quickly() -> None:
    async def _run_connect() -> None:
        connector = PostgreSQLConnector(connect_timeout=2)
        await connector.connect(_config(host="nonexistent.invalid", ssl_mode="disable"))

    with pytest.raises(
        (ConnectorConnectionError, ConnectorAuthenticationError),
    ) as exc_info:
        run_async(_run_connect())

    assert SECRET not in str(exc_info.value)
    assert "OperationalError" not in str(exc_info.value)


def test_execute_query_applies_database_row_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[object, object]] = []

    class _Column:
        name = "n"

    class _Cursor:
        def __init__(self) -> None:
            self.description = [_Column()]

        async def fetchmany(self, size: int) -> list[tuple[int]]:
            assert size == 4
            return [(1,), (2,), (3,), (4,)]

        async def close(self) -> None:
            return None

    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> _Cursor:
            executed.append((str(query), params))
            return _Cursor()

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> QueryResult:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())
        try:
            return await connector.execute_query(
                "SELECT n FROM numbers ORDER BY n", limit=3
            )
        finally:
            await connector.disconnect()

    result = run_async(_run())
    assert result.truncated is True
    assert result.rows == ((1,), (2,), (3,))
    query, params = executed[0]
    assert "LIMIT" in query.upper()
    assert "OFFSET 0" in query.upper()
    assert params == {"row_limit": 4}


def test_execute_query_timeout_does_not_hang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> None:
            await asyncio.sleep(5)

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=0.2)
        await connector.connect(_config())
        try:
            await connector.execute_query("SELECT 1")
        finally:
            await connector.disconnect()

    with pytest.raises(ConnectorQueryError, match="timed out"):
        run_async(_run())


def test_execute_query_success_does_not_log_sql(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _Column:
        name = "email"

    class _Cursor:
        def __init__(self) -> None:
            self.description = [_Column()]

        async def fetchmany(self, size: int) -> list[tuple[str]]:
            return [("a@example.com",)]

        async def close(self) -> None:
            return None

    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> _Cursor:
            return _Cursor()

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)
    logger = logging.getLogger("app.connectors.postgresql")
    sql = "SELECT email FROM users WHERE email = 'victim@example.com'"

    async def _run() -> QueryResult:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())
        try:
            return await connector.execute_query(sql, limit=1)
        finally:
            await connector.disconnect()

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = run_async(_run())

    assert result.rows == (("a@example.com",),)
    text = caplog.text
    assert sql not in text
    assert "victim@example.com" not in text
    assert "SELECT email" not in text


def test_execute_query_failure_does_not_log_sql_or_parameters(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sensitive_sql = (
        "SELECT ssn, email FROM customers "
        "WHERE email = 'victim@example.com' AND token = 'super-secret-token'"
    )
    driver_message = (
        f'syntax error at or near "AND"\n'
        f"LINE 1: {sensitive_sql}\n"
        f"HINT: parameters={{'row_limit': 2}}"
    )

    class _Connection:
        closed = False

        async def execute(self, query: object, params: object = None) -> None:
            raise psycopg.ProgrammingError(driver_message)

        async def close(self) -> None:
            return None

    async def _fake_connect(**_kwargs: object) -> _Connection:
        return _Connection()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", _fake_connect)
    logger = logging.getLogger("app.connectors.postgresql")

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=1)
        await connector.connect(_config())
        try:
            await connector.execute_query(sensitive_sql, limit=1)
        finally:
            await connector.disconnect()

    with (
        caplog.at_level(logging.WARNING, logger=logger.name),
        pytest.raises(ConnectorQueryError) as exc_info,
    ):
        run_async(_run())

    # Sanitized application error must remain unchanged for callers.
    assert str(exc_info.value) == "Unable to query the PostgreSQL data source"

    text = caplog.text
    assert "execute_query" in text
    assert "ProgrammingError" in text
    assert sensitive_sql not in text
    assert "SELECT ssn" not in text
    assert "victim@example.com" not in text
    assert "super-secret-token" not in text
    assert "row_limit" not in text
    assert "LINE 1" not in text
    assert "detail=" not in text


def test_invalid_port_fails_cleanly() -> None:
    async def _run_connect() -> None:
        connector = PostgreSQLConnector(connect_timeout=2)
        await connector.connect(_config(host="127.0.0.1", port=1, ssl_mode="disable"))

    with pytest.raises(ConnectorConnectionError) as exc_info:
        run_async(_run_connect())

    assert SECRET not in str(exc_info.value)
    assert "password" not in str(exc_info.value).lower()
