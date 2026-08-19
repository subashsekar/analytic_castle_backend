from __future__ import annotations

import ast
import asyncio
import inspect
import logging
from pathlib import Path
from typing import get_type_hints

import pytest

from app.connectors import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorAuthenticationError,
    ConnectorConfig,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorQueryError,
    DataConnector,
    PostgreSQLConnector,
    QueryResult,
    SchemaInfo,
    TableInfo,
    UnsupportedConnectorError,
    UnsupportedOperationError,
    connector_lifecycle,
    create_connector,
    implemented_types,
    is_implemented,
    recognized_types,
    register_connector,
    sanitize_connector_message,
    unregister_connector,
)
from app.connectors.base import DataConnector as DataConnectorProtocol
from app.enums import DataSourceType

SECRET = "SuperSecretCustomerPassword!"
CONNECTION_URI = f"postgresql://readonly:{SECRET}@db.internal.example:5432/analytics"

_CONNECTOR_DIR = Path(__file__).resolve().parents[1] / "app" / "connectors"
_FORBIDDEN_IMPORTS = {
    "app.db.session",
    "app.db.models",
    "app.db.models.data_source",
    "app.db.models.data_source_connection",
    "app.main",
    "app.services.credentials",
}


class FakeConnector:
    """Test double. Not a production connector."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0
        self._retained_credential: str | None = None
        self._host: str | None = None

    async def connect(self, config: ConnectorConfig) -> None:
        self.connect_calls += 1
        _ = config.credential
        self._host = config.host
        self._retained_credential = None
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False
        self._host = None
        self._retained_credential = None

    async def test_connection(self) -> ConnectionTestResult:
        if not self.connected:
            raise ConnectorConnectionError("Not connected")
        return ConnectionTestResult(success=True, message="ok")

    async def get_schemas(self) -> list[SchemaInfo]:
        return [SchemaInfo(name="public")]

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        return [TableInfo(name="events", schema=schema or "public")]

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        return [
            ColumnInfo(name="id", data_type="integer", nullable=False),
            ColumnInfo(name=table, data_type="text"),
        ]

    async def execute_query(self, query: str) -> QueryResult:
        if query.strip().upper().startswith("UNSUPPORTED"):
            raise UnsupportedOperationError("Query is not supported")
        return QueryResult(columns=("id",), rows=((1,),))


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


def test_fake_connector_satisfies_protocol() -> None:
    connector = FakeConnector()

    assert isinstance(connector, DataConnector)
    assert isinstance(connector, DataConnectorProtocol)


def test_protocol_defines_required_async_methods() -> None:
    required = {
        "connect",
        "disconnect",
        "test_connection",
        "get_schemas",
        "get_tables",
        "get_columns",
        "execute_query",
    }

    for name in required:
        assert hasattr(DataConnector, name)
        assert inspect.iscoroutinefunction(getattr(FakeConnector, name))

    connect_hints = get_type_hints(FakeConnector.connect)
    assert connect_hints["config"] is ConnectorConfig
    assert connect_hints["return"] is type(None)

    test_hints = get_type_hints(FakeConnector.test_connection)
    assert test_hints["return"] is ConnectionTestResult

    schemas_hints = get_type_hints(FakeConnector.get_schemas)
    assert schemas_hints["return"] == list[SchemaInfo]

    tables_hints = get_type_hints(FakeConnector.get_tables)
    assert tables_hints["return"] == list[TableInfo]

    columns_hints = get_type_hints(FakeConnector.get_columns)
    assert columns_hints["return"] == list[ColumnInfo]

    query_hints = get_type_hints(FakeConnector.execute_query)
    assert query_hints["return"] is QueryResult


def test_protocol_can_be_used_as_a_type_contract() -> None:
    def accept(connector: DataConnector) -> DataConnector:
        return connector

    assert accept(FakeConnector()) is not None


def test_fake_connector_methods_return_declared_types() -> None:
    async def _run() -> None:
        connector = FakeConnector()
        config = _config()
        await connector.connect(config)

        test_result = await connector.test_connection()
        schemas = await connector.get_schemas()
        tables = await connector.get_tables("public")
        columns = await connector.get_columns("events", schema="public")
        query_result = await connector.execute_query("SELECT 1")

        assert isinstance(test_result, ConnectionTestResult)
        assert test_result.success is True
        assert schemas == [SchemaInfo(name="public")]
        assert tables == [TableInfo(name="events", schema="public")]
        assert columns[0] == ColumnInfo(name="id", data_type="integer", nullable=False)
        assert query_result == QueryResult(columns=("id",), rows=((1,),))

        await connector.disconnect()

    asyncio.run(_run())


def test_registry_recognizes_known_types_and_postgresql_implementation() -> None:
    assert recognized_types() == frozenset(DataSourceType)
    assert DataSourceType.POSTGRESQL in recognized_types()
    assert DataSourceType.MYSQL in recognized_types()
    assert DataSourceType.CSV in recognized_types()
    assert implemented_types() == frozenset({DataSourceType.POSTGRESQL})
    assert is_implemented(DataSourceType.POSTGRESQL) is True
    for source_type in DataSourceType:
        if source_type is DataSourceType.POSTGRESQL:
            continue
        assert is_implemented(source_type) is False


@pytest.mark.parametrize(
    "source_type",
    [
        source_type
        for source_type in DataSourceType
        if source_type is not DataSourceType.POSTGRESQL
    ],
)
def test_unregistered_types_raise_unsupported_connector_error(
    source_type: DataSourceType,
) -> None:
    with pytest.raises(UnsupportedConnectorError, match=source_type.value):
        create_connector(source_type)


def test_postgresql_does_not_fall_back_to_another_connector() -> None:
    unregister_connector(DataSourceType.POSTGRESQL)
    register_connector(DataSourceType.MYSQL, FakeConnector)

    mysql_connector = create_connector(DataSourceType.MYSQL)
    assert isinstance(mysql_connector, FakeConnector)

    with pytest.raises(UnsupportedConnectorError, match="POSTGRESQL"):
        create_connector(DataSourceType.POSTGRESQL)

    with pytest.raises(UnsupportedConnectorError, match="CSV"):
        create_connector(DataSourceType.CSV)


def test_registered_type_resolves_to_registered_builder() -> None:
    register_connector(DataSourceType.POSTGRESQL, FakeConnector)

    connector = create_connector(DataSourceType.POSTGRESQL)

    assert is_implemented(DataSourceType.POSTGRESQL) is True
    assert DataSourceType.POSTGRESQL in implemented_types()
    assert isinstance(connector, FakeConnector)
    assert isinstance(connector, DataConnector)


def test_registry_does_not_instantiate_unrelated_type() -> None:
    created: list[str] = []

    class TrackingConnector(FakeConnector):
        def __init__(self) -> None:
            created.append("postgresql")
            super().__init__()

    register_connector(DataSourceType.POSTGRESQL, TrackingConnector)

    with pytest.raises(UnsupportedConnectorError):
        create_connector(DataSourceType.MYSQL)

    assert created == []


def test_config_repr_and_str_omit_credentials() -> None:
    config = _config()

    rendered = repr(config)
    as_string = str(config)

    assert SECRET not in rendered
    assert SECRET not in as_string
    assert "credential" not in rendered
    assert "password" not in rendered.lower()
    assert config.host in rendered
    assert config.username in rendered
    assert config.credential == SECRET


def test_connector_errors_sanitize_credentials_in_messages() -> None:
    raw = f"password={SECRET} credential={SECRET} connection_string={CONNECTION_URI}"

    errors = (
        ConnectorError(raw),
        ConnectorConnectionError(raw),
        ConnectorAuthenticationError(raw),
        ConnectorQueryError(raw),
        UnsupportedOperationError(raw),
        UnsupportedConnectorError(raw),
    )

    for error in errors:
        text = str(error)
        assert SECRET not in text
        assert "postgresql://readonly:" not in text
        assert SECRET not in repr(error)


def test_sanitize_connector_message_redacts_uris_and_assignments() -> None:
    raw = f"could not connect to {CONNECTION_URI} password={SECRET}"

    sanitized = sanitize_connector_message(raw)

    assert SECRET not in sanitized
    assert "postgresql://readonly:[REDACTED]@" in sanitized
    assert "password=[REDACTED]" in sanitized


def test_connector_config_is_safe_to_log(caplog: pytest.LogCaptureFixture) -> None:
    config = _config()
    logger = logging.getLogger("connector-security-test")

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("Using connector config %s", config)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert SECRET not in messages
    assert SECRET not in caplog.text
    assert "credential" not in messages


def test_credential_assignment_is_redacted_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("connector-security-assignment")

    with caplog.at_level(logging.INFO, logger=logger.name):
        logger.info("connecting with credential=" + SECRET + " and password=" + SECRET)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert SECRET not in messages
    assert "credential=[REDACTED]" in messages
    assert "password=[REDACTED]" in messages


def test_connector_exception_logs_do_not_include_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("connector-security-exc")

    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise ConnectorConnectionError(
                f"failed uri={CONNECTION_URI} password={SECRET}"
            )
        except ConnectorConnectionError:
            logger.exception("Customer data source connection failed")

    text = caplog.text
    assert SECRET not in text
    assert f"postgresql://readonly:{SECRET}@" not in text


def test_lifecycle_connects_then_disconnects() -> None:
    async def _run() -> None:
        connector = FakeConnector()
        config = _config()

        async with connector_lifecycle(connector, config) as active:
            assert active is connector
            assert connector.connected is True
            result = await connector.test_connection()
            assert result.success is True

        assert connector.connected is False
        assert connector.connect_calls == 1
        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_disconnects_after_operation_error() -> None:
    async def _run() -> None:
        connector = FakeConnector()

        with pytest.raises(UnsupportedOperationError):
            async with connector_lifecycle(connector, _config()):
                assert connector.connected is True
                await connector.execute_query("UNSUPPORTED")

        assert connector.connected is False
        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_disconnects_when_connect_fails() -> None:
    class FailingConnectConnector(FakeConnector):
        async def connect(self, config: ConnectorConfig) -> None:
            self.connect_calls += 1
            raise ConnectorAuthenticationError("Authentication failed")

    async def _run() -> None:
        connector = FailingConnectConnector()

        with pytest.raises(ConnectorAuthenticationError):
            async with connector_lifecycle(connector, _config()):
                pytest.fail("must not enter the connector session")

        assert connector.connected is False
        assert connector.connect_calls == 1
        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_preserves_connect_error_when_disconnect_fails() -> None:
    class FailingConnectConnector(FakeConnector):
        async def connect(self, config: ConnectorConfig) -> None:
            self.connect_calls += 1
            raise ConnectorAuthenticationError("Authentication failed")

        async def disconnect(self) -> None:
            self.disconnect_calls += 1
            raise RuntimeError("disconnect failed")

    async def _run() -> None:
        connector = FailingConnectConnector()

        with pytest.raises(ConnectorAuthenticationError):
            async with connector_lifecycle(connector, _config()):
                pytest.fail("must not enter the connector session")

        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_preserves_operation_error_when_disconnect_fails() -> None:
    class FailingDisconnectConnector(FakeConnector):
        async def disconnect(self) -> None:
            self.disconnect_calls += 1
            raise RuntimeError("disconnect failed")

    async def _run() -> None:
        connector = FailingDisconnectConnector()

        with pytest.raises(UnsupportedOperationError):
            async with connector_lifecycle(connector, _config()):
                await connector.execute_query("UNSUPPORTED")

        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_success_when_disconnect_fails() -> None:
    class FailingDisconnectConnector(FakeConnector):
        async def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.connected = False
            raise RuntimeError("disconnect failed")

    async def _run() -> None:
        connector = FailingDisconnectConnector()
        async with connector_lifecycle(connector, _config()) as active:
            result = await active.test_connection()
            assert result.success is True
        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_lifecycle_reraises_cancellation() -> None:
    class SlowDisconnectConnector(FakeConnector):
        async def disconnect(self) -> None:
            self.disconnect_calls += 1
            self.connected = False
            raise asyncio.CancelledError()

    async def _run() -> None:
        connector = SlowDisconnectConnector()

        with pytest.raises(asyncio.CancelledError):
            async with connector_lifecycle(connector, _config()):
                raise asyncio.CancelledError()

        assert connector.disconnect_calls == 1

    asyncio.run(_run())


def test_connect_does_not_retain_credential() -> None:
    async def _run() -> None:
        connector = FakeConnector()

        async with connector_lifecycle(connector, _config()):
            assert connector._retained_credential is None
            assert SECRET not in repr(connector)
            assert SECRET not in str(vars(connector))

        assert connector._retained_credential is None
        assert SECRET not in str(vars(connector))

    asyncio.run(_run())


def test_connector_package_does_not_use_application_database() -> None:
    for path in sorted(_CONNECTOR_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        leaked = imported & _FORBIDDEN_IMPORTS
        assert not leaked, f"{path.name} imports application DB modules: {leaked}"

        source = path.read_text(encoding="utf-8")
        assert "SessionLocal" not in source
        assert "get_db" not in source
        assert "settings.DATABASE_URL" not in source
        assert "DATA_SOURCE_ENCRYPTION_KEY" not in source
        assert "encrypt_secret" not in source
        assert "decrypt_secret" not in source


def test_application_registers_postgresql_connector() -> None:
    from app.main import app

    assert app.title
    assert DataSourceType.POSTGRESQL in implemented_types()
    connector = create_connector(DataSourceType.POSTGRESQL)
    assert isinstance(connector, PostgreSQLConnector)
    with pytest.raises(UnsupportedConnectorError):
        create_connector(DataSourceType.MYSQL)
