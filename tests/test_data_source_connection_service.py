from __future__ import annotations

import ast
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.connectors import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorAuthenticationError,
    ConnectorConfig,
    ConnectorConnectionError,
    PostgreSQLConnector,
    QueryResult,
    SchemaInfo,
    TableInfo,
    UnsupportedConnectorError,
    UnsupportedOperationError,
    create_connector,
    register_connector,
)
from app.core.logging import RedactingFilter
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceStatus,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.services.credentials import encrypt_secret
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionService,
    DataSourceNotFoundError,
)
from tests.conftest import run_async

CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"
CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)
_SERVICE_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "services"
    / "data_source_connections.py"
)
_FORBIDDEN_IMPORTS = {
    "fastapi",
    "app.api.deps",
    "app.api.routes",
    "app.db.session",
    "psycopg",
    "asyncpg",
    "cryptography",
    "cryptography.hazmat.primitives.ciphers.aead",
}


class RecordingConnector:
    """Test double. Not a production connector."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connected = False
        self.config: ConnectorConfig | None = None
        self.fail_connect: Exception | None = None
        self.fail_test: Exception | None = None
        self.test_success = True

    async def connect(self, config: ConnectorConfig) -> None:
        self.connect_calls += 1
        self.config = config
        if self.fail_connect is not None:
            raise self.fail_connect
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def test_connection(self) -> ConnectionTestResult:
        if not self.connected:
            raise ConnectorConnectionError("Not connected")
        if self.fail_test is not None:
            raise self.fail_test
        return ConnectionTestResult(success=self.test_success)

    async def get_schemas(self) -> list[SchemaInfo]:
        raise UnsupportedOperationError("Schema discovery is not supported")

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        raise UnsupportedOperationError("Table discovery is not supported")

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        raise UnsupportedOperationError("Column discovery is not supported")

    async def execute_query(self, query: str) -> QueryResult:
        raise UnsupportedOperationError("Query execution is not supported")


def _user() -> User:
    return User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
    )


def _seed_data_source(
    db_session: Session,
    *,
    source_type: DataSourceType = DataSourceType.POSTGRESQL,
    with_connection: bool = True,
    password: str = CUSTOMER_PASSWORD,
    encrypted_password: str | None = None,
    host: str = "db.internal.example",
    port: int = 5432,
    database_name: str = "analytics",
    username: str = "readonly",
    ssl_mode: str = "prefer",
) -> DataSource:
    user = _user()
    organization = Organization(
        name="AnalyticCastle",
        slug=f"analyticcastle-{uuid.uuid4().hex[:8]}",
    )
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Analytics",
        slug=f"analytics-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    data_source = DataSource(
        workspace_id=workspace.id,
        name="Production Analytics DB",
        type=source_type,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    if with_connection:
        ciphertext = (
            encrypted_password
            if encrypted_password is not None
            else encrypt_secret(password)
        )
        db_session.add(
            DataSourceConnection(
                data_source_id=data_source.id,
                host=host,
                port=port,
                database_name=database_name,
                username=username,
                encrypted_password=ciphertext,
                ssl_mode=ssl_mode,
            )
        )
        db_session.flush()
    return data_source


def _register_recording_connector() -> list[RecordingConnector]:
    created: list[RecordingConnector] = []

    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    return created


def test_existing_data_source_connection_succeeds(db_session: Session) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert result.success is True
    assert result.message == ""
    assert "password" not in result.__dict__
    assert CUSTOMER_PASSWORD not in repr(result)
    assert data_source.status is DataSourceStatus.ACTIVE
    assert data_source.last_tested_at is not None
    assert data_source.last_tested_at.tzinfo is not None
    assert created[0].disconnect_calls == 1
    assert created[0].connected is False


def test_missing_data_source_raises(db_session: Session) -> None:
    service = DataSourceConnectionService(db_session)

    with pytest.raises(DataSourceNotFoundError, match="not found"):
        run_async(service.test_connection(uuid.uuid4()))


def test_missing_connection_configuration_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, with_connection=False)
    service = DataSourceConnectionService(db_session)

    with pytest.raises(ConnectionConfigurationError, match="missing"):
        run_async(service.test_connection(data_source.id))

    db_session.refresh(data_source)
    assert data_source.status is DataSourceStatus.INACTIVE
    assert data_source.last_tested_at is None


def test_unsupported_connector_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, source_type=DataSourceType.MYSQL)
    service = DataSourceConnectionService(db_session)

    with pytest.raises(UnsupportedConnectorError, match="MYSQL"):
        run_async(service.test_connection(data_source.id))

    db_session.refresh(data_source)
    assert data_source.status is DataSourceStatus.INACTIVE
    assert data_source.last_tested_at is None


def test_encrypted_credential_is_decrypted_for_connector(
    db_session: Session,
) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)
    assert data_source.connection is not None
    db_session.refresh(data_source.connection)

    assert result.success is True
    assert created[0].config is not None
    assert created[0].config.credential == CUSTOMER_PASSWORD
    assert created[0].config.host == "db.internal.example"
    assert created[0].config.port == 5432
    assert created[0].config.database_name == "analytics"
    assert created[0].config.username == "readonly"
    assert created[0].config.ssl_mode == "prefer"
    assert data_source.connection.encrypted_password == ciphertext
    assert data_source.connection.encrypted_password != CUSTOMER_PASSWORD
    assert CUSTOMER_PASSWORD not in str(vars(data_source.connection))
    assert "password" not in DataSourceConnection.__table__.columns


def test_plaintext_is_not_persisted_on_the_model(db_session: Session) -> None:
    _register_recording_connector()
    data_source = _seed_data_source(db_session)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))
    assert data_source.connection is not None
    db_session.refresh(data_source.connection)

    connection = data_source.connection
    assert connection.encrypted_password == ciphertext
    assert getattr(connection, "password", None) is None
    assert CUSTOMER_PASSWORD not in repr(connection)


def test_malformed_credential_is_mapped_to_configuration_error(
    db_session: Session,
) -> None:
    data_source = _seed_data_source(
        db_session,
        encrypted_password="not-valid-ciphertext",
    )
    service = DataSourceConnectionService(db_session)

    with pytest.raises(ConnectionConfigurationError, match="credentials"):
        run_async(service.test_connection(data_source.id))

    db_session.refresh(data_source)
    assert data_source.status is DataSourceStatus.INACTIVE
    assert data_source.last_tested_at is None


def test_postgresql_connector_is_resolved_through_the_registry(
    db_session: Session,
) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))

    assert len(created) == 1
    assert created[0].connect_calls == 1
    assert not isinstance(created[0], PostgreSQLConnector)


def test_default_registry_still_builds_postgresql_connector() -> None:
    connector = create_connector(DataSourceType.POSTGRESQL)

    assert isinstance(connector, PostgreSQLConnector)
    assert create_connector(DataSourceType.POSTGRESQL) is not connector


def test_unsupported_type_does_not_instantiate_postgresql(
    db_session: Session,
) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session, source_type=DataSourceType.CSV)
    service = DataSourceConnectionService(db_session)

    with pytest.raises(UnsupportedConnectorError, match="CSV"):
        run_async(service.test_connection(data_source.id))

    assert created == []


def test_connector_is_disconnected_after_success(db_session: Session) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))

    assert created[0].connect_calls == 1
    assert created[0].disconnect_calls == 1
    assert created[0].connected is False


def test_connector_is_disconnected_after_query_failure(db_session: Session) -> None:
    created: list[RecordingConnector] = []

    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_test = ConnectorConnectionError("Unable to query")
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))

    assert result.success is False
    assert created[0].disconnect_calls == 1
    assert created[0].connected is False


def test_connection_failure_returns_safe_result(db_session: Session) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError("Unable to connect")
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert result == ConnectionTestResult(success=False)
    assert CUSTOMER_PASSWORD not in repr(result)
    assert data_source.status is DataSourceStatus.ERROR
    assert data_source.last_tested_at is not None


def test_timeout_is_mapped_to_failed_result(db_session: Session) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError(
            "PostgreSQL connection timed out"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    before = datetime.now(UTC)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert result.success is False
    assert data_source.status is DataSourceStatus.ERROR
    assert data_source.last_tested_at is not None
    assert data_source.last_tested_at >= before
    assert data_source.last_tested_at.tzinfo is not None


def test_invalid_credentials_are_mapped_to_failed_result(
    db_session: Session,
) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorAuthenticationError(
            f"password={CUSTOMER_PASSWORD} uri={CONNECTION_URI}"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    dumped = {"success": result.success}

    assert result.success is False
    assert CUSTOMER_PASSWORD not in str(dumped)
    assert CONNECTION_URI not in str(dumped)
    assert "password" not in dumped


def test_invalid_database_is_mapped_to_failed_result(db_session: Session) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError(
            "The PostgreSQL database is unavailable"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert result.success is False
    assert data_source.status is DataSourceStatus.ERROR


def test_successful_test_sets_active_and_timestamp(db_session: Session) -> None:
    _register_recording_connector()
    data_source = _seed_data_source(db_session)
    before = datetime.now(UTC)
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert data_source.status is DataSourceStatus.ACTIVE
    assert data_source.last_tested_at is not None
    assert data_source.last_tested_at >= before
    assert data_source.last_tested_at.tzinfo is not None
    assert data_source.last_tested_at.utcoffset() is not None


def test_failed_test_sets_error_and_timestamp(db_session: Session) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError("Connection refused")
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)

    assert data_source.status is DataSourceStatus.ERROR
    assert data_source.last_tested_at is not None
    assert data_source.last_tested_at.tzinfo is not None


def test_precondition_errors_do_not_update_status(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, with_connection=False)
    service = DataSourceConnectionService(db_session)

    with pytest.raises(ConnectionConfigurationError):
        run_async(service.test_connection(data_source.id))

    db_session.refresh(data_source)
    assert data_source.status is DataSourceStatus.INACTIVE
    assert data_source.last_tested_at is None


def test_operations_do_not_share_connector_instances(db_session: Session) -> None:
    created = _register_recording_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    run_async(service.test_connection(data_source.id))
    run_async(service.test_connection(data_source.id))

    assert len(created) == 2
    assert created[0] is not created[1]
    assert created[0].config is not created[1].config


def test_password_is_not_logged_or_returned(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorAuthenticationError(
            f"password={CUSTOMER_PASSWORD} connection_string={CONNECTION_URI} "
            f"encrypted_password=v1:nonce:payload"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    logger = logging.getLogger("app.services.data_source_connections")
    logger.addFilter(RedactingFilter())
    data_source = _seed_data_source(db_session)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    service = DataSourceConnectionService(db_session)

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = run_async(service.test_connection(data_source.id))

    text = caplog.text
    assert result.success is False
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert ciphertext not in text
    assert CUSTOMER_PASSWORD not in repr(result)
    assert "Data source connection test failed" in text
    assert str(data_source.id) in text
    assert "POSTGRESQL" in text


def test_service_does_not_access_encryption_or_http_internals() -> None:
    source = _SERVICE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    leaked = imported & _FORBIDDEN_IMPORTS
    assert not leaked, f"service imports forbidden modules: {leaked}"
    assert "DATA_SOURCE_ENCRYPTION_KEY" not in source
    assert "DATABASE_URL" not in source
    assert "decrypt_secret" not in source
    assert "encrypt_secret" not in source
    assert "AESGCM" not in source
    assert "Request" not in source
    assert "Response" not in source
    assert "Depends" not in source
    assert "APIRouter" not in source
    assert "connector_config_from_connection" in source
    assert "create_connector" in source
    assert "connector_lifecycle" in source
    assert "SELECT 1" not in source
    assert "SessionLocal" not in source
    assert "get_db" not in source


def test_driver_errors_are_not_raised_to_the_caller(db_session: Session) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = TimeoutError()
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))

    assert result.success is False
    assert result.message == ""


def test_connection_test_result_omits_internal_error_details(
    db_session: Session,
) -> None:
    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError(
            "OperationalError: SSL SYSCALL error"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))

    assert result == ConnectionTestResult(success=False)
    assert "OperationalError" not in result.message
    assert "SSL" not in result.message
