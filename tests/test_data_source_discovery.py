from __future__ import annotations

import logging
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorConfig,
    ConnectorQueryError,
    QueryResult,
    SchemaInfo,
    TableInfo,
    UnsupportedConnectorError,
    UnsupportedOperationError,
    register_connector,
)
from app.core.logging import RedactingFilter
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceSchema,
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
    DataSourceNotFoundError,
)
from app.services.data_source_discovery import DataSourceDiscoveryService
from app.services.discovery_exceptions import MetadataDiscoveryError
from app.services.discovery_types import DiscoveryResult
from tests.conftest import run_async
from tests.test_postgresql_discovery import FakeCatalogConnector, _sample_catalog

CUSTOMER_PASSWORD = "CustomerDiscoverySecret!@#"
CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)


class DiscoveryConnector(FakeCatalogConnector):
    """Test double. Not a production connector."""

    def __init__(self) -> None:
        super().__init__(_sample_catalog())
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connected = False
        self.config: ConnectorConfig | None = None
        self.fail_connect: Exception | None = None

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
        return ConnectionTestResult(success=True)

    async def get_schemas(self) -> list[SchemaInfo]:
        return []

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        return []

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        return []

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
    workspace: Workspace | None = None,
    name: str = "Production Analytics DB",
) -> DataSource:
    user = _user()
    if workspace is None:
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
    else:
        db_session.add(user)
        db_session.flush()
    data_source = DataSource(
        workspace_id=workspace.id,
        name=name,
        type=source_type,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    if with_connection:
        db_session.add(
            DataSourceConnection(
                data_source_id=data_source.id,
                host="db.internal.example",
                port=5432,
                database_name="analytics",
                username="readonly",
                encrypted_password=encrypt_secret(password),
                ssl_mode="prefer",
            )
        )
        db_session.flush()
    return data_source


def _register_discovery_connector() -> list[DiscoveryConnector]:
    created: list[DiscoveryConnector] = []

    def _builder() -> DiscoveryConnector:
        connector = DiscoveryConnector()
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    return created


def test_discovery_loads_decrypts_connects_and_disconnects(
    db_session: Session,
) -> None:
    created = _register_discovery_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceDiscoveryService(db_session)

    result = run_async(
        service.discover(data_source.id, workspace_id=data_source.workspace_id)
    )

    assert isinstance(result, DiscoveryResult)
    assert [schema.name for schema in result.schemas] == [
        "analytics",
        "empty_schema",
        "public",
        "reporting",
    ]
    assert created[0].connect_calls == 1
    assert created[0].disconnect_calls == 1
    assert created[0].connected is False
    assert created[0].config is not None
    assert created[0].config.credential == CUSTOMER_PASSWORD
    db_session.refresh(data_source)
    assert data_source.status is DataSourceStatus.INACTIVE
    assert data_source.last_tested_at is None


def test_discovery_does_not_persist_metadata(db_session: Session) -> None:
    _register_discovery_connector()
    data_source = _seed_data_source(db_session)
    service = DataSourceDiscoveryService(db_session)

    run_async(service.discover(data_source.id, workspace_id=data_source.workspace_id))

    remaining = db_session.scalars(select(DataSourceSchema)).all()
    assert remaining == []


def test_workspace_isolation_hides_other_data_sources(db_session: Session) -> None:
    _register_discovery_connector()
    data_source_a = _seed_data_source(db_session, name="Workspace A DB")
    organization = data_source_a.workspace.organization
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Finance",
        slug=f"finance-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()
    data_source_b = _seed_data_source(
        db_session,
        workspace=workspace_b,
        name="Workspace B DB",
    )
    service = DataSourceDiscoveryService(db_session)

    with pytest.raises(DataSourceNotFoundError, match="not found"):
        run_async(
            service.discover(data_source_b.id, workspace_id=data_source_a.workspace_id)
        )

    result = run_async(
        service.discover(data_source_a.id, workspace_id=data_source_a.workspace_id)
    )
    assert result.schemas


def test_missing_data_source_raises(db_session: Session) -> None:
    service = DataSourceDiscoveryService(db_session)
    with pytest.raises(DataSourceNotFoundError, match="not found"):
        run_async(service.discover(uuid.uuid4(), workspace_id=uuid.uuid4()))


def test_missing_connection_configuration_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, with_connection=False)
    service = DataSourceDiscoveryService(db_session)
    with pytest.raises(ConnectionConfigurationError, match="missing"):
        run_async(
            service.discover(data_source.id, workspace_id=data_source.workspace_id)
        )


def test_unsupported_connector_type_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, source_type=DataSourceType.MYSQL)
    service = DataSourceDiscoveryService(db_session)
    with pytest.raises(UnsupportedConnectorError):
        run_async(
            service.discover(data_source.id, workspace_id=data_source.workspace_id)
        )


def test_discovery_errors_and_logs_omit_credentials(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created: list[DiscoveryConnector] = []

    def _builder() -> DiscoveryConnector:
        connector = DiscoveryConnector()
        connector.fail_error = ConnectorQueryError(
            f"password={CUSTOMER_PASSWORD} connection_string={CONNECTION_URI}"
        )
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    logger = logging.getLogger("app.services.data_source_discovery")
    logger.addFilter(RedactingFilter())
    data_source = _seed_data_source(db_session)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    service = DataSourceDiscoveryService(db_session)

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(MetadataDiscoveryError) as exc_info,
    ):
        run_async(
            service.discover(data_source.id, workspace_id=data_source.workspace_id)
        )

    message = str(exc_info.value)
    text = caplog.text
    assert CUSTOMER_PASSWORD not in message
    assert CONNECTION_URI not in message
    assert "OperationalError" not in message
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert ciphertext not in text
    assert str(data_source.id) in text
    assert created[0].disconnect_calls == 1


def test_successful_discovery_logs_counts_without_secrets(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _register_discovery_connector()
    logger = logging.getLogger("app.services.data_source_discovery")
    logger.addFilter(RedactingFilter())
    data_source = _seed_data_source(db_session)
    service = DataSourceDiscoveryService(db_session)

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = run_async(
            service.discover(data_source.id, workspace_id=data_source.workspace_id)
        )

    text = caplog.text
    assert "Metadata discovery started" in text
    assert "Metadata discovery completed" in text
    assert f"schemas_discovered={len(result.schemas)}" in text
    assert f"tables_discovered={len(result.tables)}" in text
    assert f"columns_discovered={len(result.columns)}" in text
    assert f"relationships_discovered={len(result.relationships)}" in text
    assert str(data_source.id) in text
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert "encrypted_password" not in text
