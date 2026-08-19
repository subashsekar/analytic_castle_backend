from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.connectors import (
    PostgreSQLConnectionState,
    PostgreSQLConnector,
    connector_lifecycle,
)
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.services.credentials import (
    connector_config_from_connection,
    encrypt_secret,
)
from tests.conftest import run_async
from tests.integration.test_postgresql_connector_live import _integration_config

_SKIP_REASON = (
    "PostgreSQL credential integration tests: SKIPPED — test database not configured"
)


def _seed_encrypted_connection(
    db_session: Session,
    *,
    host: str,
    port: int,
    database_name: str,
    username: str,
    password: str,
    ssl_mode: str,
) -> DataSourceConnection:
    user = User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
    )
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
        name="Live PostgreSQL",
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    connection = DataSourceConnection(
        data_source_id=data_source.id,
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        encrypted_password=encrypt_secret(password),
        ssl_mode=ssl_mode,
    )
    db_session.add(connection)
    db_session.flush()
    return connection


def test_encrypted_password_decrypts_and_connects(db_session: Session) -> None:
    live = _integration_config()
    if live is None:
        pytest.skip(_SKIP_REASON)
    connection = _seed_encrypted_connection(
        db_session,
        host=live.host,
        port=live.port,
        database_name=live.database_name,
        username=live.username,
        password=live.credential,
        ssl_mode=live.ssl_mode or "prefer",
    )

    assert connection.encrypted_password != live.credential
    config = connector_config_from_connection(connection)
    assert config.credential == live.credential
    assert "credential=" not in repr(config)
    if live.credential not in {live.host, live.database_name, live.username}:
        assert live.credential not in repr(config)
        assert live.credential not in repr(connection)

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(connector, config) as active:
            assert active.connection_state is PostgreSQLConnectionState.CONNECTED
            result = await active.test_connection()
            assert result.success is True
            assert live.credential not in repr(active)
        assert connector.connection_state is PostgreSQLConnectionState.DISCONNECTED

    run_async(_exercise())
