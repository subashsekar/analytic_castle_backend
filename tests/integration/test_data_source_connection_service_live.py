from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.db.models import DataSourceStatus
from app.services.data_source_connections import DataSourceConnectionService
from tests.conftest import run_async
from tests.integration.test_credential_connector_live import _seed_encrypted_connection
from tests.integration.test_postgresql_connector_live import _integration_config

_SKIP_REASON = (
    "PostgreSQL connection service integration tests: SKIPPED — "
    "test database not configured"
)


def test_full_internal_connection_flow(db_session: Session) -> None:
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
    data_source = connection.data_source
    ciphertext = connection.encrypted_password
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(data_source.id))
    db_session.refresh(data_source)
    db_session.refresh(connection)

    assert result.success is True
    assert live.credential not in repr(result)
    assert data_source.status is DataSourceStatus.ACTIVE
    assert data_source.last_tested_at is not None
    assert data_source.last_tested_at.tzinfo is not None
    assert connection.encrypted_password == ciphertext
    assert connection.encrypted_password != live.credential
    assert live.credential not in repr(connection)


def test_invalid_credentials_fail_without_leaking_secrets(
    db_session: Session,
) -> None:
    live = _integration_config()
    if live is None:
        pytest.skip(_SKIP_REASON)

    connection = _seed_encrypted_connection(
        db_session,
        host=live.host,
        port=live.port,
        database_name=live.database_name,
        username=live.username,
        password="definitely-not-the-password",
        ssl_mode=live.ssl_mode or "prefer",
    )
    ciphertext = connection.encrypted_password
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(connection.data_source_id))
    db_session.refresh(connection.data_source)
    db_session.refresh(connection)

    assert result.success is False
    assert "definitely-not-the-password" not in repr(result)
    assert connection.data_source.status is DataSourceStatus.ERROR
    assert connection.data_source.last_tested_at is not None
    assert connection.encrypted_password == ciphertext
    assert connection.encrypted_password != "definitely-not-the-password"


def test_invalid_database_fails(db_session: Session) -> None:
    live = _integration_config()
    if live is None:
        pytest.skip(_SKIP_REASON)

    connection = _seed_encrypted_connection(
        db_session,
        host=live.host,
        port=live.port,
        database_name="analyticcastle_missing_db_xyz",
        username=live.username,
        password=live.credential,
        ssl_mode=live.ssl_mode or "prefer",
    )
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(connection.data_source_id))
    db_session.refresh(connection.data_source)

    assert result.success is False
    assert live.credential not in repr(result)
    assert connection.data_source.status is DataSourceStatus.ERROR


def test_timeout_to_unroutable_host_fails(db_session: Session) -> None:
    live = _integration_config()
    if live is None:
        pytest.skip(_SKIP_REASON)

    connection = _seed_encrypted_connection(
        db_session,
        host="192.0.2.1",
        port=5432,
        database_name="analytics",
        username="readonly",
        password="not-a-real-password",
        ssl_mode="disable",
    )
    service = DataSourceConnectionService(db_session)

    result = run_async(service.test_connection(connection.data_source_id))
    db_session.refresh(connection.data_source)

    assert result.success is False
    assert "not-a-real-password" not in repr(result)
    assert connection.data_source.status is DataSourceStatus.ERROR
    assert connection.data_source.last_tested_at is not None
