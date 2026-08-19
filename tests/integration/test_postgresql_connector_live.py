from __future__ import annotations

import os

import pytest

from app.connectors import (
    ConnectorAuthenticationError,
    ConnectorConfig,
    ConnectorConnectionError,
    PostgreSQLConnectionState,
    PostgreSQLConnector,
    connector_lifecycle,
)
from tests.conftest import run_async

_SKIP_REASON = "PostgreSQL integration tests: SKIPPED — test database not configured"


def _integration_config() -> ConnectorConfig | None:
    host = os.environ.get("TEST_POSTGRES_HOST")
    database_name = os.environ.get("TEST_POSTGRES_DATABASE")
    username = os.environ.get("TEST_POSTGRES_USERNAME")
    password = os.environ.get("TEST_POSTGRES_PASSWORD")
    if not host or not database_name or not username or password is None:
        return None
    port_text = os.environ.get("TEST_POSTGRES_PORT", "5432")
    ssl_mode = os.environ.get("TEST_POSTGRES_SSL_MODE", "prefer")
    try:
        port = int(port_text)
    except ValueError:
        return None
    return ConnectorConfig(
        host=host,
        port=port,
        database_name=database_name,
        username=username,
        credential=password,
        ssl_mode=ssl_mode,
    )


def _require_config() -> ConnectorConfig:
    config = _integration_config()
    if config is None:
        pytest.skip(_SKIP_REASON)
    return config


def test_successful_connection_select_one_and_disconnect() -> None:
    config = _require_config()

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(connector, config) as active:
            assert active.connection_state is PostgreSQLConnectionState.CONNECTED
            result = await active.test_connection()
            assert result.success is True
            assert config.credential not in repr(active)
            connector_state = vars(active)
            assert "credential" not in connector_state
            rendered = str(connector_state)
            assert f"password={config.credential}" not in rendered.lower()
            # Skip substring checks when the password equals another DSN field.
            if config.credential not in {
                config.host,
                config.database_name,
                config.username,
            }:
                assert config.credential not in rendered
        assert connector.connection_state is PostgreSQLConnectionState.DISCONNECTED
        with pytest.raises(ConnectorConnectionError, match="Not connected"):
            await connector.test_connection()

    run_async(_exercise())


def test_invalid_credentials() -> None:
    config = _require_config()
    bad = ConnectorConfig(
        host=config.host,
        port=config.port,
        database_name=config.database_name,
        username=config.username,
        credential="definitely-not-the-password",
        ssl_mode=config.ssl_mode,
    )

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        with pytest.raises(
            (ConnectorAuthenticationError, ConnectorConnectionError)
        ) as exc_info:
            await connector.connect(bad)
        assert connector.connection_state is PostgreSQLConnectionState.FAILED
        assert "definitely-not-the-password" not in str(exc_info.value)
        await connector.disconnect()

    run_async(_exercise())


def test_invalid_database() -> None:
    config = _require_config()
    bad = ConnectorConfig(
        host=config.host,
        port=config.port,
        database_name="analyticcastle_missing_db_xyz",
        username=config.username,
        credential=config.credential,
        ssl_mode=config.ssl_mode,
    )

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        with pytest.raises(ConnectorConnectionError) as exc_info:
            await connector.connect(bad)
        assert config.credential not in str(exc_info.value)
        await connector.disconnect()

    run_async(_exercise())


def test_ssl_configuration_connects() -> None:
    config = _require_config()

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(connector, config):
            result = await connector.test_connection()
            assert result.success is True

    run_async(_exercise())


def test_connection_timeout_to_unroutable_host() -> None:
    config = ConnectorConfig(
        host="192.0.2.1",
        port=5432,
        database_name="analytics",
        username="readonly",
        credential="not-a-real-password",
        ssl_mode="disable",
    )

    async def _exercise() -> None:
        connector = PostgreSQLConnector(connect_timeout=1)
        with pytest.raises(ConnectorConnectionError) as exc_info:
            await connector.connect(config)
        message = str(exc_info.value).lower()
        assert "not-a-real-password" not in message
        assert connector.connection_state is PostgreSQLConnectionState.FAILED
        await connector.disconnect()

    run_async(_exercise())
