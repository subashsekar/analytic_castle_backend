from __future__ import annotations

import pytest

from app.connectors import PostgreSQLConnector, connector_lifecycle
from app.connectors.exceptions import ConnectorQueryError
from tests.conftest import run_async
from tests.integration.test_postgresql_connector_live import _require_config

_SKIP_REASON = "PostgreSQL integration tests: SKIPPED — test database not configured"


def test_live_readonly_select_and_write_rejection() -> None:
    config = _require_config()

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(connector, config) as active:
            result = await active.execute_query("SELECT 1 AS n", limit=10)
            assert result.columns == ("n",)
            assert result.rows == ((1,),)
            assert result.truncated is False
            cte = await active.execute_query(
                "WITH data AS (SELECT 1 AS n) SELECT n FROM data",
                limit=10,
            )
            assert cte.rows == ((1,),)
            empty = await active.execute_query("SELECT 1 AS n WHERE false")
            assert empty.rows == ()
            assert empty.columns == ("n",)
            with pytest.raises(ConnectorQueryError):
                await active.execute_query("DELETE FROM pg_catalog.pg_class")

    run_async(_run())


def test_live_timeout_then_successful_query() -> None:
    config = _require_config()

    async def _run() -> None:
        timed_out = PostgreSQLConnector(connect_timeout=1)
        async with connector_lifecycle(timed_out, config) as active:
            with pytest.raises(ConnectorQueryError, match="timed out"):
                await active.execute_query("SELECT pg_sleep(5)")
        recovered = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(recovered, config) as active:
            result = await active.execute_query("SELECT 1 AS n")
            assert result.rows == ((1,),)

    run_async(_run())
