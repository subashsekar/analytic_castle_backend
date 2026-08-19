from __future__ import annotations

import os

import pytest

from app.connectors import PostgreSQLConnector, connector_lifecycle
from app.connectors.exceptions import ConnectorQueryError
from tests.conftest import run_async
from tests.integration.test_postgresql_connector_live import _require_config

_SKIP_REASON = "PostgreSQL integration tests: SKIPPED — test database not configured"


@pytest.mark.skipif(os.environ.get("TEST_POSTGRES_HOST") is None, reason=_SKIP_REASON)
def test_live_readonly_select_and_write_rejection() -> None:
    config = _require_config()

    async def _run() -> None:
        connector = PostgreSQLConnector(connect_timeout=5)
        async with connector_lifecycle(connector, config) as active:
            result = await active.execute_query("SELECT 1 AS n", limit=10)
            assert result.columns == ("n",)
            assert result.rows == ((1,),)
            assert result.truncated is False
            empty = await active.execute_query("SELECT 1 AS n WHERE false")
            assert empty.rows == ()
            with pytest.raises(ConnectorQueryError):
                await active.execute_query("DELETE FROM pg_catalog.pg_class")

    run_async(_run())
