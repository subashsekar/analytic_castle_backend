"""Phase 5.6 — MCP integration against the shared test PostgreSQL database."""

from __future__ import annotations

import os

import pytest
from sqlalchemy.orm import Session

from app.db.models import DataSourceConnection, User, Workspace
from app.enums import WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPQueryRejectedError,
    MCPQueryTimeoutError,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.servers.postgres.tools.columns import POSTGRES_GET_COLUMNS_TOOL_NAME
from app.mcp.servers.postgres.tools.relationships import (
    POSTGRES_GET_RELATIONSHIPS_TOOL_NAME,
)
from app.mcp.servers.postgres.tools.sample_rows import POSTGRES_SAMPLE_ROWS_TOOL_NAME
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from app.mcp.servers.postgres.tools.tables import (
    POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
    POSTGRES_LIST_TABLES_TOOL_NAME,
)
from app.services.credentials import encrypt_secret
from tests.conftest import run_async
from tests.mcp.conftest import add_workspace_member
from tests.test_ai_metadata import _column, _relationship, _schema, _source, _table

_SKIP_REASON = "PostgreSQL MCP integration: SKIPPED — test database DSN unavailable"


def _live_dsn_parts() -> dict[str, str | int] | None:
    """Reuse the same DATABASE_URL / TEST_DATABASE_URL strategy as conftest."""
    url = os.environ.get("DATABASE_URL") or os.environ.get("TEST_DATABASE_URL")
    if not url or "postgresql" not in url:
        return None
    # postgresql+psycopg://user:pass@host:port/db
    try:
        without_scheme = url.split("://", 1)[1]
        creds, hostpart = without_scheme.rsplit("@", 1)
        username, password = creds.split(":", 1)
        host_port, database = hostpart.split("/", 1)
        database = database.split("?", 1)[0]
        if ":" in host_port:
            host, port_text = host_port.split(":", 1)
            port = int(port_text)
        else:
            host, port = host_port, 5432
    except (ValueError, IndexError):
        return None
    return {
        "host": host,
        "port": port,
        "database_name": database,
        "username": username,
        "password": password,
    }


def _require_live_parts() -> dict[str, str | int]:
    parts = _live_dsn_parts()
    if parts is None:
        pytest.skip(_SKIP_REASON)
    return parts


def test_mcp_catalog_sample_and_query_against_test_database(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    parts = _require_live_parts()
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    source = _source(db_session, workspace, test_user)
    db_session.add(
        DataSourceConnection(
            data_source_id=source.id,
            host=str(parts["host"]),
            port=int(parts["port"]),
            database_name=str(parts["database_name"]),
            username=str(parts["username"]),
            encrypted_password=encrypt_secret(str(parts["password"])),
            ssl_mode="prefer",
        )
    )
    db_session.flush()

    schema = _schema(db_session, source, "public")
    # Persist catalog rows that mirror discoverable names; MCP catalog reads DB metadata.
    table = _table(db_session, schema, "mcp_phase56_probe")
    id_col = _column(db_session, table, "id", position=1, primary_key=True)
    name_col = _column(db_session, table, "name", position=2)
    del name_col
    _relationship(
        db_session,
        source_table=table,
        source_column=id_col,
        target_table=table,
        target_column=id_col,
        constraint_name="mcp_phase56_self_fk",
    )

    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        schemas = await client.call_tool(
            POSTGRES_LIST_SCHEMAS_TOOL_NAME,
            {"data_source_id": str(source.id)},
            context,
        )
        assert any(item.name == "public" for item in schemas.items)

        tables = await client.call_tool(
            POSTGRES_LIST_TABLES_TOOL_NAME,
            {"data_source_id": str(source.id), "schema_id": str(schema.id)},
            context,
        )
        assert any(item.name == "mcp_phase56_probe" for item in tables.items)

        described = await client.call_tool(
            POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
            {"data_source_id": str(source.id), "table_id": str(table.id)},
            context,
        )
        assert described.name == "mcp_phase56_probe"

        columns = await client.call_tool(
            POSTGRES_GET_COLUMNS_TOOL_NAME,
            {"data_source_id": str(source.id), "table_id": str(table.id)},
            context,
        )
        assert [c.name for c in columns.items] == ["id", "name"]

        relationships = await client.call_tool(
            POSTGRES_GET_RELATIONSHIPS_TOOL_NAME,
            {"data_source_id": str(source.id)},
            context,
        )
        assert any(
            item.constraint_name == "mcp_phase56_self_fk"
            for item in relationships.items
        )

        # Live read-only query against the test database itself.
        query = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {
                "data_source_id": str(source.id),
                "sql": "SELECT 1 AS n",
                "limit": 10,
            },
            context,
        )
        assert query.columns == ["n"]
        assert query.rows == [[1]]

        with pytest.raises(MCPQueryRejectedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {
                    "data_source_id": str(source.id),
                    "sql": "DELETE FROM pg_catalog.pg_class",
                },
                context,
            )

        # Sample may fail if the physical table does not exist; assert safe mapping.
        from app.mcp.exceptions import MCPError

        try:
            await client.call_tool(
                POSTGRES_SAMPLE_ROWS_TOOL_NAME,
                {
                    "data_source_id": str(source.id),
                    "table_id": str(table.id),
                    "limit": 1,
                },
                context,
            )
        except MCPError:
            pass

    run_async(_run())


def test_mcp_query_timeout_against_test_database(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = _require_live_parts()
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    source = _source(db_session, workspace, test_user)
    db_session.add(
        DataSourceConnection(
            data_source_id=source.id,
            host=str(parts["host"]),
            port=int(parts["port"]),
            database_name=str(parts["database_name"]),
            username=str(parts["username"]),
            encrypted_password=encrypt_secret(str(parts["password"])),
            ssl_mode="prefer",
        )
    )
    db_session.flush()

    from app.core.config import settings

    monkeypatch.setattr(settings, "MCP_QUERY_TIMEOUT_SECONDS", 1)

    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        from app.mcp.exceptions import MCPQueryRejectedError

        with pytest.raises(MCPQueryRejectedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {
                    "data_source_id": str(source.id),
                    "sql": "SELECT pg_sleep(5)",
                },
                context,
            )
        with pytest.raises(MCPQueryTimeoutError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {
                    "data_source_id": str(source.id),
                    "sql": "SELECT count(*) FROM generate_series(1, 200000000)",
                },
                context,
            )

    run_async(_run())


def test_migration_compatibility_metadata_tables_exist(db_session: Session) -> None:
    """MCP catalog depends on current ORM metadata tables created for tests."""
    from sqlalchemy import inspect

    from app.db.session import engine

    inspector = inspect(engine)
    required = {
        "data_sources",
        "data_source_connections",
        "data_source_schemas",
        "data_source_tables",
        "data_source_columns",
        "data_source_relationships",
    }
    existing = set(inspector.get_table_names())
    missing = required - existing
    assert not missing, f"Missing tables for MCP: {sorted(missing)}"
