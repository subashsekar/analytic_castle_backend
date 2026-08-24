from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.connectors.types import ConnectorConfig, QueryResult
from app.core.config import settings
from app.core.logging import RedactingFilter
from app.db.models import DataSourceConnection, Organization, User, Workspace
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    POSTGRES_TOOL_NAMES,
    MCPQueryError,
    MCPQueryRejectedError,
    MCPQueryRequest,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPToolContext,
    MCPToolNotFoundError,
    PostgresQueryTool,
    build_postgres_mcp,
)
from app.services.credentials import encrypt_secret
from tests.conftest import run_async
from tests.test_ai_metadata import _source

SECRET = "CustomerMcpPassword!@# 42"


def _connected_source(db_session: Session, workspace: Workspace, test_user: User):
    source = _source(db_session, workspace, test_user)
    db_session.add(
        DataSourceConnection(
            data_source_id=source.id,
            host="db.internal.example",
            port=5432,
            database_name="analytics",
            username="readonly",
            encrypted_password=encrypt_secret(SECRET),
            ssl_mode="prefer",
        )
    )
    db_session.flush()
    return source


async def _fake_ok(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
    del sql
    assert config.credential == SECRET
    rows = tuple((index,) for index in range(min(3, limit)))
    return QueryResult(columns=("id",), rows=rows, truncated=False)


def test_tool_is_registered_once(db_session: Session) -> None:
    registry, _client = build_postgres_mcp(db_session)
    schemas = registry.list_schemas()
    names = [schema.name for schema in schemas]
    assert names == sorted(POSTGRES_TOOL_NAMES)
    query = next(
        schema for schema in schemas if schema.name == POSTGRES_QUERY_TOOL_NAME
    )
    assert "read-only" in query.description.lower()
    assert "sql" in query.input_schema["properties"]
    assert "connection_string" not in str(query.input_schema)
    assert "password" not in str(query.input_schema)
    with pytest.raises(MCPToolNotFoundError):
        registry.get("postgres.write")


def test_client_invokes_query_tool(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)
    registered_tool = registry.get(POSTGRES_QUERY_TOOL_NAME)
    registered_tool._executor = _fake_ok  # inject deterministic query executor

    async def _run() -> None:
        payload = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert payload.row_count == 3
        assert payload.columns == ["id"]
        dumped = payload.model_dump()
        assert "password" not in dumped
        assert SECRET not in str(dumped)

    run_async(_run())


def test_write_sql_is_rejected_before_executor(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    called = {"count": 0}

    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        called["count"] += 1
        return QueryResult(columns=(), rows=())

    source = _connected_source(db_session, workspace, test_user)
    tool = PostgresQueryTool(db_session, executor=_executor)

    async def _run() -> None:
        with pytest.raises(MCPQueryRejectedError):
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="DELETE FROM users"),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())
    assert called["count"] == 0


def test_request_rejects_connection_overrides() -> None:
    with pytest.raises(ValidationError):
        MCPQueryRequest.model_validate(
            {
                "data_source_id": str(uuid.uuid4()),
                "sql": "SELECT 1",
                "connection_string": "postgresql://u:p@h/db",
            }
        )


def test_cross_workspace_data_source_is_not_found(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    admin_user: User,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    other = Workspace(
        organization_id=organization.id,
        name="Other",
        slug=f"other-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(other)
    db_session.flush()
    tool = PostgresQueryTool(db_session, executor=_fake_ok)

    async def _run() -> None:
        with pytest.raises(MCPQueryError, match="Data source not found"):
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                MCPToolContext(workspace_id=other.id, user_id=admin_user.id),
            )

    run_async(_run())


def test_row_limit_is_capped(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_LIMIT", 5)
    seen: dict[str, int] = {}

    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql
        seen["limit"] = limit
        return QueryResult(columns=("id",), rows=tuple((i,) for i in range(limit)))

    source = _connected_source(db_session, workspace, test_user)
    tool = PostgresQueryTool(db_session, executor=_executor)

    async def _run() -> None:
        result = await tool.execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1", limit=1_000_000),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.row_count == 5

    run_async(_run())
    assert seen["limit"] == 5


def test_empty_and_null_results(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    async def _empty(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("id", "name"), rows=())

    async def _nulls(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("id", "name"), rows=((1, None),))

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        empty = await PostgresQueryTool(db_session, executor=_empty).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1 WHERE false"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert empty.row_count == 0
        assert empty.columns == ["id", "name"]
        nulls = await PostgresQueryTool(db_session, executor=_nulls).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert nulls.rows == [[1, None]]

    run_async(_run())


def test_serialization_covers_common_types(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    sample_id = UUID("123e4567-e89b-12d3-a456-426614174000")

    async def _typed(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(
            columns=("u", "d", "ts", "n", "j", "a", "b"),
            rows=(
                (
                    sample_id,
                    date(2026, 1, 2),
                    datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
                    Decimal("12.50"),
                    {"ok": True},
                    [1, 2],
                    b"\x00\x01",
                ),
            ),
        )

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        result = await PostgresQueryTool(db_session, executor=_typed).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        row = result.rows[0]
        assert row[0] == str(sample_id)
        assert row[1] == "2026-01-02"
        assert "2026-01-02T03:04:05" in str(row[2])
        assert row[3] == "12.50"
        assert row[4] == {"ok": True}
        assert row[5] == [1, 2]
        assert row[6] == "[BINARY]"
        dumped = result.model_dump()
        assert SECRET not in str(dumped)
        json.dumps(dumped)

    run_async(_run())


def test_oversized_values_are_bounded(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 16)

    async def _wide(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("blob",), rows=(("x" * 5000,),))

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        result = await PostgresQueryTool(db_session, executor=_wide).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.row_count == 1
        assert len(json.dumps(result.rows[0])) < 80

    run_async(_run())


def test_result_payload_is_capped(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_RESULT_CHARS", 120)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 10_000)

    async def _many(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(
            columns=("blob",),
            rows=tuple(("x" * 40,) for _ in range(20)),
        )

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        result = await PostgresQueryTool(db_session, executor=_many).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.truncated is True
        assert result.row_count < 20
        assert result.row_count >= 1

    run_async(_run())


def test_single_oversized_row_fails_safely(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_RESULT_CHARS", 40)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 10_000)

    async def _huge(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("blob",), rows=(("x" * 5000,),))

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(MCPQueryResultError):
            await PostgresQueryTool(db_session, executor=_huge).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_timeout_maps_to_query_timeout(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    async def _hang(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise TimeoutError()

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(MCPQueryTimeoutError):
            await PostgresQueryTool(db_session, executor=_hang).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_timeout_then_success(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    calls = {"count": 0}

    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError()
        return QueryResult(columns=("n",), rows=((1,),))

    source = _connected_source(db_session, workspace, test_user)
    tool = PostgresQueryTool(db_session, executor=_executor)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPQueryTimeoutError):
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        result = await tool.execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            context,
        )
        assert result.rows == [[1]]

    run_async(_run())


def test_cancellation_is_not_swallowed(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise asyncio.CancelledError()

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await PostgresQueryTool(db_session, executor=_executor).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_data_source_credentials_are_isolated(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source_a = _connected_source(db_session, workspace, test_user)
    source_b = _source(db_session, workspace, test_user, name="Warehouse")
    db_session.add(
        DataSourceConnection(
            data_source_id=source_b.id,
            host="other.internal.example",
            port=5432,
            database_name="warehouse",
            username="readonly",
            encrypted_password=encrypt_secret("OtherMcpPassword!@#"),
            ssl_mode="prefer",
        )
    )
    db_session.flush()
    seen: list[str] = []

    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del sql, limit
        seen.append(config.database_name)
        return QueryResult(columns=("n",), rows=((1,),))

    tool = PostgresQueryTool(db_session, executor=_executor)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        await tool.execute(
            MCPQueryRequest(data_source_id=source_a.id, sql="SELECT 1"),
            context,
        )
        await tool.execute(
            MCPQueryRequest(data_source_id=source_b.id, sql="SELECT 1"),
            context,
        )

    run_async(_run())
    assert seen == ["analytics", "warehouse"]


def test_default_row_limit_is_used(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_DEFAULT_LIMIT", 7)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_LIMIT", 50)
    seen: dict[str, int] = {}

    async def _executor(config: ConnectorConfig, sql: str, limit: int) -> QueryResult:
        del config, sql
        seen["limit"] = limit
        return QueryResult(columns=("id",), rows=())

    source = _connected_source(db_session, workspace, test_user)

    async def _run() -> None:
        await PostgresQueryTool(db_session, executor=_executor).execute(
            MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )

    run_async(_run())
    assert seen["limit"] == 7


def test_query_logs_omit_sql_and_secrets(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    logger = logging.getLogger("app.mcp.servers.postgres.tools.query")
    logger.addFilter(RedactingFilter())

    async def _run() -> None:
        await PostgresQueryTool(db_session, executor=_fake_ok).execute(
            MCPQueryRequest(
                data_source_id=source.id,
                sql="SELECT email FROM users",
            ),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )

    with caplog.at_level(logging.INFO, logger=logger.name):
        run_async(_run())
    combined = " ".join(record.getMessage() for record in caplog.records)
    assert "SELECT email" not in combined
    assert SECRET not in combined
    assert str(source.id) in combined


def test_database_connection_failures_map_safely(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    from app.connectors.exceptions import ConnectorConnectionError, ConnectorQueryError
    from app.mcp.exceptions import MCPDatabaseError, MCPDatabaseUnavailableError

    source = _connected_source(db_session, workspace, test_user)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPDatabaseUnavailableError):
            await PostgresQueryTool(
                db_session,
                executor=_failing_executor(
                    ConnectorConnectionError("connection timed out")
                ),
            ).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        with pytest.raises(MCPDatabaseError):
            await PostgresQueryTool(
                db_session,
                executor=_failing_executor(
                    ConnectorConnectionError("authentication failed")
                ),
            ).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        with pytest.raises(MCPQueryError):
            await PostgresQueryTool(
                db_session,
                executor=_failing_executor(ConnectorQueryError("syntax error")),
            ).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        with pytest.raises(MCPQueryTimeoutError):
            await PostgresQueryTool(
                db_session,
                executor=_failing_executor(ConnectorQueryError("query timed out")),
            ).execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )

    run_async(_run())


def test_empty_and_oversized_sql_rejected(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(ValidationError):
            MCPQueryRequest(data_source_id=source.id, sql="")
        with pytest.raises(ValidationError):
            MCPQueryRequest(data_source_id=source.id, sql="x" * 100_001)
        # Valid length but invalid SQL still rejected before executor.
        calls: list[object] = []

        async def _track(config: object, sql: str, limit: int) -> QueryResult:
            calls.append(sql)
            return await _fake_ok(config, sql, limit)

        with pytest.raises(MCPQueryRejectedError):
            await PostgresQueryTool(db_session, executor=_track).execute(
                MCPQueryRequest(
                    data_source_id=source.id, sql="INSERT INTO t VALUES (1)"
                ),
                context,
            )
        assert calls == []

    run_async(_run())


def _failing_executor(error: Exception):
    async def _execute(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise error

    return _execute
