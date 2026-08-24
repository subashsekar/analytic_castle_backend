"""Phase 5.6 — concurrency, cancellation, and resource-safety tests."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.orm import Session

from app.db.models import Organization, User, Workspace
from app.enums import WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPInternalError,
    MCPQueryError,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.exceptions import MCPAccessDeniedError
from app.mcp.server.lifecycle import shutdown_mcp
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from tests.conftest import run_async
from tests.mcp.conftest import (
    add_workspace_member,
    connected_source,
    create_workspace,
    fake_query_ok,
    tracking_executor,
)
from tests.test_ai_metadata import _schema, _source


def test_concurrent_users_do_not_leak_authorization(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    admin_user: User,
) -> None:
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    workspace_b = create_workspace(db_session, organization, name="Workspace B")
    add_workspace_member(
        db_session, workspace=workspace_b, user=admin_user, role=WorkspaceRole.OWNER
    )

    source_a = _source(db_session, workspace, test_user)
    _schema(db_session, source_a, "public_a")
    source_b = _source(db_session, workspace_b, admin_user)
    _schema(db_session, source_b, "public_b")

    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        results = await asyncio.gather(
            client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source_a.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            ),
            client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source_b.id)},
                MCPToolContext(workspace_id=workspace_b.id, user_id=admin_user.id),
            ),
            return_exceptions=True,
        )
        assert not isinstance(results[0], BaseException)
        assert not isinstance(results[1], BaseException)
        assert results[0].items[0].name == "public_a"
        assert results[1].items[0].name == "public_b"

        denied = await asyncio.gather(
            client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source_b.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            ),
            client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source_a.id)},
                MCPToolContext(workspace_id=workspace_b.id, user_id=admin_user.id),
            ),
            return_exceptions=True,
        )
        assert isinstance(denied[0], MCPAccessDeniedError)
        assert isinstance(denied[1], MCPAccessDeniedError)

    run_async(_run())


def test_concurrent_query_calls_keep_executor_isolation(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    admin_user: User,
) -> None:
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    workspace_b = create_workspace(db_session, organization, name="Query WS B")
    add_workspace_member(
        db_session, workspace=workspace_b, user=admin_user, role=WorkspaceRole.OWNER
    )
    source_a = connected_source(db_session, workspace, test_user, name="A")
    source_b = connected_source(db_session, workspace_b, admin_user, name="B")

    seen_sql: list[str] = []

    async def _exec(config: object, sql: str, limit: int) -> object:
        seen_sql.append(sql)
        await asyncio.sleep(0)
        return await fake_query_ok(config, sql, limit)

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _exec

    async def _run() -> None:
        await asyncio.gather(
            client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source_a.id), "sql": "SELECT 1 AS a"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            ),
            client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source_b.id), "sql": "SELECT 2 AS b"},
                MCPToolContext(workspace_id=workspace_b.id, user_id=admin_user.id),
            ),
        )
        # Cross-user denial must not execute SQL.
        calls_before = len(seen_sql)
        with pytest.raises((MCPAccessDeniedError, MCPQueryError)):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source_b.id), "sql": "SELECT 3"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert len(seen_sql) == calls_before

    run_async(_run())
    assert set(seen_sql) == {"SELECT 1 AS a", "SELECT 2 AS b"}


def test_cancellation_during_client_call_cleans_up(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = connected_source(db_session, workspace, test_user)
    started = asyncio.Event()
    calls: list[tuple[object, ...]] = []

    async def _hang(config: object, sql: str, limit: int) -> object:
        calls.append((config, sql, limit))
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("unreachable")

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _hang

    async def _run() -> None:
        task = asyncio.create_task(
            client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # No success path; one in-flight executor call observed then cancelled.
        assert len(calls) == 1
        shutdown_mcp()

    run_async(_run())


def test_query_failure_does_not_leave_follow_up_executor_state(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = connected_source(db_session, workspace, test_user)
    calls: list[tuple[object, ...]] = []
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = tracking_executor(
        calls=calls,
        error=RuntimeError("password=secret123 boom"),
    )

    async def _run() -> None:
        with pytest.raises(MCPInternalError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = tracking_executor(
            calls=calls
        )
        result = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 2"},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.row_count >= 1

    run_async(_run())
    assert len(calls) == 2
