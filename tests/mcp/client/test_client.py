"""Phase 5.6 — MCP client invocation boundary tests."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import reset_rate_limiters
from app.db.models import Organization, User, Workspace
from app.enums import WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPErrorCode,
    MCPInternalError,
    MCPQueryError,
    MCPRateLimitError,
    MCPToolContext,
    MCPToolValidationError,
    MCPUnauthorizedError,
    build_postgres_mcp,
)
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPServerNotFoundError,
    MCPToolNotFoundError,
)
from app.mcp.security import MCPToolPermission
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from tests.conftest import run_async
from tests.mcp.conftest import (
    PermissionProbeTool,
    StaticServer,
    add_workspace_member,
    connected_source,
    create_workspace,
    fake_query_ok,
    tracking_executor,
)
from tests.test_ai_metadata import _schema, _source


def test_successful_tool_invocation(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    mcp_context: MCPToolContext,
) -> None:
    source = _source(db_session, workspace, test_user)
    _schema(db_session, source, "public")
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        result = await client.call_tool(
            POSTGRES_LIST_SCHEMAS_TOOL_NAME,
            {"data_source_id": str(source.id)},
            mcp_context,
        )
        assert result.items[0].name == "public"

    run_async(_run())


def test_unknown_tool_preserves_structured_error(
    db_session: Session,
    mcp_context: MCPToolContext,
) -> None:
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPToolNotFoundError) as exc_info:
            await client.call_tool(
                "postgres.missing",
                {"data_source_id": str(uuid.uuid4())},
                mcp_context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_TOOL_NOT_FOUND
        assert exc_info.value.request_id

    run_async(_run())


def test_unknown_server_preserves_structured_error(
    db_session: Session,
    mcp_context: MCPToolContext,
) -> None:
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPServerNotFoundError) as exc_info:
            await client.call_tool(
                "oracle.query",
                {"data_source_id": str(uuid.uuid4())},
                mcp_context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_SERVER_NOT_FOUND

    run_async(_run())


def test_invalid_arguments(
    db_session: Session,
    mcp_context: MCPToolContext,
) -> None:
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPToolValidationError) as exc_info:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": "not-a-uuid"},
                mcp_context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_TOOL_VALIDATION_FAILED

    run_async(_run())


def test_permission_denied_for_member_query(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = fake_query_ok

    async def _run() -> None:
        with pytest.raises(MCPQueryError) as exc_info:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND

    run_async(_run())


def test_workspace_denied_before_tool_body(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    mcp_context: MCPToolContext,
) -> None:
    foreign = create_workspace(db_session, organization, name="Foreign")
    source = _source(db_session, foreign, test_user)
    registry, client = build_postgres_mcp(db_session)
    invoked: list[str] = []
    original = registry.get(POSTGRES_LIST_SCHEMAS_TOOL_NAME).invoke

    async def _track(arguments: object, context: MCPToolContext) -> object:
        invoked.append("called")
        return await original(arguments, context)

    registry.get(POSTGRES_LIST_SCHEMAS_TOOL_NAME).invoke = _track  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                mcp_context,
            )

    run_async(_run())
    assert invoked == []


def test_data_source_denied_for_outsider(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    admin_user: User,
    mcp_context: MCPToolContext,
) -> None:
    del mcp_context
    source = _source(db_session, workspace, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError) as exc_info:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=admin_user.id),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND

    run_async(_run())


def test_rate_limited_request_does_not_reach_executor(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    mcp_context: MCPToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = connected_source(db_session, workspace, test_user)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_QUERY", "1/minute")
    reset_rate_limiters()

    calls: list[tuple[object, ...]] = []
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = tracking_executor(calls=calls)

    async def _run() -> None:
        await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            mcp_context,
        )
        with pytest.raises(MCPRateLimitError) as exc_info:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                mcp_context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_RATE_LIMITED

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()
    assert len(calls) == 1


def test_tool_exception_mapped_to_internal_error(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    mcp_context: MCPToolContext,
) -> None:
    source = _source(db_session, workspace, test_user)
    registry = build_postgres_mcp(db_session)[0]
    probe = PermissionProbeTool(permission=MCPToolPermission.METADATA_READ)

    async def _boom(arguments: BaseModel, context: MCPToolContext) -> BaseModel:
        del arguments, context
        raise RuntimeError("password=super-secret DATABASE_URL=postgresql://x")

    probe.invoke = _boom  # type: ignore[method-assign]
    StaticServer([probe]).register(registry)
    client = build_postgres_mcp(db_session)[1]
    # Rebuild client against the mutated registry from first build.
    from app.mcp.client import MCPClient

    client = MCPClient(registry, db_session)

    async def _run() -> None:
        with pytest.raises(MCPInternalError) as exc_info:
            await client.call_tool(
                "probe.action",
                {"data_source_id": str(source.id)},
                mcp_context,
            )
        message = str(exc_info.value)
        assert "password=" not in message
        assert "DATABASE_URL" not in message
        assert "super-secret" not in message
        assert exc_info.value.code is MCPErrorCode.MCP_INTERNAL_ERROR

    run_async(_run())


def test_mcp_error_propagation_preserves_code(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    mcp_context: MCPToolContext,
) -> None:
    source = _source(db_session, workspace, test_user)
    registry, _ = build_postgres_mcp(db_session)
    probe = PermissionProbeTool(permission=MCPToolPermission.METADATA_READ)

    async def _deny(arguments: BaseModel, context: MCPToolContext) -> BaseModel:
        del arguments, context
        raise MCPAccessDeniedError("Data source not found")

    probe.invoke = _deny  # type: ignore[method-assign]
    StaticServer([probe]).register(registry)
    from app.mcp.client import MCPClient

    client = MCPClient(registry, db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError) as exc_info:
            await client.call_tool(
                "probe.action",
                {"data_source_id": str(source.id)},
                mcp_context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND
        assert exc_info.value.request_id

    run_async(_run())


def test_cancellation_propagates(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    mcp_context: MCPToolContext,
) -> None:
    source = connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)

    async def _hang(config: object, sql: str, limit: int) -> Any:
        del config, sql, limit
        await asyncio.sleep(60)
        raise AssertionError("should have been cancelled")

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _hang

    async def _run() -> None:
        task = asyncio.create_task(
            client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                mcp_context,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run_async(_run())


def test_missing_and_inactive_principal(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = connected_source(db_session, workspace, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPUnauthorizedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=None),
            )

        test_user.is_active = False
        db_session.flush()
        with pytest.raises(MCPUnauthorizedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

        with pytest.raises(MCPUnauthorizedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=uuid.uuid4()),
            )

    run_async(_run())
