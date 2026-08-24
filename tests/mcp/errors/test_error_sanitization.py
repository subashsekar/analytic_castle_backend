"""Phase 5.6 — MCP error sanitization and credential/PII protection."""

from __future__ import annotations

import logging
import uuid

import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import User, Workspace
from app.mcp import (
    MCPErrorCode,
    MCPInternalError,
    MCPToolContext,
    build_postgres_mcp,
    error_response_dict,
    to_error_response,
)
from app.mcp.client import MCPClient
from app.mcp.error_codes import MCPErrorCategory
from app.mcp.exceptions import (
    MCPAuthorizationError,
    MCPDatabaseError,
    MCPDatabaseUnavailableError,
    MCPError,
    MCPQueryLimitError,
    MCPQueryTimeoutError,
)
from app.mcp.security import MCPToolPermission
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from tests.conftest import run_async
from tests.mcp.conftest import (
    MCP_TEST_PASSWORD,
    PermissionProbeTool,
    StaticServer,
    connected_source,
)
from tests.test_ai_metadata import _source

SENSITIVE = (
    "password=secret123",
    "DATABASE_URL=postgresql://u:p@host/db",
    "SELECT * FROM customers WHERE email='secret@example.com'",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.fake.sig",
    "api_key=sk-live-abc123",
    MCP_TEST_PASSWORD,
)


def test_forced_sensitive_exception_is_sanitized_in_response(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    registry, _ = build_postgres_mcp(db_session)
    probe = PermissionProbeTool(permission=MCPToolPermission.METADATA_READ)
    leak = " | ".join(SENSITIVE)

    async def _boom(arguments: BaseModel, context: MCPToolContext) -> BaseModel:
        del arguments, context
        raise RuntimeError(leak)

    probe.invoke = _boom  # type: ignore[method-assign]
    StaticServer([probe]).register(registry)
    client = MCPClient(registry, db_session)

    with caplog.at_level(logging.WARNING):

        async def _run() -> None:
            with pytest.raises(MCPInternalError) as exc_info:
                await client.call_tool(
                    "probe.action",
                    {"data_source_id": str(source.id)},
                    MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
                )
            payload = error_response_dict(exc_info.value)
            rendered = str(payload)
            for secret in SENSITIVE:
                assert secret not in rendered
                assert secret not in str(exc_info.value)
            assert payload["error"]["code"] == MCPErrorCode.MCP_INTERNAL_ERROR.value

        run_async(_run())

    joined = "\n".join(record.getMessage() for record in caplog.records)
    for secret in SENSITIVE:
        assert secret not in joined


def test_stable_error_categories_have_safe_messages() -> None:
    cases: list[tuple[MCPError, MCPErrorCode, MCPErrorCategory]] = [
        (
            MCPAuthorizationError(),
            MCPErrorCode.MCP_FORBIDDEN,
            MCPErrorCategory.AUTHORIZATION_ERROR,
        ),
        (
            MCPQueryTimeoutError(),
            MCPErrorCode.MCP_QUERY_TIMEOUT,
            MCPErrorCategory.QUERY_TIMEOUT,
        ),
        (
            MCPQueryLimitError(),
            MCPErrorCode.MCP_QUERY_LIMIT_EXCEEDED,
            MCPErrorCategory.LIMIT_EXCEEDED,
        ),
        (
            MCPDatabaseUnavailableError(),
            MCPErrorCode.MCP_DATABASE_UNAVAILABLE,
            MCPErrorCategory.DATA_SOURCE_ERROR,
        ),
        (
            MCPDatabaseError(),
            MCPErrorCode.MCP_DATABASE_ERROR,
            MCPErrorCategory.DATA_SOURCE_ERROR,
        ),
    ]
    for exc, code, category in cases:
        assert exc.code is code
        assert exc.category is category
        body = to_error_response(exc).model_dump()
        assert body["error"]["code"] == code.value
        assert body["error"]["category"] == category.value
        assert body["error"]["request_id"]
        assert "traceback" not in str(body).lower()
        assert "__cause__" not in str(body)


def test_registry_and_tool_metadata_omit_credentials(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    connected_source(db_session, workspace, test_user)
    registry, _ = build_postgres_mcp(db_session)
    blob = str(registry.list_schemas()) + str(registry.list_tools())
    assert MCP_TEST_PASSWORD not in blob
    assert "encrypted_password" not in blob
    assert "DATABASE_URL" not in blob
    assert "Authorization" not in blob


def test_list_schemas_error_path_omits_secrets(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPError) as exc_info:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(uuid.uuid4())},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        payload = error_response_dict(exc_info.value)
        assert MCP_TEST_PASSWORD not in str(payload)
        assert "password" not in str(payload).lower() or "Data source" in str(
            payload["error"]["message"]
        )

    run_async(_run())
