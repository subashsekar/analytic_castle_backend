"""Phase 5.5 — MCP error handling, safe responses, and secure logging."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.connectors.exceptions import ConnectorConnectionError, ConnectorQueryError
from app.connectors.types import QueryResult
from app.core.config import settings
from app.core.rate_limit import reset_rate_limiters
from app.core.request_id import bind_request_id, reset_request_id
from app.db.models import DataSourceConnection, Organization, User, Workspace
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPError,
    MCPErrorCategory,
    MCPErrorCode,
    MCPInternalError,
    MCPQueryError,
    MCPQueryRejectedError,
    MCPQueryRequest,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPRateLimitError,
    MCPSampleDataError,
    MCPServerNotFoundError,
    MCPToolContext,
    MCPToolNotFoundError,
    MCPToolValidationError,
    MCPUnauthorizedError,
    build_postgres_mcp,
    error_response_dict,
    http_status_for,
    to_error_response,
)
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPDatabaseUnavailableError,
    MCPServerError,
)
from app.mcp.registry import MCPRegistry
from app.mcp.schemas import MCPSampleRowsRequest
from app.mcp.server.lifecycle import build_postgres_mcp as build_mcp
from app.mcp.server.lifecycle import shutdown_mcp
from app.mcp.servers.postgres.tools.query import PostgresQueryTool
from app.mcp.servers.postgres.tools.sample_rows import PostgresSampleRowsTool
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from app.services.credentials import encrypt_secret
from app.services.sample_data_exceptions import (
    SampleSerializationError,
    SampleTableNotFoundError,
)
from app.services.sample_data_types import SampleDataResult
from tests.conftest import run_async
from tests.test_ai_metadata import _source

SECRET = "Phase55SecretPassword!99"
SENSITIVE_SQL = "SELECT * FROM customer_secret_table WHERE ssn = '123-45-6789'"


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


async def _fake_ok(config: object, sql: str, limit: int) -> QueryResult:
    del config, sql
    return QueryResult(
        columns=("id",),
        rows=tuple((i,) for i in range(min(2, limit))),
        truncated=False,
    )


def _assert_safe_error_payload(payload: dict[str, Any]) -> None:
    assert set(payload.keys()) == {"error"}
    error = payload["error"]
    assert set(error.keys()) <= {
        "code",
        "category",
        "message",
        "request_id",
        "retry_after",
    }
    assert "code" in error and "category" in error and "message" in error
    assert "request_id" in error
    blob = str(payload).lower()
    for forbidden in (
        "traceback",
        "password",
        SECRET.lower(),
        "database_url",
        "bearer ",
        "asyncpg",
        "psycopg",
        "customer_secret_table",
        "123-45-6789",
        "__cause__",
        "__context__",
    ):
        assert forbidden not in blob


def test_error_response_shape_and_codes() -> None:
    exc = MCPQueryTimeoutError("The query timed out", request_id="req-timeout-1")
    payload = error_response_dict(exc)
    _assert_safe_error_payload(payload)
    assert payload["error"]["code"] == MCPErrorCode.MCP_QUERY_TIMEOUT.value
    assert payload["error"]["category"] == MCPErrorCategory.QUERY_TIMEOUT.value
    assert payload["error"]["request_id"] == "req-timeout-1"
    assert http_status_for(exc) == 504


def test_unknown_server_and_tool_codes(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPServerNotFoundError) as server_exc:
            await client.call_tool(
                "unknown.query",
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                context,
            )
        assert server_exc.value.code is MCPErrorCode.MCP_SERVER_NOT_FOUND

        with pytest.raises(MCPToolNotFoundError) as tool_exc:
            await client.call_tool(
                "postgres.write",
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                context,
            )
        assert tool_exc.value.code is MCPErrorCode.MCP_TOOL_NOT_FOUND

    run_async(_run())


def test_registry_get_server_uses_server_not_found(db_session: Session) -> None:
    registry, _ = build_postgres_mcp(db_session)
    with pytest.raises(MCPServerNotFoundError) as exc_info:
        registry.get_server("missing")
    assert exc_info.value.code is MCPErrorCode.MCP_SERVER_NOT_FOUND


def test_validation_errors_are_deterministic(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPToolValidationError) as exc_info:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": "not-a-uuid", "sql": "SELECT 1"},
                context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_TOOL_VALIDATION_FAILED
        assert exc_info.value.category is MCPErrorCategory.VALIDATION_ERROR
        payload = to_error_response(exc_info.value).model_dump(mode="json")
        _assert_safe_error_payload(payload)

    run_async(_run())


def test_authentication_and_authorization_codes(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPUnauthorizedError) as auth_exc:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id),
            )
        assert auth_exc.value.code is MCPErrorCode.MCP_UNAUTHORIZED

        other = Workspace(
            organization_id=organization.id,
            name="Other WS",
            slug=f"other-{uuid.uuid4().hex[:8]}",
        )
        db_session.add(other)
        db_session.flush()
        foreign = _source(db_session, other, test_user)

        with pytest.raises(MCPAccessDeniedError) as denied:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(foreign.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert denied.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND
        assert "exists" not in str(denied.value).lower()
        assert "cannot access" not in str(denied.value).lower()

        with pytest.raises(MCPQueryError) as query_denied:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(foreign.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert query_denied.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND

    run_async(_run())


def test_query_error_classification(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    tool = PostgresQueryTool(db_session, executor=_fake_ok)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPQueryRejectedError) as rejected:
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="DELETE FROM users"),
                context,
            )
        assert rejected.value.code is MCPErrorCode.MCP_QUERY_INVALID

        async def _timeout(config: object, sql: str, limit: int) -> QueryResult:
            del config, sql, limit
            raise ConnectorQueryError("PostgreSQL query timed out")

        tool._executor = _timeout
        with pytest.raises(MCPQueryTimeoutError) as timed_out:
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        assert timed_out.value.code is MCPErrorCode.MCP_QUERY_TIMEOUT
        assert SENSITIVE_SQL not in str(timed_out.value)

        async def _unavailable(config: object, sql: str, limit: int) -> QueryResult:
            del config, sql, limit
            raise ConnectorConnectionError("The PostgreSQL database is unavailable")

        tool._executor = _unavailable
        with pytest.raises(MCPDatabaseUnavailableError) as unavailable:
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                context,
            )
        assert unavailable.value.code is MCPErrorCode.MCP_DATABASE_UNAVAILABLE

    run_async(_run())


def test_result_limit_error_code(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_RESULT_CHARS", 20)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 10_000)
    source = _connected_source(db_session, workspace, test_user)

    async def _huge(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("blob",), rows=(("x" * 100,),), truncated=False)

    tool = PostgresQueryTool(db_session, executor=_huge)

    async def _run() -> None:
        with pytest.raises(MCPQueryResultError) as exc_info:
            await tool.execute(
                MCPQueryRequest(data_source_id=source.id, sql="SELECT 1"),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_RESULT_LIMIT_EXCEEDED

    run_async(_run())


def test_sample_error_mapping(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _connected_source(db_session, workspace, test_user)

    class _Sampler:
        def __init__(self, error: Exception) -> None:
            self.error = error

        async def get_sample(self, *args: object, **kwargs: object) -> SampleDataResult:
            del args, kwargs
            raise self.error

    async def _run() -> None:
        with pytest.raises(MCPSampleDataError) as missing:
            await PostgresSampleRowsTool(
                db_session,
                sampler=_Sampler(SampleTableNotFoundError("Table not found")),
            ).invoke(
                MCPSampleRowsRequest(
                    data_source_id=source.id,
                    table_id=uuid.uuid4(),
                ),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert missing.value.code is MCPErrorCode.MCP_RESOURCE_NOT_FOUND
        assert "Table not found" in str(missing.value)

        with pytest.raises(MCPSampleDataError) as ser:
            await PostgresSampleRowsTool(
                db_session,
                sampler=_Sampler(SampleSerializationError("Unable to serialize")),
            ).invoke(
                MCPSampleRowsRequest(
                    data_source_id=source.id,
                    table_id=uuid.uuid4(),
                ),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert ser.value.code is MCPErrorCode.MCP_SAMPLE_DATA_ERROR
        assert "serialize" in str(ser.value).lower()

    run_async(_run())


def test_unexpected_tool_exception_becomes_internal_error(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)

    async def _boom(arguments: object, context: MCPToolContext) -> object:
        del arguments, context
        raise RuntimeError(f"boom sql={SENSITIVE_SQL} password={SECRET}")

    registry.get(POSTGRES_QUERY_TOOL_NAME).invoke = _boom  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(MCPInternalError) as exc_info:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_INTERNAL_ERROR
        payload = error_response_dict(exc_info.value)
        _assert_safe_error_payload(payload)
        assert SENSITIVE_SQL not in str(payload)
        assert SECRET not in str(payload)

    run_async(_run())


def test_cancellation_is_not_swallowed(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)

    async def _cancel(arguments: object, context: MCPToolContext) -> object:
        del arguments, context
        raise asyncio.CancelledError()

    registry.get(POSTGRES_QUERY_TOOL_NAME).invoke = _cancel  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(asyncio.CancelledError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_rate_limit_error_includes_safe_retry_metadata(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_QUERY", "1/minute")
    reset_rate_limiters()
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            context,
        )
        with pytest.raises(MCPRateLimitError) as exc_info:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                context,
            )
        assert exc_info.value.code is MCPErrorCode.MCP_RATE_LIMITED
        assert exc_info.value.retry_after is not None
        assert exc_info.value.retry_after >= 1
        payload = error_response_dict(exc_info.value)
        _assert_safe_error_payload(payload)
        assert payload["error"]["retry_after"] == exc_info.value.retry_after

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()


def test_fastapi_maps_mcp_errors() -> None:
    from app.main import mcp_exception_handler

    probe = FastAPI()
    probe.add_exception_handler(MCPError, mcp_exception_handler)

    @probe.get("/probe")
    async def _probe() -> None:
        raise MCPToolValidationError(
            "MCP tool arguments are invalid", request_id="probe-1"
        )

    with TestClient(probe) as client:
        response = client.get("/probe")
    assert response.status_code == 422
    payload = response.json()
    _assert_safe_error_payload(payload)
    assert payload["error"]["code"] == MCPErrorCode.MCP_TOOL_VALIDATION_FAILED.value
    assert payload["error"]["request_id"] == "probe-1"
    assert "traceback" not in str(payload).lower()


def test_fastapi_maps_rate_limit_with_retry_after() -> None:
    from app.main import mcp_exception_handler

    probe = FastAPI()
    probe.add_exception_handler(MCPError, mcp_exception_handler)

    @probe.get("/probe-rl")
    async def _probe() -> None:
        raise MCPRateLimitError(retry_after=12)

    with TestClient(probe) as client:
        response = client.get("/probe-rl")
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "12"
    payload = response.json()
    _assert_safe_error_payload(payload)
    assert payload["error"]["code"] == MCPErrorCode.MCP_RATE_LIMITED.value


def test_logging_omits_secrets_sql_and_pii(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)

    async def _boom(arguments: object, context: MCPToolContext) -> object:
        del arguments, context
        raise RuntimeError(
            f"failed sql={SENSITIVE_SQL} password={SECRET} "
            f"Authorization=Bearer eyJhbGciOiJIUzI1NiJ9.aaa.bbb "
            f"email=jane.doe@example.com"
        )

    registry.get(POSTGRES_QUERY_TOOL_NAME).invoke = _boom  # type: ignore[method-assign]
    token = bind_request_id("log-req-55")

    with caplog.at_level(logging.WARNING):

        async def _run() -> None:
            with pytest.raises(MCPInternalError):
                await client.call_tool(
                    POSTGRES_QUERY_TOOL_NAME,
                    {"data_source_id": str(source.id), "sql": SENSITIVE_SQL},
                    MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
                )

        try:
            run_async(_run())
        finally:
            reset_request_id(token)

    joined = "\n".join(record.getMessage() for record in caplog.records).lower()
    assert "error_type=" in joined
    assert "code=" in joined
    for forbidden in (
        SENSITIVE_SQL.lower(),
        SECRET.lower(),
        "bearer eyj",
        "jane.doe@example.com",
        "customer_secret_table",
    ):
        assert forbidden not in joined


def test_startup_failure_is_server_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenServer:
        name = "postgres"

        def register(self, registry: MCPRegistry) -> None:
            del registry
            raise RuntimeError("register exploded")

    monkeypatch.setattr(
        "app.mcp.server.lifecycle.PostgreSQLMCPServer",
        lambda session: _BrokenServer(),
    )
    with pytest.raises(MCPServerError) as exc_info:
        build_mcp(session=None)  # type: ignore[arg-type]
    assert exc_info.value.code is MCPErrorCode.MCP_SERVER_ERROR


def test_request_id_attached_to_errors(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    _, client = build_postgres_mcp(db_session)
    token = bind_request_id("corr-55")
    try:

        async def _run() -> None:
            with pytest.raises(MCPServerNotFoundError) as exc_info:
                await client.call_tool(
                    "missing.tool",
                    {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                    MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
                )
            assert exc_info.value.request_id == "corr-55"
            payload = to_error_response(exc_info.value)
            assert payload.error.request_id == "corr-55"

        run_async(_run())
    finally:
        reset_request_id(token)


def test_tool_permission_still_required_on_register() -> None:
    class _In(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class _BareTool:
        name = "postgres.bare"
        description = "bare"
        input_model = _In
        output_model = _In

        async def invoke(self, arguments: object, context: object) -> object:
            del arguments, context
            return {}

    registry = MCPRegistry()

    class _Server:
        name = "postgres"

    registry.register_server(_Server())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="permission"):
        registry.register(_BareTool(), server_name="postgres")  # type: ignore[arg-type]


def test_shutdown_is_idempotent() -> None:
    shutdown_mcp()
    shutdown_mcp()


def test_error_response_never_includes_exception_chain() -> None:
    try:
        try:
            raise ConnectorQueryError("PostgreSQL query timed out")
        except ConnectorQueryError as inner:
            raise MCPQueryTimeoutError("The query timed out") from inner
    except MCPQueryTimeoutError as exc:
        payload = error_response_dict(exc)
        _assert_safe_error_payload(payload)
        assert exc.__cause__ is not None
        assert "ConnectorQueryError" not in str(payload)
