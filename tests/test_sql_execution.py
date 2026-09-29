"""Unit and security tests for Phase 7.3 SQL query execution."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.ai.exceptions import (
    AIContextError,
    AIProviderError,
    AIProviderTimeoutError,
    AIRequestValidationError,
)
from app.ai.sql_execution import (
    SQLExecuteParams,
    SQLExecutionAuthorizationError,
    SQLExecutionCancelledError,
    SQLExecutionDatabaseError,
    SQLExecutionError,
    SQLExecutionErrorCode,
    SQLExecutionRejectedError,
    SQLExecutionResult,
    SQLExecutionResultLimitError,
    SQLExecutionService,
    SQLExecutionStatus,
    SQLExecutionTimeoutError,
    SQLExecutionValidationError,
    ensure_requested_row_limit,
    execute_validated_sql,
    execution_log_context,
    map_mcp_query_error,
    map_sql_execution_error,
    resolve_applied_row_limit,
    serialize_execution_error,
    serialize_execution_result,
    serialize_execution_service_result,
)
from app.ai.sql_validation import (
    SQLValidationViolation,
    SQLValidationViolationCode,
    ValidatedSQL,
)
from app.ai.state import AgentStateService
from app.connectors.exceptions import ConnectorConnectionError, ConnectorQueryError
from app.connectors.types import QueryResult
from app.core.config import settings
from app.core.logging import redact_secret
from app.core.rate_limit import reset_rate_limiters
from app.db.models import (
    DataSource,
    DataSourceConnection,
    Organization,
    User,
    Workspace,
    WorkspaceMember,
)
from app.enums import UserRole, WorkspacePermission, WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPClient,
    MCPQueryRejectedError,
    MCPQueryResult,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPRateLimitError,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.config import mcp_settings
from app.mcp.exceptions import (
    MCPAuthorizationError,
    MCPDatabaseUnavailableError,
    MCPError,
    MCPInternalError,
    MCPQueryLimitError,
    MCPServerError,
    MCPServerNotFoundError,
    MCPToolNotFoundError,
    MCPToolValidationError,
    MCPUnauthorizedError,
)
from app.mcp.servers.postgres.server import PostgreSQLMCPServer
from app.services.credentials import encrypt_secret
from tests.conftest import _create_user, run_async
from tests.test_sql_generation import (
    _fake_metadata,
    _FixedSnapshotStateService,
    _seed_orders_catalog,
)
from tests.test_supervisor import _create_session, _seed_data_source, _seed_workspace

_SQL_EXECUTION_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "ai" / "sql_execution"
)
SECRET_SNIPPET = "CustomerDbPassword!@# 42"
VALID_SQL = "SELECT region, amount FROM public.orders"


def _seed_member(
    db_session: Session,
    *,
    workspace: Workspace,
    user: User,
    role: WorkspaceRole = WorkspaceRole.OWNER,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def _attach_connection(
    db_session: Session,
    data_source: DataSource,
    *,
    password: str = SECRET_SNIPPET,
) -> DataSourceConnection:
    connection = DataSourceConnection(
        data_source_id=data_source.id,
        host="db.internal.example",
        port=5432,
        database_name="analytics",
        username="readonly",
        encrypted_password=encrypt_secret(password),
        ssl_mode="prefer",
    )
    db_session.add(connection)
    db_session.flush()
    return connection


def _authorized_params(
    *,
    user: User,
    workspace: Workspace,
    organization: Organization,
    data_source: DataSource,
    metadata,
    sql: str = VALID_SQL,
    limit: int | None = None,
    session_id: UUID | None = None,
) -> SQLExecuteParams:
    return SQLExecuteParams(
        workspace_id=workspace.id,
        organization_id=organization.id,
        user_id=user.id,
        data_source_id=data_source.id,
        sql=sql,
        metadata=metadata,
        limit=limit,
        session_id=session_id,
    )


def _service_with_executor(
    db_session: Session,
    executor: Callable[[object, str, int], Awaitable[QueryResult]],
) -> tuple[SQLExecutionService, list[tuple[object, ...]]]:
    registry, client = build_postgres_mcp(db_session)
    calls: list[tuple[object, ...]] = []

    async def _tracking(config: object, sql: str, limit: int) -> QueryResult:
        calls.append((config, sql, limit))
        return await executor(config, sql, limit)

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _tracking
    return (
        SQLExecutionService(db_session, mcp_client=client, mcp_registry=registry),
        calls,
    )


async def _ok_executor(config: object, sql: str, limit: int) -> QueryResult:
    del config, sql
    rows = tuple(("west", 10 + index) for index in range(min(2, limit)))
    return QueryResult(columns=("region", "amount"), rows=rows, truncated=False)


class _ForeignToolResponse(BaseModel):
    """Stand-in for a tool response that is not an MCPQueryResult."""

    ok: bool = True


class _StubMCPClient:
    """MCP client stub for boundary outcomes the real stack maps internally."""

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        response: BaseModel | None = None,
    ) -> None:
        self._error = error
        self._response = response or MCPQueryResult(
            columns=["region"],
            rows=[["west"]],
            row_count=1,
        )
        self.calls = 0

    async def call_tool(
        self,
        name: str,
        arguments: BaseModel,
        context: MCPToolContext,
    ) -> BaseModel:
        del name, arguments, context
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._response


def _execute_with_stub_client(
    client: _StubMCPClient,
    *,
    sql: str = VALID_SQL,
) -> SQLExecutionResult:
    metadata = _fake_metadata()
    return run_async(
        execute_validated_sql(
            client=cast(MCPClient, client),
            data_source_id=metadata.data_source_id,
            workspace_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            validated=ValidatedSQL(sql=sql),
            metadata=metadata,
        )
    )


def test_resolve_applied_row_limit_uses_mcp_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_DEFAULT_LIMIT", 25)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_LIMIT", 50)
    assert resolve_applied_row_limit(None) == 25
    assert resolve_applied_row_limit(10) == 10
    assert resolve_applied_row_limit(1_000_000) == 50


def test_map_mcp_and_sql_execution_errors() -> None:
    assert isinstance(
        map_mcp_query_error(MCPQueryTimeoutError()), SQLExecutionTimeoutError
    )
    assert isinstance(
        map_mcp_query_error(MCPQueryResultError()), SQLExecutionResultLimitError
    )
    assert isinstance(
        map_mcp_query_error(MCPQueryRejectedError()), SQLExecutionRejectedError
    )
    assert isinstance(
        map_mcp_query_error(MCPRateLimitError()), SQLExecutionRejectedError
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionAuthorizationError("denied")),
        AIContextError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionValidationError("bad")),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionTimeoutError()),
        AIProviderTimeoutError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionDatabaseError()),
        AIProviderError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionCancelledError()),
        AIProviderError,
    )
    timeout = SQLExecutionTimeoutError()
    assert timeout.status is SQLExecutionStatus.TIMEOUT
    cancelled = SQLExecutionCancelledError()
    assert cancelled.status is SQLExecutionStatus.CANCELLED


def test_successful_select_execution(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
        assert outcome.result.status is SQLExecutionStatus.SUCCEEDED
        assert outcome.result.row_count == 2
        assert outcome.result.columns == ["region", "amount"]
        assert outcome.result.rows[0] == ["west", 10]
        assert outcome.result.truncated is False
        assert outcome.result.duration_ms >= 0
        assert outcome.result.applied_row_limit == mcp_settings.MCP_QUERY_DEFAULT_LIMIT
        assert outcome.validated.sql.lower().startswith("select")
        assert outcome.data_source_id == data_source.id

    run_async(_run())
    assert len(calls) == 1
    assert "SELECT" in str(calls[0][1]).upper()


def test_validation_rejection_before_execution(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionValidationError) as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    sql="DELETE FROM public.orders",
                )
            )
        assert raised.value.code is SQLExecutionErrorCode.SQL_EXECUTION_VALIDATION_ERROR
        assert any(
            item.code
            in {
                SQLValidationViolationCode.DANGEROUS_STATEMENT,
                SQLValidationViolationCode.NOT_READONLY,
            }
            for item in raised.value.violations
        )

    run_async(_run())
    assert calls == []


def test_timeout(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _hang(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise TimeoutError()

    service, _calls = _service_with_executor(db_session, _hang)

    async def _run() -> None:
        with pytest.raises(SQLExecutionTimeoutError, match="timed out") as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        assert raised.value.status is SQLExecutionStatus.TIMEOUT
        assert raised.value.duration_ms >= 0

    run_async(_run())


def test_row_limit_is_capped_server_side(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_LIMIT", 5)
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    seen: dict[str, int] = {}

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql
        seen["limit"] = limit
        return QueryResult(
            columns=("id",),
            rows=tuple((index,) for index in range(limit)),
            truncated=False,
        )

    service, _calls = _service_with_executor(db_session, _executor)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                limit=1_000_000,
            )
        )
        assert outcome.result.row_count == 5
        assert outcome.result.applied_row_limit == 5

    run_async(_run())
    assert seen["limit"] == 5


def test_result_size_limit(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_RESULT_CHARS", 40)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 10_000)
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _huge(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("blob",), rows=(("x" * 5000,),))

    service, _calls = _service_with_executor(db_session, _huge)

    async def _run() -> None:
        with pytest.raises(SQLExecutionResultLimitError, match="too large"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())


def test_database_errors(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _fail(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise ConnectorQueryError("relation does not exist")

    service, _calls = _service_with_executor(db_session, _fail)

    async def _run() -> None:
        with pytest.raises(SQLExecutionDatabaseError) as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        assert raised.value.status is SQLExecutionStatus.FAILED
        assert raised.value.duration_ms >= 0

    run_async(_run())


def test_empty_results(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _empty(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("region", "amount"), rows=())

    service, _calls = _service_with_executor(db_session, _empty)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
        assert outcome.result.status is SQLExecutionStatus.SUCCEEDED
        assert outcome.result.row_count == 0
        assert outcome.result.rows == []
        assert outcome.result.columns == ["region", "amount"]

    run_async(_run())


def test_large_results_are_truncated(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_RESULT_CHARS", 120)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_VALUE_CHARS", 10_000)
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _many(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(
            columns=("blob",),
            rows=tuple(("x" * 40,) for _ in range(20)),
        )

    service, _calls = _service_with_executor(db_session, _many)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
        assert outcome.result.truncated is True
        assert 1 <= outcome.result.row_count < 20

    run_async(_run())


def test_member_lacks_query_permission(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="permission"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_rejects_non_member(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="not a member"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_rejects_organization_mismatch(db_session: Session) -> None:
    user, workspace, _organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    other_org = Organization(name="Other Org", slug=f"other-{uuid.uuid4().hex[:8]}")
    db_session.add(other_org)
    db_session.flush()
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="organization"):
            await service.execute(
                SQLExecuteParams(
                    workspace_id=workspace.id,
                    organization_id=other_org.id,
                    user_id=user.id,
                    data_source_id=data_source.id,
                    sql=VALID_SQL,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_rejects_foreign_data_source(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    foreign = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _attach_connection(db_session, foreign)
    metadata = _seed_orders_catalog(db_session, foreign)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="not accessible"):
            await service.execute(
                _authorized_params(
                    user=user_a,
                    workspace=workspace_a,
                    organization=organization_a,
                    data_source=foreign,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_rejects_invented_metadata_ids(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="unauthorized"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=_fake_metadata(data_source.id),
                )
            )

    run_async(_run())
    assert calls == []


def test_session_cross_workspace_denied(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    data_source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    data_source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _attach_connection(db_session, data_source_b)
    metadata_b = _seed_orders_catalog(db_session, data_source_b)
    _, snapshot = _create_session(
        db_session,
        user=user_a,
        workspace=workspace_a,
        organization=organization_a,
        data_source=data_source_a,
    )
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="not accessible"):
            await service.execute(
                _authorized_params(
                    user=user_b,
                    workspace=workspace_b,
                    organization=organization_b,
                    data_source=data_source_b,
                    metadata=metadata_b,
                    session_id=snapshot.session_id,
                )
            )

    run_async(_run())
    assert calls == []


def test_workspace_data_source_isolation_on_mcp_path(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, _ = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _attach_connection(db_session, source_a)
    _attach_connection(db_session, source_b)
    metadata_a = _seed_orders_catalog(db_session, source_a)

    async def _should_not_run(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise AssertionError("executor must not run for foreign data source")

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _should_not_run
    service = SQLExecutionService(db_session, mcp_client=client, mcp_registry=registry)

    async def _run() -> None:
        # Pre-auth uses workspace_a but points metadata/source inconsistently via
        # foreign source_b while claiming workspace_a ownership — blocked early.
        with pytest.raises(SQLExecutionAuthorizationError):
            await service.execute(
                SQLExecuteParams(
                    workspace_id=workspace_a.id,
                    organization_id=organization_a.id,
                    user_id=user_a.id,
                    data_source_id=source_b.id,
                    sql=VALID_SQL,
                    metadata=metadata_a,
                )
            )

    run_async(_run())


def test_concurrent_execution_isolation(db_session: Session) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    _attach_connection(db_session, source_a)
    _attach_connection(db_session, source_b)
    metadata_a = _seed_orders_catalog(db_session, source_a)
    metadata_b = _seed_orders_catalog(db_session, source_b)

    seen: list[str] = []

    async def _exec(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        seen.append(sql)
        await asyncio.sleep(0)
        marker = "a" if "amount" in sql else "b"
        return QueryResult(columns=("marker",), rows=((marker,),))

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _exec
    service = SQLExecutionService(db_session, mcp_client=client, mcp_registry=registry)

    async def _run() -> None:
        results = await asyncio.gather(
            service.execute(
                _authorized_params(
                    user=user_a,
                    workspace=workspace_a,
                    organization=organization_a,
                    data_source=source_a,
                    metadata=metadata_a,
                    sql="SELECT region, amount FROM public.orders",
                )
            ),
            service.execute(
                _authorized_params(
                    user=user_b,
                    workspace=workspace_b,
                    organization=organization_b,
                    data_source=source_b,
                    metadata=metadata_b,
                    sql="SELECT region FROM public.orders",
                )
            ),
        )
        assert results[0].result.rows == [["a"]]
        assert results[1].result.rows == [["b"]]
        assert results[0].data_source_id == source_a.id
        assert results[1].data_source_id == source_b.id

    run_async(_run())
    assert len(seen) == 2


def test_cancellation_and_cleanup(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    started = asyncio.Event()

    async def _slow(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        started.set()
        await asyncio.sleep(60)
        return QueryResult(columns=("id",), rows=((1,),))

    service, _calls = _service_with_executor(db_session, _slow)

    async def _run() -> None:
        task = asyncio.create_task(
            service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(SQLExecutionCancelledError) as raised:
            await task
        assert raised.value.status is SQLExecutionStatus.CANCELLED
        assert raised.value.duration_ms >= 0

    run_async(_run())


def test_safe_error_and_result_serialization(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, _calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                sql=f"{VALID_SQL} -- {SECRET_SNIPPET}",
            )
        )
        payload = serialize_execution_service_result(outcome)
        result_payload = serialize_execution_result(outcome.result)
        rendered = json.dumps(payload) + json.dumps(result_payload)
        assert SECRET_SNIPPET not in rendered
        assert "password" not in rendered.lower()
        assert "SELECT" not in rendered
        assert "db.internal.example" not in rendered
        assert payload["status"] == SQLExecutionStatus.SUCCEEDED.value
        assert "sql" not in payload

        validation_error = SQLExecutionValidationError(
            f"bad sql containing password={SECRET_SNIPPET}",
            violations=[],
        )
        error_payload = serialize_execution_error(validation_error)
        error_json = json.dumps(error_payload)
        assert SECRET_SNIPPET not in error_json
        assert "[REDACTED]" in error_json
        assert error_payload["code"] == (
            SQLExecutionErrorCode.SQL_EXECUTION_VALIDATION_ERROR.value
        )
        assert error_payload["status"] == SQLExecutionStatus.REJECTED.value
        assert "duration_ms" in error_payload

        context = execution_log_context(outcome.result)
        assert SECRET_SNIPPET not in str(context)
        assert "SELECT" not in str(context)

    run_async(_run())


def test_execution_does_not_log_sql_or_secrets(
    db_session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, _calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with caplog.at_level(logging.INFO):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    sql=f"{VALID_SQL} -- {SECRET_SNIPPET}",
                )
            )

    run_async(_run())
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert SECRET_SNIPPET not in joined
    assert "SELECT region" not in joined
    assert SECRET_SNIPPET not in redact_secret(f"password={SECRET_SNIPPET}")


def test_sql_execution_package_uses_mcp_not_direct_connector() -> None:
    import ast

    forbidden_direct = {
        "app.connectors.postgresql",
        "psycopg",
        "asyncpg",
    }
    found_mcp = False
    for path in _SQL_EXECUTION_DIR.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "execute_query" not in source, path
        if path.name == "execution.py":
            assert "POSTGRES_QUERY_TOOL_NAME" in source
            assert "call_tool" in source
            found_mcp = True
        tree_imports: set[str] = set()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                tree_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                tree_imports.add(node.module)
        assert not tree_imports.intersection(forbidden_direct), path
    assert found_mcp


def test_forged_validated_sql_is_rechecked_before_mcp(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)
    forged = ValidatedSQL(
        sql="SELECT table_name FROM information_schema.tables",
        referenced_tables=["information_schema.tables"],
        referenced_columns=["table_name"],
    )

    async def _run() -> None:
        with pytest.raises(SQLExecutionValidationError) as raised:
            await execute_validated_sql(
                client=service._mcp_client,
                data_source_id=data_source.id,
                workspace_id=workspace.id,
                user_id=user.id,
                validated=forged,
                metadata=metadata,
            )
        assert raised.value.status is SQLExecutionStatus.REJECTED
        assert {item.code for item in raised.value.violations} & {
            SQLValidationViolationCode.UNKNOWN_TABLE,
            SQLValidationViolationCode.CROSS_SCHEMA,
        }

    run_async(_run())
    assert calls == []
    del organization


def test_invalid_row_limit_is_rejected_before_execution(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(SQLExecutionValidationError, match="row limit"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    limit=0,
                )
            )

    run_async(_run())
    assert calls == []


def test_rate_limit_is_enforced_via_mcp(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_QUERY", "1/minute")
    reset_rate_limiters()
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        first = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
        assert first.result.status is SQLExecutionStatus.SUCCEEDED
        with pytest.raises(
            SQLExecutionRejectedError, match="Too many MCP requests"
        ) as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        assert raised.value.status is SQLExecutionStatus.REJECTED
        assert raised.value.duration_ms >= 0

    run_async(_run())
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mcp_error", "expected"),
    [
        (MCPToolValidationError(), SQLExecutionRejectedError),
        (MCPDatabaseUnavailableError(), SQLExecutionDatabaseError),
        (MCPQueryLimitError(), SQLExecutionDatabaseError),
        (MCPInternalError(), SQLExecutionDatabaseError),
        (MCPUnauthorizedError(), SQLExecutionDatabaseError),
        (MCPAuthorizationError(), SQLExecutionAuthorizationError),
        (MCPToolNotFoundError(), SQLExecutionAuthorizationError),
        (MCPServerNotFoundError(), SQLExecutionAuthorizationError),
    ],
)
def test_map_mcp_query_error_covers_remaining_boundary_errors(
    mcp_error: MCPError,
    expected: type[SQLExecutionError],
) -> None:
    # Registry/lookup failures land on the authorization arm because the mapper
    # matches "not found" in the message; pinned so the surface cannot drift.
    mapped = map_mcp_query_error(mcp_error)
    assert type(mapped) is expected


def test_map_sql_execution_error_covers_remaining_arms() -> None:
    assert isinstance(
        map_sql_execution_error(SQLExecutionRejectedError()),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionResultLimitError()),
        AIRequestValidationError,
    )
    assert isinstance(
        map_sql_execution_error(SQLExecutionError("boom")),
        AIProviderError,
    )


def test_mapped_mcp_errors_carry_execution_duration() -> None:
    client = _StubMCPClient(error=MCPQueryTimeoutError())
    with pytest.raises(SQLExecutionTimeoutError) as raised:
        _execute_with_stub_client(client)
    assert raised.value.status is SQLExecutionStatus.TIMEOUT
    assert raised.value.duration_ms >= 0
    assert client.calls == 1


def test_unexpected_mcp_response_type_is_rejected() -> None:
    client = _StubMCPClient(response=_ForeignToolResponse())
    with pytest.raises(SQLExecutionError) as raised:
        _execute_with_stub_client(client)
    assert raised.value.code is SQLExecutionErrorCode.SQL_EXECUTION_INTERNAL_ERROR
    assert "Unexpected MCP query response" in str(raised.value)
    assert raised.value.duration_ms >= 0


def test_execution_result_reports_validated_sql_counts() -> None:
    client = _StubMCPClient()
    sql = "SELECT region, amount FROM public.orders"
    result = _execute_with_stub_client(client, sql=sql)
    assert result.status is SQLExecutionStatus.SUCCEEDED
    assert result.sql_char_count == len(sql)
    assert result.referenced_table_count == 1
    assert result.referenced_column_count == 2
    assert result.applied_row_limit == resolve_applied_row_limit(None)


def test_ensure_requested_row_limit_rejects_invalid_limits() -> None:
    assert ensure_requested_row_limit(None) is None
    assert ensure_requested_row_limit(10) == 10
    for invalid in (0, -5, True):
        with pytest.raises(SQLExecutionValidationError, match="row limit"):
            ensure_requested_row_limit(cast(int, invalid))


def test_connection_loss_maps_to_database_error(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _drop(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise ConnectorConnectionError("The database is unavailable")

    service, _calls = _service_with_executor(db_session, _drop)

    async def _run() -> None:
        with pytest.raises(SQLExecutionDatabaseError) as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        assert raised.value.status is SQLExecutionStatus.FAILED
        assert raised.value.duration_ms >= 0

    run_async(_run())


def test_unexpected_executor_failure_is_sanitized(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)

    async def _boom(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        raise RuntimeError(f"password={SECRET_SNIPPET} host=db.internal.example")

    service, _calls = _service_with_executor(db_session, _boom)

    async def _run() -> None:
        with pytest.raises(SQLExecutionDatabaseError) as raised:
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )
        message = str(raised.value)
        assert SECRET_SNIPPET not in message
        assert "db.internal.example" not in message

    run_async(_run())


def test_service_requires_mcp_client_and_registry_together(
    db_session: Session,
) -> None:
    _registry, client = build_postgres_mcp(db_session)
    with pytest.raises(ValueError, match="provided together"):
        SQLExecutionService(db_session, mcp_client=client)


def test_service_surfaces_mcp_startup_failure(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(self: PostgreSQLMCPServer, registry: object) -> None:
        del self, registry
        raise RuntimeError("registration exploded")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _fail)
    with pytest.raises(MCPServerError, match="failed to start"):
        SQLExecutionService(db_session)


def test_raw_cancellation_from_execution_is_wrapped(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _cancel(**kwargs: object) -> SQLExecutionResult:
        del kwargs
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "app.ai.sql_execution.service.execute_validated_sql",
        _cancel,
    )

    async def _run() -> None:
        with pytest.raises(SQLExecutionCancelledError, match="cancelled"):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_applied_row_limit_matches_server_side_resolution(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MCP_QUERY_DEFAULT_LIMIT", 100)
    monkeypatch.setattr(settings, "MCP_QUERY_MAX_LIMIT", 10)
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
            )
        )
        assert outcome.result.applied_row_limit == 10

    run_async(_run())
    assert calls[0][2] == 10


def test_serialize_execution_error_caps_and_redacts_violations() -> None:
    violations = [
        SQLValidationViolation(
            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
            message=f"violation {index} password={SECRET_SNIPPET}",
            identifier=f"column_{index}",
        )
        for index in range(25)
    ]
    payload = serialize_execution_error(
        SQLExecutionValidationError("rejected", violations=violations)
    )
    assert len(payload["violations"]) == 20
    rendered = json.dumps(payload)
    assert SECRET_SNIPPET not in rendered
    assert "[REDACTED]" in rendered
    assert payload["violations"][0]["identifier"] == "column_0"

    timeout_payload = serialize_execution_error(SQLExecutionTimeoutError())
    assert "violations" not in timeout_payload
    assert timeout_payload["status"] == SQLExecutionStatus.TIMEOUT.value


def test_execution_rejects_inactive_user(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)
    user.is_active = False
    db_session.flush()

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="User is not authorized",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_execution_rejects_unknown_workspace(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    service, calls = _service_with_executor(db_session, _ok_executor)
    params = replace(
        _authorized_params(
            user=user,
            workspace=workspace,
            organization=organization,
            data_source=data_source,
            metadata=metadata,
        ),
        workspace_id=uuid.uuid4(),
    )

    async def _run() -> None:
        with pytest.raises(SQLExecutionAuthorizationError, match="Workspace is not"):
            await service.execute(params)

    run_async(_run())
    assert calls == []


def test_execution_rejects_metadata_from_another_data_source(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    other_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    other_metadata = _seed_orders_catalog(db_session, other_source)
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="Metadata context does not match",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=other_metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_execution_allows_super_admin_without_membership(db_session: Session) -> None:
    owner, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=owner)
    data_source = _seed_data_source(db_session, workspace=workspace, user=owner)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)
    params = _authorized_params(
        user=super_admin,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
        metadata=metadata,
    )
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _ok_executor

    async def _run() -> None:
        permissive = SQLExecutionService(
            db_session,
            mcp_client=client,
            mcp_registry=registry,
        )
        outcome = await permissive.execute(params)
        assert outcome.result.status is SQLExecutionStatus.SUCCEEDED

        strict = SQLExecutionService(
            db_session,
            mcp_client=client,
            mcp_registry=registry,
            allow_super_admin=False,
        )
        with pytest.raises(SQLExecutionAuthorizationError, match="not a member"):
            await strict.execute(params)

    run_async(_run())


def test_mcp_permission_check_denies_when_service_check_is_weaker(
    db_session: Session,
) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.MEMBER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    registry, client = build_postgres_mcp(db_session)
    calls: list[tuple[object, ...]] = []

    async def _tracking(config: object, sql: str, limit: int) -> QueryResult:
        calls.append((config, sql, limit))
        return await _ok_executor(config, sql, limit)

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _tracking
    service = SQLExecutionService(
        db_session,
        mcp_client=client,
        mcp_registry=registry,
        required_permission=WorkspacePermission.DATA_SOURCE_READ,
    )

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="Data source is not accessible",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert calls == []


def test_execution_session_scoped_request_succeeds(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        outcome = await service.execute(
            _authorized_params(
                user=user,
                workspace=workspace,
                organization=organization,
                data_source=data_source,
                metadata=metadata,
                session_id=snapshot.session_id,
            )
        )
        assert outcome.result.status is SQLExecutionStatus.SUCCEEDED

    run_async(_run())
    assert len(calls) == 1


def test_execution_session_without_data_source_denied(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
    )
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="no authorized data source",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )

    run_async(_run())
    assert calls == []


def test_execution_session_data_source_mismatch_denied(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    session_source = _seed_data_source(db_session, workspace=workspace, user=user)
    other_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, other_source)
    metadata = _seed_orders_catalog(db_session, other_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=session_source,
    )
    service, calls = _service_with_executor(db_session, _ok_executor)

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="does not match the analysis session",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=other_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )

    run_async(_run())
    assert calls == []


def test_execution_session_organization_drift_denied(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    _, snapshot = _create_session(
        db_session,
        user=user,
        workspace=workspace,
        organization=organization,
        data_source=data_source,
    )
    registry, client = build_postgres_mcp(db_session)
    calls: list[tuple[object, ...]] = []

    async def _tracking(config: object, sql: str, limit: int) -> QueryResult:
        calls.append((config, sql, limit))
        return await _ok_executor(config, sql, limit)

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _tracking
    drifted = snapshot.model_copy(update={"organization_id": uuid.uuid4()})
    service = SQLExecutionService(
        db_session,
        mcp_client=client,
        mcp_registry=registry,
        state_service=cast(AgentStateService, _FixedSnapshotStateService(drifted)),
    )

    async def _run() -> None:
        with pytest.raises(
            SQLExecutionAuthorizationError,
            match="organization does not match",
        ):
            await service.execute(
                _authorized_params(
                    user=user,
                    workspace=workspace,
                    organization=organization,
                    data_source=data_source,
                    metadata=metadata,
                    session_id=snapshot.session_id,
                )
            )

    run_async(_run())
    assert calls == []
