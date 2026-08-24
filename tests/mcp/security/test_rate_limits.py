"""Phase 5.6 — MCP rate-limit scope and isolation tests."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import reset_rate_limiters
from app.db.models import User, Workspace
from app.enums import ColumnSensitivity, DataSourceTableType, WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPRateLimitError,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.servers.postgres.tools.sample_rows import POSTGRES_SAMPLE_ROWS_TOOL_NAME
from app.mcp.servers.postgres.tools.schemas import POSTGRES_LIST_SCHEMAS_TOOL_NAME
from app.services.data_masking import REDACTED
from app.services.sample_data_types import SampleColumn, SampleDataResult
from tests.conftest import run_async
from tests.mcp.conftest import (
    add_workspace_member,
    connected_source,
    tracking_executor,
)
from tests.test_ai_metadata import _schema, _source, _table


class _Sampler:
    def __init__(self, source_id: object, table_id: object) -> None:
        self.source_id = source_id
        self.table_id = table_id
        self.calls = 0

    async def get_sample(self, *args: object, **kwargs: object) -> SampleDataResult:
        del args, kwargs
        self.calls += 1
        return SampleDataResult(
            data_source_id=self.source_id,  # type: ignore[arg-type]
            table_id=self.table_id,  # type: ignore[arg-type]
            schema_name="public",
            table_name="customers",
            table_type=DataSourceTableType.TABLE,
            columns=(
                SampleColumn(
                    name="email",
                    data_type="string",
                    sensitivity=ColumnSensitivity.PII,
                    masked=True,
                ),
            ),
            rows=({"email": REDACTED},),
            row_limit=10,
            truncated_columns=False,
        )


def test_sample_rate_limit_blocks_without_hitting_sampler(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del workspace_member
    source = connected_source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "customers")
    sampler = _Sampler(source.id, table.id)

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_SAMPLE", "1/minute")
    reset_rate_limiters()

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_SAMPLE_ROWS_TOOL_NAME)._sampler = sampler
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)
    args = {"data_source_id": str(source.id), "table_id": str(table.id)}

    async def _run() -> None:
        await client.call_tool(POSTGRES_SAMPLE_ROWS_TOOL_NAME, args, context)
        with pytest.raises(MCPRateLimitError):
            await client.call_tool(POSTGRES_SAMPLE_ROWS_TOOL_NAME, args, context)

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()
    assert sampler.calls == 1


def test_metadata_rate_limit_blocks_catalog_calls(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    _schema(db_session, source, "public")
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_METADATA", "1/minute")
    reset_rate_limiters()
    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        await client.call_tool(
            POSTGRES_LIST_SCHEMAS_TOOL_NAME,
            {"data_source_id": str(source.id)},
            context,
        )
        with pytest.raises(MCPRateLimitError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                context,
            )

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()


def test_rate_limits_are_isolated_per_user(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    admin_user: User,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    add_workspace_member(
        db_session, workspace=workspace, user=admin_user, role=WorkspaceRole.OWNER
    )
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
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        with pytest.raises(MCPRateLimitError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        # Different user still allowed.
        await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            MCPToolContext(workspace_id=workspace.id, user_id=admin_user.id),
        )

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()
    assert len(calls) == 2
