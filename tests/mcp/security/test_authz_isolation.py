"""Phase 5.6 — MCP authorization, isolation, IDOR, and permission tests."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.db.models import User, Workspace
from app.enums import WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPError,
    MCPErrorCode,
    MCPQueryError,
    MCPToolContext,
    MCPUnauthorizedError,
    build_postgres_mcp,
)
from app.mcp.exceptions import MCPAccessDeniedError
from app.mcp.security import MCPToolPermission
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
from tests.conftest import run_async
from tests.mcp.conftest import (
    PermissionProbeTool,
    StaticServer,
    add_workspace_member,
    connected_source,
    create_organization,
    create_workspace,
    fake_query_ok,
)
from tests.test_ai_metadata import (
    _column,
    _relationship,
    _schema,
    _source,
    _table,
)


def test_inactive_user_is_unauthorized(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    test_user.is_active = False
    db_session.flush()
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPUnauthorizedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_organization_isolation(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    other_org = create_organization(db_session, name="Isolated Org")
    other_ws = create_workspace(db_session, other_org, name="Other Org WS")
    foreign_source = _source(db_session, other_ws, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError) as exc_info:
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(foreign_source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND

    run_async(_run())


def test_same_organization_different_workspace_isolation(
    db_session: Session,
    workspace: Workspace,
    organization: object,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    from app.db.models import Organization

    assert isinstance(organization, Organization)
    other = create_workspace(db_session, organization, name="Sibling WS")
    foreign = _source(db_session, other, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(foreign.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_column_and_relationship_idor(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source_a = _source(db_session, workspace, test_user)
    source_b = _source(db_session, workspace, test_user)
    schema_b = _schema(db_session, source_b, "hr")
    table_b = _table(db_session, schema_b, "employees")
    emp_id = _column(db_session, table_b, "id", position=1, primary_key=True)
    mgr = _column(db_session, table_b, "manager_id", position=2)
    _relationship(
        db_session,
        source_table=table_b,
        source_column=mgr,
        target_table=table_b,
        target_column=emp_id,
        constraint_name="employees_manager_fk",
    )
    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPError, match="Table not found"):
            await client.call_tool(
                POSTGRES_GET_COLUMNS_TOOL_NAME,
                {
                    "data_source_id": str(source_a.id),
                    "table_id": str(table_b.id),
                },
                context,
            )
        # Relationships are scoped by data_source_id; foreign DS returns empty
        # rather than leaking source_b rows when querying source_a.
        result = await client.call_tool(
            POSTGRES_GET_RELATIONSHIPS_TOOL_NAME,
            {"data_source_id": str(source_a.id)},
            context,
        )
        assert result.items == []
        assert result.total == 0

    run_async(_run())


def test_forged_user_id_cannot_escalate(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    admin_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    # admin_user is not a workspace member; forging context.user_id as owner fails.
    source = _source(db_session, workspace, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=admin_user.id),
            )

    run_async(_run())


@pytest.mark.parametrize(
    ("tool_name", "permission", "role", "should_pass"),
    [
        (
            POSTGRES_LIST_SCHEMAS_TOOL_NAME,
            MCPToolPermission.METADATA_READ,
            WorkspaceRole.MEMBER,
            True,
        ),
        (
            POSTGRES_SAMPLE_ROWS_TOOL_NAME,
            MCPToolPermission.SAMPLE_DATA_READ,
            WorkspaceRole.MEMBER,
            True,
        ),
        (
            POSTGRES_QUERY_TOOL_NAME,
            MCPToolPermission.QUERY_READ,
            WorkspaceRole.MEMBER,
            False,
        ),
        (
            POSTGRES_QUERY_TOOL_NAME,
            MCPToolPermission.QUERY_READ,
            WorkspaceRole.ADMIN,
            True,
        ),
        (
            POSTGRES_QUERY_TOOL_NAME,
            MCPToolPermission.QUERY_READ,
            WorkspaceRole.OWNER,
            True,
        ),
    ],
)
def test_central_permission_matrix(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    tool_name: str,
    permission: MCPToolPermission,
    role: WorkspaceRole,
    should_pass: bool,
) -> None:
    del permission
    add_workspace_member(db_session, workspace=workspace, user=test_user, role=role)
    source = connected_source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "customers")
    registry, client = build_postgres_mcp(db_session)
    if tool_name == POSTGRES_QUERY_TOOL_NAME:
        registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = fake_query_ok
    if tool_name == POSTGRES_SAMPLE_ROWS_TOOL_NAME:
        from app.enums import ColumnSensitivity, DataSourceTableType
        from app.services.data_masking import REDACTED
        from app.services.sample_data_types import SampleColumn, SampleDataResult

        class _Sampler:
            async def get_sample(
                self, *args: object, **kwargs: object
            ) -> SampleDataResult:
                del args, kwargs
                return SampleDataResult(
                    data_source_id=source.id,
                    table_id=table.id,
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

        registry.get(POSTGRES_SAMPLE_ROWS_TOOL_NAME)._sampler = _Sampler()

    args: dict[str, object] = {"data_source_id": str(source.id)}
    if tool_name in {
        POSTGRES_SAMPLE_ROWS_TOOL_NAME,
        POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
        POSTGRES_GET_COLUMNS_TOOL_NAME,
        POSTGRES_LIST_TABLES_TOOL_NAME,
    }:
        if tool_name == POSTGRES_LIST_TABLES_TOOL_NAME:
            pass
        else:
            args["table_id"] = str(table.id)
    if tool_name == POSTGRES_QUERY_TOOL_NAME:
        args["sql"] = "SELECT 1"

    async def _run() -> None:
        context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)
        if should_pass:
            result = await client.call_tool(tool_name, args, context)
            assert result is not None
        else:
            with pytest.raises((MCPAccessDeniedError, MCPQueryError)):
                await client.call_tool(tool_name, args, context)

    run_async(_run())


def test_registered_tool_without_own_auth_still_protected(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    """Central client must deny QUERY_READ for MEMBER even if tool skips auth."""
    add_workspace_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = connected_source(db_session, workspace, test_user)
    registry, _ = build_postgres_mcp(db_session)
    probe = PermissionProbeTool(permission=MCPToolPermission.QUERY_READ)
    StaticServer([probe]).register(registry)
    from app.mcp.client import MCPClient

    client = MCPClient(registry, db_session)

    async def _run() -> None:
        with pytest.raises(MCPQueryError):
            await client.call_tool(
                "probe.action",
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())
    assert probe.invoked == []


def test_forged_permission_on_tool_does_not_bypass_registry(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = connected_source(db_session, workspace, test_user)
    registry, client = build_postgres_mcp(db_session)
    tool = registry.get(POSTGRES_QUERY_TOOL_NAME)
    # Mutating the live tool object must not change the frozen registry record.
    tool.permission = MCPToolPermission.METADATA_READ  # type: ignore[misc]
    tool._executor = fake_query_ok
    assert registry.get_record(POSTGRES_QUERY_TOOL_NAME).permission is (
        MCPToolPermission.QUERY_READ
    )

    async def _run() -> None:
        seen: dict[str, object] = {}
        import app.mcp.client.client as client_mod

        original = client_mod.resolve_authorized_workspace_id

        def _capture(**kwargs: object) -> uuid.UUID:
            seen.update(kwargs)
            return original(**kwargs)  # type: ignore[arg-type]

        client_mod.resolve_authorized_workspace_id = _capture  # type: ignore[assignment]
        try:
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        finally:
            client_mod.resolve_authorized_workspace_id = original  # type: ignore[assignment]
        assert seen["permission"] is MCPToolPermission.QUERY_READ

    run_async(_run())
