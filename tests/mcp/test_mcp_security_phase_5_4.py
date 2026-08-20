from __future__ import annotations

import uuid

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.connectors.types import QueryResult
from app.db.models import (
    DataSourceConnection,
    Organization,
    User,
    Workspace,
    WorkspaceMember,
)
from app.enums import ColumnSensitivity, DataSourceTableType, WorkspaceRole
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPError,
    MCPQueryError,
    MCPRateLimitError,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPServerNotFoundError,
    MCPToolNameValidationError,
    MCPToolNotFoundError,
    MCPToolValidationError,
    MCPUnauthorizedError,
)
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
from app.services.credentials import encrypt_secret
from app.services.data_masking import REDACTED
from app.services.sample_data_types import SampleColumn, SampleDataResult
from tests.conftest import run_async
from tests.test_ai_metadata import _schema, _source, _table

SECRET = "CustomerMcpPassword!@# 42"


async def _fake_ok(config: object, sql: str, limit: int) -> QueryResult:
    del config, sql
    rows = tuple((index,) for index in range(min(3, limit)))
    return QueryResult(columns=("id",), rows=rows, truncated=False)


def _add_postgres_connection(db_session: Session, *, data_source_id: uuid.UUID) -> None:
    db_session.add(
        DataSourceConnection(
            data_source_id=data_source_id,
            host="db.internal.example",
            port=5432,
            database_name="analytics",
            username="readonly",
            encrypted_password=encrypt_secret(SECRET),
            ssl_mode="prefer",
        )
    )
    db_session.flush()


def _add_member(
    db_session: Session,
    *,
    workspace: Workspace,
    user: User,
    role: WorkspaceRole,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def test_mcp_authentication_required(db_session: Session, workspace: Workspace) -> None:
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPUnauthorizedError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id),
            )

    run_async(_run())


def test_mcp_unknown_tool_and_server_are_rejected(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    _, client = build_postgres_mcp(db_session)
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPToolNotFoundError):
            await client.call_tool(
                "postgres.write",
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                context,
            )

        with pytest.raises(MCPServerNotFoundError):
            await client.call_tool(
                "unknown.query",
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                context,
            )

        with pytest.raises(MCPToolNameValidationError):
            await client.call_tool(
                "app.services.some_function",
                {"data_source_id": str(uuid.uuid4()), "sql": "SELECT 1"},
                context,
            )

    run_async(_run())


def test_workspace_isolation_for_query_tool(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    # workspace fixture is authorized; a new workspace is intentionally not.
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Workspace B",
        slug=f"ws-b-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()

    source_a = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source_a.id)

    source_b = _source(db_session, workspace_b, test_user)
    # Connection isn't needed for denial (auth fails first), but add it to
    # ensure the allowed path works consistently.
    _add_postgres_connection(db_session, data_source_id=source_b.id)

    registry, client = build_postgres_mcp(db_session)
    registry.get(
        POSTGRES_QUERY_TOOL_NAME
    )._executor = _fake_ok  # inject deterministic executor

    async def _run() -> None:
        allowed = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source_a.id), "sql": "SELECT 1"},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert allowed.row_count == 3

        with pytest.raises(MCPQueryError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source_b.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_idor_protection_for_table_and_schema_ids(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    source_a = _source(db_session, workspace, test_user)
    source_b = _source(db_session, workspace, test_user)

    schema_b = _schema(db_session, source_b, "hr")
    table_b = _table(db_session, schema_b, "employees")

    registry, client = build_postgres_mcp(db_session)
    # Metadata tools use DB metadata; no executor injection required.
    _ = registry

    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        with pytest.raises(MCPError, match="Table not found"):
            await client.call_tool(
                POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
                {
                    "data_source_id": str(source_a.id),
                    "table_id": str(table_b.id),
                },
                context,
            )

        with pytest.raises(MCPError, match="Schema not found"):
            await client.call_tool(
                POSTGRES_LIST_TABLES_TOOL_NAME,
                {
                    "data_source_id": str(source_a.id),
                    "schema_id": str(schema_b.id),
                    "table_type": DataSourceTableType.TABLE,
                },
                context,
            )

        with pytest.raises(MCPError, match="Table not found"):
            await client.call_tool(
                POSTGRES_GET_COLUMNS_TOOL_NAME,
                {
                    "data_source_id": str(source_a.id),
                    "table_id": str(table_b.id),
                },
                context,
            )

    run_async(_run())


def test_client_enforces_authorization_before_tool_invoke(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Workspace B Auth",
        slug=f"ws-b-auth-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()

    source_b = _source(db_session, workspace_b, test_user)
    invoked: list[str] = []

    registry, client = build_postgres_mcp(db_session)
    original = registry.get(POSTGRES_LIST_SCHEMAS_TOOL_NAME).invoke

    async def _tracking_invoke(arguments: object, context: MCPToolContext) -> object:
        invoked.append("called")
        return await original(arguments, context)

    registry.get(POSTGRES_LIST_SCHEMAS_TOOL_NAME).invoke = _tracking_invoke  # type: ignore[method-assign]

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source_b.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())
    assert invoked == []


def test_client_uses_registry_permission_for_authorization(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source.id)
    seen: dict[str, object] = {}

    def _capture_resolve(**kwargs: object) -> uuid.UUID:
        seen.update(kwargs)
        return workspace.id

    monkeypatch.setattr(
        "app.mcp.client.client.resolve_authorized_workspace_id",
        _capture_resolve,
    )
    monkeypatch.setattr(
        "app.mcp.client.client.enforce_mcp_rate_limit",
        lambda **_kwargs: None,
    )

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok

    async def _run() -> None:
        await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )

    run_async(_run())
    assert seen["permission"] is MCPToolPermission.QUERY_READ
    assert seen["data_source_id"] == source.id


def test_mcp_rate_limit_blocks_excessive_query_calls(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.core.config import settings
    from app.core.rate_limit import reset_rate_limiters

    source = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source.id)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_MCP_QUERY", "2/minute")
    reset_rate_limiters()

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok
    context = MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)

    async def _run() -> None:
        for _ in range(2):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                context,
            )
        with pytest.raises(MCPRateLimitError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                context,
            )

    try:
        run_async(_run())
    finally:
        reset_rate_limiters()


def test_registry_maps_tool_permissions(db_session: Session) -> None:
    registry, _client = build_postgres_mcp(db_session)
    expected = {
        POSTGRES_LIST_SCHEMAS_TOOL_NAME: MCPToolPermission.METADATA_READ,
        POSTGRES_LIST_TABLES_TOOL_NAME: MCPToolPermission.METADATA_READ,
        POSTGRES_DESCRIBE_TABLE_TOOL_NAME: MCPToolPermission.METADATA_READ,
        POSTGRES_GET_COLUMNS_TOOL_NAME: MCPToolPermission.METADATA_READ,
        POSTGRES_GET_RELATIONSHIPS_TOOL_NAME: MCPToolPermission.METADATA_READ,
        POSTGRES_SAMPLE_ROWS_TOOL_NAME: MCPToolPermission.SAMPLE_DATA_READ,
        POSTGRES_QUERY_TOOL_NAME: MCPToolPermission.QUERY_READ,
    }
    for name, permission in expected.items():
        assert registry.get_record(name).permission is permission


def test_member_with_data_source_read_cannot_query(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    """CASE 1: DATA_SOURCE_READ without QUERY_READ → query denied."""
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source.id)
    invoked: list[str] = []

    registry, client = build_postgres_mcp(db_session)
    original = registry.get(POSTGRES_QUERY_TOOL_NAME).invoke

    async def _tracking_invoke(arguments: object, context: MCPToolContext) -> object:
        invoked.append("called")
        return await original(arguments, context)

    registry.get(POSTGRES_QUERY_TOOL_NAME).invoke = _tracking_invoke  # type: ignore[method-assign]
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok

    async def _run() -> None:
        with pytest.raises(MCPQueryError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())
    assert invoked == []


def test_owner_with_query_read_can_query(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    """CASE 2: QUERY_READ (via OWNER) + data-source access → allowed."""
    source = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source.id)
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok

    async def _run() -> None:
        result = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            {"data_source_id": str(source.id), "sql": "SELECT 1"},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.row_count == 3

    run_async(_run())


def test_member_can_sample_rows_with_masking(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    """CASE 3: SAMPLE_DATA_READ without QUERY_READ → sample allowed + masked."""
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = _source(db_session, workspace, test_user)
    table_id = uuid.uuid4()
    masked = SampleDataResult(
        data_source_id=source.id,
        table_id=table_id,
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

    registry, client = build_postgres_mcp(db_session)

    async def _fake_sample(*_args: object, **_kwargs: object) -> SampleDataResult:
        return masked

    registry.get(POSTGRES_SAMPLE_ROWS_TOOL_NAME)._sampler = type(
        "_Sampler",
        (),
        {"get_sample": staticmethod(_fake_sample)},
    )()

    async def _run() -> None:
        result = await client.call_tool(
            POSTGRES_SAMPLE_ROWS_TOOL_NAME,
            {"data_source_id": str(source.id), "table_id": str(table_id)},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.rows[0]["email"] == REDACTED
        assert result.columns[0].masked is True

    run_async(_run())


def test_query_read_without_data_source_access_is_denied(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
) -> None:
    """CASE 4 / 5 / IDOR: QUERY_READ does not grant access to foreign data sources."""
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.OWNER
    )
    foreign_workspace = Workspace(
        organization_id=organization.id,
        name="Foreign WS",
        slug=f"ws-foreign-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(foreign_workspace)
    db_session.flush()
    foreign_source = _source(db_session, foreign_workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=foreign_source.id)

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok

    async def _run() -> None:
        with pytest.raises(MCPQueryError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(foreign_source.id), "sql": "SELECT 1"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_data_source_read_cannot_bypass_query_via_unmask_fields(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    """CASE 6: include_unmasked/mask/raw do not grant query access."""
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = _source(db_session, workspace, test_user)
    _add_postgres_connection(db_session, data_source_id=source.id)
    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _fake_ok

    async def _run() -> None:
        with pytest.raises(MCPToolValidationError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {
                    "data_source_id": str(source.id),
                    "sql": "SELECT email FROM users",
                    "include_unmasked": True,
                    "mask": False,
                    "raw": True,
                },
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        with pytest.raises(MCPQueryError):
            await client.call_tool(
                POSTGRES_QUERY_TOOL_NAME,
                {"data_source_id": str(source.id), "sql": "SELECT email FROM users"},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_client_denies_query_permission_for_tool_without_own_auth(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    """Central client enforcement: fake QUERY_READ tool with no internal auth."""
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.MEMBER
    )
    source = _source(db_session, workspace, test_user)
    invoked: list[str] = []

    class _FakeQueryArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        data_source_id: uuid.UUID

    class _FakeQueryResult(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ok: bool = True

    class _FakeQueryTool:
        name = "postgres.fake_query"
        permission = MCPToolPermission.QUERY_READ
        description = "Test-only query capability tool"
        input_model = _FakeQueryArgs
        output_model = _FakeQueryResult

        async def invoke(
            self, arguments: BaseModel, context: MCPToolContext
        ) -> _FakeQueryResult:
            del arguments, context
            invoked.append("called")
            return _FakeQueryResult()

    registry, client = build_postgres_mcp(db_session)
    registry.register(_FakeQueryTool(), server_name="postgres")

    async def _run() -> None:
        with pytest.raises(MCPQueryError):
            await client.call_tool(
                "postgres.fake_query",
                {"data_source_id": str(source.id)},
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())
    assert invoked == []


def test_admin_with_query_read_can_invoke_fake_query_tool(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> None:
    _add_member(
        db_session, workspace=workspace, user=test_user, role=WorkspaceRole.ADMIN
    )
    source = _source(db_session, workspace, test_user)

    class _FakeQueryArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        data_source_id: uuid.UUID

    class _FakeQueryResult(BaseModel):
        model_config = ConfigDict(extra="forbid")
        ok: bool = True

    class _FakeQueryTool:
        name = "postgres.fake_query_admin"
        permission = MCPToolPermission.QUERY_READ
        description = "Test-only query capability tool"
        input_model = _FakeQueryArgs
        output_model = _FakeQueryResult

        async def invoke(
            self, arguments: BaseModel, context: MCPToolContext
        ) -> _FakeQueryResult:
            del arguments, context
            return _FakeQueryResult()

    registry, client = build_postgres_mcp(db_session)
    registry.register(_FakeQueryTool(), server_name="postgres")

    async def _run() -> None:
        result = await client.call_tool(
            "postgres.fake_query_admin",
            {"data_source_id": str(source.id)},
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.ok is True

    run_async(_run())
