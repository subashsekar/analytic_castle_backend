from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.db.models import Organization, User, Workspace
from app.enums import DataSourceTableType
from app.mcp import (
    POSTGRES_TOOL_NAMES,
    MCPToolContext,
    build_postgres_mcp,
)
from app.mcp.schemas import (
    MCPDataSourceRequest,
    MCPDescribeTableRequest,
    MCPListTablesRequest,
    MCPTableRequest,
)
from app.mcp.servers.postgres.tools.columns import PostgresGetColumnsTool
from app.mcp.servers.postgres.tools.relationships import PostgresGetRelationshipsTool
from app.mcp.servers.postgres.tools.schemas import (
    POSTGRES_LIST_SCHEMAS_TOOL_NAME,
    PostgresListSchemasTool,
)
from app.mcp.servers.postgres.tools.tables import (
    PostgresDescribeTableTool,
    PostgresListTablesTool,
)
from app.services.metadata_catalog_types import (
    ColumnRecord,
    MetadataPage,
    RelationshipRecord,
    SchemaRecord,
    TableRecord,
)
from tests.conftest import run_async
from tests.test_ai_metadata import (
    _column,
    _relationship,
    _schema,
    _source,
    _table,
)

NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _context(workspace: Workspace, test_user: User) -> MCPToolContext:
    return MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)


def test_postgres_server_registers_all_tools(db_session: Session) -> None:
    registry, client = build_postgres_mcp(db_session)
    names = [schema.name for schema in registry.list_schemas()]
    assert names == sorted(POSTGRES_TOOL_NAMES)
    for name in POSTGRES_TOOL_NAMES:
        tool = registry.get(name)
        assert tool.name == name
        assert "password" not in str(tool.input_model.model_json_schema())
        assert "connection_string" not in str(tool.input_model.model_json_schema())
        assert client._registry.get(name) is tool


def test_list_schemas_calls_catalog_and_returns_safe_result(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    source_id = source.id
    page = MetadataPage(
        items=(
            SchemaRecord(
                id=uuid.uuid4(),
                data_source_id=source_id,
                name="public",
                table_count=2,
                created_at=NOW,
                updated_at=NOW,
            ),
        ),
        page=1,
        page_size=50,
        total=1,
    )
    catalog = _FakeCatalog(page)

    async def _run() -> None:
        result = await PostgresListSchemasTool(db_session, catalog=catalog).invoke(
            MCPDataSourceRequest(data_source_id=source_id),
            _context(workspace, test_user),
        )
        assert result.items[0].name == "public"
        assert result.total == 1
        assert catalog.calls == [
            ("list_schemas", source_id, workspace.id, 1, None),
        ]

    run_async(_run())


def test_list_schemas_rejects_foreign_workspace(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    source = _source(db_session, workspace, test_user)
    _schema(db_session, source, "public")
    other = Workspace(
        organization_id=organization.id,
        name="Other",
        slug=f"other-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(other)
    db_session.flush()

    async def _run() -> None:
        result = await PostgresListSchemasTool(db_session).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            MCPToolContext(workspace_id=other.id, user_id=test_user.id),
        )
        assert result.items[0].name == "public"

    run_async(_run())


def test_list_tables_passes_filters(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    source_id = source.id
    schema_id = uuid.uuid4()
    page = MetadataPage(
        items=(
            TableRecord(
                id=uuid.uuid4(),
                data_source_id=source_id,
                schema_id=schema_id,
                schema_name="public",
                name="customers",
                table_type=DataSourceTableType.TABLE,
                description="Customers",
                column_count=3,
                created_at=NOW,
                updated_at=NOW,
            ),
        ),
        page=1,
        page_size=50,
        total=1,
    )
    catalog = _FakeCatalog(page)

    async def _run() -> None:
        result = await PostgresListTablesTool(db_session, catalog=catalog).invoke(
            MCPListTablesRequest(
                data_source_id=source_id,
                schema_id=schema_id,
                search="cust",
                table_type=DataSourceTableType.TABLE,
            ),
            _context(workspace, test_user),
        )
        assert result.items[0].name == "customers"
        assert catalog.calls[0][0] == "list_tables"
        assert catalog.calls[0][3] == schema_id
        assert catalog.calls[0][4] == "cust"
        assert catalog.calls[0][5] is DataSourceTableType.TABLE

    run_async(_run())


def test_describe_table_reuses_catalog(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "sales")
    table = _table(db_session, schema, "orders", description="Order headers")

    async def _run() -> None:
        result = await PostgresDescribeTableTool(db_session).invoke(
            MCPDescribeTableRequest(data_source_id=source.id, table_id=table.id),
            _context(workspace, test_user),
        )
        assert result.id == table.id
        assert result.schema_name == "sales"
        assert result.name == "orders"
        assert result.table_type is DataSourceTableType.TABLE
        assert result.description == "Order headers"
        dumped = result.model_dump()
        assert "password" not in dumped
        assert "encrypted" not in str(dumped)

    run_async(_run())


def test_get_columns_returns_ordinal_order(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "customers")
    _column(db_session, table, "email", position=2, data_type="string")
    _column(db_session, table, "id", position=1, data_type="integer", primary_key=True)

    async def _run() -> None:
        result = await PostgresGetColumnsTool(db_session).invoke(
            MCPTableRequest(data_source_id=source.id, table_id=table.id),
            _context(workspace, test_user),
        )
        assert [item.name for item in result.items] == ["id", "email"]
        assert result.items[0].ordinal_position == 1
        assert result.items[0].is_primary_key is True
        assert result.items[1].is_nullable is True

    run_async(_run())


def test_get_relationships_returns_persisted_keys(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    public = _schema(db_session, source, "public")
    other = _schema(db_session, source, "hr")
    customers = _table(db_session, public, "customers")
    employees = _table(db_session, other, "employees")
    customer_id = _column(db_session, customers, "id", position=1, primary_key=True)
    employee_id = _column(db_session, employees, "id", position=1, primary_key=True)
    manager_id = _column(db_session, employees, "manager_id", position=2)
    customer_owner = _column(db_session, customers, "owner_id", position=2)
    _relationship(
        db_session,
        source_table=employees,
        source_column=manager_id,
        target_table=employees,
        target_column=employee_id,
        constraint_name="employees_manager_fk",
    )
    _relationship(
        db_session,
        source_table=customers,
        source_column=customer_owner,
        target_table=employees,
        target_column=employee_id,
        constraint_name="customers_owner_fk",
    )
    del customer_id

    async def _run() -> None:
        result = await PostgresGetRelationshipsTool(db_session).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            _context(workspace, test_user),
        )
        names = {item.constraint_name for item in result.items}
        assert names == {"employees_manager_fk", "customers_owner_fk"}
        self_ref = next(
            item
            for item in result.items
            if item.constraint_name == "employees_manager_fk"
        )
        assert self_ref.source_table_name == "employees"
        assert self_ref.target_table_name == "employees"
        cross = next(
            item
            for item in result.items
            if item.constraint_name == "customers_owner_fk"
        )
        assert cross.source_schema_name == "public"
        assert cross.target_schema_name == "hr"

    run_async(_run())


def test_catalog_requests_reject_credentials() -> None:
    with pytest.raises(ValidationError):
        MCPDataSourceRequest.model_validate(
            {
                "data_source_id": str(uuid.uuid4()),
                "password": "secret",
            }
        )
    with pytest.raises(ValidationError):
        MCPDescribeTableRequest.model_validate(
            {
                "data_source_id": str(uuid.uuid4()),
                "table_id": str(uuid.uuid4()),
                "connection_string": "postgresql://u:p@h/db",
            }
        )


def test_client_invokes_list_schemas(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    source = _source(db_session, workspace, test_user)
    _schema(db_session, source, "public")
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        payload = await client.call_tool(
            POSTGRES_LIST_SCHEMAS_TOOL_NAME,
            {"data_source_id": str(source.id)},
            _context(workspace, test_user),
        )
        assert payload.items[0].name == "public"

    run_async(_run())


class _FakeCatalog:
    def __init__(self, page: MetadataPage[object]) -> None:
        self.page = page
        self.calls: list[tuple[object, ...]] = []

    def list_schemas(
        self,
        data_source_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[object]:
        self.calls.append(
            ("list_schemas", data_source_id, workspace_id, page, page_size)
        )
        return self.page

    def list_tables(
        self,
        data_source_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        schema_id: uuid.UUID | None = None,
        search: str | None = None,
        table_type: DataSourceTableType | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[object]:
        self.calls.append(
            (
                "list_tables",
                data_source_id,
                workspace_id,
                schema_id,
                search,
                table_type,
                page,
                page_size,
            )
        )
        return self.page

    def get_table(
        self,
        data_source_id: uuid.UUID,
        table_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
    ) -> TableRecord:
        raise AssertionError("not used")

    def list_columns(
        self,
        data_source_id: uuid.UUID,
        table_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[ColumnRecord]:
        raise AssertionError("not used")

    def list_relationships(
        self,
        data_source_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        page: int = 1,
        page_size: int | None = None,
    ) -> MetadataPage[RelationshipRecord]:
        raise AssertionError("not used")
