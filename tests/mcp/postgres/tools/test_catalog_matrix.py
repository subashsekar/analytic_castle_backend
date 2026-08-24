"""Phase 5.6 — PostgreSQL MCP catalog tool failure and isolation matrices."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.db.models import Organization, User, Workspace
from app.mcp import MCPError, MCPErrorCode, MCPToolContext, build_postgres_mcp
from app.mcp.exceptions import MCPAccessDeniedError, MCPToolValidationError
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
    POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
    PostgresDescribeTableTool,
    PostgresListTablesTool,
)
from app.services.metadata_catalog_exceptions import (
    MetadataCatalogError,
    MetadataNotFoundError,
)
from app.services.metadata_catalog_types import MetadataPage, SchemaRecord
from tests.conftest import run_async
from tests.mcp.conftest import create_organization, create_workspace
from tests.test_ai_metadata import (
    _column,
    _relationship,
    _schema,
    _source,
    _table,
)

NOW = datetime(2026, 1, 2, tzinfo=UTC)


def _ctx(workspace: Workspace, user: User) -> MCPToolContext:
    return MCPToolContext(workspace_id=workspace.id, user_id=user.id)


class _FailingCatalog:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def list_schemas(self, *args: object, **kwargs: object) -> MetadataPage[object]:
        del args, kwargs
        self.calls += 1
        raise self.error

    def list_tables(self, *args: object, **kwargs: object) -> MetadataPage[object]:
        del args, kwargs
        self.calls += 1
        raise self.error

    def get_table(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        raise self.error

    def list_columns(self, *args: object, **kwargs: object) -> MetadataPage[object]:
        del args, kwargs
        self.calls += 1
        raise self.error

    def list_relationships(
        self, *args: object, **kwargs: object
    ) -> MetadataPage[object]:
        del args, kwargs
        self.calls += 1
        raise self.error


def test_list_schemas_empty_result(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)

    async def _run() -> None:
        result = await PostgresListSchemasTool(db_session).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            _ctx(workspace, test_user),
        )
        assert result.items == []
        assert result.total == 0

    run_async(_run())


def test_list_schemas_handles_catalog_failure_safely(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    catalog = _FailingCatalog(
        MetadataCatalogError("password=secret SELECT * FROM customers")
    )

    async def _run() -> None:
        with pytest.raises(MCPError) as exc_info:
            await PostgresListSchemasTool(db_session, catalog=catalog).invoke(
                MCPDataSourceRequest(data_source_id=source.id),
                _ctx(workspace, test_user),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATABASE_ERROR
        assert "password=" not in str(exc_info.value)
        assert "SELECT" not in str(exc_info.value)

    run_async(_run())


def test_list_schemas_system_schema_names_are_returned_if_persisted(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    """Catalog tools read persisted metadata; system schemas are not auto-hidden here."""
    del workspace_member
    source = _source(db_session, workspace, test_user)
    _schema(db_session, source, "public")
    _schema(db_session, source, "pg_catalog")

    async def _run() -> None:
        result = await PostgresListSchemasTool(db_session).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            _ctx(workspace, test_user),
        )
        names = {item.name for item in result.items}
        assert "public" in names
        assert "pg_catalog" in names

    run_async(_run())


def test_list_tables_empty_schema_and_invalid_schema(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    empty = _schema(db_session, source, "empty_schema")

    async def _run() -> None:
        result = await PostgresListTablesTool(db_session).invoke(
            MCPListTablesRequest(data_source_id=source.id, schema_id=empty.id),
            _ctx(workspace, test_user),
        )
        assert result.items == []
        assert result.total == 0
        with pytest.raises(MCPError, match="Schema not found") as exc_info:
            await PostgresListTablesTool(db_session).invoke(
                MCPListTablesRequest(data_source_id=source.id, schema_id=uuid.uuid4()),
                _ctx(workspace, test_user),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_RESOURCE_NOT_FOUND

    run_async(_run())


def test_list_tables_database_failure(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    catalog = _FailingCatalog(MetadataCatalogError("db down"))

    async def _run() -> None:
        with pytest.raises(MCPError) as exc_info:
            await PostgresListTablesTool(db_session, catalog=catalog).invoke(
                MCPListTablesRequest(data_source_id=source.id),
                _ctx(workspace, test_user),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_DATABASE_ERROR

    run_async(_run())


def test_describe_table_missing_unauthorized_and_cross_org(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "orders")

    other_org = create_organization(db_session)
    other_ws = create_workspace(db_session, other_org)
    foreign_source = _source(db_session, other_ws, test_user)
    foreign_schema = _schema(db_session, foreign_source, "public")
    foreign_table = _table(db_session, foreign_schema, "orders")

    _, client = build_postgres_mcp(db_session)
    context = _ctx(workspace, test_user)

    async def _run() -> None:
        ok = await client.call_tool(
            POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
            {"data_source_id": str(source.id), "table_id": str(table.id)},
            context,
        )
        assert ok.name == "orders"

        with pytest.raises(MCPError, match="Table not found"):
            await client.call_tool(
                POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
                {"data_source_id": str(source.id), "table_id": str(uuid.uuid4())},
                context,
            )

        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
                {
                    "data_source_id": str(foreign_source.id),
                    "table_id": str(foreign_table.id),
                },
                context,
            )

        with pytest.raises(MCPToolValidationError):
            await client.call_tool(
                POSTGRES_DESCRIBE_TABLE_TOOL_NAME,
                {"data_source_id": str(source.id), "table_id": "bad"},
                context,
            )

    run_async(_run())


def test_describe_table_catalog_failure(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    catalog = _FailingCatalog(MetadataNotFoundError("Table not found"))

    async def _run() -> None:
        with pytest.raises(MCPError) as exc_info:
            await PostgresDescribeTableTool(db_session, catalog=catalog).invoke(
                MCPDescribeTableRequest(
                    data_source_id=source.id, table_id=uuid.uuid4()
                ),
                _ctx(workspace, test_user),
            )
        assert exc_info.value.code is MCPErrorCode.MCP_RESOURCE_NOT_FOUND

    run_async(_run())


def test_get_columns_missing_and_cross_workspace(
    db_session: Session,
    workspace: Workspace,
    organization: Organization,
    test_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "customers")
    _column(db_session, table, "id", position=1, primary_key=True)

    sibling = create_workspace(db_session, organization, name="Sibling")
    foreign = _source(db_session, sibling, test_user)
    foreign_schema = _schema(db_session, foreign, "public")
    foreign_table = _table(db_session, foreign_schema, "customers")

    async def _run() -> None:
        ok = await PostgresGetColumnsTool(db_session).invoke(
            MCPTableRequest(data_source_id=source.id, table_id=table.id),
            _ctx(workspace, test_user),
        )
        assert [c.name for c in ok.items] == ["id"]

        with pytest.raises(MCPError, match="Table not found"):
            await PostgresGetColumnsTool(db_session).invoke(
                MCPTableRequest(data_source_id=source.id, table_id=uuid.uuid4()),
                _ctx(workspace, test_user),
            )

        with pytest.raises(MCPAccessDeniedError):
            await PostgresGetColumnsTool(db_session).invoke(
                MCPTableRequest(data_source_id=foreign.id, table_id=foreign_table.id),
                _ctx(workspace, test_user),
            )

    run_async(_run())


def test_get_relationships_composite_and_missing_table_scope(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    public = _schema(db_session, source, "public")
    sales = _schema(db_session, source, "sales")
    customers = _table(db_session, public, "customers")
    orders = _table(db_session, sales, "orders")
    cust_id = _column(db_session, customers, "id", position=1, primary_key=True)
    order_cust = _column(db_session, orders, "customer_id", position=1)
    order_region = _column(db_session, orders, "region_id", position=2)
    region = _column(db_session, customers, "region_id", position=2, primary_key=False)
    _relationship(
        db_session,
        source_table=orders,
        source_column=order_cust,
        target_table=customers,
        target_column=cust_id,
        constraint_name="orders_customer_fk",
    )
    _relationship(
        db_session,
        source_table=orders,
        source_column=order_region,
        target_table=customers,
        target_column=region,
        constraint_name="orders_region_fk",
    )

    async def _run() -> None:
        result = await PostgresGetRelationshipsTool(db_session).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            _ctx(workspace, test_user),
        )
        names = {item.constraint_name for item in result.items}
        assert names == {"orders_customer_fk", "orders_region_fk"}
        cross = next(
            item
            for item in result.items
            if item.constraint_name == "orders_customer_fk"
        )
        assert cross.source_schema_name == "sales"
        assert cross.target_schema_name == "public"
        assert cross.source_table_name == "orders"
        assert cross.target_table_name == "customers"

    run_async(_run())


def test_client_list_schemas_unauthorized_user(
    db_session: Session, workspace: Workspace, test_user: User, admin_user: User
) -> None:
    source = _source(db_session, workspace, test_user)
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_LIST_SCHEMAS_TOOL_NAME,
                {"data_source_id": str(source.id)},
                _ctx(workspace, admin_user),
            )

    run_async(_run())


def test_fake_empty_catalog_page_shape(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    page = MetadataPage[SchemaRecord](items=(), page=1, page_size=50, total=0)

    class _Empty:
        def list_schemas(
            self, *args: object, **kwargs: object
        ) -> MetadataPage[SchemaRecord]:
            del args, kwargs
            return page

    async def _run() -> None:
        result = await PostgresListSchemasTool(db_session, catalog=_Empty()).invoke(
            MCPDataSourceRequest(data_source_id=source.id),
            _ctx(workspace, test_user),
        )
        assert result.total == 0
        assert result.items == []
        assert result.page == 1

    run_async(_run())
