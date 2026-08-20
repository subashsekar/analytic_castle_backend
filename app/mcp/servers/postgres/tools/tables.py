"""PostgreSQL MCP tool: list persisted tables for a data source."""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.mcp.schemas import (
    MCPListTablesRequest,
    MCPDescribeTableRequest,
    MCPToolContext,
)
from app.mcp.security import MCPToolPermission, resolve_authorized_workspace_id
from app.mcp.servers.postgres.tools.common import (
    catalog_page,
    parse_arguments,
    run_catalog,
)
from app.schemas.metadata import (
    MetadataTableListResponse,
    MetadataTableResponse,
)
from app.services.metadata_catalog import MetadataCatalogService

POSTGRES_LIST_TABLES_TOOL_NAME = "postgres.list_tables"


class PostgresListTablesTool:
    name = POSTGRES_LIST_TABLES_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.METADATA_READ
    description = (
        "List persisted tables for an authorized PostgreSQL data source. "
        "Supports schema, type, and name filters. Does not return credentials."
    )
    input_model: type[BaseModel] = MCPListTablesRequest
    output_model: type[BaseModel] = MetadataTableListResponse

    def __init__(
        self,
        session: Session,
        *,
        catalog: MetadataCatalogService | None = None,
    ) -> None:
        self._session = session
        self._catalog = catalog

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> MetadataTableListResponse:
        request = parse_arguments(MCPListTablesRequest, arguments)
        return self.execute(request, context)

    def execute(
        self, request: MCPListTablesRequest, context: MCPToolContext
    ) -> MetadataTableListResponse:
        catalog = self._catalog or MetadataCatalogService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )
        page = run_catalog(
            lambda: catalog.list_tables(
                request.data_source_id,
                workspace_id=resolved_workspace_id,
                schema_id=request.schema_id,
                search=request.search,
                table_type=request.table_type,
                page=request.page,
                page_size=request.page_size,
            )
        )
        return catalog_page(page, MetadataTableListResponse, MetadataTableResponse)


POSTGRES_DESCRIBE_TABLE_TOOL_NAME = "postgres.describe_table"


class PostgresDescribeTableTool:
    name = POSTGRES_DESCRIBE_TABLE_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.METADATA_READ
    description = (
        "Return persisted metadata for one table in an authorized PostgreSQL "
        "data source, including schema, type, and description."
    )
    input_model: type[BaseModel] = MCPDescribeTableRequest
    output_model: type[BaseModel] = MetadataTableResponse

    def __init__(
        self,
        session: Session,
        *,
        catalog: MetadataCatalogService | None = None,
    ) -> None:
        self._session = session
        self._catalog = catalog

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> MetadataTableResponse:
        request = parse_arguments(MCPDescribeTableRequest, arguments)
        return self.execute(request, context)

    def execute(
        self, request: MCPDescribeTableRequest, context: MCPToolContext
    ) -> MetadataTableResponse:
        catalog = self._catalog or MetadataCatalogService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )
        record = run_catalog(
            lambda: catalog.get_table(
                request.data_source_id,
                request.table_id,
                workspace_id=resolved_workspace_id,
            )
        )
        return MetadataTableResponse.model_validate(record)
