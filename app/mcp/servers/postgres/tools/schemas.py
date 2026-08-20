"""PostgreSQL MCP tool: list persisted schemas for a data source."""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.mcp.schemas import MCPDataSourceRequest, MCPToolContext
from app.mcp.security import MCPToolPermission, resolve_authorized_workspace_id
from app.mcp.servers.postgres.tools.common import (
    catalog_page,
    parse_arguments,
    run_catalog,
)
from app.schemas.metadata import MetadataSchemaListResponse, MetadataSchemaResponse
from app.services.metadata_catalog import MetadataCatalogService

POSTGRES_LIST_SCHEMAS_TOOL_NAME = "postgres.list_schemas"


class PostgresListSchemasTool:
    name = POSTGRES_LIST_SCHEMAS_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.METADATA_READ
    description = (
        "List persisted schemas for an authorized PostgreSQL data source. "
        "Does not query the customer database or return credentials."
    )
    input_model: type[BaseModel] = MCPDataSourceRequest
    output_model: type[BaseModel] = MetadataSchemaListResponse

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
    ) -> MetadataSchemaListResponse:
        request = parse_arguments(MCPDataSourceRequest, arguments)
        return self.execute(request, context)

    def execute(
        self, request: MCPDataSourceRequest, context: MCPToolContext
    ) -> MetadataSchemaListResponse:
        catalog = self._catalog or MetadataCatalogService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )
        page = run_catalog(
            lambda: catalog.list_schemas(
                request.data_source_id,
                workspace_id=resolved_workspace_id,
                page=request.page,
                page_size=request.page_size,
            )
        )
        return catalog_page(page, MetadataSchemaListResponse, MetadataSchemaResponse)
