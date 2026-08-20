"""PostgreSQL MCP tool: list persisted columns for a table."""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.mcp.schemas import MCPTableRequest, MCPToolContext
from app.mcp.security import MCPToolPermission, resolve_authorized_workspace_id
from app.mcp.servers.postgres.tools.common import (
    catalog_page,
    parse_arguments,
    run_catalog,
)
from app.schemas.metadata import MetadataColumnListResponse, MetadataColumnResponse
from app.services.metadata_catalog import MetadataCatalogService

POSTGRES_GET_COLUMNS_TOOL_NAME = "postgres.get_columns"


class PostgresGetColumnsTool:
    name = POSTGRES_GET_COLUMNS_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.METADATA_READ
    description = (
        "List persisted columns for one table in an authorized PostgreSQL "
        "data source, ordered by ordinal position. Does not return credentials."
    )
    input_model: type[BaseModel] = MCPTableRequest
    output_model: type[BaseModel] = MetadataColumnListResponse

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
    ) -> MetadataColumnListResponse:
        request = parse_arguments(MCPTableRequest, arguments)
        return self.execute(request, context)

    def execute(
        self, request: MCPTableRequest, context: MCPToolContext
    ) -> MetadataColumnListResponse:
        catalog = self._catalog or MetadataCatalogService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )
        page = run_catalog(
            lambda: catalog.list_columns(
                request.data_source_id,
                request.table_id,
                workspace_id=resolved_workspace_id,
                page=request.page,
                page_size=request.page_size,
            )
        )
        return catalog_page(page, MetadataColumnListResponse, MetadataColumnResponse)
