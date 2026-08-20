"""PostgreSQL MCP tool: list persisted relationships for a data source."""

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
from app.schemas.metadata import (
    MetadataRelationshipListResponse,
    MetadataRelationshipResponse,
)
from app.services.metadata_catalog import MetadataCatalogService

POSTGRES_GET_RELATIONSHIPS_TOOL_NAME = "postgres.get_relationships"


class PostgresGetRelationshipsTool:
    name = POSTGRES_GET_RELATIONSHIPS_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.METADATA_READ
    description = (
        "List persisted foreign-key relationships for an authorized PostgreSQL "
        "data source, including composite, self-referencing, and cross-schema "
        "keys. Relationships are not inferred from names."
    )
    input_model: type[BaseModel] = MCPDataSourceRequest
    output_model: type[BaseModel] = MetadataRelationshipListResponse

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
    ) -> MetadataRelationshipListResponse:
        request = parse_arguments(MCPDataSourceRequest, arguments)
        return self.execute(request, context)

    def execute(
        self, request: MCPDataSourceRequest, context: MCPToolContext
    ) -> MetadataRelationshipListResponse:
        catalog = self._catalog or MetadataCatalogService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )
        page = run_catalog(
            lambda: catalog.list_relationships(
                request.data_source_id,
                workspace_id=resolved_workspace_id,
                page=request.page,
                page_size=request.page_size,
            )
        )
        return catalog_page(
            page,
            MetadataRelationshipListResponse,
            MetadataRelationshipResponse,
        )
