"""PostgreSQL MCP server. Registers catalog, sample, and query tools."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.mcp.registry import MCPRegistry
from app.mcp.servers.postgres.tools.columns import PostgresGetColumnsTool
from app.mcp.servers.postgres.tools.query import PostgresQueryTool
from app.mcp.servers.postgres.tools.relationships import PostgresGetRelationshipsTool
from app.mcp.servers.postgres.tools.sample_rows import PostgresSampleRowsTool
from app.mcp.servers.postgres.tools.schemas import PostgresListSchemasTool
from app.mcp.servers.postgres.tools.tables import (
    PostgresDescribeTableTool,
    PostgresListTablesTool,
)


class PostgreSQLMCPServer:
    name = "postgres"

    def __init__(self, session: Session) -> None:
        self._tools = (
            PostgresListSchemasTool(session),
            PostgresListTablesTool(session),
            PostgresDescribeTableTool(session),
            PostgresGetColumnsTool(session),
            PostgresGetRelationshipsTool(session),
            PostgresSampleRowsTool(session),
            PostgresQueryTool(session),
        )

    def register(self, registry: MCPRegistry) -> None:
        registry.register_server(self)
        for tool in self._tools:
            registry.register(tool, server_name=self.name)
