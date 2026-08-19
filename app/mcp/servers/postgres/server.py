"""PostgreSQL MCP server. Registers the read-only query tool."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.mcp.registry import MCPRegistry
from app.mcp.servers.postgres.tools.query import PostgresQueryTool


class PostgreSQLMCPServer:
    def __init__(self, session: Session) -> None:
        self._tool = PostgresQueryTool(session)

    def register(self, registry: MCPRegistry) -> None:
        registry.register(self._tool)
