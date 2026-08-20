"""In-process MCP lifecycle. Servers are created per request/session."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry
from app.mcp.servers.postgres.server import PostgreSQLMCPServer


def build_postgres_mcp(session: Session) -> tuple[MCPRegistry, MCPClient]:
    registry = MCPRegistry()
    PostgreSQLMCPServer(session).register(registry)
    return registry, MCPClient(registry, session)


def shutdown_mcp() -> None:
    """In-process MCP holds no long-lived connections or pools."""
    return
