from sqlalchemy.orm import Session

from app.mcp.client import MCPClient
from app.mcp.exceptions import (
    MCPError,
    MCPQueryError,
    MCPQueryRejectedError,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPToolNotFoundError,
    MCPToolValidationError,
)
from app.mcp.registry import MCPRegistry
from app.mcp.servers.postgres.server import PostgreSQLMCPServer
from app.mcp.servers.postgres.tools.query import (
    POSTGRES_QUERY_TOOL_NAME,
    PostgresQueryTool,
)
from app.mcp.types import MCPQueryRequest, MCPQueryResult, MCPToolContext, MCPToolSchema


def build_postgres_mcp(session: Session) -> tuple[MCPRegistry, MCPClient]:
    registry = MCPRegistry()
    PostgreSQLMCPServer(session).register(registry)
    return registry, MCPClient(registry)


__all__ = [
    "MCPClient",
    "MCPError",
    "MCPQueryError",
    "MCPQueryRejectedError",
    "MCPQueryRequest",
    "MCPQueryResult",
    "MCPQueryResultError",
    "MCPQueryTimeoutError",
    "MCPRegistry",
    "MCPToolContext",
    "MCPToolNotFoundError",
    "MCPToolSchema",
    "MCPToolValidationError",
    "POSTGRES_QUERY_TOOL_NAME",
    "PostgreSQLMCPServer",
    "PostgresQueryTool",
    "build_postgres_mcp",
]
