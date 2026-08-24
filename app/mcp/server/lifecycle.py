"""In-process MCP lifecycle. Servers are created per request/session."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.mcp.client import MCPClient
from app.mcp.exceptions import MCPConfigurationError, MCPServerError
from app.mcp.registry import MCPRegistry
from app.mcp.servers.postgres.server import PostgreSQLMCPServer

logger = logging.getLogger(__name__)


def build_postgres_mcp(session: Session) -> tuple[MCPRegistry, MCPClient]:
    """Build a per-session PostgreSQL MCP registry and client.

    Startup failures are converted into safe MCP configuration/server errors.
    A failed build does not leave registered servers in a shared global registry
    because the registry is local to this call.
    """

    registry = MCPRegistry()
    try:
        PostgreSQLMCPServer(session).register(registry)
    except MCPConfigurationError:
        raise
    except MCPServerError:
        raise
    except Exception as exc:
        logger.warning(
            "MCP postgres server startup failed error_type=%s",
            type(exc).__name__,
        )
        raise MCPServerError("MCP server failed to start") from exc
    return registry, MCPClient(registry, session)


def shutdown_mcp() -> None:
    """In-process MCP holds no long-lived connections or pools."""

    return
