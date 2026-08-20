"""Shared MCP server contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.mcp.registry import MCPRegistry


class MCPServer(Protocol):
    name: str

    def register(self, registry: MCPRegistry) -> None: ...
