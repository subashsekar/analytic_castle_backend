"""Named MCP server and tool registry. One entry per name."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from app.mcp.exceptions import MCPToolNotFoundError
from app.mcp.exceptions import MCPServerNotFoundError, MCPToolNameValidationError
from app.mcp.schemas import MCPToolSchema
from app.mcp.security import MCPToolPermission
from app.mcp.tools.base import MCPTool

if TYPE_CHECKING:
    from app.mcp.server.base import MCPServer


class MCPRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, _MCPRegisteredTool] = {}

    def register_server(self, server: MCPServer) -> None:
        if server.name in self._servers:
            raise ValueError(f"MCP server already registered: {server.name}")
        self._servers[server.name] = server

    def get_server(self, name: str) -> MCPServer:
        try:
            return self._servers[name]
        except KeyError as exc:
            raise MCPToolNotFoundError("MCP server was not found") from exc

    def list_servers(self) -> tuple[MCPServer, ...]:
        return tuple(self._servers[name] for name in sorted(self._servers))

    def register(self, tool: MCPTool, *, server_name: str) -> None:
        if tool.name in self._tools:
            raise ValueError(f"MCP tool already registered: {tool.name}")
        self._tools[tool.name] = _MCPRegisteredTool.from_tool(
            tool=tool, server_name=server_name
        )

    def get(self, name: str) -> MCPTool:
        return self.get_record(name).handler

    def get_record(self, name: str) -> "_MCPRegisteredTool":
        self._validate_tool_name(name)
        server = name.split(".", 1)[0]
        if server not in self._servers:
            raise MCPServerNotFoundError()
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MCPToolNotFoundError("MCP tool was not found") from exc

    def list_tools(self) -> tuple[MCPTool, ...]:
        return tuple(self._tools[name].handler for name in sorted(self._tools))

    def list_schemas(self) -> tuple[MCPToolSchema, ...]:
        schemas: list[MCPToolSchema] = []
        for tool in self.list_tools():
            schemas.append(
                MCPToolSchema(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.input_model.model_json_schema(),
                    output_schema=tool.output_model.model_json_schema(),
                )
            )
        return tuple(schemas)

    _TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")

    def _validate_tool_name(self, name: str) -> None:
        if not isinstance(name, str) or not self._TOOL_NAME_RE.match(name):
            raise MCPToolNameValidationError()


@dataclass(frozen=True)
class _MCPRegisteredTool:
    name: str
    server_name: str
    permission: MCPToolPermission
    handler: MCPTool
    description: str
    input_model: type[object]
    output_model: type[object]

    @classmethod
    def from_tool(cls, *, tool: MCPTool, server_name: str) -> "_MCPRegisteredTool":
        permission = getattr(tool, "permission", None)
        if not isinstance(permission, MCPToolPermission):
            raise ValueError(f"MCP tool {tool.name} has no permission policy")
        return cls(
            name=tool.name,
            server_name=server_name,
            permission=permission,
            handler=tool,
            description=tool.description,
            input_model=tool.input_model,
            output_model=tool.output_model,
        )
