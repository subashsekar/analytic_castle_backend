"""Named MCP tool registry. One tool per name."""

from __future__ import annotations

from app.mcp.exceptions import MCPToolNotFoundError
from app.mcp.types import MCPTool, MCPToolSchema


class MCPRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"MCP tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> MCPTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise MCPToolNotFoundError("MCP tool was not found") from exc

    def list_tools(self) -> tuple[MCPTool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

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
