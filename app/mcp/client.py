"""In-process MCP client. Invokes registered tools only."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from app.mcp.exceptions import MCPToolValidationError
from app.mcp.registry import MCPRegistry
from app.mcp.types import MCPToolContext


class MCPClient:
    def __init__(self, registry: MCPRegistry) -> None:
        self._registry = registry

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | BaseModel,
        context: MCPToolContext,
    ) -> BaseModel:
        tool = self._registry.get(name)
        try:
            if isinstance(arguments, tool.input_model):
                payload = arguments
            elif isinstance(arguments, BaseModel):
                payload = tool.input_model.model_validate(arguments.model_dump())
            else:
                payload = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise MCPToolValidationError("MCP tool arguments are invalid") from exc
        return await tool.invoke(payload, context)
