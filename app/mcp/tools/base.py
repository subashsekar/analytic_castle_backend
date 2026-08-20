"""Shared MCP tool contract."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from app.mcp.schemas import MCPToolContext


class MCPTool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> BaseModel: ...
