"""In-process MCP types. These are not HTTP schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class MCPToolContext:
    workspace_id: UUID
    user_id: UUID | None = None


class MCPTool(Protocol):
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> BaseModel: ...


class MCPToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]


class MCPQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID
    sql: str = Field(min_length=1, max_length=100_000)
    limit: int | None = Field(default=None, ge=1, le=10_000)


class MCPQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool = False
