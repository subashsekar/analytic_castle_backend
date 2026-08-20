"""In-process MCP client. Invokes registered tools only."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPQueryError,
    MCPToolValidationError,
)
from app.mcp.registry import MCPRegistry
from app.mcp.schemas import MCPToolContext
from app.mcp.security import (
    MCPToolPermission,
    enforce_mcp_rate_limit,
    resolve_authorized_workspace_id,
)


class MCPClient:
    """Registry-backed MCP invocation boundary.

    Authorization and rate limiting run here so a future tool cannot execute
    merely because it is registered.
    """

    def __init__(self, registry: MCPRegistry, session: Session) -> None:
        self._registry = registry
        self._session = session

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | BaseModel,
        context: MCPToolContext,
    ) -> BaseModel:
        record = self._registry.get_record(name)
        tool = record.handler
        try:
            if isinstance(arguments, tool.input_model):
                payload = arguments
            elif isinstance(arguments, BaseModel):
                payload = tool.input_model.model_validate(arguments.model_dump())
            else:
                payload = tool.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise MCPToolValidationError("MCP tool arguments are invalid") from exc

        data_source_id = getattr(payload, "data_source_id", None)
        if not isinstance(data_source_id, UUID):
            raise MCPToolValidationError("MCP tool arguments are invalid")

        try:
            resolve_authorized_workspace_id(
                session=self._session,
                context=context,
                data_source_id=data_source_id,
                permission=record.permission,
            )
        except MCPAccessDeniedError as exc:
            # Preserve the historical query-tool denial surface for callers.
            if record.permission is MCPToolPermission.QUERY_READ:
                raise MCPQueryError(str(exc)) from exc
            raise

        enforce_mcp_rate_limit(
            permission=record.permission,
            user_id=context.user_id,
        )
        return await tool.invoke(payload, context)
