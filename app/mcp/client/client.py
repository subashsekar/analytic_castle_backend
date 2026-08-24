"""In-process MCP client. Invokes registered tools only."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.mcp.error_codes import MCPErrorCategory, MCPErrorCode
from app.mcp.errors import (
    attach_request_id,
    log_mcp_failure,
    map_unexpected_to_mcp_error,
)
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPError,
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
        started = time.perf_counter()
        tool_name = name
        server_name: str | None = None
        data_source_id: UUID | None = None
        try:
            record = self._registry.get_record(name)
            tool_name = record.name
            server_name = record.server_name
            tool = record.handler
            try:
                if isinstance(arguments, tool.input_model):
                    payload = arguments
                elif isinstance(arguments, BaseModel):
                    payload = tool.input_model.model_validate(arguments.model_dump())
                else:
                    payload = tool.input_model.model_validate(arguments)
            except ValidationError as exc:
                raise attach_request_id(
                    MCPToolValidationError("MCP tool arguments are invalid")
                ) from exc

            raw_data_source_id = getattr(payload, "data_source_id", None)
            if not isinstance(raw_data_source_id, UUID):
                raise attach_request_id(
                    MCPToolValidationError("MCP tool arguments are invalid")
                )
            data_source_id = raw_data_source_id

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
                    raise attach_request_id(
                        MCPQueryError(
                            str(exc),
                            code=MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND,
                            category=MCPErrorCategory.NOT_FOUND,
                        )
                    ) from exc
                raise attach_request_id(exc) from exc

            enforce_mcp_rate_limit(
                permission=record.permission,
                user_id=context.user_id,
            )
            try:
                return await tool.invoke(payload, context)
            except MCPError as exc:
                raise attach_request_id(exc) from exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise map_unexpected_to_mcp_error(exc) from exc
        except MCPError as exc:
            attach_request_id(exc)
            log_mcp_failure(
                exc=exc,
                tool_name=tool_name,
                server_name=server_name,
                data_source_id=data_source_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        except asyncio.CancelledError:
            raise
