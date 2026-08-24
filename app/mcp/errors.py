"""Central MCP error mapping and safe client responses.

Converts exceptions into stable codes/categories without leaking SQL,
credentials, stack traces, or internal exception chains.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.core.request_id import MISSING_REQUEST_ID, get_request_id, new_request_id
from app.mcp.error_codes import MCPErrorCategory, MCPErrorCode
from app.mcp.exceptions import (
    MCPError,
    MCPInternalError,
)

logger = logging.getLogger(__name__)


class MCPErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: MCPErrorCode
    category: MCPErrorCategory
    message: str
    request_id: str
    retry_after: int | None = Field(default=None, ge=1)


class MCPErrorResponse(BaseModel):
    """Normalized MCP error envelope for clients."""

    model_config = ConfigDict(extra="forbid")

    error: MCPErrorBody


def resolve_mcp_request_id(request_id: str | None = None) -> str:
    if request_id:
        return request_id
    contextual = get_request_id()
    if contextual and contextual != MISSING_REQUEST_ID:
        return contextual
    return new_request_id()


def attach_request_id(exc: MCPError, request_id: str | None = None) -> MCPError:
    """Ensure an MCP error carries a correlation id (mutates in place)."""

    if not exc.request_id:
        exc.request_id = resolve_mcp_request_id(request_id)
    return exc


def to_error_response(
    exc: MCPError,
    *,
    request_id: str | None = None,
) -> MCPErrorResponse:
    """Serialize an MCP error into a safe client payload.

    Never includes traceback, ``__cause__``, SQL, or credentials.
    """

    rid = resolve_mcp_request_id(request_id or exc.request_id)
    body = MCPErrorBody(
        code=_error_code(exc),
        category=_error_category(exc),
        message=str(exc),
        request_id=rid,
        retry_after=exc.retry_after
        if exc.retry_after and exc.retry_after > 0
        else None,
    )
    return MCPErrorResponse(error=body)


def http_status_for(exc: MCPError) -> int:
    return int(getattr(exc, "http_status", 500))


def map_unexpected_to_mcp_error(
    exc: BaseException,
    *,
    request_id: str | None = None,
) -> MCPError:
    """Map an unexpected failure to ``MCPInternalError``.

    Preserves existing ``MCPError`` instances. Callers must not pass
    ``CancelledError`` here — cancellation must propagate.
    """

    if isinstance(exc, MCPError):
        return attach_request_id(exc, request_id)
    return attach_request_id(
        MCPInternalError("An unexpected MCP error occurred"),
        request_id,
    )


def log_mcp_failure(
    *,
    exc: BaseException,
    tool_name: str | None = None,
    server_name: str | None = None,
    data_source_id: UUID | str | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log MCP failures with safe structured fields only.

    Does not log SQL, parameters, credentials, JWT, PII, or sample rows.
    """

    request_id = MISSING_REQUEST_ID
    code: str | None = None
    category: str | None = None
    if isinstance(exc, MCPError):
        request_id = resolve_mcp_request_id(exc.request_id)
        code = str(_error_code(exc).value)
        category = str(_error_category(exc).value)
    else:
        request_id = resolve_mcp_request_id()

    logger.warning(
        "MCP failure request_id=%s error_type=%s code=%s category=%s "
        "tool_name=%s server_name=%s data_source_id=%s duration_ms=%s",
        request_id,
        type(exc).__name__,
        code or "-",
        category or "-",
        tool_name or "-",
        server_name or "-",
        data_source_id if data_source_id is not None else "-",
        f"{duration_ms:.0f}" if duration_ms is not None else "-",
    )


def error_response_dict(
    exc: MCPError, *, request_id: str | None = None
) -> dict[str, Any]:
    """Dict form of ``to_error_response`` for FastAPI / JSON clients."""

    payload = to_error_response(exc, request_id=request_id)
    return payload.model_dump(mode="json", exclude_none=True)


def _error_code(exc: MCPError) -> MCPErrorCode:
    code = getattr(exc, "code", MCPErrorCode.MCP_INTERNAL_ERROR)
    if isinstance(code, MCPErrorCode):
        return code
    try:
        return MCPErrorCode(str(code))
    except ValueError:
        return MCPErrorCode.MCP_INTERNAL_ERROR


def _error_category(exc: MCPError) -> MCPErrorCategory:
    category = getattr(exc, "category", MCPErrorCategory.INTERNAL_ERROR)
    if isinstance(category, MCPErrorCategory):
        return category
    try:
        return MCPErrorCategory(str(category))
    except ValueError:
        return MCPErrorCategory.INTERNAL_ERROR
