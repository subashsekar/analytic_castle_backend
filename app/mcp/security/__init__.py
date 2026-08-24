"""Phase 5.4 — MCP security primitives.

This module centralizes:
- Tool permission model (small + typed)
- Workspace/data-source isolation authorization
- MCP rate limiting (reuses in-process limiter)
- Safe, non-enumerating errors
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.authorization import is_super_admin, workspace_role_has_permission
from app.core.config import parse_rate_limit, settings
from app.core.rate_limit import rate_limiter
from app.db.models import DataSource, User, WorkspaceMember
from app.enums import WorkspacePermission
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPRateLimitError,
    MCPUnauthorizedError,
)
from app.mcp.schemas import MCPToolContext


class MCPToolPermission(str, Enum):
    """Small permission model for MCP tools."""

    METADATA_READ = "METADATA_READ"
    SAMPLE_DATA_READ = "SAMPLE_DATA_READ"
    QUERY_READ = "QUERY_READ"


def _workspace_permission_for_mcp(permission: MCPToolPermission) -> WorkspacePermission:
    """Map MCP tool capabilities onto application workspace permissions.

    METADATA_READ / SAMPLE_DATA_READ reuse DATA_SOURCE_READ (masked/catalog).
    QUERY_READ requires the stricter DATA_SOURCE_QUERY capability so ordinary
    DATA_SOURCE_READ members cannot bypass sample masking via SQL.
    """

    if permission is MCPToolPermission.QUERY_READ:
        return WorkspacePermission.DATA_SOURCE_QUERY
    return WorkspacePermission.DATA_SOURCE_READ


def resolve_authorized_workspace_id(
    *,
    session: Session,
    context: MCPToolContext,
    data_source_id: UUID,
    permission: MCPToolPermission,
) -> UUID:
    """Resolve the data source's workspace after verifying user authorization.

    Security boundary:
    - Tool arguments may include `data_source_id`, but authorization is derived
      from the authoritative `DataSource.workspace_id` and workspace membership.
    - `context.workspace_id` is treated as non-authoritative and not used for
      isolation.
    """

    if context.user_id is None:
        raise MCPUnauthorizedError()

    user = session.get(User, context.user_id)
    if user is None or not user.is_active:
        raise MCPUnauthorizedError()

    data_source = session.get(DataSource, data_source_id)
    if data_source is None:
        # Avoid enumerating whether the data source exists.
        raise MCPAccessDeniedError()

    if is_super_admin(user.role):
        return data_source.workspace_id

    member_role = session.scalar(
        select(WorkspaceMember.role).where(
            WorkspaceMember.workspace_id == data_source.workspace_id,
            WorkspaceMember.user_id == user.id,
        )
    )
    if member_role is None:
        raise MCPAccessDeniedError()

    required_permission = _workspace_permission_for_mcp(permission)
    if not workspace_role_has_permission(member_role, required_permission):
        raise MCPAccessDeniedError()

    return data_source.workspace_id


def _rate_limit_scope_for_permission(permission: MCPToolPermission) -> str:
    if permission is MCPToolPermission.QUERY_READ:
        return "mcp-query"
    if permission is MCPToolPermission.SAMPLE_DATA_READ:
        return "mcp-sample"
    return "mcp-metadata"


def enforce_mcp_rate_limit(
    *,
    permission: MCPToolPermission,
    user_id: UUID | None,
) -> None:
    """Apply in-process rate limits to high-risk MCP tool families.

    Reuses the existing ``InMemoryRateLimiter``. Does not introduce Redis.
    """

    if not settings.RATE_LIMIT_ENABLED:
        return
    if user_id is None:
        # Authentication is enforced separately; do not rate-limit anonymously.
        return

    scope = _rate_limit_scope_for_permission(permission)
    limit, window_seconds = parse_rate_limit(settings.rate_limit_spec(scope))
    result = rate_limiter.hit(f"mcp:{scope}:{user_id}", limit, window_seconds)
    if not result.allowed:
        raise MCPRateLimitError(retry_after=result.retry_after)


__all__ = [
    "MCPToolPermission",
    "enforce_mcp_rate_limit",
    "resolve_authorized_workspace_id",
]
