"""Execute validated SQL exclusively through the PostgreSQL MCP query tool.

Timeouts, row limits, and result-size limits are enforced by the existing MCP
`postgres.query` path and PostgreSQL connector. This module does not open a
second database execution path. Caller-supplied ValidatedSQL is re-checked
against Chapter 7.2 before any MCP invocation.
"""

from __future__ import annotations

import asyncio
import time
from uuid import UUID

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_execution.errors import (
    SQLExecutionCancelledError,
    SQLExecutionError,
    SQLExecutionValidationError,
)
from app.ai.sql_execution.errors_mapping import map_mcp_query_error
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.sql_validation.models import ValidatedSQL
from app.ai.sql_validation.validation import validate_generated_sql
from app.mcp import (
    POSTGRES_QUERY_TOOL_NAME,
    MCPClient,
    MCPError,
    MCPQueryRequest,
    MCPQueryResult,
    MCPToolContext,
)
from app.mcp.config import mcp_settings


def resolve_applied_row_limit(limit: int | None) -> int:
    """Mirror the MCP query tool's server-side row-limit resolution."""
    maximum = mcp_settings.MCP_QUERY_MAX_LIMIT
    if limit is None:
        return min(mcp_settings.MCP_QUERY_DEFAULT_LIMIT, maximum)
    return min(limit, maximum)


def ensure_requested_row_limit(limit: int | None) -> int | None:
    """Reject invalid caller limits before they reach MCP argument validation."""
    if limit is None:
        return None
    if isinstance(limit, bool) or limit < 1:
        raise SQLExecutionValidationError("Query row limit is invalid")
    return limit


def _ensure_validated_sql(
    validated: ValidatedSQL,
    metadata: ResolvedMetadataContext,
) -> ValidatedSQL:
    """Re-run Chapter 7.2 so forged ValidatedSQL cannot skip the allowlist."""
    result = validate_generated_sql(validated.sql, metadata)
    if not result.is_valid or result.validated is None:
        raise SQLExecutionValidationError(
            "SQL failed validation and was not executed",
            violations=result.violations,
        )
    return result.validated


async def execute_validated_sql(
    *,
    client: MCPClient,
    data_source_id: UUID,
    workspace_id: UUID,
    user_id: UUID,
    validated: ValidatedSQL,
    metadata: ResolvedMetadataContext,
    limit: int | None = None,
) -> SQLExecutionResult:
    """Invoke `postgres.query` only after 7.2 re-validation succeeds."""
    requested_limit = ensure_requested_row_limit(limit)
    trusted = _ensure_validated_sql(validated, metadata)
    applied_limit = resolve_applied_row_limit(requested_limit)
    started = time.perf_counter()
    try:
        raw = await client.call_tool(
            POSTGRES_QUERY_TOOL_NAME,
            MCPQueryRequest(
                data_source_id=data_source_id,
                sql=trusted.sql,
                limit=requested_limit,
            ),
            MCPToolContext(workspace_id=workspace_id, user_id=user_id),
        )
    except asyncio.CancelledError as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        raise SQLExecutionCancelledError(
            "SQL execution was cancelled",
            duration_ms=duration_ms,
        ) from exc
    except MCPError as exc:
        mapped = map_mcp_query_error(exc)
        mapped.duration_ms = (time.perf_counter() - started) * 1000
        raise mapped from exc
    except SQLExecutionError:
        raise

    if not isinstance(raw, MCPQueryResult):
        raise SQLExecutionError(
            "Unexpected MCP query response type",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    duration_ms = (time.perf_counter() - started) * 1000
    return SQLExecutionResult(
        status=SQLExecutionStatus.SUCCEEDED,
        columns=list(raw.columns),
        rows=[list(row) for row in raw.rows],
        row_count=raw.row_count,
        truncated=raw.truncated,
        duration_ms=duration_ms,
        applied_row_limit=applied_limit,
        sql_char_count=len(trusted.sql),
        referenced_table_count=len(trusted.referenced_tables),
        referenced_column_count=len(trusted.referenced_columns),
    )
