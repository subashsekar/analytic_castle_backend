"""Map SQL execution and MCP errors to AI analyst exceptions for API handling."""

from __future__ import annotations

from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIProviderError,
    AIProviderTimeoutError,
    AIRequestValidationError,
)
from app.ai.sql_execution.errors import (
    SQLExecutionAuthorizationError,
    SQLExecutionCancelledError,
    SQLExecutionDatabaseError,
    SQLExecutionError,
    SQLExecutionRejectedError,
    SQLExecutionResultLimitError,
    SQLExecutionTimeoutError,
    SQLExecutionValidationError,
)
from app.mcp.exceptions import (
    MCPDatabaseUnavailableError,
    MCPError,
    MCPQueryRejectedError,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPRateLimitError,
    MCPToolValidationError,
)


def map_mcp_query_error(exc: MCPError) -> SQLExecutionError:
    """Translate MCP query-boundary errors into SQL execution errors."""
    if isinstance(exc, MCPQueryTimeoutError):
        return SQLExecutionTimeoutError(str(exc) or "The query timed out")
    if isinstance(exc, MCPQueryResultError):
        return SQLExecutionResultLimitError(str(exc) or "The query result is too large")
    if isinstance(exc, (MCPQueryRejectedError, MCPToolValidationError)):
        return SQLExecutionRejectedError(
            str(exc) or "The query was rejected by the execution boundary"
        )
    if isinstance(exc, MCPDatabaseUnavailableError):
        return SQLExecutionDatabaseError(str(exc) or "The database is unavailable")
    if isinstance(exc, MCPRateLimitError):
        return SQLExecutionRejectedError(str(exc) or "Query rate limit exceeded")
    message = str(exc) or "The database query failed"
    # Auth denials are intentionally surfaced as not-found by the MCP query path.
    if "not found" in message.lower() or "not authorized" in message.lower():
        return SQLExecutionAuthorizationError("Data source is not accessible")
    return SQLExecutionDatabaseError(message)


def map_sql_execution_error(exc: SQLExecutionError) -> AIError:
    if isinstance(exc, SQLExecutionAuthorizationError):
        return AIContextError(str(exc))
    if isinstance(exc, SQLExecutionValidationError):
        return AIRequestValidationError(str(exc))
    if isinstance(exc, SQLExecutionTimeoutError):
        return AIProviderTimeoutError(str(exc))
    if isinstance(exc, (SQLExecutionResultLimitError, SQLExecutionRejectedError)):
        return AIRequestValidationError(str(exc))
    if isinstance(exc, SQLExecutionCancelledError):
        return AIProviderError(str(exc))
    if isinstance(exc, SQLExecutionDatabaseError):
        return AIProviderError(str(exc))
    return AIProviderError(str(exc))
