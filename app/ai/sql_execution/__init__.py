"""SQL query execution for validated analytical SQL.

Runs Chapter 7.2 validation, then executes exclusively through the existing
PostgreSQL MCP `postgres.query` tool. Does not correct SQL (Chapter 7.4) or
record query history (Chapter 7.5).
"""

from app.ai.sql_execution.errors import (
    SQLExecutionAuthorizationError,
    SQLExecutionCancelledError,
    SQLExecutionConfigurationError,
    SQLExecutionDatabaseError,
    SQLExecutionError,
    SQLExecutionErrorCode,
    SQLExecutionRejectedError,
    SQLExecutionResultLimitError,
    SQLExecutionTimeoutError,
    SQLExecutionValidationError,
)
from app.ai.sql_execution.errors_mapping import (
    map_mcp_query_error,
    map_sql_execution_error,
)
from app.ai.sql_execution.execution import (
    ensure_requested_row_limit,
    execute_validated_sql,
    resolve_applied_row_limit,
)
from app.ai.sql_execution.logging_helpers import execution_log_context
from app.ai.sql_execution.models import (
    SQLExecuteParams,
    SQLExecutionResult,
    SQLExecutionServiceResult,
    SQLExecutionStatus,
)
from app.ai.sql_execution.serialization import (
    serialize_execution_error,
    serialize_execution_result,
    serialize_execution_service_result,
)
from app.ai.sql_execution.service import SQLExecutionService

__all__ = [
    "SQLExecuteParams",
    "SQLExecutionAuthorizationError",
    "SQLExecutionCancelledError",
    "SQLExecutionConfigurationError",
    "SQLExecutionDatabaseError",
    "SQLExecutionError",
    "SQLExecutionErrorCode",
    "SQLExecutionRejectedError",
    "SQLExecutionResult",
    "SQLExecutionResultLimitError",
    "SQLExecutionService",
    "SQLExecutionServiceResult",
    "SQLExecutionStatus",
    "SQLExecutionTimeoutError",
    "SQLExecutionValidationError",
    "ensure_requested_row_limit",
    "execute_validated_sql",
    "execution_log_context",
    "map_mcp_query_error",
    "map_sql_execution_error",
    "resolve_applied_row_limit",
    "serialize_execution_error",
    "serialize_execution_result",
    "serialize_execution_service_result",
]
