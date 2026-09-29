"""Safe serialization for SQL execution results and errors.

Cell values are already sanitized by the MCP query tool. This module only
exposes status/metadata and never includes credentials, connection details,
or raw SQL text.
"""

from __future__ import annotations

from typing import Any

from app.ai.sql_execution.errors import SQLExecutionError, SQLExecutionValidationError
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionServiceResult
from app.core.logging import redact_secret


def serialize_execution_result(result: SQLExecutionResult) -> dict[str, Any]:
    """Return a JSON-safe payload for a successful or terminal execution result."""
    return {
        "status": result.status.value,
        "columns": list(result.columns),
        "rows": [list(row) for row in result.rows],
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": result.duration_ms,
        "applied_row_limit": result.applied_row_limit,
        "sql_char_count": result.sql_char_count,
        "referenced_table_count": result.referenced_table_count,
        "referenced_column_count": result.referenced_column_count,
    }


def serialize_execution_service_result(
    outcome: SQLExecutionServiceResult,
) -> dict[str, Any]:
    payload = serialize_execution_result(outcome.result)
    payload["data_source_id"] = str(outcome.data_source_id)
    payload["workspace_id"] = str(outcome.workspace_id)
    payload["organization_id"] = str(outcome.organization_id)
    # ValidatedSQL contains SQL text; expose only non-sensitive counts already
    # present on the result. Never include sql or identifier lists here.
    return payload


def serialize_execution_error(exc: SQLExecutionError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": exc.code.value,
        "status": exc.status.value,
        "duration_ms": exc.duration_ms,
        "message": redact_secret(str(exc)),
    }
    if isinstance(exc, SQLExecutionValidationError) and exc.violations:
        payload["violations"] = [
            {
                "code": item.code.value,
                "message": redact_secret(item.message),
                "identifier": item.identifier,
            }
            for item in exc.violations[:20]
        ]
    return payload
