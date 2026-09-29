"""Safe observability helpers that never emit SQL, credentials, or row values."""

from __future__ import annotations

from app.ai.sql_execution.models import SQLExecutionResult


def execution_log_context(result: SQLExecutionResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": round(result.duration_ms, 3),
        "applied_row_limit": result.applied_row_limit,
        "sql_char_count": result.sql_char_count,
        "referenced_table_count": result.referenced_table_count,
        "referenced_column_count": result.referenced_column_count,
        "column_count": len(result.columns),
    }
