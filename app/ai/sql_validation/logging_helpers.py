"""Safe observability helpers that never emit SQL or schema identifiers."""

from __future__ import annotations

from app.ai.sql_validation.models import SQLValidationResult


def validation_log_context(result: SQLValidationResult) -> dict[str, object]:
    return {
        "is_valid": result.is_valid,
        "violation_count": len(result.violations),
        "violation_codes": [item.code.value for item in result.violations[:10]],
        "has_validated_sql": result.validated is not None,
        "referenced_table_count": (
            len(result.validated.referenced_tables) if result.validated else 0
        ),
        "referenced_column_count": (
            len(result.validated.referenced_columns) if result.validated else 0
        ),
        "sql_char_count": (len(result.validated.sql) if result.validated else None),
    }
