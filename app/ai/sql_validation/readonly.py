"""Reuse Phase 5 read-only SQL security checks without duplication."""

from __future__ import annotations

from app.ai.sql_validation.errors import SQLValidationReadonlyError
from app.connectors.exceptions import ConnectorQueryError
from app.connectors.readonly_sql import validate_readonly_sql


def ensure_readonly_sql(sql: str) -> str:
    """Return a single stripped read-only statement, or raise a typed error.

    Delegates to ``validate_readonly_sql`` (Phase 5). Does not execute SQL.
    """
    try:
        return validate_readonly_sql(sql)
    except ConnectorQueryError as exc:
        raise SQLValidationReadonlyError(
            "SQL is not a single read-only SELECT statement"
        ) from exc
