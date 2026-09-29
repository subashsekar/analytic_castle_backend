"""Core SQL validation pipeline for untrusted generated SQL.

Does not execute SQL. Reuses Phase 5 read-only checks and validates identifiers
against authorized Phase 4 metadata only.
"""

from __future__ import annotations

from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.sql_validation.allowlist import (
    build_schema_allowlist,
    has_usable_allowlist,
)
from app.ai.sql_validation.errors import (
    SQLValidationError,
    SQLValidationParseError,
    SQLValidationReadonlyError,
    SQLValidationSchemaError,
)
from app.ai.sql_validation.extract import extract_sql_references
from app.ai.sql_validation.identifiers import validate_identifiers
from app.ai.sql_validation.models import (
    SQLValidationResult,
    SQLValidationViolation,
    SQLValidationViolationCode,
    ValidatedSQL,
)
from app.ai.sql_validation.readonly import ensure_readonly_sql
from app.core.config import settings


def validate_generated_sql(
    sql: str,
    metadata: ResolvedMetadataContext,
    *,
    default_schema: str | None = None,
    max_sql_chars: int | None = None,
) -> SQLValidationResult:
    """Validate untrusted SQL as a single read-only statement against metadata."""
    violations: list[SQLValidationViolation] = []
    char_limit = (
        max_sql_chars if max_sql_chars is not None else settings.AI_SQL_MAX_SQL_CHARS
    )
    schema_default = (
        default_schema
        if default_schema is not None
        else settings.AI_SQL_VALIDATION_DEFAULT_SCHEMA
    )

    if not isinstance(sql, str) or not sql.strip():
        return SQLValidationResult(
            is_valid=False,
            violations=[
                SQLValidationViolation(
                    code=SQLValidationViolationCode.EMPTY_SQL,
                    message="SQL is empty",
                )
            ],
        )

    if len(sql) > char_limit:
        return SQLValidationResult(
            is_valid=False,
            violations=[
                SQLValidationViolation(
                    code=SQLValidationViolationCode.SQL_TOO_LONG,
                    message=f"SQL exceeds maximum length of {char_limit} characters",
                )
            ],
        )

    allowlist = build_schema_allowlist(metadata)
    if not has_usable_allowlist(allowlist):
        raise SQLValidationSchemaError(
            "Authorized schema metadata is required for SQL validation"
        )

    try:
        normalized = ensure_readonly_sql(sql)
    except SQLValidationReadonlyError as exc:
        code = _readonly_violation_code(sql)
        return SQLValidationResult(
            is_valid=False,
            violations=[
                SQLValidationViolation(
                    code=code,
                    message=str(exc),
                )
            ],
        )

    try:
        references = extract_sql_references(
            normalized,
            default_schema=schema_default,
        )
    except SQLValidationParseError as exc:
        return SQLValidationResult(
            is_valid=False,
            violations=[
                SQLValidationViolation(
                    code=SQLValidationViolationCode.PARSE_ERROR,
                    message=str(exc),
                )
            ],
        )

    violations.extend(validate_identifiers(references, allowlist))
    if violations:
        return SQLValidationResult(is_valid=False, violations=violations)

    referenced_tables = sorted({table.key() for table in references.tables})
    referenced_columns = sorted(
        {key for column in references.columns if (key := column.key()) is not None}
        | {
            column.column_name
            for column in references.columns
            if column.key() is None and column.table_name is None
        }
    )
    return SQLValidationResult(
        is_valid=True,
        validated=ValidatedSQL(
            sql=normalized,
            referenced_tables=referenced_tables,
            referenced_columns=referenced_columns,
        ),
        violations=[],
    )


def validate_generated_sql_or_raise(
    sql: str,
    metadata: ResolvedMetadataContext,
    *,
    default_schema: str | None = None,
    max_sql_chars: int | None = None,
) -> ValidatedSQL:
    """Validate SQL and raise a typed error when invalid."""
    result = validate_generated_sql(
        sql,
        metadata,
        default_schema=default_schema,
        max_sql_chars=max_sql_chars,
    )
    if result.is_valid and result.validated is not None:
        return result.validated
    _raise_from_result(result)
    raise AssertionError("unreachable")  # pragma: no cover


def _raise_from_result(result: SQLValidationResult) -> None:
    if not result.violations:
        raise SQLValidationError("SQL validation failed")
    first = result.violations[0]
    if first.code in {
        SQLValidationViolationCode.NOT_READONLY,
        SQLValidationViolationCode.MULTI_STATEMENT,
        SQLValidationViolationCode.DANGEROUS_STATEMENT,
    }:
        raise SQLValidationReadonlyError(first.message)
    if first.code == SQLValidationViolationCode.PARSE_ERROR:
        raise SQLValidationParseError(first.message)
    if first.code in {
        SQLValidationViolationCode.UNKNOWN_TABLE,
        SQLValidationViolationCode.CROSS_SCHEMA,
    }:
        from app.ai.sql_validation.errors import SQLValidationTableError

        raise SQLValidationTableError(first.message)
    if first.code in {
        SQLValidationViolationCode.UNKNOWN_COLUMN,
        SQLValidationViolationCode.UNAUTHORIZED_IDENTIFIER,
        SQLValidationViolationCode.AMBIGUOUS_COLUMN,
        SQLValidationViolationCode.STAR_SELECTION,
    }:
        from app.ai.sql_validation.errors import SQLValidationColumnError

        raise SQLValidationColumnError(first.message)
    if first.code == SQLValidationViolationCode.INSUFFICIENT_SCHEMA:
        raise SQLValidationSchemaError(first.message)
    raise SQLValidationError(first.message)


def _readonly_violation_code(sql: str) -> SQLValidationViolationCode:
    lowered = sql.lower()
    if ";" in sql.strip().rstrip(";"):
        return SQLValidationViolationCode.MULTI_STATEMENT
    write_tokens = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
        "merge",
        "copy",
        "call",
        "do ",
    )
    dangerous_functions = (
        "pg_sleep(",
        "pg_read_file(",
        "pg_write_file(",
        "pg_ls_dir(",
        "pg_read_binary_file(",
        "dblink(",
        "dblink_exec(",
        "lo_import(",
        "lo_export(",
        "set_config(",
        "current_setting(",
    )
    if any(token in lowered for token in write_tokens) or any(
        token in lowered for token in dangerous_functions
    ):
        return SQLValidationViolationCode.DANGEROUS_STATEMENT
    return SQLValidationViolationCode.NOT_READONLY
