"""Compare extracted SQL identifiers against the authorized schema allowlist."""

from __future__ import annotations

from app.ai.sql_validation.models import (
    ExtractedReferences,
    QualifiedColumn,
    SchemaAllowlist,
    SQLValidationViolation,
    SQLValidationViolationCode,
)


def validate_identifiers(
    references: ExtractedReferences,
    allowlist: SchemaAllowlist,
) -> list[SQLValidationViolation]:
    """Return violations for unknown/unauthorized tables and columns."""
    violations: list[SQLValidationViolation] = []

    if references.has_star_selection:
        violations.append(
            SQLValidationViolation(
                code=SQLValidationViolationCode.STAR_SELECTION,
                message="Star column selections are not allowed",
            )
        )

    if not references.tables:
        violations.append(
            SQLValidationViolation(
                code=SQLValidationViolationCode.INSUFFICIENT_SCHEMA,
                message="SQL must reference at least one authorized catalog table",
            )
        )
        # Still check columns for clearer multi-violation diagnostics, but without
        # the global column-name fallback (no authorized tables in scope).
        for column in references.columns:
            violation = _column_violation(
                column,
                allowlist=allowlist,
                authorized_table_keys=set(),
            )
            if violation is not None:
                violations.append(violation)
        return _dedupe_violations(violations)

    for table in references.tables:
        key = table.key()
        if key in allowlist.tables:
            continue
        schema = table.schema_name.lower()
        code = (
            SQLValidationViolationCode.CROSS_SCHEMA
            if schema not in allowlist.schemas
            else SQLValidationViolationCode.UNKNOWN_TABLE
        )
        violations.append(
            SQLValidationViolation(
                code=code,
                message="Unauthorized or unknown table reference",
            )
        )

    authorized_table_keys = {
        table.key() for table in references.tables if table.key() in allowlist.tables
    }

    for column in references.columns:
        violation = _column_violation(
            column,
            allowlist=allowlist,
            authorized_table_keys=authorized_table_keys,
        )
        if violation is not None:
            violations.append(violation)

    return _dedupe_violations(violations)


def _column_violation(
    column: QualifiedColumn,
    *,
    allowlist: SchemaAllowlist,
    authorized_table_keys: set[str],
) -> SQLValidationViolation | None:
    col = column.column_name.lower()

    # Fully resolved against a physical table.
    full_key = column.key()
    if full_key is not None:
        if full_key in allowlist.columns:
            return None
        table_key = f"{column.schema_name}.{column.table_name}".lower()
        if table_key not in allowlist.tables:
            # Table violation already reported (or will be).
            return None
        # Identifier is echoed only for authorized tables so correction can target it.
        return SQLValidationViolation(
            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
            message="Unauthorized or unknown column reference",
            identifier=col,
        )

    # Qualifier present but unresolved (unknown alias / invented table).
    if column.table_name is not None:
        return SQLValidationViolation(
            code=SQLValidationViolationCode.UNAUTHORIZED_IDENTIFIER,
            message="Unauthorized or unknown column qualifier",
        )

    # Unqualified column: must resolve to exactly one authorized referenced table.
    if not authorized_table_keys:
        return SQLValidationViolation(
            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
            message="Unauthorized or unknown column reference",
        )

    matches = [
        table_key
        for table_key in authorized_table_keys
        if col in allowlist.columns_by_table.get(table_key, frozenset())
    ]
    if len(matches) == 1:
        return None
    if len(matches) > 1:
        return SQLValidationViolation(
            code=SQLValidationViolationCode.AMBIGUOUS_COLUMN,
            message="Ambiguous unqualified column reference",
            identifier=col,
        )
    return SQLValidationViolation(
        code=SQLValidationViolationCode.UNKNOWN_COLUMN,
        message="Unauthorized or unknown column reference",
        identifier=col,
    )


def _dedupe_violations(
    violations: list[SQLValidationViolation],
) -> list[SQLValidationViolation]:
    seen: set[tuple[str, str, str | None]] = set()
    unique: list[SQLValidationViolation] = []
    for item in violations:
        marker = (item.code.value, item.message, item.identifier)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique
