"""Classify SQL validation and execution failures as correctable or not.

Correction is only attempted after an actual failure. Dangerous, write, and
authorization failures are never sent to the LLM.
"""

from __future__ import annotations

from app.ai.sql_correction.errors import SQLCorrectionValidationError
from app.ai.sql_correction.models import (
    SQLErrorAnalysis,
    SQLErrorCorrectability,
    SQLErrorSource,
)
from app.ai.sql_validation.models import (
    SQLValidationViolation,
    SQLValidationViolationCode,
)

NON_CORRECTABLE_VIOLATION_CODES = frozenset(
    {
        SQLValidationViolationCode.NOT_READONLY,
        SQLValidationViolationCode.MULTI_STATEMENT,
        SQLValidationViolationCode.DANGEROUS_STATEMENT,
        SQLValidationViolationCode.SQL_TOO_LONG,
        SQLValidationViolationCode.INSUFFICIENT_SCHEMA,
    }
)

CORRECTABLE_VIOLATION_CODES = frozenset(
    {
        SQLValidationViolationCode.UNKNOWN_TABLE,
        SQLValidationViolationCode.UNKNOWN_COLUMN,
        SQLValidationViolationCode.AMBIGUOUS_COLUMN,
        SQLValidationViolationCode.STAR_SELECTION,
        SQLValidationViolationCode.PARSE_ERROR,
        SQLValidationViolationCode.CROSS_SCHEMA,
        SQLValidationViolationCode.UNAUTHORIZED_IDENTIFIER,
        SQLValidationViolationCode.EMPTY_SQL,
    }
)

NON_CORRECTABLE_EXECUTION_CODES = frozenset(
    {
        "SQL_EXECUTION_AUTHORIZATION_ERROR",
        "SQL_EXECUTION_TIMEOUT_ERROR",
        "SQL_EXECUTION_RESULT_LIMIT_ERROR",
        "SQL_EXECUTION_REJECTED_ERROR",
        "SQL_EXECUTION_CANCELLED_ERROR",
        "SQL_EXECUTION_CONFIGURATION_ERROR",
    }
)

CORRECTABLE_EXECUTION_CODES = frozenset(
    {
        "SQL_EXECUTION_DATABASE_ERROR",
        "SQL_EXECUTION_VALIDATION_ERROR",
    }
)

_NON_CORRECTABLE_MESSAGE_MARKERS = (
    "permission denied",
    "not authorized",
    "authentication failed",
    "timed out",
    "timeout",
    "cancelled",
    "too large",
    "rate limit",
)


def has_correction_failure(
    *,
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
) -> bool:
    return (
        bool(violations) or bool(execution_error_code) or bool(execution_error_message)
    )


def require_correction_failure(
    *,
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
) -> None:
    """Reject correction requests that are not backed by an actual failure."""
    if not has_correction_failure(
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
    ):
        raise SQLCorrectionValidationError(
            "SQL correction requires a validation or execution failure"
        )


def analyze_sql_failure(
    *,
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
) -> SQLErrorAnalysis:
    """Classify a failed query. Any non-correctable signal wins."""
    require_correction_failure(
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
    )

    codes: list[str] = [item.code.value for item in violations]
    if execution_error_code:
        codes.append(execution_error_code)

    source = SQLErrorSource.VALIDATION if violations else SQLErrorSource.EXECUTION

    if _has_non_correctable_violations(violations):
        return SQLErrorAnalysis(
            correctability=SQLErrorCorrectability.NON_CORRECTABLE,
            source=SQLErrorSource.VALIDATION,
            codes=codes,
            reason="SQL failed a non-correctable safety or size check",
        )

    if _is_non_correctable_execution(execution_error_code, execution_error_message):
        return SQLErrorAnalysis(
            correctability=SQLErrorCorrectability.NON_CORRECTABLE,
            source=SQLErrorSource.EXECUTION,
            codes=codes,
            reason="Execution failed with a non-correctable boundary error",
        )

    if violations or execution_error_code or execution_error_message:
        return SQLErrorAnalysis(
            correctability=SQLErrorCorrectability.CORRECTABLE,
            source=source,
            codes=codes,
            reason="SQL failed a schema, syntax, or database check that may be rewritten",
        )

    return SQLErrorAnalysis(
        correctability=SQLErrorCorrectability.NON_CORRECTABLE,
        source=source,
        codes=codes,
        reason="Failure could not be classified as correctable",
    )


def _has_non_correctable_violations(
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...],
) -> bool:
    for item in violations:
        if item.code in NON_CORRECTABLE_VIOLATION_CODES:
            return True
        if item.code not in CORRECTABLE_VIOLATION_CODES:
            return True
    return False


def _is_non_correctable_execution(
    execution_error_code: str | None,
    execution_error_message: str | None,
) -> bool:
    if not execution_error_code and not execution_error_message:
        return False
    code = (execution_error_code or "").strip().upper()
    if code in NON_CORRECTABLE_EXECUTION_CODES:
        return True
    if code and code not in CORRECTABLE_EXECUTION_CODES:
        return True
    message = (execution_error_message or "").strip().lower()
    return any(marker in message for marker in _NON_CORRECTABLE_MESSAGE_MARKERS)
