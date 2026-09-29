"""SQL safety validation for untrusted generated SQL.

This package validates read-only shape and authorized catalog identifiers.
It does not execute SQL, correct SQL (Chapter 7.4), or record query history
(Chapter 7.5).
"""

from app.ai.sql_validation.allowlist import (
    build_schema_allowlist,
    has_usable_allowlist,
)
from app.ai.sql_validation.errors import (
    SQLValidationAuthorizationError,
    SQLValidationColumnError,
    SQLValidationConfigurationError,
    SQLValidationError,
    SQLValidationErrorCode,
    SQLValidationParseError,
    SQLValidationReadonlyError,
    SQLValidationSchemaError,
    SQLValidationTableError,
)
from app.ai.sql_validation.errors_mapping import map_sql_validation_error
from app.ai.sql_validation.extract import extract_sql_references
from app.ai.sql_validation.logging_helpers import validation_log_context
from app.ai.sql_validation.models import (
    ExtractedReferences,
    QualifiedColumn,
    QualifiedTable,
    SchemaAllowlist,
    SQLValidateParams,
    SQLValidationResult,
    SQLValidationServiceResult,
    SQLValidationViolation,
    SQLValidationViolationCode,
    ValidatedSQL,
)
from app.ai.sql_validation.readonly import ensure_readonly_sql
from app.ai.sql_validation.service import SQLValidationService
from app.ai.sql_validation.validation import (
    validate_generated_sql,
    validate_generated_sql_or_raise,
)

__all__ = [
    "ExtractedReferences",
    "QualifiedColumn",
    "QualifiedTable",
    "SQLValidateParams",
    "SQLValidationAuthorizationError",
    "SQLValidationColumnError",
    "SQLValidationConfigurationError",
    "SQLValidationError",
    "SQLValidationErrorCode",
    "SQLValidationParseError",
    "SQLValidationReadonlyError",
    "SQLValidationResult",
    "SQLValidationSchemaError",
    "SQLValidationService",
    "SQLValidationServiceResult",
    "SQLValidationTableError",
    "SQLValidationViolation",
    "SQLValidationViolationCode",
    "SchemaAllowlist",
    "ValidatedSQL",
    "build_schema_allowlist",
    "ensure_readonly_sql",
    "extract_sql_references",
    "has_usable_allowlist",
    "map_sql_validation_error",
    "validate_generated_sql",
    "validate_generated_sql_or_raise",
    "validation_log_context",
]
