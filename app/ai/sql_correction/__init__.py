"""SQL correction and bounded retry for failed generated or executed queries.

This package classifies validation/execution failures, asks an LLM to rewrite
untrusted SQL, and re-validates every draft with Chapter 7.2. It does not
record query history (Chapter 7.5) and never bypasses MCP security limits.
"""

from app.ai.sql_correction.analysis import (
    CORRECTABLE_EXECUTION_CODES,
    CORRECTABLE_VIOLATION_CODES,
    NON_CORRECTABLE_EXECUTION_CODES,
    NON_CORRECTABLE_VIOLATION_CODES,
    analyze_sql_failure,
    has_correction_failure,
    require_correction_failure,
)
from app.ai.sql_correction.correction import correct_sql
from app.ai.sql_correction.errors import (
    SQLCorrectionAuthorizationError,
    SQLCorrectionConfigurationError,
    SQLCorrectionError,
    SQLCorrectionErrorCode,
    SQLCorrectionLLMError,
    SQLCorrectionSchemaError,
    SQLCorrectionValidationError,
)
from app.ai.sql_correction.errors_mapping import map_sql_correction_error
from app.ai.sql_correction.feedback import build_correction_feedback
from app.ai.sql_correction.logging_helpers import (
    analysis_log_context,
    correction_log_context,
)
from app.ai.sql_correction.models import (
    SQLCorrectionOutcome,
    SQLCorrectionServiceResult,
    SQLCorrectionStatus,
    SQLCorrectParams,
    SQLErrorAnalysis,
    SQLErrorCorrectability,
    SQLErrorSource,
)
from app.ai.sql_correction.prompts import (
    SQL_CORRECTION_BUNDLE_VERSION,
    SQL_CORRECTION_SYSTEM_PROMPT_ID,
    SQL_CORRECTION_USER_TEMPLATE_ID,
    SQLCorrectionVariables,
    build_sql_correction_prompt_registry,
)
from app.ai.sql_correction.serialization import (
    serialize_correction_error,
    serialize_correction_outcome,
    serialize_correction_service_result,
)
from app.ai.sql_correction.service import SQLCorrectionService
from app.ai.sql_correction.validation import (
    normalize_correction_output,
    parse_llm_correction,
    resolve_max_attempts,
    sql_is_unchanged,
)
from app.ai.sql_correction.workflow import (
    correct_failed_sql,
    execute_corrected_sql,
    require_validated_correction,
)

__all__ = [
    "CORRECTABLE_EXECUTION_CODES",
    "CORRECTABLE_VIOLATION_CODES",
    "NON_CORRECTABLE_EXECUTION_CODES",
    "NON_CORRECTABLE_VIOLATION_CODES",
    "SQL_CORRECTION_BUNDLE_VERSION",
    "SQL_CORRECTION_SYSTEM_PROMPT_ID",
    "SQL_CORRECTION_USER_TEMPLATE_ID",
    "SQLCorrectParams",
    "SQLCorrectionAuthorizationError",
    "SQLCorrectionConfigurationError",
    "SQLCorrectionError",
    "SQLCorrectionErrorCode",
    "SQLCorrectionLLMError",
    "SQLCorrectionOutcome",
    "SQLCorrectionSchemaError",
    "SQLCorrectionService",
    "SQLCorrectionServiceResult",
    "SQLCorrectionStatus",
    "SQLCorrectionValidationError",
    "SQLCorrectionVariables",
    "SQLErrorAnalysis",
    "SQLErrorCorrectability",
    "SQLErrorSource",
    "analysis_log_context",
    "analyze_sql_failure",
    "build_correction_feedback",
    "build_sql_correction_prompt_registry",
    "correct_failed_sql",
    "correct_sql",
    "correction_log_context",
    "execute_corrected_sql",
    "has_correction_failure",
    "map_sql_correction_error",
    "normalize_correction_output",
    "parse_llm_correction",
    "require_correction_failure",
    "require_validated_correction",
    "resolve_max_attempts",
    "serialize_correction_error",
    "serialize_correction_outcome",
    "serialize_correction_service_result",
    "sql_is_unchanged",
]
