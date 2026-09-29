"""Schema-aware SQL generation for the AI analyst.

This package produces structured, untrusted SQL drafts from authorized Phase 4
catalog metadata. It does not validate SQL safety or execute queries.
"""

from app.ai.sql_generation.catalog import ensure_metadata_catalog_authorized
from app.ai.sql_generation.errors import (
    SQLGenerationAuthorizationError,
    SQLGenerationConfigurationError,
    SQLGenerationError,
    SQLGenerationErrorCode,
    SQLGenerationLLMError,
    SQLGenerationSchemaError,
    SQLGenerationValidationError,
)
from app.ai.sql_generation.errors_mapping import map_sql_generation_error
from app.ai.sql_generation.generation import generate_sql
from app.ai.sql_generation.logging_helpers import (
    generated_sql_log_context,
    generation_log_context,
)
from app.ai.sql_generation.models import (
    GeneratedSQL,
    LLMSQLOutput,
    SQLDialect,
    SQLGenerateParams,
    SQLGenerationConfidence,
    SQLGenerationOutcome,
    SQLGenerationResult,
)
from app.ai.sql_generation.prompts import (
    SQL_GENERATION_BUNDLE_VERSION,
    SQL_GENERATION_SYSTEM_PROMPT_ID,
    SQL_GENERATION_USER_TEMPLATE_ID,
    SQLGenerationVariables,
    build_sql_generation_prompt_registry,
)
from app.ai.sql_generation.schema_context import (
    SchemaPromptContext,
    build_schema_prompt_context,
    has_usable_schema,
)
from app.ai.sql_generation.service import SQLGenerationService
from app.ai.sql_generation.validation import normalize_sql_output, parse_llm_sql

__all__ = [
    "SQL_GENERATION_BUNDLE_VERSION",
    "SQL_GENERATION_SYSTEM_PROMPT_ID",
    "SQL_GENERATION_USER_TEMPLATE_ID",
    "GeneratedSQL",
    "LLMSQLOutput",
    "SQLDialect",
    "SQLGenerateParams",
    "SQLGenerationAuthorizationError",
    "SQLGenerationConfidence",
    "SQLGenerationConfigurationError",
    "SQLGenerationError",
    "SQLGenerationErrorCode",
    "SQLGenerationLLMError",
    "SQLGenerationOutcome",
    "SQLGenerationResult",
    "SQLGenerationSchemaError",
    "SQLGenerationService",
    "SQLGenerationValidationError",
    "SQLGenerationVariables",
    "SchemaPromptContext",
    "build_schema_prompt_context",
    "build_sql_generation_prompt_registry",
    "ensure_metadata_catalog_authorized",
    "generate_sql",
    "generated_sql_log_context",
    "generation_log_context",
    "has_usable_schema",
    "map_sql_generation_error",
    "normalize_sql_output",
    "parse_llm_sql",
]
