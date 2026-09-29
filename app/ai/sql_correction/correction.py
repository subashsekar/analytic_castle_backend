"""LLM-backed schema-aware SQL correction for a single attempt.

Produces an untrusted SQL draft. Does not execute SQL and does not call MCP.
"""

from __future__ import annotations

import logging

from app.ai.intent_types import AIIntent
from app.ai.llm import (
    AsyncLLMClient,
    LLMError,
    LLMRequest,
    LLMResponseValidationError,
)
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.prompt import (
    PromptRegistry,
    bundle_to_llm_messages,
    structured_output_from_model,
)
from app.ai.sql_correction.errors import (
    SQLCorrectionLLMError,
    SQLCorrectionSchemaError,
    SQLCorrectionValidationError,
)
from app.ai.sql_correction.feedback import build_correction_feedback
from app.ai.sql_correction.logging_helpers import correction_log_context
from app.ai.sql_correction.models import SQLCorrectionOutcome, SQLCorrectionStatus
from app.ai.sql_correction.prompts import (
    SQL_CORRECTION_BUNDLE_VERSION,
    SQL_CORRECTION_SYSTEM_PROMPT_ID,
    SQL_CORRECTION_USER_TEMPLATE_ID,
    SQLCorrectionVariables,
)
from app.ai.sql_correction.validation import (
    normalize_correction_output,
    parse_llm_correction,
)
from app.ai.sql_generation.errors import SQLGenerationValidationError
from app.ai.sql_generation.models import LLMSQLOutput, SQLGenerationOutcome
from app.ai.sql_generation.schema_cache import cached_schema_prompt_context
from app.ai.sql_generation.schema_context import (
    SchemaPromptContext,
    has_usable_schema,
)
from app.ai.sql_validation.models import SQLValidationViolation
from app.core.config import settings

logger = logging.getLogger(__name__)


async def correct_sql(
    *,
    client: AsyncLLMClient,
    registry: PromptRegistry,
    previous_sql: str,
    metadata: ResolvedMetadataContext,
    feedback: str,
    attempt: int,
    message: str | None = None,
    intent: AIIntent | None = None,
    plan_summary: str | None = None,
    data_source_name: str | None = None,
    schema_context: SchemaPromptContext | None = None,
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
) -> SQLGenerationOutcome:
    """Ask the LLM to rewrite failed SQL. Does not execute the result."""
    if not has_usable_schema(metadata):
        raise SQLCorrectionSchemaError("Schema context is required for SQL correction")

    schema = schema_context or cached_schema_prompt_context(metadata)
    if not schema.text.strip() or schema.table_count + schema.column_count == 0:
        raise SQLCorrectionSchemaError("Schema context is required for SQL correction")

    max_message = settings.AI_MAX_MESSAGE_CHARS
    bounded_message = (message or "none").strip() or "none"
    if len(bounded_message) > max_message:
        bounded_message = bounded_message[:max_message]

    intent_value = intent.intent.value if intent is not None else "none"
    operation_value = (
        intent.operation.value
        if intent is not None and intent.operation is not None
        else "none"
    )
    subject_value = intent.subject if intent is not None and intent.subject else "none"
    plan_text = (plan_summary or "none").strip() or "none"
    max_plan_chars = settings.AI_SQL_MAX_PLAN_SUMMARY_CHARS
    if len(plan_text) > max_plan_chars:
        plan_text = plan_text[:max_plan_chars].rstrip() + "...[truncated]"

    max_sql_chars = settings.AI_SQL_MAX_SQL_CHARS
    bounded_sql = previous_sql.strip()
    if len(bounded_sql) > max_sql_chars:
        bounded_sql = bounded_sql[:max_sql_chars].rstrip() + "...[truncated]"

    feedback_text = feedback.strip() or build_correction_feedback(
        violations=violations,
        execution_error_code=execution_error_code,
        execution_error_message=execution_error_message,
    )

    bundle = registry.build_bundle(
        bundle_version=SQL_CORRECTION_BUNDLE_VERSION,
        system_prompt_id=SQL_CORRECTION_SYSTEM_PROMPT_ID,
        user_template_id=SQL_CORRECTION_USER_TEMPLATE_ID,
        user_variables=SQLCorrectionVariables(
            message=bounded_message,
            intent=intent_value,
            operation=operation_value,
            subject=subject_value,
            data_source_name=data_source_name or "none",
            plan_summary=plan_text,
            suggested_limit=str(settings.AI_MAX_RESULT_LIMIT),
            attempt=str(attempt),
            feedback=feedback_text,
            previous_sql=bounded_sql,
            schema_context=schema.text,
        ),
        structured_output=structured_output_from_model(LLMSQLOutput),
    )
    request = LLMRequest(
        model=client.config.model,
        messages=bundle_to_llm_messages(bundle),
        structured_output=bundle.structured_output,
    )
    try:
        response = await client.complete(request)
    except LLMError as exc:
        raise SQLCorrectionLLMError(str(exc)) from exc

    if not response.content:
        raise SQLCorrectionLLMError("SQL correction response was empty")

    try:
        llm_result = parse_llm_correction(response.content)
        outcome = normalize_correction_output(
            llm_result=llm_result,
            schema_truncated=schema.truncated,
        )
    except (LLMResponseValidationError, SQLGenerationValidationError) as exc:
        raise SQLCorrectionLLMError(str(exc)) from exc
    except SQLCorrectionValidationError as exc:
        raise SQLCorrectionLLMError(str(exc)) from exc

    logger.info(
        "sql correction attempt completed",
        extra=correction_log_context(
            SQLCorrectionOutcome(
                status=(
                    SQLCorrectionStatus.CLARIFICATION_REQUIRED
                    if outcome.requires_clarification
                    else SQLCorrectionStatus.UNCORRECTED
                ),
                generated=outcome.generated,
                attempt_count=attempt,
                requires_clarification=outcome.requires_clarification,
                clarification_question=outcome.clarification_question,
                schema_truncated=outcome.schema_truncated,
            )
        ),
    )
    return outcome
