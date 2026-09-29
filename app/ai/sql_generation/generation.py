"""LLM-backed schema-aware SQL generation."""

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
from app.ai.sql_generation.errors import (
    SQLGenerationLLMError,
    SQLGenerationSchemaError,
    SQLGenerationValidationError,
)
from app.ai.sql_generation.logging_helpers import generation_log_context
from app.ai.sql_generation.models import LLMSQLOutput, SQLGenerationOutcome
from app.ai.sql_generation.prompts import (
    SQL_GENERATION_BUNDLE_VERSION,
    SQL_GENERATION_SYSTEM_PROMPT_ID,
    SQL_GENERATION_USER_TEMPLATE_ID,
    SQLGenerationVariables,
)
from app.ai.sql_generation.schema_cache import cached_schema_prompt_context
from app.ai.sql_generation.schema_context import (
    SchemaPromptContext,
    has_usable_schema,
)
from app.ai.sql_generation.validation import normalize_sql_output, parse_llm_sql
from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate_sql(
    *,
    client: AsyncLLMClient,
    registry: PromptRegistry,
    message: str,
    metadata: ResolvedMetadataContext,
    intent: AIIntent | None = None,
    plan_summary: str | None = None,
    data_source_name: str | None = None,
    schema_context: SchemaPromptContext | None = None,
    conversation_context: str | None = None,
) -> SQLGenerationOutcome:
    """Generate structured SQL from authorized metadata. Does not execute SQL."""
    if not has_usable_schema(metadata):
        raise SQLGenerationSchemaError("Schema context is required for SQL generation")

    schema = schema_context or cached_schema_prompt_context(metadata)
    if not schema.text.strip() or schema.table_count + schema.column_count == 0:
        raise SQLGenerationSchemaError("Schema context is required for SQL generation")

    max_message = settings.AI_MAX_MESSAGE_CHARS
    bounded_message = message.strip()
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
    conversation_text = (conversation_context or "none").strip() or "none"
    if len(conversation_text) > max_message:
        conversation_text = conversation_text[-max_message:]

    bundle = registry.build_bundle(
        bundle_version=SQL_GENERATION_BUNDLE_VERSION,
        system_prompt_id=SQL_GENERATION_SYSTEM_PROMPT_ID,
        user_template_id=SQL_GENERATION_USER_TEMPLATE_ID,
        user_variables=SQLGenerationVariables(
            message=bounded_message,
            intent=intent_value,
            operation=operation_value,
            subject=subject_value,
            data_source_name=data_source_name or "none",
            plan_summary=plan_text,
            suggested_limit=str(settings.AI_MAX_RESULT_LIMIT),
            schema_context=schema.text,
            conversation_context=conversation_text,
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
        raise SQLGenerationLLMError(str(exc)) from exc

    if not response.content:
        raise SQLGenerationLLMError("SQL generation response was empty")

    try:
        llm_result = parse_llm_sql(response.content)
        outcome = normalize_sql_output(
            llm_result=llm_result,
            schema_truncated=schema.truncated,
        )
    except (LLMResponseValidationError, SQLGenerationValidationError) as exc:
        raise SQLGenerationLLMError(str(exc)) from exc

    logger.info(
        "sql generation completed",
        extra=generation_log_context(outcome),
    )
    return outcome
