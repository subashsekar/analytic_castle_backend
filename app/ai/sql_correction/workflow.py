"""Bounded SQL correction retry workflow.

Corrects only after a validation or execution failure. Re-validates every
draft with Chapter 7.2. Execution is optional and always goes through
Chapter 7.3, which re-validates before MCP.
"""

from __future__ import annotations

import logging

from app.ai.llm import AsyncLLMClient
from app.ai.prompt import PromptRegistry
from app.ai.sql_correction.analysis import (
    analyze_sql_failure,
    require_correction_failure,
)
from app.ai.sql_correction.correction import correct_sql
from app.ai.sql_correction.errors import (
    SQLCorrectionSchemaError,
    SQLCorrectionValidationError,
)
from app.ai.sql_correction.feedback import build_correction_feedback
from app.ai.sql_correction.logging_helpers import correction_log_context
from app.ai.sql_correction.models import (
    SQLCorrectionOutcome,
    SQLCorrectionStatus,
    SQLCorrectParams,
    SQLErrorCorrectability,
)
from app.ai.sql_correction.validation import resolve_max_attempts, sql_is_unchanged
from app.ai.sql_execution.execution import execute_validated_sql
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.sql_generation.schema_cache import cached_schema_prompt_context
from app.ai.sql_generation.schema_context import has_usable_schema
from app.ai.sql_validation.validation import validate_generated_sql
from app.mcp import MCPClient

logger = logging.getLogger(__name__)


async def correct_failed_sql(
    *,
    client: AsyncLLMClient,
    registry: PromptRegistry,
    params: SQLCorrectParams,
) -> SQLCorrectionOutcome:
    """Retry-correct failed SQL up to a configured maximum. Never executes SQL."""
    require_correction_failure(
        violations=params.violations,
        execution_error_code=params.execution_error_code,
        execution_error_message=params.execution_error_message,
    )
    if not has_usable_schema(params.metadata):
        raise SQLCorrectionSchemaError("Schema context is required for SQL correction")

    max_attempts = resolve_max_attempts(params.max_attempts)
    analysis = analyze_sql_failure(
        violations=params.violations,
        execution_error_code=params.execution_error_code,
        execution_error_message=params.execution_error_message,
    )
    if analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE:
        outcome = SQLCorrectionOutcome(
            status=SQLCorrectionStatus.UNCORRECTED,
            violations=list(params.violations),
            attempt_count=0,
            max_attempts=max_attempts,
            analysis=analysis,
        )
        logger.info(
            "sql correction skipped as non-correctable",
            extra=correction_log_context(outcome),
        )
        return outcome

    schema = cached_schema_prompt_context(params.metadata)
    current_sql = params.sql
    current_violations = list(params.violations)
    current_exec_code = params.execution_error_code
    current_exec_message = params.execution_error_message
    last_generated = None
    latest_analysis = analysis

    for attempt in range(1, max_attempts + 1):
        feedback = build_correction_feedback(
            violations=current_violations,
            execution_error_code=current_exec_code,
            execution_error_message=current_exec_message,
        )
        generation = await correct_sql(
            client=client,
            registry=registry,
            previous_sql=current_sql,
            metadata=params.metadata,
            feedback=feedback,
            attempt=attempt,
            message=params.message,
            intent=params.intent,
            plan_summary=params.plan_summary,
            data_source_name=params.data_source_name,
            schema_context=schema,
            violations=current_violations,
            execution_error_code=current_exec_code,
            execution_error_message=current_exec_message,
        )

        if generation.requires_clarification:
            outcome = SQLCorrectionOutcome(
                status=SQLCorrectionStatus.CLARIFICATION_REQUIRED,
                generated=generation.generated,
                violations=current_violations,
                attempt_count=attempt,
                max_attempts=max_attempts,
                requires_clarification=True,
                clarification_question=generation.clarification_question,
                schema_truncated=generation.schema_truncated or schema.truncated,
                analysis=latest_analysis,
            )
            logger.info(
                "sql correction requested clarification",
                extra=correction_log_context(outcome),
            )
            return outcome

        generated = generation.generated
        if generated is None:
            continue

        last_generated = generated
        if sql_is_unchanged(current_sql, generated.sql):
            outcome = SQLCorrectionOutcome(
                status=SQLCorrectionStatus.UNCHANGED,
                generated=generated,
                violations=current_violations,
                attempt_count=attempt,
                max_attempts=max_attempts,
                schema_truncated=generation.schema_truncated or schema.truncated,
                analysis=latest_analysis,
            )
            logger.info(
                "sql correction left SQL unchanged",
                extra=correction_log_context(outcome),
            )
            return outcome

        validation = validate_generated_sql(generated.sql, params.metadata)
        if validation.is_valid and validation.validated is not None:
            outcome = SQLCorrectionOutcome(
                status=SQLCorrectionStatus.CORRECTED,
                validated=validation.validated,
                generated=generated,
                violations=[],
                attempt_count=attempt,
                max_attempts=max_attempts,
                schema_truncated=generation.schema_truncated or schema.truncated,
                analysis=latest_analysis,
            )
            logger.info(
                "sql correction produced validated SQL",
                extra=correction_log_context(outcome),
            )
            return outcome

        current_sql = generated.sql
        current_violations = list(validation.violations)
        current_exec_code = None
        current_exec_message = None
        latest_analysis = analyze_sql_failure(violations=current_violations)
        if latest_analysis.correctability is SQLErrorCorrectability.NON_CORRECTABLE:
            outcome = SQLCorrectionOutcome(
                status=SQLCorrectionStatus.UNCORRECTED,
                generated=generated,
                violations=current_violations,
                attempt_count=attempt,
                max_attempts=max_attempts,
                schema_truncated=generation.schema_truncated or schema.truncated,
                analysis=latest_analysis,
            )
            logger.info(
                "sql correction produced a non-correctable draft",
                extra=correction_log_context(outcome),
            )
            return outcome

    outcome = SQLCorrectionOutcome(
        status=SQLCorrectionStatus.UNCORRECTED,
        generated=last_generated,
        violations=current_violations,
        attempt_count=max_attempts,
        max_attempts=max_attempts,
        schema_truncated=schema.truncated,
        analysis=latest_analysis,
    )
    logger.info(
        "sql correction exhausted retry attempts", extra=correction_log_context(outcome)
    )
    return outcome


def require_validated_correction(outcome: SQLCorrectionOutcome):
    """Gate execution on a fully validated correction result."""
    if outcome.status is not SQLCorrectionStatus.CORRECTED or outcome.validated is None:
        raise SQLCorrectionValidationError(
            "Corrected SQL must pass validation before execution"
        )
    return outcome.validated


async def execute_corrected_sql(
    *,
    client: MCPClient,
    params: SQLCorrectParams,
    outcome: SQLCorrectionOutcome,
    limit: int | None = None,
) -> SQLExecutionResult:
    """Execute corrected SQL only after 7.2 validation, using the 7.3 path."""
    validated = require_validated_correction(outcome)
    return await execute_validated_sql(
        client=client,
        data_source_id=params.data_source_id,
        workspace_id=params.workspace_id,
        user_id=params.user_id,
        validated=validated,
        metadata=params.metadata,
        limit=limit,
    )
