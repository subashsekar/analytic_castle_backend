"""Parse and normalize structured SQL generation LLM output.

This validates structure and bounds only. It does not perform SQL safety
validation (Chapter 7.2) or execute queries (Chapter 7.3).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.ai.llm.errors import LLMResponseValidationError
from app.ai.sql_generation.errors import SQLGenerationValidationError
from app.ai.sql_generation.models import (
    GeneratedSQL,
    LLMSQLOutput,
    SQLGenerationOutcome,
)
from app.ai.sql_generation.prompts import SQL_GENERATION_BUNDLE_VERSION
from app.core.config import settings


def parse_llm_sql(raw_content: str) -> LLMSQLOutput:
    try:
        from app.ai.llm.content import extract_json_object

        payload = extract_json_object(raw_content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMResponseValidationError(
            "SQL generation response is not valid JSON",
            raw_content=raw_content,
        ) from exc
    try:
        return LLMSQLOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            "SQL generation response failed schema validation",
            raw_content=raw_content,
        ) from exc


def normalize_sql_output(
    *,
    llm_result: LLMSQLOutput,
    schema_truncated: bool = False,
) -> SQLGenerationOutcome:
    if llm_result.requires_clarification:
        question = llm_result.clarification_question
        if not question:
            raise SQLGenerationValidationError(
                "Clarification required but no question provided"
            )
        return SQLGenerationOutcome(
            generated=None,
            requires_clarification=True,
            clarification_question=question,
            schema_truncated=schema_truncated,
        )

    sql = llm_result.sql
    if not sql:
        raise SQLGenerationValidationError("SQL generation output is missing sql")

    max_sql_chars = settings.AI_SQL_MAX_SQL_CHARS
    if len(sql) > max_sql_chars:
        raise SQLGenerationValidationError(
            f"Generated SQL exceeds maximum length of {max_sql_chars} characters"
        )

    max_explanation = settings.AI_SQL_MAX_EXPLANATION_CHARS
    explanation = llm_result.explanation
    if explanation is not None and len(explanation) > max_explanation:
        explanation = explanation[:max_explanation].rstrip()

    max_assumptions = settings.AI_SQL_MAX_ASSUMPTIONS
    assumptions = list(llm_result.assumptions[:max_assumptions])

    suggested_limit = llm_result.suggested_limit
    if suggested_limit is not None:
        suggested_limit = min(suggested_limit, settings.AI_MAX_RESULT_LIMIT)

    generated = GeneratedSQL(
        sql=sql,
        dialect=llm_result.dialect,
        referenced_tables=list(llm_result.referenced_tables),
        referenced_columns=list(llm_result.referenced_columns),
        explanation=explanation,
        confidence=llm_result.confidence,
        assumptions=assumptions,
        suggested_limit=suggested_limit,
        generation_version=SQL_GENERATION_BUNDLE_VERSION.value,
        schema_truncated=schema_truncated,
    )
    return SQLGenerationOutcome(
        generated=generated,
        requires_clarification=False,
        clarification_question=None,
        schema_truncated=schema_truncated,
    )
