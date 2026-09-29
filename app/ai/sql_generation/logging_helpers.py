"""Safe observability helpers that never emit SQL, schema, or prompts."""

from __future__ import annotations

from app.ai.sql_generation.models import GeneratedSQL, SQLGenerationOutcome


def generation_log_context(outcome: SQLGenerationOutcome) -> dict[str, object]:
    context: dict[str, object] = {
        "requires_clarification": outcome.requires_clarification,
        "schema_truncated": outcome.schema_truncated,
        "has_generated_sql": outcome.generated is not None,
    }
    if outcome.generated is not None:
        context.update(generated_sql_log_context(outcome.generated))
    return context


def generated_sql_log_context(generated: GeneratedSQL) -> dict[str, object]:
    return {
        "dialect": generated.dialect.value,
        "confidence": generated.confidence.value,
        "sql_char_count": len(generated.sql),
        "referenced_table_count": len(generated.referenced_tables),
        "referenced_column_count": len(generated.referenced_columns),
        "assumption_count": len(generated.assumptions),
        "suggested_limit": generated.suggested_limit,
        "generation_version": generated.generation_version,
        "schema_truncated": generated.schema_truncated,
        "has_explanation": generated.explanation is not None,
    }
