"""Safe observability helpers that never emit SQL, schema, or prompts."""

from __future__ import annotations

from app.ai.sql_correction.models import SQLCorrectionOutcome, SQLErrorAnalysis


def analysis_log_context(analysis: SQLErrorAnalysis) -> dict[str, object]:
    return {
        "correctability": analysis.correctability.value,
        "error_source": analysis.source.value,
        "error_codes": list(analysis.codes[:10]),
    }


def correction_log_context(outcome: SQLCorrectionOutcome) -> dict[str, object]:
    context: dict[str, object] = {
        "status": outcome.status.value,
        "attempt_count": outcome.attempt_count,
        "max_attempts": outcome.max_attempts,
        "requires_clarification": outcome.requires_clarification,
        "schema_truncated": outcome.schema_truncated,
        "has_validated_sql": outcome.validated is not None,
        "has_generated_sql": outcome.generated is not None,
        "violation_count": len(outcome.violations),
        "violation_codes": [item.code.value for item in outcome.violations[:10]],
        "sql_char_count": (
            len(outcome.validated.sql)
            if outcome.validated is not None
            else (len(outcome.generated.sql) if outcome.generated is not None else None)
        ),
    }
    if outcome.analysis is not None:
        context.update(analysis_log_context(outcome.analysis))
    return context
