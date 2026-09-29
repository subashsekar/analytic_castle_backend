"""Parse structured correction output and bound retry attempts.

Structured LLM drafts reuse Chapter 7.1 parsers. Safety checks stay in
Chapter 7.2; this module only validates correction workflow inputs.
"""

from __future__ import annotations

from app.ai.llm.errors import LLMResponseValidationError
from app.ai.sql_correction.errors import SQLCorrectionValidationError
from app.ai.sql_correction.prompts import SQL_CORRECTION_BUNDLE_VERSION
from app.ai.sql_generation.errors import SQLGenerationValidationError
from app.ai.sql_generation.models import SQLGenerationOutcome
from app.ai.sql_generation.validation import normalize_sql_output, parse_llm_sql
from app.core.config import settings


def parse_llm_correction(raw_content: str):
    """Parse structured correction JSON using the 7.1 SQL output schema."""
    return parse_llm_sql(raw_content)


def normalize_correction_output(
    *,
    llm_result,
    schema_truncated: bool = False,
) -> SQLGenerationOutcome:
    """Normalize structured correction output. SQL remains untrusted."""
    try:
        outcome = normalize_sql_output(
            llm_result=llm_result,
            schema_truncated=schema_truncated,
        )
    except SQLGenerationValidationError as exc:
        raise SQLCorrectionValidationError(str(exc)) from exc
    except LLMResponseValidationError:
        raise

    if outcome.generated is not None:
        outcome = outcome.model_copy(
            update={
                "generated": outcome.generated.model_copy(
                    update={"generation_version": SQL_CORRECTION_BUNDLE_VERSION.value}
                )
            }
        )
    return outcome


def resolve_max_attempts(requested: int | None) -> int:
    configured = settings.AI_SQL_CORRECTION_MAX_ATTEMPTS
    if requested is None:
        return configured
    if requested < 1:
        raise SQLCorrectionValidationError("Correction max attempts must be at least 1")
    return min(requested, configured)


def sql_is_unchanged(previous: str, candidate: str) -> bool:
    return _normalize_sql_text(previous) == _normalize_sql_text(candidate)


def _normalize_sql_text(value: str) -> str:
    return " ".join(value.strip().rstrip(";").split()).lower()
