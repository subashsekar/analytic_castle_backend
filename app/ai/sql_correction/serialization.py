"""Safe serialization for SQL correction outcomes and errors.

Never includes SQL text, schema, prompts, credentials, or row values.
"""

from __future__ import annotations

from typing import Any

from app.ai.sql_correction.errors import SQLCorrectionError
from app.ai.sql_correction.models import (
    SQLCorrectionOutcome,
    SQLCorrectionServiceResult,
)
from app.core.logging import redact_secret


def serialize_correction_outcome(outcome: SQLCorrectionOutcome) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": outcome.status.value,
        "attempt_count": outcome.attempt_count,
        "max_attempts": outcome.max_attempts,
        "requires_clarification": outcome.requires_clarification,
        "schema_truncated": outcome.schema_truncated,
        "has_validated_sql": outcome.validated is not None,
        "has_generated_sql": outcome.generated is not None,
        "violation_count": len(outcome.violations),
        "violation_codes": [item.code.value for item in outcome.violations[:20]],
    }
    if outcome.analysis is not None:
        payload["correctability"] = outcome.analysis.correctability.value
        payload["error_source"] = outcome.analysis.source.value
        payload["error_codes"] = list(outcome.analysis.codes[:20])
    return payload


def serialize_correction_service_result(
    result: SQLCorrectionServiceResult,
) -> dict[str, Any]:
    payload = serialize_correction_outcome(result.outcome)
    payload["data_source_id"] = str(result.data_source_id)
    payload["workspace_id"] = str(result.workspace_id)
    payload["organization_id"] = str(result.organization_id)
    return payload


def serialize_correction_error(exc: SQLCorrectionError) -> dict[str, Any]:
    return {
        "code": exc.code.value,
        "message": redact_secret(str(exc)),
    }
