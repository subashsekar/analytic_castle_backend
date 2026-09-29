"""Bounded, redacted feedback for SQL correction prompts."""

from __future__ import annotations

import re

from app.ai.sql_validation.models import SQLValidationViolation
from app.core.config import settings
from app.core.logging import redact_secret

_IDENTIFIER_LIKE = re.compile(
    r"\b(public|[a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?\b",
    re.IGNORECASE,
)
_QUOTED_LITERAL = re.compile(r"'(?:''|[^'])*'")


def build_correction_feedback(
    *,
    violations: list[SQLValidationViolation] | tuple[SQLValidationViolation, ...] = (),
    execution_error_code: str | None = None,
    execution_error_message: str | None = None,
    max_chars: int | None = None,
) -> str:
    """Render violation/execution feedback without secrets or credentials."""
    char_limit = (
        max_chars
        if max_chars is not None
        else settings.AI_SQL_CORRECTION_MAX_FEEDBACK_CHARS
    )
    lines: list[str] = []
    for item in list(violations)[:20]:
        identifier = f" identifier={item.identifier}" if item.identifier else ""
        lines.append(
            _sanitize_feedback_text(f"- {item.code.value}{identifier}: {item.message}")
        )
    if execution_error_code or execution_error_message:
        code = execution_error_code or "EXECUTION_ERROR"
        # Prefer stable codes; strip schema/table hints from raw DB messages.
        message = "The database query failed"
        if execution_error_message:
            message = _sanitize_feedback_text(execution_error_message)
        lines.append(_sanitize_feedback_text(f"- {code}: {message}"))
    if not lines:
        lines.append("- UNKNOWN: SQL needs correction")
    text = "\n".join(lines)
    if len(text) > char_limit:
        return text[: char_limit - 15].rstrip() + "...[truncated]"
    return text


def _sanitize_feedback_text(value: str) -> str:
    redacted = redact_secret(value)
    redacted = _QUOTED_LITERAL.sub("'[REDACTED]'", redacted)
    return _IDENTIFIER_LIKE.sub("[REDACTED_IDENTIFIER]", redacted)
