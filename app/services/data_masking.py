"""Deterministic masking for sensitive sample values.

Secrets are fully redacted. PII and sensitive values are masked before they
enter a sample result. NULL stays NULL.
"""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID

from app.enums import ColumnSensitivity
from app.services.pii_detection import MaskKind, mask_kind_for_column

REDACTED = "[REDACTED]"
MASKED = "[MASKED]"


def mask_value(
    value: object,
    *,
    sensitivity: ColumnSensitivity,
    column_name: str = "",
    kind: MaskKind | None = None,
) -> object:
    """Return a safe representation of ``value`` for the given sensitivity."""
    if value is None:
        return None
    if sensitivity is ColumnSensitivity.SECRET:
        return REDACTED
    if sensitivity is ColumnSensitivity.PUBLIC:
        return value
    resolved = kind or mask_kind_for_column(column_name)
    if resolved is MaskKind.SECRET:
        return REDACTED
    return _mask_kind(value, resolved)


def mask_email(value: object) -> object:
    if value is None:
        return None
    text = _as_text(value)
    if text is None:
        return MASKED
    local, separator, domain = text.partition("@")
    if not separator or not local or not domain:
        return _generic_mask(text)
    return f"{local[0]}***@{domain}"


def mask_phone(value: object) -> object:
    if value is None:
        return None
    text = _as_text(value)
    if text is None:
        return MASKED
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) < 4:
        return "*" * max(len(digits), 4)
    hidden = len(digits) - 4
    return f"{'*' * hidden}{digits[-4:]}"


def mask_card(value: object) -> object:
    if value is None:
        return None
    text = _as_text(value)
    if text is None:
        return MASKED
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) < 4:
        return "*" * max(len(digits), 4)
    hidden = len(digits) - 4
    return f"{'*' * hidden}{digits[-4:]}"


def mask_name(value: object) -> object:
    if value is None:
        return None
    text = _as_text(value)
    if text is None:
        return MASKED
    parts = text.split()
    if not parts:
        return text
    return " ".join(_mask_name_part(part) for part in parts)


def mask_generic(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (datetime, date, time)):
        return MASKED
    text = _as_text(value)
    if text is None:
        return MASKED
    return _generic_mask(text)


def _mask_kind(value: object, kind: MaskKind) -> object:
    if kind is MaskKind.EMAIL:
        return mask_email(value)
    if kind is MaskKind.PHONE:
        return mask_phone(value)
    if kind is MaskKind.CARD:
        return mask_card(value)
    if kind is MaskKind.NAME:
        return mask_name(value)
    if kind is MaskKind.SECRET:
        return REDACTED
    return mask_generic(value)


def _mask_name_part(part: str) -> str:
    if not part:
        return part
    hidden = max(len(part) - 1, 0)
    return f"{part[0]}{'*' * hidden}"


def _generic_mask(text: str) -> str:
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 6 and sum(char.isdigit() for char in text) >= len(text) // 2:
        masked = mask_phone(text)
        return masked if isinstance(masked, str) else MASKED
    if not text:
        return text
    return f"{text[0]}{'*' * min(max(len(text) - 1, 3), 8)}"


def _as_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return None
