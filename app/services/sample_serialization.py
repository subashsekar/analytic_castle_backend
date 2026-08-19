"""Safe serialization of sample-cell values.

Preserves JSON-friendly scalars and semantic representations for UUID, dates,
decimals, and JSON. Binary and oversized values are replaced, not dumped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.services.sample_data_exceptions import SampleSerializationError

BINARY_PLACEHOLDER = "[BINARY]"
TRUNCATED_SUFFIX = "…[TRUNCATED]"
TRUNCATED_JSON = "[TRUNCATED JSON]"
UNAVAILABLE = "[UNAVAILABLE]"
UNSUPPORTED = "[UNSUPPORTED]"


def serialize_sample_value(
    value: object,
    *,
    max_value_chars: int | None = None,
    max_json_chars: int | None = None,
) -> object:
    """Convert a driver value into a bounded, JSON-safe representation."""
    value_limit = (
        settings.SAMPLE_DATA_MAX_VALUE_CHARS
        if max_value_chars is None
        else max_value_chars
    )
    json_limit = (
        settings.SAMPLE_DATA_MAX_JSON_CHARS
        if max_json_chars is None
        else max_json_chars
    )
    try:
        return _serialize(value, max_value_chars=value_limit, max_json_chars=json_limit)
    except SampleSerializationError:
        raise
    except Exception as exc:
        raise SampleSerializationError("Unable to serialize sample data") from exc


def _serialize(
    value: object,
    *,
    max_value_chars: int,
    max_json_chars: int,
) -> object:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return BINARY_PLACEHOLDER
    if isinstance(value, (IPv4Address, IPv6Address)):
        return _truncate_text(str(value), max_value_chars)
    if isinstance(value, str):
        return _truncate_text(value, max_value_chars)
    if isinstance(value, Mapping):
        converted = {
            str(key): _serialize(
                item,
                max_value_chars=max_value_chars,
                max_json_chars=max_json_chars,
            )
            for key, item in value.items()
        }
        return _bounded_json(converted, max_json_chars=max_json_chars)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        converted_list = [
            _serialize(
                item,
                max_value_chars=max_value_chars,
                max_json_chars=max_json_chars,
            )
            for item in value
        ]
        return _bounded_json(converted_list, max_json_chars=max_json_chars)
    return UNSUPPORTED


def _truncate_text(value: str, max_value_chars: int) -> str:
    if len(value) <= max_value_chars:
        return value
    return f"{value[:max_value_chars]}{TRUNCATED_SUFFIX}"


def _bounded_json(value: object, *, max_json_chars: int) -> object:
    try:
        encoded = json.dumps(value, default=_json_default, ensure_ascii=False)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if len(encoded) > max_json_chars:
        return TRUNCATED_JSON
    return value


def _json_default(value: Any) -> str:
    return UNSUPPORTED
