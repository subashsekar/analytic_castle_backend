from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal
from uuid import UUID

from app.services.sample_serialization import (
    BINARY_PLACEHOLDER,
    TRUNCATED_JSON,
    TRUNCATED_SUFFIX,
    serialize_sample_value,
)


def test_uuid_serialization() -> None:
    value = UUID("550e8400-e29b-41d4-a716-446655440000")
    assert serialize_sample_value(value) == "550e8400-e29b-41d4-a716-446655440000"


def test_datetime_and_date_serialization() -> None:
    moment = datetime(2024, 3, 15, 12, 30, 0, tzinfo=UTC)
    assert serialize_sample_value(moment) == moment.isoformat()
    assert serialize_sample_value(date(2024, 3, 15)) == "2024-03-15"
    assert serialize_sample_value(time(12, 30, 0)) == "12:30:00"


def test_decimal_serialization_preserves_precision() -> None:
    assert serialize_sample_value(Decimal("19.90")) == "19.90"
    assert serialize_sample_value(Decimal("0.10")) == "0.10"
    assert serialize_sample_value(Decimal("12345678901234567890.123")) == (
        "12345678901234567890.123"
    )


def test_json_and_jsonb_like_structures() -> None:
    payload = {"ok": True, "count": 2, "nested": {"city": "Paris"}}
    assert serialize_sample_value(payload) == payload
    assert serialize_sample_value(["a", 1, None]) == ["a", 1, None]


def test_binary_is_placeholder() -> None:
    assert serialize_sample_value(b"\x00\x01secret") == BINARY_PLACEHOLDER
    assert serialize_sample_value(bytearray(b"abc")) == BINARY_PLACEHOLDER
    assert serialize_sample_value(memoryview(b"abc")) == BINARY_PLACEHOLDER
    assert b"secret" not in str(serialize_sample_value(b"secret")).encode()


def test_large_text_is_truncated() -> None:
    huge = "x" * 20_000
    serialized = serialize_sample_value(huge, max_value_chars=64)
    assert isinstance(serialized, str)
    assert serialized.endswith(TRUNCATED_SUFFIX)
    assert len(serialized) < 100
    assert huge not in serialized


def test_large_json_is_bounded() -> None:
    payload = {"notes": "n" * 20_000}
    serialized = serialize_sample_value(payload, max_json_chars=128)
    assert serialized == TRUNCATED_JSON


def test_null_and_scalars_pass_through() -> None:
    assert serialize_sample_value(None) is None
    assert serialize_sample_value(True) is True
    assert serialize_sample_value(12) == 12
    assert serialize_sample_value(1.5) == 1.5
    assert serialize_sample_value("ok") == "ok"


def test_nonfinite_floats_become_null() -> None:
    assert serialize_sample_value(float("inf")) is None
    assert serialize_sample_value(float("nan")) is None
