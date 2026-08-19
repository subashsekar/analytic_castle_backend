from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from app.enums import ColumnSensitivity
from app.services.data_masking import (
    MASKED,
    REDACTED,
    mask_card,
    mask_email,
    mask_name,
    mask_phone,
    mask_value,
)


def test_email_masking() -> None:
    assert mask_email("john.doe@example.com") == "j***@example.com"
    assert mask_email("a@b.co") == "a***@b.co"
    assert mask_email(None) is None


def test_phone_masking() -> None:
    assert mask_phone("9876543210") == "******3210"
    assert mask_phone("+1 415-555-2671") == "*******2671"
    assert mask_phone(None) is None


def test_card_masking() -> None:
    assert mask_card("4111111111111111") == "************1111"
    assert mask_card("4111-1111-1111-1111") == "************1111"
    assert mask_card(None) is None


def test_name_masking() -> None:
    assert mask_name("John Doe") == "J*** D**"
    assert mask_name("John") == "J***"
    assert mask_name(None) is None


def test_secret_redaction() -> None:
    assert (
        mask_value(
            "hunter2",
            sensitivity=ColumnSensitivity.SECRET,
            column_name="password",
        )
        == REDACTED
    )
    assert (
        mask_value(
            "super-secret-token",
            sensitivity=ColumnSensitivity.SECRET,
            column_name="api_key",
        )
        == REDACTED
    )


def test_null_values_stay_null() -> None:
    assert mask_email(None) is None
    assert mask_phone(None) is None
    assert mask_name(None) is None
    assert mask_card(None) is None
    assert (
        mask_value(None, sensitivity=ColumnSensitivity.PII, column_name="email") is None
    )
    assert (
        mask_value(None, sensitivity=ColumnSensitivity.SECRET, column_name="password")
        is None
    )


def test_pii_masking_hides_complete_values() -> None:
    email = "john.doe@example.com"
    phone = "9876543210"
    name = "John Doe"
    card = "4111111111111111"
    secret = "hunter2"

    assert mask_email(email) != email
    assert "john.doe" not in str(mask_email(email))
    assert mask_phone(phone) != phone
    assert "987654" not in str(mask_phone(phone))
    assert mask_name(name) != name
    assert "ohn" not in str(mask_name(name))
    assert mask_card(card) != card
    assert "411111111111" not in str(mask_card(card))
    assert (
        mask_value(secret, sensitivity=ColumnSensitivity.SECRET, column_name="password")
        != secret
    )


def test_generic_pii_masks_dates() -> None:
    assert (
        mask_value(
            date(1990, 5, 1),
            sensitivity=ColumnSensitivity.PII,
            column_name="date_of_birth",
        )
        == MASKED
    )
    assert (
        mask_value(
            datetime(1990, 5, 1, tzinfo=UTC),
            sensitivity=ColumnSensitivity.PII,
            column_name="dob",
        )
        == MASKED
    )


def test_sensitive_values_are_masked() -> None:
    masked = mask_value(
        "4111111111111111",
        sensitivity=ColumnSensitivity.SENSITIVE,
        column_name="card_number",
    )
    assert masked == "************1111"


def test_public_values_pass_through() -> None:
    value = UUID("550e8400-e29b-41d4-a716-446655440000")
    assert (
        mask_value(value, sensitivity=ColumnSensitivity.PUBLIC, column_name="id")
        is value
    )
    assert mask_value(
        Decimal("12.50"),
        sensitivity=ColumnSensitivity.PUBLIC,
        column_name="amount",
    ) == Decimal("12.50")


def test_masking_does_not_emit_original_uuid_as_text_for_secrets() -> None:
    token = str(uuid4())
    assert (
        mask_value(token, sensitivity=ColumnSensitivity.SECRET, column_name="token")
        == REDACTED
    )
