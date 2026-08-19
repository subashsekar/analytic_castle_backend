from __future__ import annotations

from app.enums import ColumnSensitivity
from app.services.pii_detection import (
    MaskKind,
    classify_column,
    detect_value_pii,
    mask_kind_for_column,
)


def test_email_column_is_pii() -> None:
    assert classify_column("email") is ColumnSensitivity.PII
    assert classify_column("user_email") is ColumnSensitivity.PII
    assert classify_column("email_address") is ColumnSensitivity.PII
    assert mask_kind_for_column("email") is MaskKind.EMAIL


def test_phone_column_is_pii() -> None:
    assert classify_column("phone") is ColumnSensitivity.PII
    assert classify_column("mobile") is ColumnSensitivity.PII
    assert classify_column("phone_number") is ColumnSensitivity.PII
    assert mask_kind_for_column("mobile") is MaskKind.PHONE


def test_name_column_is_pii() -> None:
    assert classify_column("first_name") is ColumnSensitivity.PII
    assert classify_column("last_name") is ColumnSensitivity.PII
    assert classify_column("full_name") is ColumnSensitivity.PII
    assert classify_column("firstName") is ColumnSensitivity.PII
    assert mask_kind_for_column("full_name") is MaskKind.NAME


def test_address_column_is_pii() -> None:
    assert classify_column("address") is ColumnSensitivity.PII
    assert classify_column("street_address") is ColumnSensitivity.PII


def test_ssn_like_column_is_pii() -> None:
    assert classify_column("ssn") is ColumnSensitivity.PII
    assert classify_column("aadhaar") is ColumnSensitivity.PII
    assert classify_column("pan") is ColumnSensitivity.PII
    assert classify_column("passport") is ColumnSensitivity.PII
    assert classify_column("date_of_birth") is ColumnSensitivity.PII
    assert classify_column("dob") is ColumnSensitivity.PII


def test_password_and_token_are_secret() -> None:
    assert classify_column("password") is ColumnSensitivity.SECRET
    assert classify_column("secret") is ColumnSensitivity.SECRET
    assert classify_column("token") is ColumnSensitivity.SECRET
    assert classify_column("api_key") is ColumnSensitivity.SECRET
    assert classify_column("access_token") is ColumnSensitivity.SECRET
    assert classify_column("refresh_token") is ColumnSensitivity.SECRET
    assert mask_kind_for_column("password") is MaskKind.SECRET


def test_normal_id_is_public() -> None:
    assert classify_column("id") is ColumnSensitivity.PUBLIC
    assert classify_column("customer_id") is ColumnSensitivity.PUBLIC
    assert classify_column("order_id") is ColumnSensitivity.PUBLIC
    assert classify_column("country_code") is ColumnSensitivity.PUBLIC
    assert classify_column("quantity") is ColumnSensitivity.PUBLIC
    assert classify_column("created_at") is ColumnSensitivity.PUBLIC


def test_description_can_mark_pii() -> None:
    assert (
        classify_column("contact", description="Primary email for the customer")
        is ColumnSensitivity.PII
    )


def test_short_patterns_do_not_false_positive() -> None:
    assert classify_column("company") is ColumnSensitivity.PUBLIC
    assert classify_column("japan") is ColumnSensitivity.PUBLIC
    assert classify_column("tokenizer") is ColumnSensitivity.PUBLIC
    assert classify_column("panel") is ColumnSensitivity.PUBLIC


def test_value_level_email_phone_and_card() -> None:
    assert detect_value_pii("john.doe@example.com") is MaskKind.EMAIL
    assert detect_value_pii("9876543210") is MaskKind.PHONE
    assert detect_value_pii("4111111111111111") is MaskKind.CARD
    assert detect_value_pii("4111-1111-1111-1111") is MaskKind.CARD


def test_value_level_ignores_ordinary_ids_and_text() -> None:
    assert detect_value_pii(42) is None
    assert detect_value_pii(1234567890) is None
    assert detect_value_pii("ORD-1001") is None
    assert detect_value_pii("not an email") is None
    assert detect_value_pii("123") is None
    assert detect_value_pii("please email me later") is None
    assert detect_value_pii("4111111111111112") is None
