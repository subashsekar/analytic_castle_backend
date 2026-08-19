import logging
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.core.config import settings
from app.core.security import (
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
    InvalidTokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_secure_token,
    hash_password,
    hash_refresh_token,
    hash_token,
    tokens_match,
    verify_password,
)


def test_hash_password_is_not_plaintext() -> None:
    password = "SecurePassword123!"
    password_hash = hash_password(password)

    assert password_hash != password
    assert password_hash.startswith("$argon2")


def test_verify_password_accepts_correct_password() -> None:
    password = "SecurePassword123!"
    password_hash = hash_password(password)

    assert verify_password(password, password_hash) is True


def test_verify_password_rejects_incorrect_password() -> None:
    password_hash = hash_password("SecurePassword123!")

    assert verify_password("WrongPassword123!", password_hash) is False


def test_verify_password_rejects_invalid_hash() -> None:
    assert verify_password("SecurePassword123!", "not-a-valid-hash") is False


def test_access_and_refresh_tokens_have_distinct_types() -> None:
    user_id = uuid.uuid4()
    access_token = create_access_token(user_id)
    refresh_token, expires_at = create_refresh_token(user_id)

    access_payload = decode_token(access_token, expected_type=TOKEN_TYPE_ACCESS)
    refresh_payload = decode_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)

    assert access_payload["sub"] == str(user_id)
    assert access_payload["type"] == TOKEN_TYPE_ACCESS
    assert refresh_payload["sub"] == str(user_id)
    assert refresh_payload["type"] == TOKEN_TYPE_REFRESH
    assert expires_at > datetime.now(UTC)


def test_decode_token_rejects_wrong_type() -> None:
    token = create_access_token(uuid.uuid4())

    with pytest.raises(InvalidTokenError):
        decode_token(token, expected_type=TOKEN_TYPE_REFRESH)


def test_decode_token_rejects_expired_token() -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": TOKEN_TYPE_ACCESS,
            "iat": now - timedelta(hours=2),
            "exp": now - timedelta(hours=1),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_token(token, expected_type=TOKEN_TYPE_ACCESS)


def test_decode_token_rejects_invalid_signature() -> None:
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "type": TOKEN_TYPE_ACCESS,
            "iat": datetime.now(UTC),
            "exp": datetime.now(UTC) + timedelta(minutes=15),
        },
        "a-different-secret-that-is-long-enough",
        algorithm=settings.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_token(token, expected_type=TOKEN_TYPE_ACCESS)


def test_refresh_token_hash_is_not_raw_token() -> None:
    token, _expires_at = create_refresh_token(uuid.uuid4())
    token_hash = hash_refresh_token(token)

    assert token_hash != token
    assert len(token_hash) == 64


def test_generate_secure_token_is_unpredictable() -> None:
    first = generate_secure_token()
    second = generate_secure_token()

    assert first != second
    assert len(first) >= 32


def test_hash_token_is_not_raw_token() -> None:
    token = generate_secure_token()
    token_hash = hash_token(token)

    assert token_hash != token
    assert len(token_hash) == 64
    assert tokens_match(token_hash, token) is True
    assert tokens_match(token_hash, generate_secure_token()) is False


def test_decode_token_rejects_invalid_subject() -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": "not-a-uuid",
            "type": TOKEN_TYPE_ACCESS,
            "iat": now,
            "exp": now + timedelta(minutes=15),
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    with pytest.raises(InvalidTokenError):
        decode_token(token, expected_type=TOKEN_TYPE_ACCESS)


def test_decode_token_rejects_none_algorithm() -> None:
    token = (
        "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0."
        "eyJzdWIiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDAiLCJ0eXBlIjoiYWNjZXNzIn0."
    )

    with pytest.raises(InvalidTokenError):
        decode_token(token, expected_type=TOKEN_TYPE_ACCESS)


def test_access_token_payload_has_no_sensitive_claims() -> None:
    user_id = uuid.uuid4()
    payload = decode_token(
        create_access_token(user_id), expected_type=TOKEN_TYPE_ACCESS
    )

    assert set(payload) <= {"sub", "type", "iat", "exp"}
    assert "password" not in payload
    assert "password_hash" not in payload


def test_password_helpers_do_not_log_secrets(caplog: pytest.LogCaptureFixture) -> None:
    password = "SecurePassword123!"
    with caplog.at_level(logging.DEBUG):
        password_hash = hash_password(password)
        verify_password(password, password_hash)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert password not in messages
    assert password_hash not in messages
