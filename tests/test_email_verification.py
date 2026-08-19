import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import generate_secure_token, hash_token
from app.db.models import EmailVerificationToken, User
from app.services.email_service import email_service

VALID_PASSWORD = "SecurePassword123!"
AUTH_PREFIX = "/api/v1/auth"
_TOKEN_RE = re.compile(r"token=([A-Za-z0-9_-]+)")


def _register_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "first_name": "John",
        "last_name": "Doe",
        "email": f"john-{uuid.uuid4().hex[:8]}@example.com",
        "password": VALID_PASSWORD,
    }
    payload.update(overrides)
    return payload


def _register(
    client: TestClient, **overrides: object
) -> tuple[dict[str, object], object]:
    payload = _register_payload(**overrides)
    response = client.post(f"{AUTH_PREFIX}/register", json=payload)
    return payload, response


def _token_from_last_email() -> str:
    assert email_service.outbox
    match = _TOKEN_RE.search(email_service.outbox[-1].body)
    assert match is not None
    return match.group(1)


def _backdate_verification_tokens(db_session: Session, user_id: uuid.UUID) -> None:
    now = datetime.now(UTC)
    tokens = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).all()
    for token in tokens:
        token.created_at = now - timedelta(minutes=5)
    db_session.flush()


def test_verification_token_is_hashed_on_register(
    client: TestClient,
    db_session: Session,
) -> None:
    _payload, response = _register(client)
    assert response.status_code == 201
    assert "token" not in response.json()
    assert _TOKEN_RE.search(response.text) is None

    user_id = uuid.UUID(response.json()["id"])
    raw_token = _token_from_last_email()
    stored = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).all()

    assert len(stored) == 1
    assert stored[0].token_hash != raw_token
    assert stored[0].token_hash == hash_token(raw_token)
    assert stored[0].used_at is None
    assert stored[0].expires_at > datetime.now(UTC)


def test_register_sends_verification_email_without_exposing_token(
    client: TestClient,
) -> None:
    payload, response = _register(client)

    assert response.status_code == 201
    assert response.json()["is_verified"] is False
    assert len(email_service.outbox) == 1
    message = email_service.outbox[0]
    assert message.to == payload["email"]
    assert message.sender == settings.EMAIL_FROM
    assert settings.FRONTEND_URL in message.body
    assert _token_from_last_email() not in response.text


def test_verify_email_succeeds(client: TestClient, db_session: Session) -> None:
    _payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    raw_token = _token_from_last_email()

    verify_response = client.post(
        f"{AUTH_PREFIX}/verify-email",
        json={"token": raw_token},
    )

    assert verify_response.status_code == 200
    assert verify_response.json()["detail"] == "Email verified successfully"
    user = db_session.get(User, user_id)
    assert user is not None
    assert user.is_verified is True
    stored = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).one()
    assert stored.used_at is not None


def test_verify_email_rejects_invalid_token(client: TestClient) -> None:
    _payload, response = _register(client)
    assert response.status_code == 201

    verify_response = client.post(
        f"{AUTH_PREFIX}/verify-email",
        json={"token": generate_secure_token()},
    )

    assert verify_response.status_code == 401
    assert verify_response.json()["detail"] == "Invalid or expired verification token"


def test_verify_email_rejects_expired_token(
    client: TestClient,
    db_session: Session,
) -> None:
    _payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    raw_token = _token_from_last_email()
    stored = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).one()
    stored.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    db_session.flush()

    verify_response = client.post(
        f"{AUTH_PREFIX}/verify-email",
        json={"token": raw_token},
    )

    assert verify_response.status_code == 401
    assert verify_response.json()["detail"] == "Invalid or expired verification token"
    user = db_session.get(User, user_id)
    assert user is not None
    assert user.is_verified is False


def test_verify_email_rejects_used_token(client: TestClient) -> None:
    _payload, _response = _register(client)
    raw_token = _token_from_last_email()

    first = client.post(f"{AUTH_PREFIX}/verify-email", json={"token": raw_token})
    second = client.post(f"{AUTH_PREFIX}/verify-email", json={"token": raw_token})

    assert first.status_code == 200
    assert second.status_code == 401
    assert second.json()["detail"] == "Invalid or expired verification token"


def test_verify_email_handles_already_verified_user(
    client: TestClient,
    db_session: Session,
) -> None:
    _payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    user = db_session.get(User, user_id)
    assert user is not None
    user.is_verified = True
    db_session.flush()

    verify_response = client.post(
        f"{AUTH_PREFIX}/verify-email",
        json={"token": _token_from_last_email()},
    )

    assert verify_response.status_code == 200
    assert verify_response.json()["detail"] == "Email is already verified"
    stored = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).one()
    assert stored.used_at is not None


def test_resend_verification_for_unverified_user(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    original = _token_from_last_email()
    _backdate_verification_tokens(db_session, user_id)

    resend = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": payload["email"]},
    )

    assert resend.status_code == 200
    assert "unverified" in resend.json()["detail"].lower()
    assert original not in resend.text

    tokens = db_session.scalars(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user_id)
    ).all()
    assert len(tokens) == 2
    used = [token for token in tokens if token.used_at is not None]
    unused = [token for token in tokens if token.used_at is None]
    assert len(used) == 1
    assert len(unused) == 1
    new_token = _token_from_last_email()
    assert new_token != original
    assert unused[0].token_hash == hash_token(new_token)
    assert used[0].token_hash == hash_token(original)


def test_resend_verification_verified_user_returns_generic_response(
    client: TestClient,
) -> None:
    payload, _response = _register(client)
    client.post(f"{AUTH_PREFIX}/verify-email", json={"token": _token_from_last_email()})

    resend = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": payload["email"]},
    )
    missing = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": "missing@example.com"},
    )

    assert resend.status_code == missing.status_code == 200
    assert resend.json() == missing.json()
    assert len(email_service.outbox) == 1


def test_verify_email_send_alias_matches_resend_verification(
    client: TestClient,
) -> None:
    payload, _response = _register(client)
    missing = client.post(
        f"{AUTH_PREFIX}/verify-email/send",
        json={"email": "missing@example.com"},
    )
    existing = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": payload["email"]},
    )
    aliased = client.post(
        f"{AUTH_PREFIX}/verify-email/send",
        json={"email": payload["email"]},
    )

    assert missing.status_code == existing.status_code == aliased.status_code == 200
    assert missing.json() == existing.json() == aliased.json()
    assert len(email_service.outbox) == 1


def test_resend_verification_unknown_email_does_not_enumerate(
    client: TestClient,
) -> None:
    response = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": "missing@example.com"},
    )

    assert response.status_code == 200
    assert "unverified" in response.json()["detail"].lower()
    assert email_service.outbox == []


def test_resend_verification_enforces_cooldown(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])

    immediate = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": payload["email"]},
    )
    assert immediate.status_code == 200
    assert len(email_service.outbox) == 1

    _backdate_verification_tokens(db_session, user_id)
    after_wait = client.post(
        f"{AUTH_PREFIX}/resend-verification",
        json={"email": payload["email"]},
    )
    assert after_wait.status_code == 200
    assert after_wait.json() == immediate.json()
    assert len(email_service.outbox) == 2


def test_previous_verification_token_cannot_be_used_after_resend(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    original = _token_from_last_email()
    _backdate_verification_tokens(db_session, user_id)
    client.post(f"{AUTH_PREFIX}/resend-verification", json={"email": payload["email"]})

    reuse = client.post(f"{AUTH_PREFIX}/verify-email", json={"token": original})
    assert reuse.status_code == 401

    current = _token_from_last_email()
    verify = client.post(f"{AUTH_PREFIX}/verify-email", json={"token": current})
    assert verify.status_code == 200


def test_verification_tokens_are_not_logged(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        _payload, _response = _register(client)
        raw_token = _token_from_last_email()
        client.post(f"{AUTH_PREFIX}/verify-email", json={"token": raw_token})

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert raw_token not in messages
    assert hash_token(raw_token) not in messages
