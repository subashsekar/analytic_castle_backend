import logging
import re
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import generate_secure_token, hash_token, verify_password
from app.db.models import PasswordResetToken, RefreshToken, User
from app.services.email_service import email_service

VALID_PASSWORD = "SecurePassword123!"
NEW_PASSWORD = "NewSecurePassword123!"
AUTH_PREFIX = "/api/v1/auth"
FORGOT_DETAIL = "If the account exists, a password reset link has been sent."
_TOKEN_RE = re.compile(r"token=([A-Za-z0-9_-]+)")


def _register(
    client: TestClient, **overrides: object
) -> tuple[dict[str, object], object]:
    payload: dict[str, object] = {
        "first_name": "John",
        "last_name": "Doe",
        "email": f"john-{uuid.uuid4().hex[:8]}@example.com",
        "password": VALID_PASSWORD,
    }
    payload.update(overrides)
    response = client.post(f"{AUTH_PREFIX}/register", json=payload)
    return payload, response


def _login(client: TestClient, email: object, password: str = VALID_PASSWORD):
    return client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": password},
    )


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _token_from_last_email() -> str:
    assert email_service.outbox
    match = _TOKEN_RE.search(email_service.outbox[-1].body)
    assert match is not None
    return match.group(1)


def test_forgot_password_existing_email_creates_hashed_token(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])

    forgot = client.post(
        f"{AUTH_PREFIX}/forgot-password", json={"email": payload["email"]}
    )

    assert forgot.status_code == 200
    assert forgot.json()["detail"] == FORGOT_DETAIL
    assert "token" not in forgot.json()
    raw_token = _token_from_last_email()
    assert raw_token not in forgot.text

    stored = db_session.scalars(
        select(PasswordResetToken).where(PasswordResetToken.user_id == user_id)
    ).all()
    assert len(stored) == 1
    assert stored[0].token_hash != raw_token
    assert stored[0].token_hash == hash_token(raw_token)
    assert stored[0].used_at is None
    assert stored[0].expires_at > datetime.now(UTC)


def test_forgot_password_unknown_email_returns_generic_response(
    client: TestClient,
    db_session: Session,
) -> None:
    response = client.post(
        f"{AUTH_PREFIX}/forgot-password",
        json={"email": "missing@example.com"},
    )

    assert response.status_code == 200
    assert response.json()["detail"] == FORGOT_DETAIL
    assert email_service.outbox == []
    assert db_session.scalars(select(PasswordResetToken)).all() == []


def test_forgot_password_does_not_enumerate_accounts(client: TestClient) -> None:
    payload, _response = _register(client)
    existing = client.post(
        f"{AUTH_PREFIX}/forgot-password",
        json={"email": payload["email"]},
    )
    missing = client.post(
        f"{AUTH_PREFIX}/forgot-password",
        json={"email": "missing@example.com"},
    )

    assert existing.status_code == missing.status_code == 200
    assert existing.json() == missing.json()


def test_reset_password_succeeds_and_revokes_refresh_tokens(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    login = _login(client, payload["email"]).json()
    client.post(f"{AUTH_PREFIX}/forgot-password", json={"email": payload["email"]})
    raw_token = _token_from_last_email()
    user = db_session.get(User, user_id)
    assert user is not None
    old_hash = user.password_hash

    reset = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": raw_token, "new_password": NEW_PASSWORD},
    )

    assert reset.status_code == 200
    assert reset.json()["detail"] == "Password has been reset successfully"
    assert "password" not in reset.json()
    assert "password_hash" not in reset.text

    user = db_session.get(User, user_id)
    assert user is not None
    assert user.password_hash != old_hash
    assert verify_password(NEW_PASSWORD, user.password_hash)
    assert not verify_password(VALID_PASSWORD, user.password_hash)

    stored = db_session.scalars(
        select(PasswordResetToken).where(PasswordResetToken.user_id == user_id)
    ).one()
    assert stored.used_at is not None

    refresh_tokens = db_session.scalars(
        select(RefreshToken).where(RefreshToken.user_id == user_id)
    ).all()
    assert refresh_tokens
    assert all(token.revoked_at is not None for token in refresh_tokens)

    reuse_refresh = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": login["refresh_token"]},
    )
    assert reuse_refresh.status_code == 401

    old_login = _login(client, payload["email"])
    new_login = _login(client, payload["email"], password=NEW_PASSWORD)
    assert old_login.status_code == 401
    assert new_login.status_code == 200


def test_reset_password_rejects_invalid_token(client: TestClient) -> None:
    _payload, _response = _register(client)

    response = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": generate_secure_token(), "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired reset token"


def test_reset_password_rejects_expired_token(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    client.post(f"{AUTH_PREFIX}/forgot-password", json={"email": payload["email"]})
    raw_token = _token_from_last_email()
    stored = db_session.scalars(
        select(PasswordResetToken).where(PasswordResetToken.user_id == user_id)
    ).one()
    stored.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    db_session.flush()

    reset = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": raw_token, "new_password": NEW_PASSWORD},
    )

    assert reset.status_code == 401
    user = db_session.get(User, user_id)
    assert user is not None
    assert verify_password(VALID_PASSWORD, user.password_hash)


def test_reset_password_rejects_used_token(client: TestClient) -> None:
    payload, _response = _register(client)
    client.post(f"{AUTH_PREFIX}/forgot-password", json={"email": payload["email"]})
    raw_token = _token_from_last_email()

    first = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": raw_token, "new_password": NEW_PASSWORD},
    )
    second = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": raw_token, "new_password": "AnotherPassword123!"},
    )

    assert first.status_code == 200
    assert second.status_code == 401
    assert second.json()["detail"] == "Invalid or expired reset token"


def test_reset_password_rejects_weak_password(client: TestClient) -> None:
    payload, _response = _register(client)
    client.post(f"{AUTH_PREFIX}/forgot-password", json={"email": payload["email"]})

    response = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": _token_from_last_email(), "new_password": "weak"},
    )

    assert response.status_code == 422


def test_change_password_succeeds_and_revokes_refresh_tokens(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    tokens = _login(client, payload["email"]).json()
    user = db_session.get(User, user_id)
    assert user is not None
    old_hash = user.password_hash

    change = client.post(
        f"{AUTH_PREFIX}/change-password",
        headers=_auth_header(tokens["access_token"]),
        json={"current_password": VALID_PASSWORD, "new_password": NEW_PASSWORD},
    )

    assert change.status_code == 200
    assert change.json()["detail"] == "Password changed successfully"
    assert "password_hash" not in change.text

    user = db_session.get(User, user_id)
    assert user is not None
    assert user.password_hash != old_hash
    assert verify_password(NEW_PASSWORD, user.password_hash)

    refresh_tokens = db_session.scalars(
        select(RefreshToken).where(RefreshToken.user_id == user_id)
    ).all()
    assert refresh_tokens
    assert all(token.revoked_at is not None for token in refresh_tokens)

    reuse_refresh = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert reuse_refresh.status_code == 401
    assert _login(client, payload["email"]).status_code == 401
    assert _login(client, payload["email"], password=NEW_PASSWORD).status_code == 200


def test_change_password_rejects_incorrect_current_password(client: TestClient) -> None:
    payload, _response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.post(
        f"{AUTH_PREFIX}/change-password",
        headers=_auth_header(tokens["access_token"]),
        json={"current_password": "WrongPassword123!", "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect current password"


def test_change_password_rejects_weak_new_password(client: TestClient) -> None:
    payload, _response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.post(
        f"{AUTH_PREFIX}/change-password",
        headers=_auth_header(tokens["access_token"]),
        json={"current_password": VALID_PASSWORD, "new_password": "nouppercase1!"},
    )

    assert response.status_code == 422


def test_change_password_requires_authentication(client: TestClient) -> None:
    response = client.post(
        f"{AUTH_PREFIX}/change-password",
        json={"current_password": VALID_PASSWORD, "new_password": NEW_PASSWORD},
    )

    assert response.status_code == 401


def test_password_endpoints_do_not_expose_or_log_secrets(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.DEBUG):
        payload, _response = _register(client)
        tokens = _login(client, payload["email"]).json()
        forgot = client.post(
            f"{AUTH_PREFIX}/forgot-password",
            json={"email": payload["email"]},
        )
        raw_token = _token_from_last_email()
        reset = client.post(
            f"{AUTH_PREFIX}/reset-password",
            json={"token": raw_token, "new_password": NEW_PASSWORD},
        )
        login = _login(client, payload["email"], password=NEW_PASSWORD)
        change = client.post(
            f"{AUTH_PREFIX}/change-password",
            headers=_auth_header(login.json()["access_token"]),
            json={
                "current_password": NEW_PASSWORD,
                "new_password": "AnotherPassword123!",
            },
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert raw_token not in messages
    assert VALID_PASSWORD not in messages
    assert NEW_PASSWORD not in messages
    assert "password_hash" not in messages
    for response in (forgot, reset, change):
        assert "password_hash" not in response.text
        assert raw_token not in response.text
        assert tokens["refresh_token"] not in response.text
