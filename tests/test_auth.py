import logging
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
    hash_refresh_token,
    verify_password,
)
from app.db.models import RefreshToken, User, UserRole

VALID_PASSWORD = "SecurePassword123!"
AUTH_PREFIX = "/api/v1/auth"


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
) -> tuple[dict[str, object], dict]:
    payload = _register_payload(**overrides)
    response = client.post(f"{AUTH_PREFIX}/register", json=payload)
    return payload, response


def _login(client: TestClient, email: object, password: str = VALID_PASSWORD):
    return client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": password},
    )


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _encode_token(
    user_id: uuid.UUID,
    token_type: str,
    *,
    expired: bool = False,
    secret: str | None = None,
) -> str:
    now = datetime.now(UTC)
    exp = now - timedelta(minutes=5) if expired else now + timedelta(minutes=15)
    return jwt.encode(
        {
            "sub": str(user_id),
            "type": token_type,
            "iat": now,
            "exp": exp,
            "jti": str(uuid.uuid4()),
        },
        secret or settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def test_register_succeeds(client: TestClient) -> None:
    payload, response = _register(client)

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == payload["email"]
    assert body["first_name"] == "John"
    assert body["last_name"] == "Doe"
    assert body["role"] == UserRole.USER.value
    assert body["is_verified"] is False
    assert body["is_active"] is True
    assert "access_token" not in body
    assert "refresh_token" not in body


def test_register_rejects_duplicate_email(client: TestClient) -> None:
    _payload, first = _register(client, email="duplicate@example.com")
    assert first.status_code == 201

    _, second = _register(client, email="duplicate@example.com")
    assert second.status_code == 409
    assert second.json()["detail"] == "An account with this email already exists"

    _, mixed_case = _register(client, email="Duplicate@example.com")
    assert mixed_case.status_code == 409


def test_register_rejects_invalid_email(client: TestClient) -> None:
    _, response = _register(client, email="not-an-email")

    assert response.status_code == 422


def test_register_rejects_weak_password(client: TestClient) -> None:
    weak_passwords = [
        "short1!",
        "nouppercase1!",
        "NOLOWERCASE1!",
        "NoNumber!",
        "NoSpecial123",
    ]
    for password in weak_passwords:
        _, response = _register(client, password=password)
        assert response.status_code == 422, password


def test_register_hashes_password_and_omits_hash(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, response = _register(client)

    assert response.status_code == 201
    body = response.json()
    assert "password" not in body
    assert "password_hash" not in body

    user = db_session.get(User, uuid.UUID(body["id"]))
    assert user is not None
    assert user.password_hash != payload["password"]
    assert user.password_hash.startswith("$argon2")
    assert verify_password(VALID_PASSWORD, user.password_hash)


def test_register_assigns_default_role_and_status(
    client: TestClient,
    db_session: Session,
) -> None:
    _, response = _register(client)
    user = db_session.get(User, uuid.UUID(response.json()["id"]))

    assert user is not None
    assert user.role == UserRole.USER
    assert user.is_verified is False
    assert user.is_active is True


def test_login_succeeds(client: TestClient) -> None:
    payload, register_response = _register(client)
    response = _login(client, payload["email"])

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["user"]["id"] == register_response.json()["id"]
    assert body["user"]["email"] == payload["email"]
    assert "password" not in body
    assert "password_hash" not in body
    assert "password_hash" not in body["user"]


def test_login_rejects_incorrect_password(client: TestClient) -> None:
    payload, _response = _register(client)
    response = _login(client, payload["email"], password="WrongPassword123!")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_rejects_unknown_email(client: TestClient) -> None:
    response = _login(client, "missing@example.com")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_rejects_inactive_user(client: TestClient, db_session: Session) -> None:
    payload, register_response = _register(client)
    user = db_session.get(User, uuid.UUID(register_response.json()["id"]))
    assert user is not None
    user.is_active = False
    db_session.flush()

    response = _login(client, payload["email"])

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_generates_access_and_refresh_tokens(client: TestClient) -> None:
    payload, register_response = _register(client)
    response = _login(client, payload["email"])
    body = response.json()

    access_payload = jwt.decode(
        body["access_token"],
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )
    refresh_payload = jwt.decode(
        body["refresh_token"],
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
    )

    assert access_payload["type"] == TOKEN_TYPE_ACCESS
    assert refresh_payload["type"] == TOKEN_TYPE_REFRESH
    assert access_payload["sub"] == register_response.json()["id"]
    assert refresh_payload["sub"] == register_response.json()["id"]


def test_me_returns_authenticated_user(client: TestClient) -> None:
    payload, register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.get(
        f"{AUTH_PREFIX}/me", headers=_auth_header(tokens["access_token"])
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == register_response.json()["id"]
    assert body["email"] == payload["email"]
    assert "password_hash" not in body


def test_me_rejects_missing_token(client: TestClient) -> None:
    response = client.get(f"{AUTH_PREFIX}/me")

    assert response.status_code == 401


def test_me_rejects_invalid_token(client: TestClient) -> None:
    response = client.get(f"{AUTH_PREFIX}/me", headers=_auth_header("not-a-jwt"))

    assert response.status_code == 401


def test_me_rejects_expired_access_token(client: TestClient) -> None:
    _payload, register_response = _register(client)
    token = _encode_token(
        uuid.UUID(register_response.json()["id"]),
        TOKEN_TYPE_ACCESS,
        expired=True,
    )

    response = client.get(f"{AUTH_PREFIX}/me", headers=_auth_header(token))

    assert response.status_code == 401


def test_me_rejects_invalid_signature(client: TestClient) -> None:
    _payload, register_response = _register(client)
    token = _encode_token(
        uuid.UUID(register_response.json()["id"]),
        TOKEN_TYPE_ACCESS,
        secret="a-different-secret-that-is-long-enough",
    )

    response = client.get(f"{AUTH_PREFIX}/me", headers=_auth_header(token))

    assert response.status_code == 401


def test_me_rejects_refresh_token_as_access_token(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.get(
        f"{AUTH_PREFIX}/me",
        headers=_auth_header(tokens["refresh_token"]),
    )

    assert response.status_code == 401


def test_me_rejects_inactive_user(client: TestClient, db_session: Session) -> None:
    payload, register_response = _register(client)
    tokens = _login(client, payload["email"]).json()
    user = db_session.get(User, uuid.UUID(register_response.json()["id"]))
    assert user is not None
    user.is_active = False
    db_session.flush()

    response = client.get(
        f"{AUTH_PREFIX}/me", headers=_auth_header(tokens["access_token"])
    )

    assert response.status_code == 401


def test_refresh_returns_new_token_pair(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["refresh_token"] != tokens["refresh_token"]
    assert body["token_type"] == "bearer"


def test_refresh_rejects_expired_refresh_token(client: TestClient) -> None:
    _payload, register_response = _register(client)
    token = _encode_token(
        uuid.UUID(register_response.json()["id"]),
        TOKEN_TYPE_REFRESH,
        expired=True,
    )

    response = client.post(f"{AUTH_PREFIX}/refresh", json={"refresh_token": token})

    assert response.status_code == 401


def test_refresh_rejects_invalid_refresh_token(client: TestClient) -> None:
    response = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": "not-a-valid-refresh-token"},
    )

    assert response.status_code == 401


def test_refresh_rejects_access_token(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["access_token"]},
    )

    assert response.status_code == 401


def test_refresh_rejects_revoked_token(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    logout_response = client.post(
        f"{AUTH_PREFIX}/logout",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert logout_response.status_code == 200

    response = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert response.status_code == 401


def test_refresh_rotates_and_invalidates_old_token(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    first = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert first.status_code == 200
    rotated = first.json()

    reuse_old = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": tokens["refresh_token"]},
    )
    assert reuse_old.status_code == 401

    second = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": rotated["refresh_token"]},
    )
    assert second.status_code == 200
    assert second.json()["refresh_token"] != rotated["refresh_token"]


def test_logout_succeeds(client: TestClient) -> None:
    payload, _register_response = _register(client)
    tokens = _login(client, payload["email"]).json()

    response = client.post(
        f"{AUTH_PREFIX}/logout",
        json={"refresh_token": tokens["refresh_token"]},
    )

    assert response.status_code == 200
    assert response.json()["detail"] == "Successfully logged out"


def test_logout_rejects_invalid_token(client: TestClient) -> None:
    response = client.post(
        f"{AUTH_PREFIX}/logout",
        json={"refresh_token": "not-a-valid-refresh-token"},
    )

    assert response.status_code == 401


def test_raw_refresh_token_is_not_stored(
    client: TestClient,
    db_session: Session,
) -> None:
    payload, register_response = _register(client)
    tokens = _login(client, payload["email"]).json()
    raw_token = tokens["refresh_token"]
    user_id = uuid.UUID(register_response.json()["id"])

    stored = db_session.scalars(
        select(RefreshToken).where(RefreshToken.user_id == user_id)
    ).all()
    assert len(stored) == 1
    assert stored[0].token_hash != raw_token
    assert stored[0].token_hash == hash_refresh_token(raw_token)


def test_auth_responses_never_expose_password_hash(client: TestClient) -> None:
    payload, register_response = _register(client)
    login_response = _login(client, payload["email"])
    me_response = client.get(
        f"{AUTH_PREFIX}/me",
        headers=_auth_header(login_response.json()["access_token"]),
    )
    refresh_response = client.post(
        f"{AUTH_PREFIX}/refresh",
        json={"refresh_token": login_response.json()["refresh_token"]},
    )

    for response in (register_response, login_response, me_response, refresh_response):
        dumped = response.text
        assert "password_hash" not in dumped
        assert VALID_PASSWORD not in dumped
        body = response.json()
        assert "password" not in body
        assert "password_hash" not in body
        if "user" in body:
            assert "password_hash" not in body["user"]


def test_auth_endpoints_do_not_log_secrets(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    password = "SecurePassword123!"
    with caplog.at_level(logging.DEBUG):
        payload, _register_response = _register(client, password=password)
        login_response = _login(client, payload["email"], password=password)
        client.get(
            f"{AUTH_PREFIX}/me",
            headers=_auth_header(login_response.json()["access_token"]),
        )
        client.post(
            f"{AUTH_PREFIX}/refresh",
            json={"refresh_token": login_response.json()["refresh_token"]},
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert password not in messages
    assert "password_hash" not in messages
    assert login_response.json()["access_token"] not in messages
    assert login_response.json()["refresh_token"] not in messages


def test_register_does_not_return_verification_token(client: TestClient) -> None:
    _payload, response = _register(client)

    assert response.status_code == 201
    assert "token" not in response.json()
    assert response.json()["is_verified"] is False
