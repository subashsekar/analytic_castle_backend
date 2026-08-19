import uuid

from fastapi.testclient import TestClient

from tests.conftest import VALID_TEST_PASSWORD, token_from_last_email

AUTH_PREFIX = "/api/v1/auth"
NEW_PASSWORD = "NewSecurePassword123!"


def test_forgot_password_reset_and_login_with_new_password(client: TestClient) -> None:
    email = f"reset-{uuid.uuid4().hex[:8]}@example.com"
    register = client.post(
        f"{AUTH_PREFIX}/register",
        json={
            "first_name": "Reset",
            "last_name": "User",
            "email": email,
            "password": VALID_TEST_PASSWORD,
        },
    )
    assert register.status_code == 201

    forgot = client.post(f"{AUTH_PREFIX}/forgot-password", json={"email": email})
    assert forgot.status_code == 200
    assert "token" not in forgot.json()
    raw_token = token_from_last_email()

    reset = client.post(
        f"{AUTH_PREFIX}/reset-password",
        json={"token": raw_token, "new_password": NEW_PASSWORD},
    )
    assert reset.status_code == 200

    old_login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": VALID_TEST_PASSWORD},
    )
    new_login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": NEW_PASSWORD},
    )
    me = client.get(
        f"{AUTH_PREFIX}/me",
        headers={"Authorization": f"Bearer {new_login.json()['access_token']}"},
    )

    assert old_login.status_code == 401
    assert new_login.status_code == 200
    assert me.status_code == 200
    assert me.json()["email"] == email
    assert raw_token not in reset.text
    assert NEW_PASSWORD not in reset.text
