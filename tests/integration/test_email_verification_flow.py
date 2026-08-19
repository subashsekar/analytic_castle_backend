import uuid

from fastapi.testclient import TestClient

from tests.conftest import VALID_TEST_PASSWORD, token_from_last_email

AUTH_PREFIX = "/api/v1/auth"


def test_email_verification_marks_user_verified(client: TestClient) -> None:
    email = f"verify-{uuid.uuid4().hex[:8]}@example.com"
    register = client.post(
        f"{AUTH_PREFIX}/register",
        json={
            "first_name": "Verify",
            "last_name": "User",
            "email": email,
            "password": VALID_TEST_PASSWORD,
        },
    )
    assert register.status_code == 201
    assert register.json()["is_verified"] is False
    raw_token = token_from_last_email()

    verify = client.post(f"{AUTH_PREFIX}/verify-email", json={"token": raw_token})
    assert verify.status_code == 200
    assert verify.json()["detail"] == "Email verified successfully"

    login = client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": VALID_TEST_PASSWORD},
    )
    assert login.status_code == 200
    me = client.get(
        f"{AUTH_PREFIX}/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )

    assert me.status_code == 200
    assert me.json()["is_verified"] is True
    assert raw_token not in register.text
    assert raw_token not in verify.text
