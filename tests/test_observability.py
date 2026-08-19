import asyncio
import json
import logging
import re
import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core.config import settings
from app.core.request_id import REQUEST_ID_HEADER, normalize_request_id
from app.main import unhandled_exception_handler

AUTH_PREFIX = "/api/v1/auth"
VALID_PASSWORD = "SecurePassword123!"


def test_request_id_is_generated_when_missing(client: TestClient) -> None:
    response = client.get("/health/live")

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id
    assert uuid.UUID(request_id)


def test_request_id_is_preserved_when_valid(client: TestClient) -> None:
    incoming = f"client-{uuid.uuid4()}"
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: incoming})

    assert response.headers[REQUEST_ID_HEADER] == incoming


def test_invalid_request_id_is_replaced(client: TestClient) -> None:
    incoming = "not a valid id / password=secret"
    response = client.get("/health/live", headers={REQUEST_ID_HEADER: incoming})

    request_id = response.headers[REQUEST_ID_HEADER]
    assert request_id != incoming
    assert "password" not in request_id
    assert " " not in request_id


def test_request_id_is_not_an_authentication_mechanism(client: TestClient) -> None:
    response = client.get(
        f"{AUTH_PREFIX}/me",
        headers={REQUEST_ID_HEADER: str(uuid.uuid4())},
    )

    assert response.status_code == 401
    assert response.headers[REQUEST_ID_HEADER]


def test_request_completion_is_logged_with_timing(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.request"):
        response = client.get("/health/live")

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "event=request_completed" in messages
    assert "method=GET" in messages
    assert "path=/health/live" in messages
    assert "status_code=200" in messages
    assert re.search(r"duration_ms=\d+", messages)
    assert any(
        getattr(record, "request_id", None) == response.headers[REQUEST_ID_HEADER]
        for record in caplog.records
    )


def test_request_logs_do_not_include_credentials(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    email = f"obs-{uuid.uuid4().hex[:8]}@example.com"
    with caplog.at_level(logging.INFO):
        register = client.post(
            f"{AUTH_PREFIX}/register",
            json={
                "first_name": "Obs",
                "last_name": "User",
                "email": email,
                "password": VALID_PASSWORD,
            },
        )
        login = client.post(
            f"{AUTH_PREFIX}/login",
            json={"email": email, "password": VALID_PASSWORD},
        )
        token = login.json()["access_token"]
        client.get(
            f"{AUTH_PREFIX}/me",
            headers={"Authorization": f"Bearer {token}"},
        )

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert register.status_code == 201
    assert login.status_code == 200
    assert VALID_PASSWORD not in messages
    assert token not in messages
    assert login.json()["refresh_token"] not in messages
    assert "Authorization=Bearer" not in messages
    assert settings.DATABASE_URL not in messages


def test_unhandled_exception_includes_request_id_and_hides_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "DEBUG", False)
    monkeypatch.setattr(settings, "APP_ENV", "production")
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "state": {"request_id": "diag-request-id"},
        }
    )

    with caplog.at_level(logging.ERROR):
        response = asyncio.run(
            unhandled_exception_handler(
                request,
                RuntimeError("database password=supersecret at /secret/path.sql"),
            )
        )
    body = json.loads(response.body)
    logged = " ".join(record.getMessage() for record in caplog.records)

    assert response.status_code == 500
    assert body["detail"] == "Internal server error"
    assert body["request_id"] == "diag-request-id"
    assert response.headers[REQUEST_ID_HEADER] == "diag-request-id"
    assert b"supersecret" not in response.body
    assert b"/secret/path.sql" not in response.body
    assert "error_type=RuntimeError" in logged
    assert "path=/health" in logged
    assert "request_id=diag-request-id" in logged
    assert "supersecret" not in logged


def test_debug_exception_details_are_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "DEBUG", True)
    monkeypatch.setattr(settings, "APP_ENV", "local")
    request = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/health",
            "raw_path": b"/health",
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
    )

    response = asyncio.run(
        unhandled_exception_handler(
            request,
            RuntimeError(
                "password=supersecret DATABASE_URL=postgresql://user:pass@db/app"
            ),
        )
    )
    body = json.loads(response.body)

    assert response.status_code == 500
    assert "supersecret" not in body["detail"]
    assert "postgresql://user:pass@db/app" not in body["detail"]
    assert "password=[REDACTED]" in body["detail"]
    assert body["request_id"]


def test_normalize_request_id_rejects_blank_and_oversized_values() -> None:
    generated = normalize_request_id("")
    oversized = normalize_request_id("a" * 129)

    assert generated
    assert generated != ""
    assert oversized != "a" * 129
    assert normalize_request_id("trace-123") == "trace-123"
