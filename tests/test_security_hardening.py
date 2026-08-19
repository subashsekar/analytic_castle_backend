import asyncio
import logging
import uuid

import jwt
import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.orm import Session
from starlette.requests import Request
from starlette.types import Message, Receive, Scope, Send

from app.core.config import Settings, settings
from app.core.logging import RedactingFilter, redact_secret
from app.core.middleware import RequestSizeLimitMiddleware
from app.core.rate_limit import (
    InMemoryRateLimiter,
    LoginProtection,
    login_protection,
    reset_rate_limiters,
)
from app.core.security import TOKEN_TYPE_ACCESS, create_access_token, hash_password
from app.db.models import Organization, User, UserRole, Workspace, WorkspaceMember
from app.enums import WorkspaceRole
from app.main import unhandled_exception_handler

AUTH_PREFIX = "/api/v1/auth"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"
ALLOWED_ORIGIN = "http://localhost:3000"
DISALLOWED_ORIGIN = "https://evil.example"


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
) -> tuple[dict[str, object], Response]:
    payload = _register_payload(**overrides)
    return payload, client.post(f"{AUTH_PREFIX}/register", json=payload)


def _login(
    client: TestClient, email: object, password: str = VALID_PASSWORD
) -> Response:
    return client.post(
        f"{AUTH_PREFIX}/login",
        json={"email": email, "password": password},
    )


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_user(db_session: Session) -> User:
    user = User(
        first_name="Test",
        last_name="User",
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _enable_rate_limits(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)
    reset_rate_limiters()


def test_rate_limit_allows_requests_under_the_limit(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(monkeypatch, RATE_LIMIT_REGISTER="3/minute")

    responses = [
        client.post(f"{AUTH_PREFIX}/register", json=_register_payload())
        for _ in range(3)
    ]

    assert [response.status_code for response in responses] == [201, 201, 201]


def test_rate_limit_returns_429_and_retry_after(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(monkeypatch, RATE_LIMIT_REGISTER="2/minute")

    assert (
        client.post(f"{AUTH_PREFIX}/register", json=_register_payload()).status_code
        == 201
    )
    assert (
        client.post(f"{AUTH_PREFIX}/register", json=_register_payload()).status_code
        == 201
    )
    limited = client.post(f"{AUTH_PREFIX}/register", json=_register_payload())

    assert limited.status_code == 429
    assert limited.json()["detail"] == "Too many requests"
    assert limited.headers["Retry-After"]
    assert "password" not in limited.text


def test_login_rate_limit_is_per_endpoint(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(
        monkeypatch,
        RATE_LIMIT_LOGIN="2/minute",
        LOGIN_MAX_FAILED_ATTEMPTS=100,
    )
    payload, response = _register(client)
    assert response.status_code == 201

    assert _login(client, payload["email"]).status_code == 200
    assert _login(client, payload["email"]).status_code == 200
    limited = _login(client, payload["email"])

    assert limited.status_code == 429
    assert limited.json()["detail"] == "Too many requests"


def test_repeated_failed_login_is_throttled_then_recovers(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(
        monkeypatch,
        RATE_LIMIT_LOGIN="100/minute",
        LOGIN_MAX_FAILED_ATTEMPTS=3,
        LOGIN_LOCKOUT_SECONDS=30,
    )
    payload, response = _register(client)
    assert response.status_code == 201

    clock = {"now": 1_000.0}
    login_protection._clock = lambda: clock["now"]

    failures = [
        _login(client, payload["email"], password="WrongPassword123!") for _ in range(3)
    ]
    throttled = _login(client, payload["email"], password="WrongPassword123!")
    still_locked = _login(client, payload["email"])

    assert [item.status_code for item in failures] == [401, 401, 401]
    assert throttled.status_code == 429
    assert still_locked.status_code == 429
    assert throttled.json()["detail"] == "Too many requests"

    clock["now"] = 1_040.0
    recovered = _login(client, payload["email"])

    assert recovered.status_code == 200
    assert recovered.json()["access_token"]


def test_login_throttle_does_not_enumerate_users(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(
        monkeypatch,
        RATE_LIMIT_LOGIN="100/minute",
        LOGIN_MAX_FAILED_ATTEMPTS=2,
        LOGIN_LOCKOUT_SECONDS=60,
    )
    payload, response = _register(client)
    assert response.status_code == 201

    existing = [
        _login(client, payload["email"], password="WrongPassword123!") for _ in range(3)
    ]
    missing = [
        _login(client, "missing@example.com", password="WrongPassword123!")
        for _ in range(3)
    ]

    assert [item.status_code for item in existing] == [401, 401, 429]
    assert [item.status_code for item in missing] == [401, 401, 429]
    assert existing[0].json() == missing[0].json()
    assert existing[2].json() == missing[2].json()


def test_cors_allows_configured_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_cors_rejects_unknown_origin(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": DISALLOWED_ORIGIN})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") != DISALLOWED_ORIGIN


def test_cors_preflight_allows_configured_origin(client: TestClient) -> None:
    allowed = client.options(
        f"{AUTH_PREFIX}/login",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    denied = client.options(
        f"{AUTH_PREFIX}/login",
        headers={
            "Origin": DISALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert denied.headers.get("access-control-allow-origin") != DISALLOWED_ORIGIN


def test_production_cors_strips_wildcard_when_credentials_enabled() -> None:
    configured = Settings(
        APP_ENV="production",
        DEBUG=False,
        DATABASE_URL=settings.DATABASE_URL,
        JWT_SECRET_KEY=settings.JWT_SECRET_KEY,
        CORS_ALLOWED_ORIGINS="*,https://app.example.com",
        CORS_ALLOW_CREDENTIALS=True,
    )

    assert "*" not in configured.cors_allowed_origins
    assert configured.cors_allowed_origins == ["https://app.example.com"]


def test_security_headers_are_present_without_hsts_in_development(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "geolocation=()" in response.headers["permissions-policy"]
    assert "strict-transport-security" not in response.headers


def test_hsts_is_enabled_only_in_production(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "APP_ENV", "production")

    response = client.get("/health")

    assert response.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains"
    )


def test_expired_invalid_and_malformed_jwts_are_rejected(client: TestClient) -> None:
    payload, response = _register(client)
    user_id = uuid.UUID(response.json()["id"])
    expired = jwt.encode(
        {
            "sub": str(user_id),
            "type": TOKEN_TYPE_ACCESS,
            "iat": 1,
            "exp": 2,
        },
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    missing = client.get(f"{AUTH_PREFIX}/me")
    malformed = client.get(f"{AUTH_PREFIX}/me", headers=_auth_header("not-a-jwt"))
    expired_response = client.get(f"{AUTH_PREFIX}/me", headers=_auth_header(expired))
    tokens = _login(client, payload["email"]).json()
    wrong_type = client.get(
        f"{AUTH_PREFIX}/me",
        headers=_auth_header(tokens["refresh_token"]),
    )

    assert missing.status_code == 401
    assert malformed.status_code == 401
    assert expired_response.status_code == 401
    assert wrong_type.status_code == 401


def test_auth_endpoints_reject_wrong_http_methods(client: TestClient) -> None:
    for path in ("/login", "/register", "/refresh", "/logout"):
        response = client.get(f"{AUTH_PREFIX}{path}")
        assert response.status_code == 405, path


def test_register_rejects_extra_fields_and_oversized_names(client: TestClient) -> None:
    extra = client.post(
        f"{AUTH_PREFIX}/register",
        json=_register_payload(role="SUPER_ADMIN"),
    )
    too_long = client.post(
        f"{AUTH_PREFIX}/register",
        json=_register_payload(first_name="J" * 256),
    )

    assert extra.status_code == 422
    assert too_long.status_code == 422


def test_oversized_request_body_is_rejected(client: TestClient) -> None:
    body = b'{"email":"a@example.com","password":"' + (b"A" * (1024 * 1024)) + b'"}'

    response = client.post(
        f"{AUTH_PREFIX}/login",
        content=body,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"


def test_unhandled_exception_hides_internal_details(
    monkeypatch: pytest.MonkeyPatch,
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
        }
    )

    response = asyncio.run(
        unhandled_exception_handler(
            request,
            RuntimeError("database password=supersecret at /secret/path.sql"),
        )
    )
    body = response.body

    assert response.status_code == 500
    assert b'"detail":"Internal server error"' in body
    assert b"request_id" in body
    assert b"supersecret" not in body
    assert b"/secret/path.sql" not in body


def test_cross_organization_and_workspace_access_is_denied(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    organization = client.post(
        ORG_PREFIX,
        headers=_auth_header(create_access_token(owner.id)),
        json={"name": "Private Org"},
    ).json()
    workspace = client.get(
        WS_PREFIX,
        headers=_auth_header(create_access_token(owner.id)),
        params={"organization_id": organization["id"]},
    ).json()[0]

    unauthenticated = client.get(f"{ORG_PREFIX}/{organization['id']}")
    foreign_org = client.get(
        f"{ORG_PREFIX}/{organization['id']}",
        headers=_auth_header(create_access_token(outsider.id)),
    )
    foreign_workspace = client.get(
        f"{WS_PREFIX}/{workspace['id']}",
        headers=_auth_header(create_access_token(outsider.id)),
    )
    foreign_members = client.get(
        f"{WS_PREFIX}/{workspace['id']}/members",
        headers=_auth_header(create_access_token(outsider.id)),
    )

    assert unauthenticated.status_code == 401
    assert foreign_org.status_code == 404
    assert foreign_workspace.status_code == 403
    assert foreign_members.status_code == 403


def test_responses_do_not_expose_sensitive_fields(client: TestClient) -> None:
    payload, register_response = _register(client)
    login_response = _login(client, payload["email"])
    me_response = client.get(
        f"{AUTH_PREFIX}/me",
        headers=_auth_header(login_response.json()["access_token"]),
    )

    forbidden = (
        "password_hash",
        '"password"',
        "reset_token",
        "verification_token",
    )
    for response in (register_response, login_response, me_response):
        text = response.text
        for item in forbidden:
            assert item not in text
        assert VALID_PASSWORD not in text
        body = response.json()
        assert "password" not in body
        assert "password_hash" not in body


def test_logs_redact_secrets(caplog: pytest.LogCaptureFixture) -> None:
    raw = (
        "Authorization=Bearer abc.def.ghi password=Secret123! "
        "refresh_token=raw-refresh verification_token=raw-verify"
    )
    logger = logging.getLogger("security-hardening-test")

    with caplog.at_level(logging.INFO):
        logger.info(raw)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Secret123!" not in messages
    assert "raw-refresh" not in messages
    assert "raw-verify" not in messages
    assert "abc.def.ghi" not in messages
    assert redact_secret("password=Secret123!") == "password=[REDACTED]"


def test_user_repr_omits_password_hash(db_session: Session) -> None:
    user = _create_user(db_session)

    rendered = repr(user)
    assert "password" not in rendered
    assert user.password_hash not in rendered


def test_workspace_member_responses_omit_password_hash(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = Organization(name="Org", slug=f"org-{uuid.uuid4().hex[:8]}")
    db_session.add(organization)
    db_session.flush()
    workspace = Workspace(
        organization_id=organization.id,
        name="Workspace",
        slug=f"ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    db_session.add(
        WorkspaceMember(
            workspace_id=workspace.id,
            user_id=owner.id,
            role=WorkspaceRole.OWNER,
        )
    )
    db_session.flush()

    response = client.get(
        f"{WS_PREFIX}/{workspace.id}/members",
        headers=_auth_header(create_access_token(owner.id)),
    )

    assert response.status_code == 200
    assert "password_hash" not in response.text
    assert VALID_PASSWORD not in response.text


def test_auth_rate_limits_cover_remaining_sensitive_endpoints(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(
        monkeypatch,
        RATE_LIMIT_REFRESH="2/minute",
        RATE_LIMIT_PASSWORD_RESET="2/minute",
        RATE_LIMIT_VERIFY_EMAIL_SEND="2/minute",
        RATE_LIMIT_CHANGE_PASSWORD="2/minute",
        RATE_LIMIT_REGISTER="100/minute",
        RATE_LIMIT_LOGIN="100/minute",
        LOGIN_MAX_FAILED_ATTEMPTS=100,
    )
    payload, response = _register(client)
    assert response.status_code == 201
    tokens = _login(client, payload["email"]).json()

    cases = (
        (
            f"{AUTH_PREFIX}/refresh",
            {"refresh_token": "not-a-valid-refresh-token"},
            None,
        ),
        (
            f"{AUTH_PREFIX}/forgot-password",
            {"email": payload["email"]},
            None,
        ),
        (
            f"{AUTH_PREFIX}/reset-password",
            {"token": "invalid-reset-token", "new_password": "AnotherPassword123!"},
            None,
        ),
        (
            f"{AUTH_PREFIX}/resend-verification",
            {"email": payload["email"]},
            None,
        ),
        (
            f"{AUTH_PREFIX}/verify-email/send",
            {"email": payload["email"]},
            None,
        ),
        (
            f"{AUTH_PREFIX}/change-password",
            {
                "current_password": VALID_PASSWORD,
                "new_password": "AnotherPassword123!",
            },
            _auth_header(tokens["access_token"]),
        ),
    )
    for path, body, headers in cases:
        reset_rate_limiters()
        first = client.post(path, json=body, headers=headers)
        second = client.post(path, json=body, headers=headers)
        limited = client.post(path, json=body, headers=headers)
        assert first.status_code != 429, path
        assert second.status_code != 429, path
        assert limited.status_code == 429, path
        assert limited.json()["detail"] == "Too many requests"
        assert limited.headers["Retry-After"]


def test_rate_limiter_evicts_idle_keys() -> None:
    clock = {"now": 0.0}
    limiter = InMemoryRateLimiter(clock=lambda: clock["now"])
    limiter.hit("old", 5, 60)
    assert "old" in limiter._windows

    clock["now"] = 4_000.0
    limiter._hits_since_evict = 63
    limiter.hit("new", 5, 60)

    assert "old" not in limiter._windows
    assert "new" in limiter._windows


def test_login_failure_counts_expire_after_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "LOGIN_MAX_FAILED_ATTEMPTS", 3)
    monkeypatch.setattr(settings, "LOGIN_LOCKOUT_SECONDS", 30)
    clock = {"now": 1_000.0}
    protection = LoginProtection(clock=lambda: clock["now"])

    protection.record_failure("127.0.0.1", "user@example.com")
    protection.record_failure("127.0.0.1", "user@example.com")
    clock["now"] = 1_031.0

    assert protection.allow("127.0.0.1", "user@example.com") is True
    protection.record_failure("127.0.0.1", "user@example.com")
    assert protection.allow("127.0.0.1", "user@example.com") is True


def test_chunked_body_without_content_length_is_rejected() -> None:
    async def _run() -> int:
        status_codes: list[int] = []

        async def inner(_scope: Scope, receive: Receive, send: Send) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_codes.append(int(message["status"]))

        chunks = [b"x" * 600_000, b"x" * 600_000]
        index = {"value": 0}

        async def receive() -> Message:
            current = index["value"]
            if current >= len(chunks):
                return {"type": "http.request", "body": b"", "more_body": False}
            index["value"] += 1
            return {
                "type": "http.request",
                "body": chunks[current],
                "more_body": current < len(chunks) - 1,
            }

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/v1/auth/login",
            "raw_path": b"/api/v1/auth/login",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }
        await RequestSizeLimitMiddleware(inner)(scope, receive, send)
        return status_codes[0]

    assert asyncio.run(_run()) == 413


def test_exception_tracebacks_and_database_url_are_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("security-hardening-exc")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise RuntimeError(
                "password=SuperSecret123! DATABASE_URL=postgresql://user:pass@db/app"
            )
        except RuntimeError:
            logger.exception("Unhandled error")

    record = caplog.records[-1]
    formatted = record.exc_text or ""
    assert "SuperSecret123!" not in formatted
    assert "postgresql://user:pass@db/app" not in formatted
    assert "password=[REDACTED]" in formatted
    assert "DATABASE_URL=[REDACTED]" in formatted
    assert (
        redact_secret("DATABASE_URL=postgresql://user:pass@db/app")
        == "DATABASE_URL=[REDACTED]"
    )
