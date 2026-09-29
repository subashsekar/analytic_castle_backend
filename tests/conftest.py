import asyncio
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Coroutine, Generator
from pathlib import Path
from typing import TypeVar

# Local PostgreSQL fallbacks for tests when DATABASE_URL is unset.
# Same non-secret local role as .env.example — not a production credential.
# Do not inherit a developer .env DATABASE_URL (it may point at a remote host).
_LOCAL_TEST_DATABASE_URL = (
    "postgresql+psycopg://analyticcastle:analyticcastle@127.0.0.1:5432/analyticcastle"
)
_LOCAL_TEST_FALLBACK_DATABASE_URL = (
    "postgresql+psycopg://analyticcastle:analyticcastle@127.0.0.1:5432/postgres"
)
_DOCKER_TEST_DATABASE_URL = (
    "postgresql+psycopg://analyticcastle:analyticcastle@127.0.0.1:5433/analyticcastle"
)
_DOCKER_CONTAINER = "analyticcastle-pytest-pg"


def _psycopg_dsn(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def _database_is_ready(sqlalchemy_url: str) -> bool:
    import psycopg

    try:
        with psycopg.connect(
            _psycopg_dsn(sqlalchemy_url),
            connect_timeout=2,
        ) as connection:
            connection.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 — readiness probe must not raise
        return False


def _ensure_database_exists(sqlalchemy_url: str) -> bool:
    from urllib.parse import urlparse

    import psycopg

    if _database_is_ready(sqlalchemy_url):
        return True

    parsed = urlparse(_psycopg_dsn(sqlalchemy_url))
    db_name = parsed.path.lstrip("/")
    if not db_name:
        return False
    admin_path = "/postgres" if db_name else parsed.path
    admin_dsn = parsed._replace(path=admin_path).geturl()
    try:
        with psycopg.connect(admin_dsn, connect_timeout=2) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,),
                )
                if cursor.fetchone() is None:
                    cursor.execute(f'CREATE DATABASE "{db_name}"')
        return _database_is_ready(sqlalchemy_url)
    except Exception:  # noqa: BLE001 — bootstrap helper must not raise
        return False


def _configured_remote_database_url() -> str | None:
    """Read TEST_DATABASE_URL or DATABASE_URL from the project .env file."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return None
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values.get("TEST_DATABASE_URL") or values.get("DATABASE_URL")


def _start_docker_test_postgres() -> bool:
    try:
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", _DOCKER_CONTAINER],
            check=False,
            capture_output=True,
            text=True,
        )
        if inspect.returncode == 0:
            if inspect.stdout.strip() != "true":
                subprocess.run(
                    ["docker", "start", _DOCKER_CONTAINER],
                    check=True,
                    capture_output=True,
                    text=True,
                )
        else:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    _DOCKER_CONTAINER,
                    "-e",
                    "POSTGRES_USER=analyticcastle",
                    "-e",
                    "POSTGRES_PASSWORD=analyticcastle",
                    "-e",
                    "POSTGRES_DB=analyticcastle",
                    "-p",
                    "5433:5432",
                    "postgres:16-alpine",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    for _ in range(30):
        if _database_is_ready(_DOCKER_TEST_DATABASE_URL):
            return True
        time.sleep(1)
    return False


def _configure_test_database_url() -> None:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        os.environ["DATABASE_URL"] = explicit
        return
    if _database_is_ready(_LOCAL_TEST_DATABASE_URL):
        os.environ["DATABASE_URL"] = _LOCAL_TEST_DATABASE_URL
        return
    if _ensure_database_exists(_LOCAL_TEST_DATABASE_URL):
        os.environ["DATABASE_URL"] = _LOCAL_TEST_DATABASE_URL
        return
    remote_url = _configured_remote_database_url()
    if remote_url and _database_is_ready(remote_url):
        os.environ["DATABASE_URL"] = remote_url
        return
    if _database_is_ready(_LOCAL_TEST_FALLBACK_DATABASE_URL):
        os.environ["DATABASE_URL"] = _LOCAL_TEST_FALLBACK_DATABASE_URL
        return
    if _database_is_ready(_DOCKER_TEST_DATABASE_URL) or _start_docker_test_postgres():
        os.environ["DATABASE_URL"] = _DOCKER_TEST_DATABASE_URL
        return
    if _ensure_database_exists(_DOCKER_TEST_DATABASE_URL):
        os.environ["DATABASE_URL"] = _DOCKER_TEST_DATABASE_URL
        return
    current = os.environ.get("DATABASE_URL")
    if current and _database_is_ready(current):
        return
    os.environ["DATABASE_URL"] = _LOCAL_TEST_DATABASE_URL


_configure_test_database_url()
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-jwt-secret-key-not-for-production",
)
# 32-byte AES-256 test key as hex. Not a production secret.
os.environ.setdefault(
    "DATA_SOURCE_ENCRYPTION_KEY",
    "00112233445566778899aabbccddeeffffeeddccbbaa99887766554433221100",
)
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("EMAIL_PROVIDER", "console")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-fake-openrouter-key-not-real")
if not os.environ.get("LLM_API_KEY", "").strip():
    os.environ["LLM_API_KEY"] = "sk-test-fake-llm-key-not-real"
if not os.environ.get("OPENROUTER_API_KEY", "").strip():
    os.environ["OPENROUTER_API_KEY"] = "sk-test-fake-openrouter-key-not-real"
os.environ.setdefault(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.connectors.postgresql import build_postgresql_connector
from app.connectors.registry import register_connector, unregister_connector
from app.core.rate_limit import reset_rate_limiters
from app.core.security import create_access_token, hash_password
from app.db.models import (  # noqa: F401
    AgentState,
    AnalysisSession,
    ConversationContext,
    DataSource,
    DataSourceColumn,
    DataSourceConnection,
    DataSourceMetadataSync,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
    EmailVerificationToken,
    Organization,
    PasswordResetToken,
    RefreshToken,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
)
from app.db.session import Base, engine, get_db
from app.enums import DataSourceType, WorkspaceRole
from app.main import app
from app.services.email_service import email_service
from app.services.metadata_sync import reset_metadata_sync_locks
from tests.authorization_routes import router as authz_test_router

VALID_TEST_PASSWORD = "SecurePassword123!"
_TOKEN_RE = re.compile(r"token=([A-Za-z0-9_-]+)")
_T = TypeVar("_T")


_INTEGRATION_DIRS = ("integration",)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if any(part in _INTEGRATION_DIRS for part in item.path.parts):
            item.add_marker(pytest.mark.integration)


def run_async(coro: Coroutine[object, object, _T]) -> _T:
    """Run a coroutine on a psycopg-compatible event loop."""
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(coro)
    return asyncio.run(coro)


def _reset_connector_registry() -> None:
    for source_type in DataSourceType:
        unregister_connector(source_type)
    register_connector(DataSourceType.POSTGRESQL, build_postgresql_connector)


@pytest.fixture(autouse=True)
def restore_connector_registry() -> Generator[None, None, None]:
    _reset_connector_registry()
    yield
    _reset_connector_registry()


@pytest.fixture(autouse=True)
def clear_email_outbox() -> Generator[None, None, None]:
    email_service.outbox.clear()
    yield
    email_service.outbox.clear()


@pytest.fixture(autouse=True)
def clear_rate_limiters() -> Generator[None, None, None]:
    reset_rate_limiters()
    yield
    reset_rate_limiters()


@pytest.fixture(autouse=True)
def clear_metadata_sync_locks() -> Generator[None, None, None]:
    reset_metadata_sync_locks()
    yield
    reset_metadata_sync_locks()


class RealNetworkBlockedError(RuntimeError):
    """Raised when a test tries to reach a real HTTP endpoint (LLM, OpenRouter, ...)."""


@pytest.fixture(autouse=True)
def block_real_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail fast instead of calling real LLM/OpenRouter endpoints.

    Only the network-backed httpx transports are blocked; ``httpx.MockTransport``
    and Starlette's ``TestClient`` transport keep working.
    """
    import httpx

    def _blocked(self: object, request: httpx.Request) -> httpx.Response:
        del self
        raise httpx.ConnectError(
            f"real network access is disabled in tests: {request.url.host}",
            request=request,
        )

    async def _blocked_async(self: object, request: httpx.Request) -> httpx.Response:
        return _blocked(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked_async)


@pytest.fixture(autouse=True)
def fast_ai_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound retries, backoff, and agent budgets so a failure can never stall a test.

    Tests that exercise retry/timeout behavior set their own values explicitly.
    """
    from app.core.config import settings

    for name, value in {
        "LLM_MAX_RETRIES": 0,
        "LLM_TIMEOUT_SECONDS": 5.0,
        "LLM_RETRY_BASE_BACKOFF_SECONDS": 0.01,
        "LLM_RETRY_MAX_BACKOFF_SECONDS": 0.05,
        "AI_CHAT_PHASE8_BUDGET_SECONDS": 20.0,
        "AI_CHAT_PHASE8_AGENT_TIMEOUT_SECONDS": 10.0,
        "AI_CHAT_PHASE8_RCA_TIMEOUT_SECONDS": 15.0,
    }.items():
        monkeypatch.setattr(settings, name, value)


@pytest.fixture(scope="session")
def database_schema() -> None:
    """Create tables once per run; per-test isolation comes from the rolled-back transaction."""
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def db_session(database_schema: None) -> Generator[Session, None, None]:
    del database_schema
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def authz_client(db_session: Session) -> Generator[TestClient, None, None]:
    authz_app = FastAPI()
    authz_app.include_router(authz_test_router)

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    authz_app.dependency_overrides[get_db] = override_get_db
    with TestClient(authz_app) as test_client:
        yield test_client
    authz_app.dependency_overrides.clear()


@pytest.fixture
def application() -> FastAPI:
    return app


def _create_user(
    db_session: Session,
    *,
    role: UserRole,
    first_name: str = "Test",
    last_name: str = "User",
) -> User:
    user = User(
        first_name=first_name,
        last_name=last_name,
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_TEST_PASSWORD),
        role=role,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture
def test_user(db_session: Session) -> User:
    return _create_user(db_session, role=UserRole.USER)


@pytest.fixture
def admin_user(db_session: Session) -> User:
    return _create_user(db_session, role=UserRole.ADMIN, first_name="Admin")


@pytest.fixture
def super_admin_user(db_session: Session) -> User:
    return _create_user(
        db_session,
        role=UserRole.SUPER_ADMIN,
        first_name="Super",
        last_name="Admin",
    )


@pytest.fixture
def organization(db_session: Session) -> Organization:
    org = Organization(name="Test Org", slug=f"test-org-{uuid.uuid4().hex[:8]}")
    db_session.add(org)
    db_session.flush()
    return org


@pytest.fixture
def workspace(db_session: Session, organization: Organization) -> Workspace:
    item = Workspace(
        organization_id=organization.id,
        name="Test Workspace",
        slug=f"test-ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(item)
    db_session.flush()
    return item


@pytest.fixture
def workspace_member(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=test_user.id,
        role=WorkspaceRole.OWNER,
    )
    db_session.add(member)
    db_session.flush()
    return member


@pytest.fixture
def access_token(test_user: User) -> str:
    return create_access_token(test_user.id)


@pytest.fixture
def auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def login_tokens(client: TestClient, test_user: User) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": test_user.email, "password": VALID_TEST_PASSWORD},
    )
    assert response.status_code == 200
    body = response.json()
    return {
        "access_token": body["access_token"],
        "refresh_token": body["refresh_token"],
    }


def token_from_last_email() -> str:
    assert email_service.outbox
    match = _TOKEN_RE.search(email_service.outbox[-1].body)
    assert match is not None
    return match.group(1)
