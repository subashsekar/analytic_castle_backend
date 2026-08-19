from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password
from app.db.models import DataSource, DataSourceStatus, User, UserRole
from app.services.credentials import decrypt_secret
from tests.integration.test_postgresql_connector_live import _integration_config

_SKIP_REASON = "PostgreSQL API integration tests: SKIPPED"
PREFIX = "/api/v1/data-sources"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _create_user(db_session: Session) -> User:
    user = User(
        first_name="Live",
        last_name="Tester",
        email=f"live-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=UserRole.USER,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def test_create_data_source_and_test_connection_against_live_postgres(
    client: TestClient,
    db_session: Session,
) -> None:
    live = _integration_config()
    if live is None:
        pytest.skip(_SKIP_REASON)

    owner = _create_user(db_session)
    organization = client.post(
        ORG_PREFIX,
        headers=_auth_header(owner),
        json={"name": "Live Org"},
    )
    assert organization.status_code == 201
    workspaces = client.get(
        WS_PREFIX,
        headers=_auth_header(owner),
        params={"organization_id": organization.json()["id"]},
    )
    workspace_id = workspaces.json()[0]["id"]
    password = live.credential

    created = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json={
            "workspace_id": workspace_id,
            "name": "Live PostgreSQL",
            "type": "POSTGRESQL",
            "connection": {
                "host": live.host,
                "port": live.port,
                "database_name": live.database_name,
                "username": live.username,
                "password": password,
                "ssl_mode": live.ssl_mode or "prefer",
            },
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == DataSourceStatus.INACTIVE.value
    assert password not in created.text
    assert "encrypted_password" not in created.text

    stored = db_session.get(DataSource, uuid.UUID(body["id"]))
    assert stored is not None
    assert stored.connection is not None
    ciphertext = stored.connection.encrypted_password
    assert ciphertext != password
    assert decrypt_secret(ciphertext) == password

    tested = client.post(
        f"{PREFIX}/{body['id']}/test-connection",
        headers=_auth_header(owner),
    )
    assert tested.status_code == 200
    assert tested.json() == {
        "success": True,
        "message": "Connection successful.",
    }
    assert password not in tested.text
    db_session.refresh(stored)
    assert stored.status is DataSourceStatus.ACTIVE
    assert stored.last_tested_at is not None
    assert stored.connection.encrypted_password == ciphertext
