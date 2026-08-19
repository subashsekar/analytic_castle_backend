import ast
import logging
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors import (
    ConnectionTestResult,
    ConnectorAuthenticationError,
    ConnectorConfig,
    ConnectorConnectionError,
    UnsupportedOperationError,
    register_connector,
)
from app.connectors.types import ColumnInfo, QueryResult, SchemaInfo, TableInfo
from app.core.logging import RedactingFilter
from app.core.rate_limit import reset_rate_limiters
from app.core.security import create_access_token, hash_password
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceStatus,
    DataSourceType,
    User,
    UserRole,
    Workspace,
    WorkspaceMember,
    WorkspaceRole,
)
from app.services.credentials import CIPHERTEXT_VERSION, decrypt_secret

PREFIX = "/api/v1/data-sources"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"
CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)
_ROUTE_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "api" / "routes" / "data_sources.py"
)
_FORBIDDEN_IMPORTS = {
    "psycopg",
    "asyncpg",
    "cryptography",
    "cryptography.hazmat.primitives.ciphers.aead",
}


class RecordingConnector:
    """Test double. Not a production connector."""

    def __init__(self) -> None:
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connected = False
        self.config: ConnectorConfig | None = None
        self.fail_connect: Exception | None = None
        self.test_success = True

    async def connect(self, config: ConnectorConfig) -> None:
        self.connect_calls += 1
        self.config = config
        if self.fail_connect is not None:
            raise self.fail_connect
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def test_connection(self) -> ConnectionTestResult:
        if not self.connected:
            raise ConnectorConnectionError("Not connected")
        if not self.test_success:
            return ConnectionTestResult(success=False)
        return ConnectionTestResult(success=True)

    async def get_schemas(self) -> list[SchemaInfo]:
        raise UnsupportedOperationError("Schema discovery is not supported")

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        raise UnsupportedOperationError("Table discovery is not supported")

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        raise UnsupportedOperationError("Column discovery is not supported")

    async def execute_query(self, query: str) -> QueryResult:
        raise UnsupportedOperationError("Query execution is not supported")


def _auth_header(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}


def _create_user(db_session: Session, *, role: UserRole = UserRole.USER) -> User:
    user = User(
        first_name="Test",
        last_name="User",
        email=f"user-{uuid.uuid4().hex[:8]}@example.com",
        password_hash=hash_password(VALID_PASSWORD),
        role=role,
        is_verified=True,
        is_active=True,
    )
    db_session.add(user)
    db_session.flush()
    return user


def _create_organization(client: TestClient, user: User, name: str = "Org") -> dict:
    response = client.post(
        ORG_PREFIX,
        headers=_auth_header(user),
        json={"name": name},
    )
    assert response.status_code == 201
    return response.json()


def _founding_workspace(client: TestClient, user: User, organization_id: str) -> dict:
    response = client.get(
        WS_PREFIX,
        headers=_auth_header(user),
        params={"organization_id": organization_id},
    )
    assert response.status_code == 200
    assert response.json()
    return response.json()[0]


def _add_member(
    db_session: Session,
    workspace_id: uuid.UUID,
    user: User,
    role: WorkspaceRole,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def _connection_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "host": "db.example.com",
        "port": 5432,
        "database_name": "analytics",
        "username": "analytics_user",
        "password": CUSTOMER_PASSWORD,
        "ssl_mode": "require",
    }
    payload.update(overrides)
    return payload


def _create_payload(
    workspace_id: str,
    **overrides: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "workspace_id": workspace_id,
        "name": "Production Analytics",
        "type": DataSourceType.POSTGRESQL.value,
        "connection": _connection_payload(),
    }
    payload.update(overrides)
    return payload


def _create_data_source(
    client: TestClient,
    user: User,
    workspace_id: str,
    **overrides: object,
) -> dict:
    response = client.post(
        PREFIX,
        headers=_auth_header(user),
        json=_create_payload(workspace_id, **overrides),
    )
    assert response.status_code == 201, response.text
    return response.json()


def _register_recording_connector() -> list[RecordingConnector]:
    created: list[RecordingConnector] = []

    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    return created


def _enable_rate_limits(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)
    reset_rate_limiters()


def _assert_safe_metadata(body: dict) -> None:
    assert "password" not in body
    assert "encrypted_password" not in body
    assert "connection" not in body
    assert CUSTOMER_PASSWORD not in str(body)
    assert "v1:" not in str(body)
    for key in (
        "id",
        "workspace_id",
        "name",
        "type",
        "status",
        "created_by",
        "created_at",
        "updated_at",
        "last_tested_at",
    ):
        assert key in body


def test_create_data_source_encrypts_password_and_omits_secrets(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])

    response = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(workspace["id"]),
    )

    assert response.status_code == 201
    body = response.json()
    _assert_safe_metadata(body)
    assert body["name"] == "Production Analytics"
    assert body["type"] == DataSourceType.POSTGRESQL.value
    assert body["status"] == DataSourceStatus.INACTIVE.value
    assert body["created_by"] == str(owner.id)
    assert body["workspace_id"] == workspace["id"]
    assert body["last_tested_at"] is None

    stored = db_session.get(DataSource, uuid.UUID(body["id"]))
    assert stored is not None
    assert stored.connection is not None
    ciphertext = stored.connection.encrypted_password
    assert ciphertext != CUSTOMER_PASSWORD
    assert ciphertext.startswith(f"{CIPHERTEXT_VERSION}:")
    assert decrypt_secret(ciphertext) == CUSTOMER_PASSWORD
    assert stored.status is DataSourceStatus.INACTIVE


def test_create_rejects_unauthenticated_invalid_and_unauthorized_workspace(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    payload = _create_payload(workspace["id"])

    unauthenticated = client.post(PREFIX, json=payload)
    missing = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(str(uuid.uuid4())),
    )
    foreign = client.post(PREFIX, headers=_auth_header(outsider), json=payload)

    assert unauthenticated.status_code == 401
    assert missing.status_code == 404
    assert foreign.status_code == 403


def test_member_cannot_create_data_source(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        member_user,
        WorkspaceRole.MEMBER,
    )

    response = client.post(
        PREFIX,
        headers=_auth_header(member_user),
        json=_create_payload(workspace["id"]),
    )

    assert response.status_code == 403
    assert (
        db_session.scalar(
            select(DataSource).where(DataSource.name == "Production Analytics")
        )
        is None
    )


def test_create_rejects_unsupported_connector_and_invalid_postgres_config(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])

    unsupported = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(workspace["id"], type=DataSourceType.MYSQL.value),
    )
    unknown_type = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(workspace["id"], type="NOT_A_TYPE"),
    )
    empty_host = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(
            workspace["id"],
            connection=_connection_payload(host="   "),
        ),
    )
    bad_port = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(
            workspace["id"],
            connection=_connection_payload(port=0),
        ),
    )
    bad_ssl = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(
            workspace["id"],
            connection=_connection_payload(ssl_mode="not-a-mode"),
        ),
    )
    missing_password = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(
            workspace["id"],
            connection={
                "host": "db.example.com",
                "database_name": "analytics",
                "username": "analytics_user",
            },
        ),
    )

    assert unsupported.status_code == 400
    assert unsupported.json()["detail"] == "Unsupported connector type"
    assert unknown_type.status_code == 422
    assert empty_host.status_code == 422
    assert bad_port.status_code == 422
    assert bad_ssl.status_code == 422
    assert missing_password.status_code == 422


def test_create_rejects_client_controlled_status_and_ownership(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    other = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    payload = _create_payload(workspace["id"])
    payload["status"] = DataSourceStatus.ACTIVE.value
    payload["created_by"] = str(other.id)

    response = client.post(PREFIX, headers=_auth_header(owner), json=payload)

    assert response.status_code == 422


def test_encryption_failure_does_not_create_data_source(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.services.credentials import CredentialEncryptionError

    def _boom(value: str, *, key: bytes | None = None) -> str:
        raise CredentialEncryptionError("Unable to encrypt credential")

    monkeypatch.setattr("app.services.data_sources.encrypt_secret", _boom)
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])

    response = client.post(
        PREFIX,
        headers=_auth_header(owner),
        json=_create_payload(workspace["id"]),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to store data source credentials"
    assert db_session.scalar(select(DataSource)) is None


def test_list_requires_authentication(client: TestClient) -> None:
    assert client.get(PREFIX).status_code == 401


def test_list_returns_accessible_data_sources_and_isolates_workspaces(
    client: TestClient,
    db_session: Session,
) -> None:
    owner_a = _create_user(db_session)
    owner_b = _create_user(db_session)
    org_a = _create_organization(client, owner_a, "Org A")
    org_b = _create_organization(client, owner_b, "Org B")
    workspace_a = _founding_workspace(client, owner_a, org_a["id"])
    workspace_b = _founding_workspace(client, owner_b, org_b["id"])
    second = client.post(
        WS_PREFIX,
        headers=_auth_header(owner_a),
        json={"organization_id": org_a["id"], "name": "Finance"},
    )
    assert second.status_code == 201
    workspace_a2 = second.json()

    source_a = _create_data_source(client, owner_a, workspace_a["id"], name="Sales")
    _create_data_source(client, owner_a, workspace_a2["id"], name="Finance DB")
    source_b = _create_data_source(client, owner_b, workspace_b["id"], name="Other")

    listed = client.get(PREFIX, headers=_auth_header(owner_a))
    filtered = client.get(
        PREFIX,
        headers=_auth_header(owner_a),
        params={"workspace_id": workspace_a["id"]},
    )
    empty_workspace = client.post(
        WS_PREFIX,
        headers=_auth_header(owner_a),
        json={"organization_id": org_a["id"], "name": "Empty"},
    )
    assert empty_workspace.status_code == 201
    empty = client.get(
        PREFIX,
        headers=_auth_header(owner_a),
        params={"workspace_id": empty_workspace.json()["id"]},
    )
    cross_filter = client.get(
        PREFIX,
        headers=_auth_header(owner_a),
        params={"workspace_id": workspace_b["id"]},
    )

    assert listed.status_code == 200
    names = {item["name"] for item in listed.json()}
    assert names == {"Sales", "Finance DB"}
    assert source_b["id"] not in {item["id"] for item in listed.json()}
    assert {item["id"] for item in filtered.json()} == {source_a["id"]}
    assert empty.json() == []
    assert cross_filter.status_code == 403
    for item in listed.json():
        _assert_safe_metadata(item)


def test_get_data_source_protects_against_idor_and_missing_ids(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    member_user = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        member_user,
        WorkspaceRole.MEMBER,
    )

    allowed = client.get(f"{PREFIX}/{created['id']}", headers=_auth_header(owner))
    member_allowed = client.get(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(member_user),
    )
    missing = client.get(f"{PREFIX}/{uuid.uuid4()}", headers=_auth_header(owner))
    idor = client.get(f"{PREFIX}/{created['id']}", headers=_auth_header(outsider))
    unauthenticated = client.get(f"{PREFIX}/{created['id']}")

    assert allowed.status_code == 200
    _assert_safe_metadata(allowed.json())
    assert allowed.json()["id"] == created["id"]
    assert CUSTOMER_PASSWORD not in allowed.text
    assert "encrypted_password" not in allowed.text
    assert member_allowed.status_code == 200
    assert member_allowed.json()["id"] == created["id"]
    assert missing.status_code == 404
    assert idor.status_code == 403
    assert "Production Analytics" not in idor.text
    assert unauthenticated.status_code == 401


def test_update_allows_name_and_rejects_protected_fields(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    other_workspace = client.post(
        WS_PREFIX,
        headers=_auth_header(owner),
        json={"organization_id": organization["id"], "name": "Other"},
    ).json()
    created = _create_data_source(client, owner, workspace["id"])
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        member_user,
        WorkspaceRole.MEMBER,
    )

    updated = client.patch(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(owner),
        json={"name": "Production Analytics DB"},
    )
    member_denied = client.patch(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(member_user),
        json={"name": "Hijacked"},
    )
    status_change = client.patch(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(owner),
        json={"status": DataSourceStatus.ACTIVE.value},
    )
    workspace_change = client.patch(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(owner),
        json={"workspace_id": other_workspace["id"]},
    )
    created_by_change = client.patch(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(owner),
        json={"created_by": str(member_user.id)},
    )

    assert updated.status_code == 200
    assert updated.json()["name"] == "Production Analytics DB"
    assert updated.json()["status"] == DataSourceStatus.INACTIVE.value
    assert updated.json()["created_by"] == str(owner.id)
    assert updated.json()["workspace_id"] == workspace["id"]
    _assert_safe_metadata(updated.json())
    assert member_denied.status_code == 403
    assert status_change.status_code == 422
    assert workspace_change.status_code == 422
    assert created_by_change.status_code == 422

    stored = db_session.get(DataSource, uuid.UUID(created["id"]))
    assert stored is not None
    assert stored.name == "Production Analytics DB"
    assert stored.workspace_id == uuid.UUID(workspace["id"])
    assert stored.created_by == owner.id
    assert stored.status is DataSourceStatus.INACTIVE


def test_delete_removes_connection_and_rejects_member(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    member_user = _create_user(db_session)
    outsider = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    source_id = uuid.UUID(created["id"])
    stored = db_session.get(DataSource, source_id)
    assert stored is not None
    connection_id = stored.connection.id if stored.connection is not None else None
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        member_user,
        WorkspaceRole.MEMBER,
    )

    member_denied = client.delete(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(member_user),
    )
    unauthorized = client.delete(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(outsider),
    )
    deleted = client.delete(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(owner),
    )

    assert member_denied.status_code == 403
    assert unauthorized.status_code == 403
    assert deleted.status_code == 200
    assert deleted.json()["detail"] == "Data source deleted"
    db_session.expire_all()
    assert db_session.get(DataSource, source_id) is None
    assert (
        db_session.scalar(
            select(DataSourceConnection).where(DataSourceConnection.id == connection_id)
        )
        is None
    )
    remaining_workspace = db_session.get(Workspace, uuid.UUID(workspace["id"]))
    assert remaining_workspace is not None


def test_admin_can_delete_data_source(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    admin_user = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        admin_user,
        WorkspaceRole.ADMIN,
    )

    response = client.delete(
        f"{PREFIX}/{created['id']}",
        headers=_auth_header(admin_user),
    )

    assert response.status_code == 200
    assert db_session.get(DataSource, uuid.UUID(created["id"])) is None


def test_test_connection_success_failure_and_safe_errors(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    created_connectors = _register_recording_connector()

    success = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(owner),
        json={"host": "evil.example", "password": "hacked-password"},
    )

    assert success.status_code == 200
    assert success.json() == {
        "success": True,
        "message": "Connection successful.",
    }
    assert "hacked-password" not in success.text
    assert CUSTOMER_PASSWORD not in success.text
    assert created_connectors[0].config is not None
    assert created_connectors[0].config.credential == CUSTOMER_PASSWORD
    assert created_connectors[0].config.host == "db.example.com"
    assert created_connectors[0].disconnect_calls == 1

    db_session.expire_all()
    stored = db_session.get(DataSource, uuid.UUID(created["id"]))
    assert stored is not None
    assert stored.status is DataSourceStatus.ACTIVE
    assert stored.last_tested_at is not None
    assert stored.connection is not None
    assert stored.connection.encrypted_password != CUSTOMER_PASSWORD


def test_test_connection_invalid_credentials_timeout_and_host_are_safe(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])

    def _auth_failure() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorAuthenticationError(
            f"password={CUSTOMER_PASSWORD} uri={CONNECTION_URI}"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _auth_failure)
    invalid = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(owner),
    )

    def _timeout() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = TimeoutError(
            f"timed out connecting to {CONNECTION_URI}"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _timeout)
    timeout = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(owner),
    )

    def _bad_host() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorConnectionError(
            "Unable to resolve the PostgreSQL host"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _bad_host)
    invalid_host = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(owner),
    )

    for response in (invalid, timeout, invalid_host):
        assert response.status_code == 200
        assert response.json() == {
            "success": False,
            "message": "Unable to connect to the data source.",
        }
        assert CUSTOMER_PASSWORD not in response.text
        assert CONNECTION_URI not in response.text
        assert "password" not in response.json()
        assert "encrypted_password" not in response.text


def test_test_connection_rejects_unauthorized_and_unsupported_connector(
    client: TestClient,
    db_session: Session,
) -> None:
    owner = _create_user(db_session)
    outsider = _create_user(db_session)
    member_user = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    _add_member(
        db_session,
        uuid.UUID(workspace["id"]),
        member_user,
        WorkspaceRole.MEMBER,
    )
    _register_recording_connector()

    member_allowed = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(member_user),
    )
    unauthorized = client.post(
        f"{PREFIX}/{created['id']}/test-connection",
        headers=_auth_header(outsider),
    )
    unauthenticated = client.post(f"{PREFIX}/{created['id']}/test-connection")

    mysql_source = DataSource(
        workspace_id=uuid.UUID(workspace["id"]),
        name="Legacy MySQL",
        type=DataSourceType.MYSQL,
        created_by=owner.id,
    )
    db_session.add(mysql_source)
    db_session.flush()
    db_session.add(
        DataSourceConnection(
            data_source_id=mysql_source.id,
            host="mysql.example.com",
            database_name="legacy",
            username="reader",
            encrypted_password="v1:nonce:payload",
        )
    )
    db_session.flush()

    unsupported = client.post(
        f"{PREFIX}/{mysql_source.id}/test-connection",
        headers=_auth_header(owner),
    )

    assert member_allowed.status_code == 200
    assert unauthorized.status_code == 403
    assert unauthenticated.status_code == 401
    assert unsupported.status_code == 400
    assert unsupported.json()["detail"] == "Unsupported connector type"


def test_test_connection_is_rate_limited(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(monkeypatch, RATE_LIMIT_TEST_CONNECTION="2/minute")
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])
    _register_recording_connector()
    path = f"{PREFIX}/{created['id']}/test-connection"
    headers = _auth_header(owner)

    first = client.post(path, headers=headers)
    second = client.post(path, headers=headers)
    limited = client.post(path, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Too many requests"


def test_create_does_not_log_password_or_connection_string(
    client: TestClient,
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    logger = logging.getLogger("app.services.data_sources")
    logger.addFilter(RedactingFilter())

    with caplog.at_level(logging.INFO, logger=logger.name):
        response = client.post(
            PREFIX,
            headers=_auth_header(owner),
            json=_create_payload(workspace["id"]),
        )

    assert response.status_code == 201
    assert CUSTOMER_PASSWORD not in caplog.text
    assert CONNECTION_URI not in caplog.text
    assert "encrypted_password" not in caplog.text


def test_test_connection_does_not_log_secrets(
    client: TestClient,
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner)
    workspace = _founding_workspace(client, owner, organization["id"])
    created = _create_data_source(client, owner, workspace["id"])

    def _builder() -> RecordingConnector:
        connector = RecordingConnector()
        connector.fail_connect = ConnectorAuthenticationError(
            f"password={CUSTOMER_PASSWORD} connection_string={CONNECTION_URI}"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    logger = logging.getLogger("app.services.data_source_connections")
    logger.addFilter(RedactingFilter())

    with caplog.at_level(logging.INFO, logger=logger.name):
        response = client.post(
            f"{PREFIX}/{created['id']}/test-connection",
            headers=_auth_header(owner),
        )

    assert response.status_code == 200
    assert CUSTOMER_PASSWORD not in caplog.text
    assert CONNECTION_URI not in caplog.text
    assert CUSTOMER_PASSWORD not in response.text


def test_openapi_documents_data_source_endpoints_without_secret_examples(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    assert f"{PREFIX}" in paths
    assert f"{PREFIX}/{{data_source_id}}" in paths
    assert f"{PREFIX}/{{data_source_id}}/test-connection" in paths
    create = paths[f"{PREFIX}"]["post"]
    assert "security" in create or "HTTPBearer" in str(schema.get("components", {}))
    components = schema["components"]["schemas"]
    read_props = components["DataSourceRead"]["properties"]
    assert "password" not in read_props
    assert "encrypted_password" not in read_props
    connection = components["ConnectionConfigCreate"]["properties"]["password"]
    assert connection.get("writeOnly") is True
    assert connection.get("format") == "password"
    assert CUSTOMER_PASSWORD not in str(schema)


def test_routes_do_not_import_drivers_or_crypto() -> None:
    tree = ast.parse(_ROUTE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported.isdisjoint(_FORBIDDEN_IMPORTS)
