from __future__ import annotations

import ast
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from app.connectors import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorConfig,
    ConnectorQueryError,
    QueryResult,
    SchemaInfo,
    TableInfo,
    UnsupportedOperationError,
    register_connector,
)
from app.core.logging import RedactingFilter
from app.core.rate_limit import reset_rate_limiters
from app.core.security import create_access_token, hash_password
from app.db.models import (
    DataSourceColumn,
    DataSourceMetadataSync,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
    DataSourceType,
    User,
    UserRole,
    WorkspaceMember,
    WorkspaceRole,
)
from app.enums import (
    DataSourceRelationshipType,
    DataSourceTableType,
    MetadataSyncStatus,
)
from app.services.data_masking import REDACTED
from app.services.metadata_sync_exceptions import (
    ConcurrentMetadataSyncError,
    MetadataSyncError,
)
from app.services.metadata_sync_types import MetadataSyncResult

PREFIX = "/api/v1/data-sources"
ORG_PREFIX = "/api/v1/organizations"
WS_PREFIX = "/api/v1/workspaces"
VALID_PASSWORD = "SecurePassword123!"
CUSTOMER_PASSWORD = "CustomerDbPassword!@# 42"
CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)
RAW_EMAIL = "john.doe@example.com"
RAW_PASSWORD = "hunter2-password"
_ROUTE_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "api" / "routes" / "metadata.py"
)
_FORBIDDEN_IMPORTS = {
    "psycopg",
    "asyncpg",
    "cryptography",
    "cryptography.hazmat.primitives.ciphers.aead",
}


class SampleConnector:
    """Test double. Not a production connector."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])
        self.calls: list[tuple[str, str, tuple[str, ...], int]] = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connected = False
        self.config: ConnectorConfig | None = None
        self.fail_error: Exception | None = None

    async def connect(self, config: ConnectorConfig) -> None:
        self.connect_calls += 1
        self.config = config
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        self.connected = False

    async def test_connection(self) -> ConnectionTestResult:
        return ConnectionTestResult(success=True)

    async def get_schemas(self) -> list[SchemaInfo]:
        return []

    async def get_tables(self, schema: str | None = None) -> list[TableInfo]:
        return []

    async def get_columns(
        self,
        table: str,
        schema: str | None = None,
    ) -> list[ColumnInfo]:
        return []

    async def execute_query(self, query: str) -> QueryResult:
        raise UnsupportedOperationError("Query execution is not supported")

    async def _fetch_sample_rows(
        self,
        schema_name: str,
        table_name: str,
        column_names: Sequence[str],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        self.calls.append((schema_name, table_name, tuple(column_names), limit))
        if self.fail_error is not None:
            raise self.fail_error
        selected: list[dict[str, Any]] = []
        for row in self.rows[:limit]:
            selected.append({name: row.get(name) for name in column_names})
        return selected


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


def _create_data_source(
    client: TestClient,
    user: User,
    workspace_id: str,
    *,
    name: str = "Production Analytics",
) -> dict:
    response = client.post(
        PREFIX,
        headers=_auth_header(user),
        json={
            "workspace_id": workspace_id,
            "name": name,
            "type": DataSourceType.POSTGRESQL.value,
            "connection": {
                "host": "db.example.com",
                "port": 5432,
                "database_name": "analytics",
                "username": "analytics_user",
                "password": CUSTOMER_PASSWORD,
                "ssl_mode": "require",
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _setup_source(
    client: TestClient,
    db_session: Session,
    *,
    org_name: str = "Org",
    source_name: str = "Production Analytics",
) -> tuple[User, dict, dict]:
    owner = _create_user(db_session)
    organization = _create_organization(client, owner, org_name)
    workspace = _founding_workspace(client, owner, organization["id"])
    source = _create_data_source(client, owner, workspace["id"], name=source_name)
    return owner, workspace, source


def _add_schema(
    db_session: Session, data_source_id: UUID, name: str
) -> DataSourceSchema:
    schema = DataSourceSchema(data_source_id=data_source_id, name=name)
    db_session.add(schema)
    db_session.flush()
    return schema


def _add_table(
    db_session: Session,
    schema_id: UUID,
    name: str,
    *,
    table_type: DataSourceTableType = DataSourceTableType.TABLE,
    description: str | None = None,
) -> DataSourceTable:
    table = DataSourceTable(
        schema_id=schema_id,
        name=name,
        table_type=table_type,
        description=description,
    )
    db_session.add(table)
    db_session.flush()
    return table


def _add_column(
    db_session: Session,
    table_id: UUID,
    name: str,
    *,
    position: int,
    data_type: str = "string",
    database_type: str = "text",
    nullable: bool = True,
    primary_key: bool | None = None,
    unique: bool | None = None,
    default_value: str | None = None,
    description: str | None = None,
) -> DataSourceColumn:
    is_pk = position == 1 if primary_key is None else primary_key
    column = DataSourceColumn(
        table_id=table_id,
        name=name,
        ordinal_position=position,
        data_type=data_type,
        database_type=database_type,
        is_nullable=nullable,
        is_primary_key=is_pk,
        is_unique=is_pk if unique is None else unique,
        default_value=default_value,
        description=description,
    )
    db_session.add(column)
    db_session.flush()
    return column


def _add_relationship(
    db_session: Session,
    *,
    source_table: DataSourceTable,
    source_column: DataSourceColumn,
    target_table: DataSourceTable,
    target_column: DataSourceColumn,
    relationship_type: DataSourceRelationshipType = DataSourceRelationshipType.MANY_TO_ONE,
    constraint_name: str | None = None,
) -> DataSourceRelationship:
    relation = DataSourceRelationship(
        source_table_id=source_table.id,
        source_column_id=source_column.id,
        target_table_id=target_table.id,
        target_column_id=target_column.id,
        relationship_type=relationship_type,
        constraint_name=constraint_name,
    )
    db_session.add(relation)
    db_session.flush()
    return relation


def _seed_catalog(
    db_session: Session,
    data_source_id: UUID,
) -> tuple[DataSourceSchema, DataSourceTable, list[DataSourceColumn]]:
    public_schema = _add_schema(db_session, data_source_id, "public")
    customers = _add_table(
        db_session,
        public_schema.id,
        "customers",
        description="Customer accounts",
    )
    columns = [
        _add_column(
            db_session,
            customers.id,
            "id",
            position=1,
            data_type="integer",
            database_type="int4",
            nullable=False,
            default_value="nextval('customers_id_seq')",
        ),
        _add_column(
            db_session,
            customers.id,
            "email",
            position=2,
            data_type="string",
            database_type="text",
            nullable=False,
            primary_key=False,
            unique=True,
        ),
        _add_column(
            db_session,
            customers.id,
            "name",
            position=3,
            data_type="string",
            database_type="text",
            primary_key=False,
            unique=False,
        ),
    ]
    return public_schema, customers, columns


def _assert_no_secrets(body: object) -> None:
    rendered = str(body)
    assert CUSTOMER_PASSWORD not in rendered
    assert CONNECTION_URI not in rendered
    assert "encrypted_password" not in rendered
    if isinstance(body, dict):
        assert "password" not in body
        assert "encrypted_password" not in body
        assert "connection" not in body


def _enable_rate_limits(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    for name, value in overrides.items():
        monkeypatch.setattr(settings, name, value)
    reset_rate_limiters()


def test_list_and_get_schemas(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    public_schema = _add_schema(db_session, source_id, "public")
    analytics = _add_schema(db_session, source_id, "analytics")
    _add_table(db_session, public_schema.id, "customers")
    _add_table(db_session, public_schema.id, "orders")

    listed = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(owner),
    )
    detail = client.get(
        f"{PREFIX}/{source['id']}/schemas/{public_schema.id}",
        headers=_auth_header(owner),
    )

    assert listed.status_code == 200
    body = listed.json()
    _assert_no_secrets(body)
    assert body["page"] == 1
    assert body["page_size"] == 50
    assert body["total"] == 2
    names = [item["name"] for item in body["items"]]
    assert names == ["analytics", "public"]
    public = next(item for item in body["items"] if item["name"] == "public")
    assert public["id"] == str(public_schema.id)
    assert public["data_source_id"] == source["id"]
    assert public["table_count"] == 2
    analytics_item = next(item for item in body["items"] if item["name"] == "analytics")
    assert analytics_item["id"] == str(analytics.id)
    assert analytics_item["table_count"] == 0

    assert detail.status_code == 200
    schema_body = detail.json()
    assert schema_body["id"] == str(public_schema.id)
    assert schema_body["name"] == "public"
    assert schema_body["table_count"] == 2
    _assert_no_secrets(schema_body)


def test_list_schemas_pagination_and_empty_results(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    for index in range(3):
        _add_schema(db_session, source_id, f"schema_{index:02d}")

    first = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(owner),
        params={"page": 1, "page_size": 2},
    )
    second = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(owner),
        params={"page": 2, "page_size": 2},
    )
    empty_source = _create_data_source(
        client, owner, source["workspace_id"], name="Empty Source"
    )
    empty = client.get(
        f"{PREFIX}/{empty_source['id']}/schemas",
        headers=_auth_header(owner),
    )
    oversized = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(owner),
        params={"page_size": 101},
    )

    assert first.status_code == 200
    assert first.json()["total"] == 3
    assert len(first.json()["items"]) == 2
    assert second.json()["items"][0]["name"] == "schema_02"
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "page": 1, "page_size": 50, "total": 0}
    assert oversized.status_code == 422


def test_list_and_get_tables_with_filters(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    public_schema = _add_schema(db_session, source_id, "public")
    reporting = _add_schema(db_session, source_id, "reporting")
    customers = _add_table(
        db_session, public_schema.id, "customers", description="Customer accounts"
    )
    _add_column(db_session, customers.id, "id", position=1, data_type="integer")
    orders = _add_table(db_session, public_schema.id, "orders")
    _add_column(db_session, orders.id, "id", position=1, data_type="integer")
    summary = _add_table(
        db_session,
        reporting.id,
        "customer_summary",
        table_type=DataSourceTableType.VIEW,
    )
    _add_column(db_session, summary.id, "id", position=1, data_type="integer")
    _add_table(db_session, public_schema.id, "products")

    listed = client.get(
        f"{PREFIX}/{source['id']}/tables",
        headers=_auth_header(owner),
    )
    filtered = client.get(
        f"{PREFIX}/{source['id']}/tables",
        headers=_auth_header(owner),
        params={"schema_id": str(public_schema.id)},
    )
    searched = client.get(
        f"{PREFIX}/{source['id']}/tables",
        headers=_auth_header(owner),
        params={"search": "customer"},
    )
    views = client.get(
        f"{PREFIX}/{source['id']}/tables",
        headers=_auth_header(owner),
        params={"table_type": DataSourceTableType.VIEW.value},
    )
    detail = client.get(
        f"{PREFIX}/{source['id']}/tables/{customers.id}",
        headers=_auth_header(owner),
    )

    assert listed.status_code == 200
    assert listed.json()["total"] == 4
    assert filtered.json()["total"] == 3
    assert {item["name"] for item in searched.json()["items"]} == {
        "customers",
        "customer_summary",
    }
    assert views.json()["total"] == 1
    assert views.json()["items"][0]["name"] == "customer_summary"
    assert views.json()["items"][0]["table_type"] == DataSourceTableType.VIEW.value

    assert detail.status_code == 200
    body = detail.json()
    assert body["id"] == str(customers.id)
    assert body["schema_id"] == str(public_schema.id)
    assert body["schema_name"] == "public"
    assert body["name"] == "customers"
    assert body["table_type"] == DataSourceTableType.TABLE.value
    assert body["description"] == "Customer accounts"
    assert body["column_count"] == 1
    _assert_no_secrets(body)


def test_list_columns_preserves_ordinal_and_metadata(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    _, customers, columns = _seed_catalog(db_session, source_id)
    zeta = _add_column(
        db_session,
        customers.id,
        "zeta",
        position=4,
        data_type="string",
        primary_key=False,
        unique=False,
        nullable=True,
    )

    response = client.get(
        f"{PREFIX}/{source['id']}/tables/{customers.id}/columns",
        headers=_auth_header(owner),
    )

    assert response.status_code == 200
    body = response.json()
    names = [item["name"] for item in body["items"]]
    assert names == ["id", "email", "name", "zeta"]
    assert [item["ordinal_position"] for item in body["items"]] == [1, 2, 3, 4]
    primary = body["items"][0]
    assert primary["id"] == str(columns[0].id)
    assert primary["data_type"] == "integer"
    assert primary["database_type"] == "int4"
    assert primary["is_primary_key"] is True
    assert primary["is_nullable"] is False
    assert primary["default_value"] == "nextval('customers_id_seq')"
    email = body["items"][1]
    assert email["is_unique"] is True
    assert email["is_primary_key"] is False
    assert zeta.name == "zeta"
    _assert_no_secrets(body)


def test_list_relationships_covers_fk_shapes(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    public_schema = _add_schema(db_session, source_id, "public")
    reporting = _add_schema(db_session, source_id, "reporting")
    customers = _add_table(db_session, public_schema.id, "customers")
    orders = _add_table(db_session, public_schema.id, "orders")
    facts = _add_table(db_session, reporting.id, "customer_facts")
    customer_id = _add_column(db_session, customers.id, "id", position=1)
    parent_id = _add_column(
        db_session,
        customers.id,
        "parent_id",
        position=2,
        primary_key=False,
        unique=False,
    )
    order_customer = _add_column(
        db_session,
        orders.id,
        "customer_id",
        position=1,
        primary_key=False,
        unique=False,
    )
    order_org = _add_column(
        db_session,
        orders.id,
        "org_id",
        position=2,
        primary_key=False,
        unique=False,
    )
    fact_customer = _add_column(
        db_session,
        facts.id,
        "customer_id",
        position=1,
        primary_key=False,
        unique=False,
    )
    _add_relationship(
        db_session,
        source_table=orders,
        source_column=order_customer,
        target_table=customers,
        target_column=customer_id,
        constraint_name="fk_orders_customer",
    )
    _add_relationship(
        db_session,
        source_table=orders,
        source_column=order_org,
        target_table=customers,
        target_column=customer_id,
        constraint_name="fk_orders_customer",
    )
    _add_relationship(
        db_session,
        source_table=customers,
        source_column=parent_id,
        target_table=customers,
        target_column=customer_id,
        constraint_name="fk_customers_parent",
        relationship_type=DataSourceRelationshipType.ONE_TO_MANY,
    )
    _add_relationship(
        db_session,
        source_table=facts,
        source_column=fact_customer,
        target_table=customers,
        target_column=customer_id,
        constraint_name="fk_facts_customer",
    )

    response = client.get(
        f"{PREFIX}/{source['id']}/relationships",
        headers=_auth_header(owner),
    )

    assert response.status_code == 200
    items = response.json()["items"]
    assert response.json()["total"] == 4
    composite = [
        item for item in items if item["constraint_name"] == "fk_orders_customer"
    ]
    assert len(composite) == 2
    assert {item["source_column_name"] for item in composite} == {
        "customer_id",
        "org_id",
    }
    self_ref = next(
        item for item in items if item["constraint_name"] == "fk_customers_parent"
    )
    assert self_ref["source_table_id"] == self_ref["target_table_id"]
    assert self_ref["source_table_name"] == "customers"
    cross = next(
        item for item in items if item["constraint_name"] == "fk_facts_customer"
    )
    assert cross["source_schema_name"] == "reporting"
    assert cross["target_schema_name"] == "public"
    _assert_no_secrets(response.json())


def test_metadata_search_api(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    public_schema = _add_schema(db_session, source_id, "public")
    _add_table(db_session, public_schema.id, "customers")
    _add_table(db_session, public_schema.id, "customer_orders")
    _add_table(db_session, public_schema.id, "products")
    orders = _add_table(db_session, public_schema.id, "OrdersArchive")
    _add_column(db_session, orders.id, "customer_email", position=1)

    exact = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customers"},
    )
    partial = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "cust"},
    )
    case_insensitive = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "ORDERSARCHIVE"},
    )
    typed = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customer", "metadata_type": "COLUMN"},
    )
    limited = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customer", "limit": 1},
    )
    missing = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "no-such-metadata"},
    )
    empty = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "   "},
    )
    unbounded = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customer", "limit": 101},
    )
    injection = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customer%' OR 1=1 --"},
    )

    assert exact.status_code == 200
    exact_names = {
        (item["metadata_type"], item["table_name"]) for item in exact.json()["items"]
    }
    assert ("TABLE", "customers") in exact_names
    assert partial.json()["total"] >= 2
    assert any(
        item["table_name"] == "OrdersArchive"
        for item in case_insensitive.json()["items"]
    )
    assert typed.json()["items"]
    assert all(item["metadata_type"] == "COLUMN" for item in typed.json()["items"])
    assert limited.json()["limit"] == 1
    assert limited.json()["total"] == 1
    assert limited.json()["truncated"] is True
    assert missing.json()["items"] == []
    assert missing.json()["total"] == 0
    assert empty.status_code == 400
    assert unbounded.status_code == 422
    assert injection.status_code == 200
    assert injection.json()["items"] == []
    assert db_session.scalar(select(func.count()).select_from(DataSourceTable)) >= 1
    _assert_no_secrets(exact.json())


def test_sync_api_authorized_status_and_failures(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    outsider = _create_user(db_session)
    member = _create_user(db_session)
    _add_member(
        db_session,
        uuid.UUID(source["workspace_id"]),
        member,
        WorkspaceRole.MEMBER,
    )
    success = MetadataSyncResult(
        data_source_id=uuid.UUID(source["id"]),
        status=MetadataSyncStatus.SUCCESS,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        schema_count=5,
        table_count=42,
        column_count=380,
        relationship_count=27,
        error_message=None,
    )

    async def _synchronize(
        self: object, *args: object, **kwargs: object
    ) -> MetadataSyncResult:
        del self, args, kwargs
        return success

    monkeypatch.setattr(
        "app.api.routes.metadata.MetadataSyncService.synchronize",
        _synchronize,
    )

    authorized = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(owner),
    )
    member_denied = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(member),
    )
    outsider_denied = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(outsider),
    )
    unauthenticated = client.post(f"{PREFIX}/{source['id']}/metadata/sync")

    db_session.add(
        DataSourceMetadataSync(
            data_source_id=uuid.UUID(source["id"]),
            status=MetadataSyncStatus.SUCCESS,
            started_at=success.started_at,
            completed_at=success.completed_at,
            schema_count=5,
            table_count=42,
            column_count=380,
            relationship_count=27,
        )
    )
    db_session.flush()
    status_response = client.get(
        f"{PREFIX}/{source['id']}/metadata/sync-status",
        headers=_auth_header(owner),
    )

    async def _fail(
        self: object, *args: object, **kwargs: object
    ) -> MetadataSyncResult:
        del self, args, kwargs
        raise MetadataSyncError(
            f"asyncpg.exceptions.UndefinedTable password={CUSTOMER_PASSWORD} "
            f"uri={CONNECTION_URI}"
        )

    monkeypatch.setattr(
        "app.api.routes.metadata.MetadataSyncService.synchronize",
        _fail,
    )
    failed = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(owner),
    )

    async def _conflict(
        self: object, *args: object, **kwargs: object
    ) -> MetadataSyncResult:
        del self, args, kwargs
        raise ConcurrentMetadataSyncError()

    monkeypatch.setattr(
        "app.api.routes.metadata.MetadataSyncService.synchronize",
        _conflict,
    )
    conflicted = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(owner),
    )

    assert authorized.status_code == 200
    assert authorized.json()["status"] == "SUCCESS"
    assert authorized.json()["schemas"] == 5
    assert authorized.json()["tables"] == 42
    assert authorized.json()["columns"] == 380
    assert authorized.json()["relationships"] == 27
    assert member_denied.status_code == 403
    assert outsider_denied.status_code == 403
    assert unauthenticated.status_code == 401
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "SUCCESS"
    assert status_response.json()["schemas"] == 5
    assert failed.status_code == 502
    assert failed.json()["detail"] == "Metadata synchronization failed"
    assert CUSTOMER_PASSWORD not in failed.text
    assert CONNECTION_URI not in failed.text
    assert "asyncpg" not in failed.text
    assert conflicted.status_code == 409
    _assert_no_secrets(authorized.json())


def test_sample_data_api(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    _, customers, _ = _seed_catalog(db_session, source_id)
    created: list[SampleConnector] = []

    def _builder() -> SampleConnector:
        connector = SampleConnector(
            rows=[
                {
                    "id": 1,
                    "email": RAW_EMAIL,
                    "name": "Jane Doe",
                },
                {
                    "id": 2,
                    "email": "second@example.com",
                    "name": "password=super-secret",
                },
                {
                    "id": 3,
                    "email": "third@example.com",
                    "name": "Third",
                },
            ]
        )
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)

    valid = client.post(
        f"{PREFIX}/{source['id']}/tables/{customers.id}/sample",
        headers=_auth_header(owner),
        json={"limit": 2},
    )
    default_limit = client.post(
        f"{PREFIX}/{source['id']}/tables/{customers.id}/sample",
        headers=_auth_header(owner),
        json={},
    )
    bypass = client.post(
        f"{PREFIX}/{source['id']}/tables/{customers.id}/sample",
        headers=_auth_header(owner),
        json={"limit": 2, "include_unmasked": True},
    )
    missing_table = client.post(
        f"{PREFIX}/{source['id']}/tables/{uuid.uuid4()}/sample",
        headers=_auth_header(owner),
        json={"limit": 1},
    )

    assert valid.status_code == 200
    body = valid.json()
    assert body["row_count"] == 2
    assert body["row_limit"] == 2
    assert created[0].calls[0][3] == 2
    emails = [row["email"] for row in body["rows"]]
    assert RAW_EMAIL not in str(body)
    assert all(email != RAW_EMAIL for email in emails)
    assert REDACTED in str(body) or any("*" in str(email) for email in emails)
    assert body["columns"][1]["name"] == "email"
    assert body["columns"][1]["masked"] is True
    assert default_limit.status_code == 200
    assert default_limit.json()["row_limit"] == 10
    assert created[1].calls[0][3] == 10
    assert bypass.status_code == 422
    assert missing_table.status_code == 404
    _assert_no_secrets(body)


def test_sample_data_safe_error_and_secret_redaction(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    _, customers, _ = _seed_catalog(db_session, source_id)

    def _builder() -> SampleConnector:
        connector = SampleConnector()
        connector.fail_error = ConnectorQueryError(
            f"asyncpg.exceptions.UndefinedTable password={CUSTOMER_PASSWORD} "
            f"uri={CONNECTION_URI} relation public.users does not exist"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    response = client.post(
        f"{PREFIX}/{source['id']}/tables/{customers.id}/sample",
        headers=_auth_header(owner),
        json={"limit": 1},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Unable to retrieve sample data"
    assert CUSTOMER_PASSWORD not in response.text
    assert CONNECTION_URI not in response.text
    assert "asyncpg" not in response.text
    assert "UndefinedTable" not in response.text


def test_workspace_and_data_source_isolation(
    client: TestClient,
    db_session: Session,
) -> None:
    owner_a, workspace_a, source_a = _setup_source(
        client, db_session, org_name="Org A", source_name="Source A"
    )
    _, _, source_b = _setup_source(
        client, db_session, org_name="Org B", source_name="Source B"
    )
    schema_a, table_a, columns_a = _seed_catalog(db_session, uuid.UUID(source_a["id"]))
    schema_b, table_b, columns_b = _seed_catalog(db_session, uuid.UUID(source_b["id"]))
    relation_b = _add_relationship(
        db_session,
        source_table=table_b,
        source_column=columns_b[0],
        target_table=table_b,
        target_column=columns_b[0],
        constraint_name="fk_self",
    )
    same_workspace_source = _create_data_source(
        client, owner_a, workspace_a["id"], name="Source A2"
    )
    schema_a2 = _add_schema(db_session, uuid.UUID(same_workspace_source["id"]), "other")
    table_a2 = _add_table(db_session, schema_a2.id, "hidden")
    _add_column(db_session, table_a2.id, "id", position=1)

    allowed = client.get(
        f"{PREFIX}/{source_a['id']}/schemas",
        headers=_auth_header(owner_a),
    )
    cross_workspace = client.get(
        f"{PREFIX}/{source_b['id']}/schemas",
        headers=_auth_header(owner_a),
    )
    schema_idor = client.get(
        f"{PREFIX}/{source_a['id']}/schemas/{schema_b.id}",
        headers=_auth_header(owner_a),
    )
    table_idor = client.get(
        f"{PREFIX}/{source_a['id']}/tables/{table_b.id}",
        headers=_auth_header(owner_a),
    )
    same_workspace_idor = client.get(
        f"{PREFIX}/{source_a['id']}/tables/{table_a2.id}",
        headers=_auth_header(owner_a),
    )
    foreign_schema_filter = client.get(
        f"{PREFIX}/{source_a['id']}/tables",
        headers=_auth_header(owner_a),
        params={"schema_id": str(schema_b.id)},
    )
    column_idor = client.get(
        f"{PREFIX}/{source_a['id']}/tables/{table_b.id}/columns",
        headers=_auth_header(owner_a),
    )
    search_b = client.get(
        f"{PREFIX}/{source_a['id']}/metadata/search",
        headers=_auth_header(owner_a),
        params={"q": "customers"},
    )
    relationships_a = client.get(
        f"{PREFIX}/{source_a['id']}/relationships",
        headers=_auth_header(owner_a),
    )
    sample_idor = client.post(
        f"{PREFIX}/{source_a['id']}/tables/{table_b.id}/sample",
        headers=_auth_header(owner_a),
        json={"limit": 1},
    )
    unauthenticated = client.get(f"{PREFIX}/{source_a['id']}/schemas")

    assert allowed.status_code == 200
    assert {item["id"] for item in allowed.json()["items"]} == {str(schema_a.id)}
    assert cross_workspace.status_code == 403
    assert schema_idor.status_code == 404
    assert table_idor.status_code == 404
    assert same_workspace_idor.status_code == 404
    assert foreign_schema_filter.status_code == 404
    assert column_idor.status_code == 404
    assert all(item.get("schema_name") != "other" for item in search_b.json()["items"])
    assert relationships_a.json()["items"] == []
    assert str(relation_b.id) not in str(relationships_a.json())
    assert sample_idor.status_code == 404
    assert unauthenticated.status_code == 401
    assert str(schema_b.id) not in allowed.text
    assert str(columns_a[0].id) not in str(cross_workspace.json())
    assert str(table_a.id) not in str(table_idor.json())


def test_member_can_read_metadata_but_cannot_sync(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    member = _create_user(db_session)
    _add_member(
        db_session,
        uuid.UUID(source["workspace_id"]),
        member,
        WorkspaceRole.MEMBER,
    )
    _seed_catalog(db_session, uuid.UUID(source["id"]))

    listed = client.get(
        f"{PREFIX}/{source['id']}/tables",
        headers=_auth_header(member),
    )
    sync = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync",
        headers=_auth_header(member),
    )
    owner_status = client.get(
        f"{PREFIX}/{source['id']}/metadata/sync-status",
        headers=_auth_header(owner),
    )

    assert listed.status_code == 200
    assert sync.status_code == 403
    assert owner_status.status_code == 200


def test_super_admin_can_read_foreign_workspace_metadata(
    client: TestClient,
    db_session: Session,
) -> None:
    _, _, source = _setup_source(client, db_session)
    _seed_catalog(db_session, uuid.UUID(source["id"]))
    super_admin = _create_user(db_session, role=UserRole.SUPER_ADMIN)

    response = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(super_admin),
    )

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_metadata_reads_do_not_use_connectors(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    schema, table, _ = _seed_catalog(db_session, source_id)
    calls: list[str] = []

    def _forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append("create_connector")
        raise AssertionError("connector should not be used")

    monkeypatch.setattr("app.connectors.registry.create_connector", _forbidden)
    monkeypatch.setattr("app.connectors.create_connector", _forbidden)

    schemas = client.get(
        f"{PREFIX}/{source['id']}/schemas",
        headers=_auth_header(owner),
    )
    tables = client.get(
        f"{PREFIX}/{source['id']}/tables/{table.id}",
        headers=_auth_header(owner),
    )
    columns = client.get(
        f"{PREFIX}/{source['id']}/tables/{table.id}/columns",
        headers=_auth_header(owner),
    )
    search = client.get(
        f"{PREFIX}/{source['id']}/metadata/search",
        headers=_auth_header(owner),
        params={"q": "customers"},
    )

    assert schemas.status_code == 200
    assert tables.status_code == 200
    assert columns.status_code == 200
    assert search.status_code == 200
    assert calls == []
    assert schema.name == "public"


def test_list_endpoints_avoid_n_plus_one_queries(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    for index in range(12):
        schema = _add_schema(db_session, source_id, f"schema_{index:02d}")
        table = _add_table(db_session, schema.id, f"table_{index:02d}")
        _add_column(db_session, table.id, "id", position=1)
    captured: list[str] = []
    bind = db_session.get_bind()

    def _capture(
        _conn: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        captured.append(statement)

    event.listen(bind, "before_cursor_execute", _capture)
    try:
        response = client.get(
            f"{PREFIX}/{source['id']}/schemas",
            headers=_auth_header(owner),
        )
    finally:
        event.remove(bind, "before_cursor_execute", _capture)

    assert response.status_code == 200
    assert response.json()["total"] == 12
    assert len(captured) <= 8


def test_search_and_sample_and_sync_are_rate_limited(
    client: TestClient,
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_rate_limits(
        monkeypatch,
        RATE_LIMIT_METADATA_SEARCH="2/minute",
        RATE_LIMIT_METADATA_SYNC="1/minute",
        RATE_LIMIT_SAMPLE_DATA="1/minute",
    )
    owner, _, source = _setup_source(client, db_session)
    _, table, _ = _seed_catalog(db_session, uuid.UUID(source["id"]))
    search_path = f"{PREFIX}/{source['id']}/metadata/search"
    headers = _auth_header(owner)

    first = client.get(search_path, headers=headers, params={"q": "customers"})
    second = client.get(search_path, headers=headers, params={"q": "customers"})
    limited_search = client.get(search_path, headers=headers, params={"q": "customers"})

    async def _synchronize(
        self: object, *args: object, **kwargs: object
    ) -> MetadataSyncResult:
        del self, args, kwargs
        return MetadataSyncResult(
            data_source_id=uuid.UUID(source["id"]),
            status=MetadataSyncStatus.SUCCESS,
            started_at=None,
            completed_at=None,
            schema_count=0,
            table_count=0,
            column_count=0,
            relationship_count=0,
            error_message=None,
        )

    monkeypatch.setattr(
        "app.api.routes.metadata.MetadataSyncService.synchronize",
        _synchronize,
    )
    sync = client.post(f"{PREFIX}/{source['id']}/metadata/sync", headers=headers)
    limited_sync = client.post(
        f"{PREFIX}/{source['id']}/metadata/sync", headers=headers
    )

    register_connector(DataSourceType.POSTGRESQL, SampleConnector)
    sample = client.post(
        f"{PREFIX}/{source['id']}/tables/{table.id}/sample",
        headers=headers,
        json={"limit": 1},
    )
    limited_sample = client.post(
        f"{PREFIX}/{source['id']}/tables/{table.id}/sample",
        headers=headers,
        json={"limit": 1},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited_search.status_code == 429
    assert sync.status_code == 200
    assert limited_sync.status_code == 429
    assert sample.status_code == 200
    assert limited_sample.status_code == 429


def test_sample_and_sync_logs_omit_secrets_and_rows(
    client: TestClient,
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    _, table, _ = _seed_catalog(db_session, uuid.UUID(source["id"]))
    register_connector(
        DataSourceType.POSTGRESQL,
        lambda: SampleConnector(
            rows=[{"id": 1, "email": RAW_EMAIL, "name": RAW_PASSWORD}]
        ),
    )
    sample_logger = logging.getLogger("app.services.sample_data")
    sample_logger.addFilter(RedactingFilter())
    route_logger = logging.getLogger("app.api.routes.metadata")
    route_logger.addFilter(RedactingFilter())

    with caplog.at_level(logging.INFO):
        sample = client.post(
            f"{PREFIX}/{source['id']}/tables/{table.id}/sample",
            headers=_auth_header(owner),
            json={"limit": 1},
        )

    async def _fail(
        self: object, *args: object, **kwargs: object
    ) -> MetadataSyncResult:
        del self, args, kwargs
        raise MetadataSyncError(f"password={CUSTOMER_PASSWORD} uri={CONNECTION_URI}")

    monkeypatch.setattr(
        "app.api.routes.metadata.MetadataSyncService.synchronize",
        _fail,
    )
    with caplog.at_level(logging.INFO):
        sync = client.post(
            f"{PREFIX}/{source['id']}/metadata/sync",
            headers=_auth_header(owner),
        )

    text = caplog.text
    assert sample.status_code == 200
    assert sync.status_code == 502
    assert RAW_EMAIL not in text
    assert RAW_PASSWORD not in text
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert "Bearer " not in text


def test_default_value_is_not_executed_and_secrets_are_redacted(
    client: TestClient,
    db_session: Session,
) -> None:
    owner, _, source = _setup_source(client, db_session)
    source_id = uuid.UUID(source["id"])
    schema = _add_schema(db_session, source_id, "public")
    table = _add_table(db_session, schema.id, "users")
    _add_column(
        db_session,
        table.id,
        "token",
        position=1,
        default_value=f"password={CUSTOMER_PASSWORD}",
        primary_key=False,
        unique=False,
    )

    response = client.get(
        f"{PREFIX}/{source['id']}/tables/{table.id}/columns",
        headers=_auth_header(owner),
    )

    assert response.status_code == 200
    default_value = response.json()["items"][0]["default_value"]
    assert default_value is not None
    assert CUSTOMER_PASSWORD not in default_value
    assert "[REDACTED]" in default_value


def test_openapi_documents_metadata_endpoints_without_secrets(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert f"{PREFIX}/{{data_source_id}}/schemas" in paths
    assert f"{PREFIX}/{{data_source_id}}/tables/{{table_id}}" in paths
    assert f"{PREFIX}/{{data_source_id}}/tables/{{table_id}}/columns" in paths
    assert f"{PREFIX}/{{data_source_id}}/relationships" in paths
    assert f"{PREFIX}/{{data_source_id}}/metadata/search" in paths
    assert f"{PREFIX}/{{data_source_id}}/metadata/sync" in paths
    assert f"{PREFIX}/{{data_source_id}}/metadata/sync-status" in paths
    assert f"{PREFIX}/{{data_source_id}}/tables/{{table_id}}/sample" in paths
    search = paths[f"{PREFIX}/{{data_source_id}}/metadata/search"]["get"]
    assert search["summary"]
    assert "security" in search or "HTTPBearer" in str(schema.get("components", {}))
    sample = paths[f"{PREFIX}/{{data_source_id}}/tables/{{table_id}}/sample"]["post"]
    assert "SampleDataRequest" in str(sample) or "limit" in str(sample)
    rendered = str(schema)
    assert CUSTOMER_PASSWORD not in rendered
    assert "include_unmasked" not in rendered
    components = schema["components"]["schemas"]
    assert "password" not in components["MetadataColumnResponse"]["properties"]
    assert "encrypted_password" not in components["SampleDataResponse"]["properties"]


def test_metadata_routes_do_not_import_drivers_or_crypto() -> None:
    tree = ast.parse(_ROUTE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported.isdisjoint(_FORBIDDEN_IMPORTS)
    source = _ROUTE_PATH.read_text(encoding="utf-8")
    assert "include_unmasked" not in source
    assert "create_connector" not in source
    assert "discover(" not in source
