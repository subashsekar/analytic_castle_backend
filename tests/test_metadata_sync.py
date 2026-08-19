from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceConnection,
    DataSourceMetadataSync,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.enums import (
    DataSourceRelationshipType,
    DataSourceTableType,
    MetadataSyncStatus,
)
from app.services.credentials import encrypt_secret
from app.services.data_source_connections import DataSourceNotFoundError
from app.services.discovery_types import (
    DiscoveredColumn,
    DiscoveredRelationship,
    DiscoveredSchema,
    DiscoveredTable,
    DiscoveryLimits,
    DiscoveryResult,
    SchemaFilter,
)
from app.services.metadata_sync import MetadataSyncService
from tests.conftest import run_async

CUSTOMER_PASSWORD = "CustomerSyncSecret!@#"


class StubDiscovery:
    """Test double. Not a production discovery service."""

    def __init__(
        self,
        result: DiscoveryResult | None = None,
        *,
        fail: Exception | None = None,
    ) -> None:
        self.result = result
        self.fail = fail
        self.calls = 0
        self.workspace_ids: list[uuid.UUID] = []

    async def discover(
        self,
        data_source_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        schema_filter: SchemaFilter | None = None,
        limits: DiscoveryLimits | None = None,
    ) -> DiscoveryResult:
        del data_source_id, schema_filter, limits
        self.calls += 1
        self.workspace_ids.append(workspace_id)
        if self.fail is not None:
            raise self.fail
        assert self.result is not None
        return self.result


def _user() -> User:
    return User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
    )


def _seed_data_source(
    db_session: Session,
    *,
    workspace: Workspace | None = None,
    name: str = "Production Analytics DB",
    with_connection: bool = True,
) -> DataSource:
    user = _user()
    if workspace is None:
        organization = Organization(
            name="AnalyticCastle",
            slug=f"analyticcastle-{uuid.uuid4().hex[:8]}",
        )
        db_session.add_all([user, organization])
        db_session.flush()
        workspace = Workspace(
            organization_id=organization.id,
            name="Analytics",
            slug=f"analytics-{uuid.uuid4().hex[:8]}",
        )
        db_session.add(workspace)
        db_session.flush()
    else:
        db_session.add(user)
        db_session.flush()
    data_source = DataSource(
        workspace_id=workspace.id,
        name=name,
        type=DataSourceType.POSTGRESQL,
        created_by=user.id,
    )
    db_session.add(data_source)
    db_session.flush()
    if with_connection:
        db_session.add(
            DataSourceConnection(
                data_source_id=data_source.id,
                host="db.internal.example",
                port=5432,
                database_name="analytics",
                username="readonly",
                encrypted_password=encrypt_secret(CUSTOMER_PASSWORD),
                ssl_mode="prefer",
            )
        )
        db_session.flush()
    return data_source


def _column(
    schema: str,
    table: str,
    name: str,
    *,
    position: int,
    data_type: str = "integer",
    database_type: str = "int4",
    nullable: bool = False,
    default: str | None = None,
    primary_key: bool = False,
    unique: bool = False,
) -> DiscoveredColumn:
    return DiscoveredColumn(
        schema_name=schema,
        table_name=table,
        column_name=name,
        ordinal_position=position,
        data_type=data_type,
        database_type=database_type,
        is_nullable=nullable,
        default_value=default,
        is_primary_key=primary_key,
        is_unique=unique,
    )


def _relationship(
    source_schema: str,
    source_table: str,
    source_column: str,
    target_schema: str,
    target_table: str,
    target_column: str,
    *,
    constraint_name: str | None = "fk_rel",
    relationship_type: DataSourceRelationshipType = DataSourceRelationshipType.MANY_TO_ONE,
    ordinal_position: int = 1,
    constraint_column_count: int = 1,
) -> DiscoveredRelationship:
    return DiscoveredRelationship(
        source_schema=source_schema,
        source_table=source_table,
        source_column=source_column,
        target_schema=target_schema,
        target_table=target_table,
        target_column=target_column,
        relationship_type=relationship_type,
        constraint_name=constraint_name,
        ordinal_position=ordinal_position,
        constraint_column_count=constraint_column_count,
    )


def _result(
    *,
    schemas: Sequence[str] = ("public",),
    tables: Sequence[tuple[str, str, DataSourceTableType]] = (
        ("public", "users", DataSourceTableType.TABLE),
        ("public", "orders", DataSourceTableType.TABLE),
    ),
    columns: Sequence[DiscoveredColumn] | None = None,
    relationships: Sequence[DiscoveredRelationship] | None = None,
) -> DiscoveryResult:
    if columns is None:
        columns = (
            _column("public", "users", "id", position=1, primary_key=True, unique=True),
            _column(
                "public",
                "users",
                "email",
                position=2,
                data_type="string",
                database_type="text",
                unique=True,
            ),
            _column(
                "public", "orders", "id", position=1, primary_key=True, unique=True
            ),
            _column("public", "orders", "user_id", position=2),
        )
    if relationships is None:
        relationships = (
            _relationship(
                "public",
                "orders",
                "user_id",
                "public",
                "users",
                "id",
                constraint_name="fk_orders_user_id",
            ),
        )
    return DiscoveryResult(
        schemas=tuple(DiscoveredSchema(name=name) for name in schemas),
        tables=tuple(
            DiscoveredTable(schema_name=schema, table_name=name, table_type=table_type)
            for schema, name, table_type in tables
        ),
        columns=tuple(columns),
        relationships=tuple(relationships),
    )


def _sync(
    db_session: Session,
    data_source: DataSource,
    result: DiscoveryResult,
) -> MetadataSyncService:
    service = MetadataSyncService(db_session, discovery_service=StubDiscovery(result))
    outcome = run_async(
        service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )
    assert outcome.status is MetadataSyncStatus.SUCCESS
    return service


def _schema_names(db_session: Session, data_source_id: uuid.UUID) -> list[str]:
    return list(
        db_session.scalars(
            select(DataSourceSchema.name)
            .where(DataSourceSchema.data_source_id == data_source_id)
            .order_by(DataSourceSchema.name)
        ).all()
    )


def _table_keys(
    db_session: Session, data_source_id: uuid.UUID
) -> list[tuple[str, str]]:
    rows = db_session.execute(
        select(DataSourceSchema.name, DataSourceTable.name)
        .join(DataSourceTable, DataSourceTable.schema_id == DataSourceSchema.id)
        .where(DataSourceSchema.data_source_id == data_source_id)
        .order_by(DataSourceSchema.name, DataSourceTable.name)
    ).all()
    return [(str(schema), str(table)) for schema, table in rows]


def _column_keys(
    db_session: Session,
    data_source_id: uuid.UUID,
) -> list[tuple[str, str, str]]:
    rows = db_session.execute(
        select(DataSourceSchema.name, DataSourceTable.name, DataSourceColumn.name)
        .join(DataSourceTable, DataSourceTable.schema_id == DataSourceSchema.id)
        .join(DataSourceColumn, DataSourceColumn.table_id == DataSourceTable.id)
        .where(DataSourceSchema.data_source_id == data_source_id)
        .order_by(DataSourceSchema.name, DataSourceTable.name, DataSourceColumn.name)
    ).all()
    return [(str(schema), str(table), str(column)) for schema, table, column in rows]


def _relationship_keys(
    db_session: Session,
    data_source_id: uuid.UUID,
) -> list[tuple[str, str, str, str, str, str, str | None]]:
    items = (
        db_session.scalars(
            select(DataSourceRelationship)
            .options(
                joinedload(DataSourceRelationship.source_table).joinedload(
                    DataSourceTable.schema
                ),
                joinedload(DataSourceRelationship.source_column),
                joinedload(DataSourceRelationship.target_table).joinedload(
                    DataSourceTable.schema
                ),
                joinedload(DataSourceRelationship.target_column),
            )
            .join(
                DataSourceTable,
                DataSourceRelationship.source_table_id == DataSourceTable.id,
            )
            .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(DataSourceSchema.data_source_id == data_source_id)
        )
        .unique()
        .all()
    )
    keys = [
        (
            relation.source_table.schema.name,
            relation.source_table.name,
            relation.source_column.name,
            relation.target_table.schema.name,
            relation.target_table.name,
            relation.target_column.name,
            relation.constraint_name,
        )
        for relation in items
    ]
    keys.sort()
    return keys


def test_initial_sync_populates_hierarchy(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    result = _result()
    _sync(db_session, data_source, result)

    assert _schema_names(db_session, data_source.id) == ["public"]
    assert _table_keys(db_session, data_source.id) == [
        ("public", "orders"),
        ("public", "users"),
    ]
    assert ("public", "orders", "user_id") in _column_keys(db_session, data_source.id)
    assert _relationship_keys(db_session, data_source.id) == [
        (
            "public",
            "orders",
            "user_id",
            "public",
            "users",
            "id",
            "fk_orders_user_id",
        )
    ]

    status = MetadataSyncService(db_session).get_status(
        data_source.id, workspace_id=data_source.workspace_id
    )
    assert status.status is MetadataSyncStatus.SUCCESS
    assert status.schema_count == len(result.schemas)
    assert status.table_count == len(result.tables)
    assert status.column_count == len(result.columns)
    assert status.relationship_count == len(result.relationships)
    assert status.started_at is not None
    assert status.completed_at is not None
    assert status.error_message is None


def test_repeated_sync_is_idempotent(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    result = _result()
    service = MetadataSyncService(db_session, discovery_service=StubDiscovery(result))
    first = run_async(
        service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )
    schema_ids = set(db_session.scalars(select(DataSourceSchema.id)).all())
    table_ids = set(db_session.scalars(select(DataSourceTable.id)).all())
    column_ids = set(db_session.scalars(select(DataSourceColumn.id)).all())
    relationship_ids = set(db_session.scalars(select(DataSourceRelationship.id)).all())

    second = run_async(
        service.synchronize(data_source.id, workspace_id=data_source.workspace_id)
    )

    assert first.status is MetadataSyncStatus.SUCCESS
    assert second.status is MetadataSyncStatus.SUCCESS
    assert db_session.scalar(select(func.count()).select_from(DataSourceSchema)) == 1
    assert db_session.scalar(select(func.count()).select_from(DataSourceTable)) == 2
    assert db_session.scalar(select(func.count()).select_from(DataSourceColumn)) == 4
    assert (
        db_session.scalar(select(func.count()).select_from(DataSourceRelationship)) == 1
    )
    assert set(db_session.scalars(select(DataSourceSchema.id)).all()) == schema_ids
    assert set(db_session.scalars(select(DataSourceTable.id)).all()) == table_ids
    assert set(db_session.scalars(select(DataSourceColumn.id)).all()) == column_ids
    assert (
        set(db_session.scalars(select(DataSourceRelationship.id)).all())
        == relationship_ids
    )


def test_sync_adds_schema_table_column_and_relationship(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(db_session, data_source, _result())

    expanded = _result(
        schemas=("analytics", "public"),
        tables=(
            ("analytics", "events", DataSourceTableType.TABLE),
            ("public", "orders", DataSourceTableType.TABLE),
            ("public", "products", DataSourceTableType.TABLE),
            ("public", "users", DataSourceTableType.TABLE),
        ),
        columns=(
            _column("public", "users", "id", position=1, primary_key=True, unique=True),
            _column(
                "public",
                "users",
                "email",
                position=2,
                data_type="string",
                database_type="text",
                unique=True,
            ),
            _column(
                "public",
                "users",
                "name",
                position=3,
                data_type="string",
                database_type="text",
                nullable=True,
            ),
            _column(
                "public", "orders", "id", position=1, primary_key=True, unique=True
            ),
            _column("public", "orders", "user_id", position=2),
            _column(
                "public", "products", "id", position=1, primary_key=True, unique=True
            ),
            _column(
                "analytics", "events", "id", position=1, primary_key=True, unique=True
            ),
            _column("analytics", "events", "user_id", position=2),
        ),
        relationships=(
            _relationship(
                "public",
                "orders",
                "user_id",
                "public",
                "users",
                "id",
                constraint_name="fk_orders_user_id",
            ),
            _relationship(
                "analytics",
                "events",
                "user_id",
                "public",
                "users",
                "id",
                constraint_name="fk_events_user_id",
            ),
        ),
    )
    _sync(db_session, data_source, expanded)

    assert _schema_names(db_session, data_source.id) == ["analytics", "public"]
    assert ("public", "products") in _table_keys(db_session, data_source.id)
    assert ("public", "users", "name") in _column_keys(db_session, data_source.id)
    assert (
        "analytics",
        "events",
        "user_id",
        "public",
        "users",
        "id",
        "fk_events_user_id",
    ) in _relationship_keys(db_session, data_source.id)


def test_sync_removes_stale_schema_table_column_and_relationship(
    db_session: Session,
) -> None:
    data_source = _seed_data_source(db_session)
    _sync(
        db_session,
        data_source,
        _result(
            schemas=("public", "reporting"),
            tables=(
                ("public", "orders", DataSourceTableType.TABLE),
                ("public", "products", DataSourceTableType.TABLE),
                ("public", "users", DataSourceTableType.TABLE),
                ("reporting", "users", DataSourceTableType.VIEW),
            ),
            columns=(
                _column(
                    "public", "users", "id", position=1, primary_key=True, unique=True
                ),
                _column(
                    "public",
                    "users",
                    "email",
                    position=2,
                    data_type="string",
                    database_type="text",
                ),
                _column(
                    "public", "orders", "id", position=1, primary_key=True, unique=True
                ),
                _column("public", "orders", "user_id", position=2),
                _column("public", "products", "id", position=1, primary_key=True),
                _column("reporting", "users", "id", position=1, primary_key=True),
            ),
            relationships=(
                _relationship(
                    "public",
                    "orders",
                    "user_id",
                    "public",
                    "users",
                    "id",
                    constraint_name="fk_orders_user_id",
                ),
            ),
        ),
    )

    _sync(db_session, data_source, _result())

    assert _schema_names(db_session, data_source.id) == ["public"]
    assert _table_keys(db_session, data_source.id) == [
        ("public", "orders"),
        ("public", "users"),
    ]
    assert ("public", "products") not in _table_keys(db_session, data_source.id)
    assert ("public", "users", "email") in _column_keys(db_session, data_source.id)
    assert len(_relationship_keys(db_session, data_source.id)) == 1


def test_sync_applies_column_and_table_changes(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(
        db_session,
        data_source,
        _result(
            tables=(("public", "users", DataSourceTableType.TABLE),),
            columns=(
                _column(
                    "public",
                    "users",
                    "id",
                    position=1,
                    data_type="integer",
                    database_type="int4",
                    nullable=False,
                    default=None,
                    primary_key=True,
                    unique=True,
                ),
                _column(
                    "public",
                    "users",
                    "label",
                    position=2,
                    data_type="string",
                    database_type="text",
                    nullable=True,
                ),
            ),
            relationships=(),
        ),
    )
    schema = db_session.scalar(select(DataSourceSchema))
    assert schema is not None
    table = db_session.scalar(select(DataSourceTable))
    assert table is not None
    table.description = "keep me"
    label = db_session.scalar(
        select(DataSourceColumn).where(DataSourceColumn.name == "label")
    )
    assert label is not None
    label.description = "human label"
    db_session.flush()

    _sync(
        db_session,
        data_source,
        _result(
            tables=(("public", "users", DataSourceTableType.VIEW),),
            columns=(
                _column(
                    "public",
                    "users",
                    "id",
                    position=1,
                    data_type="string",
                    database_type="uuid",
                    nullable=True,
                    default="gen_random_uuid()",
                    primary_key=False,
                    unique=False,
                ),
                _column(
                    "public",
                    "users",
                    "label",
                    position=2,
                    data_type="string",
                    database_type="varchar",
                    nullable=False,
                    unique=True,
                ),
            ),
            relationships=(),
        ),
    )

    db_session.refresh(table)
    db_session.refresh(label)
    identity = db_session.scalar(
        select(DataSourceColumn).where(DataSourceColumn.name == "id")
    )
    assert identity is not None
    assert table.table_type is DataSourceTableType.VIEW
    assert table.description == "keep me"
    assert identity.data_type == "string"
    assert identity.database_type == "uuid"
    assert identity.is_nullable is True
    assert identity.default_value == "gen_random_uuid()"
    assert identity.is_primary_key is False
    assert identity.is_unique is False
    assert label.is_nullable is False
    assert label.is_unique is True
    assert label.description == "human label"


def test_sync_preserves_other_data_source_metadata(db_session: Session) -> None:
    data_source_a = _seed_data_source(db_session, name="Workspace A DB")
    data_source_b = _seed_data_source(
        db_session,
        workspace=data_source_a.workspace,
        name="Workspace B DB",
    )
    _sync(db_session, data_source_a, _result())
    _sync(
        db_session,
        data_source_b,
        _result(
            schemas=("public",),
            tables=(("public", "users", DataSourceTableType.TABLE),),
            columns=(
                _column(
                    "public", "users", "id", position=1, primary_key=True, unique=True
                ),
            ),
            relationships=(),
        ),
    )
    b_schema_ids = set(
        db_session.scalars(
            select(DataSourceSchema.id).where(
                DataSourceSchema.data_source_id == data_source_b.id
            )
        ).all()
    )

    _sync(
        db_session,
        data_source_a,
        _result(
            tables=(("public", "users", DataSourceTableType.TABLE),),
            columns=(
                _column(
                    "public", "users", "id", position=1, primary_key=True, unique=True
                ),
            ),
            relationships=(),
        ),
    )

    assert _schema_names(db_session, data_source_b.id) == ["public"]
    assert _table_keys(db_session, data_source_b.id) == [("public", "users")]
    assert (
        set(
            db_session.scalars(
                select(DataSourceSchema.id).where(
                    DataSourceSchema.data_source_id == data_source_b.id
                )
            ).all()
        )
        == b_schema_ids
    )
    assert _table_keys(db_session, data_source_a.id) == [("public", "users")]


def test_workspace_isolation_hides_other_data_sources(db_session: Session) -> None:
    data_source_a = _seed_data_source(db_session, name="Workspace A DB")
    organization = data_source_a.workspace.organization
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Finance",
        slug=f"finance-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()
    data_source_b = _seed_data_source(
        db_session,
        workspace=workspace_b,
        name="Workspace B DB",
    )
    service = MetadataSyncService(
        db_session, discovery_service=StubDiscovery(_result())
    )

    with pytest.raises(DataSourceNotFoundError, match="not found"):
        run_async(
            service.synchronize(
                data_source_b.id, workspace_id=data_source_a.workspace_id
            )
        )
    with pytest.raises(DataSourceNotFoundError, match="not found"):
        service.get_status(data_source_b.id, workspace_id=data_source_a.workspace_id)

    outcome = run_async(
        service.synchronize(data_source_a.id, workspace_id=data_source_a.workspace_id)
    )
    assert outcome.status is MetadataSyncStatus.SUCCESS
    assert _schema_names(db_session, data_source_b.id) == []


def test_sync_status_pending_before_first_run(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    status = MetadataSyncService(db_session).get_status(
        data_source.id, workspace_id=data_source.workspace_id
    )
    assert status.status is MetadataSyncStatus.PENDING
    assert status.schema_count is None
    assert db_session.scalar(select(DataSourceMetadataSync)) is None


def test_deleting_data_source_cascades_sync_state(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(db_session, data_source, _result())
    data_source_id = data_source.id
    db_session.delete(data_source)
    db_session.flush()
    assert (
        db_session.scalar(
            select(DataSourceMetadataSync).where(
                DataSourceMetadataSync.data_source_id == data_source_id
            )
        )
        is None
    )


def test_self_referencing_and_composite_relationships(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    result = _result(
        tables=(
            ("public", "employees", DataSourceTableType.TABLE),
            ("public", "tenant_customers", DataSourceTableType.TABLE),
            ("public", "tenant_orders", DataSourceTableType.TABLE),
        ),
        columns=(
            _column(
                "public", "employees", "id", position=1, primary_key=True, unique=True
            ),
            _column("public", "employees", "manager_id", position=2, nullable=True),
            _column(
                "public",
                "tenant_customers",
                "tenant_id",
                position=1,
                data_type="uuid",
                database_type="uuid",
                primary_key=True,
            ),
            _column(
                "public",
                "tenant_customers",
                "id",
                position=2,
                primary_key=True,
            ),
            _column(
                "public",
                "tenant_orders",
                "tenant_id",
                position=1,
                data_type="uuid",
                database_type="uuid",
            ),
            _column("public", "tenant_orders", "customer_id", position=2),
        ),
        relationships=(
            _relationship(
                "public",
                "employees",
                "manager_id",
                "public",
                "employees",
                "id",
                constraint_name="fk_employees_manager_id",
            ),
            _relationship(
                "public",
                "tenant_orders",
                "tenant_id",
                "public",
                "tenant_customers",
                "tenant_id",
                constraint_name="fk_tenant_orders_customer",
                ordinal_position=1,
                constraint_column_count=2,
            ),
            _relationship(
                "public",
                "tenant_orders",
                "customer_id",
                "public",
                "tenant_customers",
                "id",
                constraint_name="fk_tenant_orders_customer",
                ordinal_position=2,
                constraint_column_count=2,
            ),
        ),
    )
    _sync(db_session, data_source, result)
    keys = _relationship_keys(db_session, data_source.id)
    assert (
        "public",
        "employees",
        "manager_id",
        "public",
        "employees",
        "id",
        "fk_employees_manager_id",
    ) in keys
    assert len(keys) == 3


def test_get_status_does_not_trust_caller_workspace(db_session: Session) -> None:
    data_source = _seed_data_source(db_session)
    _sync(db_session, data_source, _result())
    with pytest.raises(DataSourceNotFoundError):
        MetadataSyncService(db_session).get_status(
            data_source.id, workspace_id=uuid.uuid4()
        )
