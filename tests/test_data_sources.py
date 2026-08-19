import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session

from app.core.logging import redact_secret
from app.db.models import (
    DataSource,
    DataSourceConnection,
    DataSourceStatus,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.schemas import DataSourceConnectionRead, DataSourceRead

ENCRYPTED_PASSWORD = "enc:not-a-plaintext-password"


def _user(**overrides: object) -> User:
    values: dict[str, object] = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": f"ada-{uuid.uuid4().hex[:8]}@example.com",
        "password_hash": "hashed-password",
        "role": UserRole.USER,
    }
    values.update(overrides)
    return User(**values)


def _organization(**overrides: object) -> Organization:
    values: dict[str, object] = {
        "name": "AnalyticCastle",
        "slug": f"analyticcastle-{uuid.uuid4().hex[:8]}",
    }
    values.update(overrides)
    return Organization(**values)


def _workspace(organization_id: uuid.UUID, **overrides: object) -> Workspace:
    values: dict[str, object] = {
        "organization_id": organization_id,
        "name": "Analytics",
        "slug": f"analytics-{uuid.uuid4().hex[:8]}",
    }
    values.update(overrides)
    return Workspace(**values)


def _data_source(
    workspace_id: uuid.UUID,
    created_by: uuid.UUID,
    **overrides: object,
) -> DataSource:
    values: dict[str, object] = {
        "workspace_id": workspace_id,
        "name": "Production Analytics DB",
        "type": DataSourceType.POSTGRESQL,
        "created_by": created_by,
    }
    values.update(overrides)
    return DataSource(**values)


def _connection(data_source_id: uuid.UUID, **overrides: object) -> DataSourceConnection:
    values: dict[str, object] = {
        "data_source_id": data_source_id,
        "host": "db.internal.example",
        "database_name": "analytics",
        "username": "readonly",
        "encrypted_password": ENCRYPTED_PASSWORD,
    }
    values.update(overrides)
    return DataSourceConnection(**values)


def _seed_workspace(db_session: Session) -> tuple[User, Organization, Workspace]:
    user = _user()
    organization = _organization()
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = _workspace(organization.id)
    db_session.add(workspace)
    db_session.flush()
    return user, organization, workspace


def test_data_source_can_be_created(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)

    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    assert data_source.id is not None
    assert isinstance(data_source.id, uuid.UUID)
    assert data_source.workspace_id == workspace.id
    assert data_source.name == "Production Analytics DB"
    assert data_source.type == DataSourceType.POSTGRESQL
    assert data_source.status == DataSourceStatus.INACTIVE
    assert data_source.created_by == user.id
    assert data_source.last_tested_at is None
    assert data_source.created_at.tzinfo is not None
    assert data_source.updated_at.tzinfo is not None
    assert data_source.created_at <= datetime.now(UTC)


def test_data_source_belongs_to_workspace(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)

    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    db_session.refresh(workspace)

    assert data_source.workspace.id == workspace.id
    assert data_source in workspace.data_sources


def test_data_source_belongs_to_creator(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)

    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    db_session.refresh(user)

    assert data_source.created_by_user.id == user.id
    assert data_source in user.created_data_sources


def test_data_source_name_is_required(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    db_session.add(
        DataSource(
            workspace_id=workspace.id,
            type=DataSourceType.POSTGRESQL,
            created_by=user.id,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_data_source_type_is_required(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    db_session.add(
        DataSource(
            workspace_id=workspace.id,
            name="Sales Database",
            created_by=user.id,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_data_source_name_cannot_be_empty(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)

    with pytest.raises(ValueError, match="name cannot be empty"):
        _data_source(workspace.id, user.id, name="   ")


def test_database_rejects_blank_data_source_name(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)

    with pytest.raises(IntegrityError):
        db_session.execute(
            text(
                "INSERT INTO data_sources "
                "(id, workspace_id, name, type, created_by) "
                "VALUES (gen_random_uuid(), :workspace_id, '   ', "
                "'POSTGRESQL', :user_id)"
            ),
            {"workspace_id": workspace.id, "user_id": user.id},
        )


def test_data_source_status_defaults_to_inactive(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    assert data_source.status == DataSourceStatus.INACTIVE

    loaded = db_session.execute(
        text("SELECT status::text FROM data_sources WHERE id = :id"),
        {"id": data_source.id},
    ).scalar_one()
    assert loaded == DataSourceStatus.INACTIVE.value


def test_data_source_type_enum_accepts_postgresql(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(
        workspace.id,
        user.id,
        type=DataSourceType.POSTGRESQL,
    )
    db_session.add(data_source)
    db_session.flush()

    assert data_source.type == DataSourceType.POSTGRESQL
    assert DataSourceType.MYSQL.value == "MYSQL"
    assert DataSourceType.CSV.value == "CSV"
    assert DataSourceType.EXCEL.value == "EXCEL"
    assert DataSourceType.GOOGLE_SHEETS.value == "GOOGLE_SHEETS"
    assert DataSourceType.REST_API.value == "REST_API"


def test_data_source_type_enum_rejects_unknown_value(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    with pytest.raises(DBAPIError):
        db_session.execute(
            text("UPDATE data_sources SET type = 'NOT_A_TYPE' WHERE id = :id"),
            {"id": data_source.id},
        )
        db_session.flush()


def test_data_source_invalid_workspace_fk_is_rejected(db_session: Session) -> None:
    user = _user()
    db_session.add(user)
    db_session.flush()

    data_source = _data_source(uuid.uuid4(), user.id)
    db_session.add(data_source)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_data_source_connection_can_be_created(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()

    assert connection.id is not None
    assert isinstance(connection.id, uuid.UUID)
    assert connection.data_source_id == data_source.id
    assert connection.host == "db.internal.example"
    assert connection.port == 5432
    assert connection.database_name == "analytics"
    assert connection.username == "readonly"
    assert connection.encrypted_password == ENCRYPTED_PASSWORD
    assert connection.ssl_mode == "prefer"
    assert connection.created_at.tzinfo is not None
    assert connection.updated_at.tzinfo is not None


def test_data_source_connection_is_one_to_one(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()
    db_session.refresh(data_source)

    assert data_source.connection is connection
    assert connection.data_source.id == data_source.id


def test_duplicate_data_source_connection_is_rejected(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()

    db_session.add(_connection(data_source.id))
    db_session.flush()
    db_session.add(
        _connection(
            data_source.id,
            host="other.example",
            database_name="other",
            username="other",
            encrypted_password="enc:other",
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_data_source_connection_invalid_fk_is_rejected(db_session: Session) -> None:
    connection = _connection(uuid.uuid4())
    db_session.add(connection)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_data_source_cascades_connection(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()
    data_source_id = data_source.id
    connection_id = connection.id

    db_session.delete(data_source)
    db_session.flush()

    assert (
        db_session.scalar(select(DataSource).where(DataSource.id == data_source_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceConnection).where(DataSourceConnection.id == connection_id)
        )
        is None
    )


def test_deleting_workspace_cascades_data_sources(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()
    workspace_id = workspace.id
    data_source_id = data_source.id
    connection_id = connection.id

    db_session.execute(delete(Workspace).where(Workspace.id == workspace_id))
    db_session.flush()

    assert (
        db_session.scalar(select(DataSource).where(DataSource.id == data_source_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceConnection).where(DataSourceConnection.id == connection_id)
        )
        is None
    )


def test_data_source_created_by_is_required(db_session: Session) -> None:
    _, _, workspace = _seed_workspace(db_session)
    db_session.add(
        DataSource(
            workspace_id=workspace.id,
            name="Finance PostgreSQL",
            type=DataSourceType.POSTGRESQL,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_deleting_user_does_not_cascade_to_data_source(db_session: Session) -> None:
    creator = _user()
    organization = _organization()
    db_session.add_all([creator, organization])
    db_session.flush()
    workspace = _workspace(organization.id)
    db_session.add(workspace)
    db_session.flush()
    data_source = _data_source(workspace.id, creator.id)
    db_session.add(data_source)
    db_session.flush()
    data_source_id = data_source.id
    creator_id = creator.id

    with pytest.raises(IntegrityError), db_session.begin_nested():
        db_session.execute(delete(User).where(User.id == creator_id))
        db_session.flush()

    remaining = db_session.scalar(
        select(DataSource).where(DataSource.id == data_source_id)
    )
    assert remaining is not None
    assert remaining.created_by == creator_id
    assert remaining.workspace_id == workspace.id


def test_data_source_is_isolated_to_its_workspace(db_session: Session) -> None:
    user, organization, workspace_a = _seed_workspace(db_session)
    workspace_b = _workspace(organization.id, name="Finance", slug="finance")
    db_session.add(workspace_b)
    db_session.flush()

    data_source_a = _data_source(workspace_a.id, user.id, name="Sales Database")
    db_session.add(data_source_a)
    db_session.flush()
    db_session.refresh(workspace_a)
    db_session.refresh(workspace_b)

    assert data_source_a.workspace_id == workspace_a.id
    assert data_source_a in workspace_a.data_sources
    assert data_source_a not in workspace_b.data_sources
    assert data_source_a.workspace is workspace_a
    assert data_source_a.workspace is not workspace_b


def test_data_source_read_schema_excludes_credentials(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()

    payload = DataSourceRead.model_validate(data_source)
    dumped = payload.model_dump()

    assert "password" not in DataSourceRead.model_fields
    assert "encrypted_password" not in DataSourceRead.model_fields
    assert "password" not in dumped
    assert "encrypted_password" not in dumped
    assert dumped["id"] == data_source.id
    assert dumped["workspace_id"] == workspace.id
    assert dumped["name"] == data_source.name
    assert dumped["type"] == DataSourceType.POSTGRESQL
    assert dumped["status"] == DataSourceStatus.INACTIVE
    assert dumped["created_by"] == user.id
    assert dumped["last_tested_at"] is None


def test_connection_read_schema_excludes_credentials(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()

    payload = DataSourceConnectionRead.model_validate(connection)
    dumped = payload.model_dump()

    assert "password" not in DataSourceConnectionRead.model_fields
    assert "encrypted_password" not in DataSourceConnectionRead.model_fields
    assert "password" not in dumped
    assert "encrypted_password" not in dumped
    assert ENCRYPTED_PASSWORD not in str(dumped)
    assert dumped["host"] == connection.host
    assert dumped["port"] == 5432
    assert dumped["database_name"] == connection.database_name
    assert dumped["username"] == connection.username
    assert dumped["ssl_mode"] == "prefer"


def test_connection_model_has_no_plaintext_password_column() -> None:
    column_names = set(DataSourceConnection.__table__.columns.keys())

    assert "password" not in column_names
    assert "encrypted_password" in column_names


def test_connection_repr_omits_credentials(db_session: Session) -> None:
    user, _, workspace = _seed_workspace(db_session)
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()

    rendered = repr(connection)
    assert "password" not in rendered
    assert "encrypted_password" not in rendered
    assert ENCRYPTED_PASSWORD not in rendered


def test_encrypted_password_is_redacted_in_logs() -> None:
    raw = f"encrypted_password={ENCRYPTED_PASSWORD} password=SuperSecret123!"

    redacted = redact_secret(raw)

    assert ENCRYPTED_PASSWORD not in redacted
    assert "SuperSecret123!" not in redacted
    assert "encrypted_password=[REDACTED]" in redacted
    assert "password=[REDACTED]" in redacted
