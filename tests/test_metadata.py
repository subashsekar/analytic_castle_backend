import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateIndex

from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceConnection,
    DataSourceRelationship,
    DataSourceRelationshipType,
    DataSourceSchema,
    DataSourceTable,
    DataSourceTableType,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.schemas import (
    DataSourceColumnRead,
    DataSourceRelationshipRead,
    DataSourceSchemaRead,
    DataSourceTableRead,
)

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


def _schema(data_source_id: uuid.UUID, **overrides: object) -> DataSourceSchema:
    values: dict[str, object] = {
        "data_source_id": data_source_id,
        "name": "public",
    }
    values.update(overrides)
    return DataSourceSchema(**values)


def _table(schema_id: uuid.UUID, **overrides: object) -> DataSourceTable:
    values: dict[str, object] = {
        "schema_id": schema_id,
        "name": "users",
        "table_type": DataSourceTableType.TABLE,
    }
    values.update(overrides)
    return DataSourceTable(**values)


def _column(table_id: uuid.UUID, **overrides: object) -> DataSourceColumn:
    values: dict[str, object] = {
        "table_id": table_id,
        "name": "id",
        "ordinal_position": 1,
        "data_type": "integer",
        "database_type": "int4",
        "is_nullable": False,
        "is_primary_key": True,
        "is_unique": True,
    }
    values.update(overrides)
    return DataSourceColumn(**values)


def _seed_data_source(
    db_session: Session,
) -> tuple[User, Organization, Workspace, DataSource]:
    user = _user()
    organization = _organization()
    db_session.add_all([user, organization])
    db_session.flush()
    workspace = _workspace(organization.id)
    db_session.add(workspace)
    db_session.flush()
    data_source = _data_source(workspace.id, user.id)
    db_session.add(data_source)
    db_session.flush()
    return user, organization, workspace, data_source


def test_schema_can_be_created(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)

    schema = _schema(data_source.id, name="public")
    db_session.add(schema)
    db_session.flush()

    assert schema.id is not None
    assert isinstance(schema.id, uuid.UUID)
    assert schema.data_source_id == data_source.id
    assert schema.name == "public"
    assert schema.created_at.tzinfo is not None
    assert schema.updated_at.tzinfo is not None
    assert schema.created_at <= datetime.now(UTC)


def test_schema_belongs_to_data_source(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)

    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    db_session.refresh(data_source)

    assert schema.data_source.id == data_source.id
    assert schema in data_source.schemas


def test_duplicate_schema_name_within_data_source_is_rejected(
    db_session: Session,
) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    db_session.add(_schema(data_source.id, name="public"))
    db_session.flush()
    db_session.add(_schema(data_source.id, name="public"))

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_schema_invalid_data_source_fk_is_rejected(db_session: Session) -> None:
    schema = _schema(uuid.uuid4(), name="public")
    db_session.add(schema)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_same_schema_name_in_different_data_sources_is_allowed(
    db_session: Session,
) -> None:
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()

    schema_a = _schema(data_source_a.id, name="public")
    schema_b = _schema(data_source_b.id, name="public")
    db_session.add_all([schema_a, schema_b])
    db_session.flush()

    assert schema_a.id != schema_b.id
    assert schema_a.name == schema_b.name
    assert schema_a.data_source_id != schema_b.data_source_id


def test_table_can_be_created(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()

    table = _table(schema.id, name="users")
    db_session.add(table)
    db_session.flush()

    assert table.id is not None
    assert isinstance(table.id, uuid.UUID)
    assert table.schema_id == schema.id
    assert table.name == "users"
    assert table.table_type == DataSourceTableType.TABLE
    assert table.description is None
    assert table.created_at.tzinfo is not None
    assert table.updated_at.tzinfo is not None


def test_table_belongs_to_schema(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()

    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    db_session.refresh(schema)

    assert table.schema.id == schema.id
    assert table in schema.tables


def test_duplicate_table_name_within_schema_is_rejected(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    db_session.add(_table(schema.id, name="users"))
    db_session.flush()
    db_session.add(_table(schema.id, name="users"))

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_table_invalid_schema_fk_is_rejected(db_session: Session) -> None:
    table = _table(uuid.uuid4(), name="users")
    db_session.add(table)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_same_table_name_in_different_schemas_is_allowed(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    public_schema = _schema(data_source.id, name="public")
    analytics_schema = _schema(data_source.id, name="analytics")
    db_session.add_all([public_schema, analytics_schema])
    db_session.flush()

    public_users = _table(public_schema.id, name="users")
    analytics_users = _table(analytics_schema.id, name="users")
    db_session.add_all([public_users, analytics_users])
    db_session.flush()

    assert public_users.id != analytics_users.id
    assert public_users.name == analytics_users.name
    assert public_users.schema_id != analytics_users.schema_id


def test_table_type_accepts_table_and_view(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()

    table = _table(schema.id, name="orders", table_type=DataSourceTableType.TABLE)
    view = _table(
        schema.id, name="order_summaries", table_type=DataSourceTableType.VIEW
    )
    db_session.add_all([table, view])
    db_session.flush()

    assert table.table_type == DataSourceTableType.TABLE
    assert view.table_type == DataSourceTableType.VIEW
    assert DataSourceTableType.TABLE.value == "TABLE"
    assert DataSourceTableType.VIEW.value == "VIEW"


def test_table_type_enum_rejects_unknown_value(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()

    with pytest.raises(DBAPIError):
        db_session.execute(
            text(
                "UPDATE data_source_tables SET table_type = 'MATERIALIZED_VIEW' "
                "WHERE id = :id"
            ),
            {"id": table.id},
        )
        db_session.flush()


def test_column_can_be_created(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()

    column = _column(
        table.id,
        name="email",
        ordinal_position=2,
        data_type="string",
        database_type="varchar",
        is_nullable=False,
        default_value=None,
        is_primary_key=False,
        is_unique=True,
        description="Login email",
    )
    db_session.add(column)
    db_session.flush()

    assert column.id is not None
    assert isinstance(column.id, uuid.UUID)
    assert column.table_id == table.id
    assert column.name == "email"
    assert column.ordinal_position == 2
    assert column.data_type == "string"
    assert column.database_type == "varchar"
    assert column.is_nullable is False
    assert column.default_value is None
    assert column.is_primary_key is False
    assert column.is_unique is True
    assert column.description == "Login email"
    assert column.created_at.tzinfo is not None
    assert column.updated_at.tzinfo is not None


def test_column_invalid_table_fk_is_rejected(db_session: Session) -> None:
    column = _column(uuid.uuid4())
    db_session.add(column)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_duplicate_column_name_within_table_is_rejected(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    db_session.add(_column(table.id, name="email", ordinal_position=1))
    db_session.flush()
    db_session.add(
        _column(
            table.id,
            name="email",
            ordinal_position=2,
            is_primary_key=False,
            is_unique=True,
        )
    )

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_column_belongs_to_table(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()

    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    db_session.refresh(table)

    assert column.table.id == table.id
    assert column in table.columns


def test_column_preserves_ordinal_position_and_types(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()

    id_column = _column(table.id, name="id", ordinal_position=1)
    first_name = _column(
        table.id,
        name="first_name",
        ordinal_position=2,
        data_type="string",
        database_type="varchar",
        is_nullable=True,
        is_primary_key=False,
        is_unique=False,
    )
    last_name = _column(
        table.id,
        name="last_name",
        ordinal_position=3,
        data_type="string",
        database_type="varchar",
        is_nullable=True,
        is_primary_key=False,
        is_unique=False,
    )
    email = _column(
        table.id,
        name="email",
        ordinal_position=4,
        data_type="string",
        database_type="varchar",
        is_nullable=False,
        is_primary_key=False,
        is_unique=True,
    )
    db_session.add_all([id_column, first_name, last_name, email])
    db_session.flush()
    db_session.refresh(table)

    ordered = sorted(table.columns, key=lambda item: item.ordinal_position)
    assert [column.name for column in ordered] == [
        "id",
        "first_name",
        "last_name",
        "email",
    ]
    assert id_column.data_type == "integer"
    assert id_column.database_type == "int4"


def test_column_nullable_primary_key_and_unique_flags(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()

    id_column = _column(
        table.id,
        name="id",
        ordinal_position=1,
        is_nullable=False,
        is_primary_key=True,
        is_unique=False,
    )
    tenant_id = _column(
        table.id,
        name="tenant_id",
        ordinal_position=2,
        data_type="uuid",
        database_type="uuid",
        is_nullable=False,
        is_primary_key=True,
        is_unique=False,
    )
    email = _column(
        table.id,
        name="email",
        ordinal_position=3,
        data_type="string",
        database_type="varchar",
        is_nullable=True,
        is_primary_key=False,
        is_unique=True,
        default_value="NULL",
    )
    db_session.add_all([id_column, tenant_id, email])
    db_session.flush()

    assert id_column.is_primary_key is True
    assert tenant_id.is_primary_key is True
    assert email.is_primary_key is False
    assert email.is_unique is True
    assert email.is_nullable is True
    assert id_column.is_unique is False
    assert email.default_value == "NULL"


def test_relationship_can_be_created(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    customers = _table(schema.id, name="customers")
    orders = _table(schema.id, name="orders")
    db_session.add_all([customers, orders])
    db_session.flush()
    customer_id = _column(customers.id, name="id")
    order_customer_id = _column(
        orders.id,
        name="customer_id",
        ordinal_position=2,
        is_primary_key=False,
        is_unique=False,
    )
    db_session.add_all([customer_id, order_customer_id])
    db_session.flush()

    relation = DataSourceRelationship(
        source_table_id=orders.id,
        source_column_id=order_customer_id.id,
        target_table_id=customers.id,
        target_column_id=customer_id.id,
        relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
        constraint_name="fk_orders_customer_id",
    )
    db_session.add(relation)
    db_session.flush()
    db_session.refresh(orders)
    db_session.refresh(customers)

    assert relation.id is not None
    assert relation.source_table.id == orders.id
    assert relation.target_table.id == customers.id
    assert relation.source_column.id == order_customer_id.id
    assert relation.target_column.id == customer_id.id
    assert relation.relationship_type == DataSourceRelationshipType.MANY_TO_ONE
    assert relation.constraint_name == "fk_orders_customer_id"
    assert relation in orders.source_relationships
    assert relation in customers.target_relationships
    assert relation.created_at.tzinfo is not None
    assert relation.updated_at.tzinfo is not None


def test_self_referencing_relationship_is_allowed(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    employees = _table(schema.id, name="employees")
    db_session.add(employees)
    db_session.flush()
    employee_id = _column(employees.id, name="id")
    manager_id = _column(
        employees.id,
        name="manager_id",
        ordinal_position=2,
        is_primary_key=False,
        is_unique=False,
        is_nullable=True,
    )
    db_session.add_all([employee_id, manager_id])
    db_session.flush()

    relation = DataSourceRelationship(
        source_table_id=employees.id,
        source_column_id=manager_id.id,
        target_table_id=employees.id,
        target_column_id=employee_id.id,
        relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
        constraint_name="fk_employees_manager_id",
    )
    db_session.add(relation)
    db_session.flush()
    db_session.refresh(employees)

    assert relation.source_table_id == relation.target_table_id
    assert relation.source_table is relation.target_table
    assert relation in employees.source_relationships
    assert relation in employees.target_relationships


def test_relationship_invalid_foreign_keys_are_rejected(db_session: Session) -> None:
    relation = DataSourceRelationship(
        source_table_id=uuid.uuid4(),
        source_column_id=uuid.uuid4(),
        target_table_id=uuid.uuid4(),
        target_column_id=uuid.uuid4(),
        relationship_type=DataSourceRelationshipType.ONE_TO_MANY,
    )
    db_session.add(relation)

    with pytest.raises(IntegrityError):
        db_session.flush()


def test_relationship_rejects_column_from_a_different_table(
    db_session: Session,
) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    orders = _table(schema.id, name="orders")
    customers = _table(schema.id, name="customers")
    db_session.add_all([orders, customers])
    db_session.flush()
    order_id = _column(orders.id)
    customer_id = _column(customers.id)
    db_session.add_all([order_id, customer_id])
    db_session.flush()

    db_session.add(
        DataSourceRelationship(
            source_table_id=orders.id,
            source_column_id=customer_id.id,
            target_table_id=customers.id,
            target_column_id=customer_id.id,
            relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_cross_data_source_relationship_is_not_enforced_at_database_layer(
    db_session: Session,
) -> None:
    """Same-data-source membership is application/service validation.

    Composite foreign keys only guarantee that each column belongs to its
    declared table. Metadata sync must reject cross-data-source relationships.
    """
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()
    schema_a = _schema(data_source_a.id, name="public")
    schema_b = _schema(data_source_b.id, name="public")
    db_session.add_all([schema_a, schema_b])
    db_session.flush()
    orders = _table(schema_a.id, name="orders")
    customers = _table(schema_b.id, name="customers")
    db_session.add_all([orders, customers])
    db_session.flush()
    order_customer_id = _column(
        orders.id,
        name="customer_id",
        is_primary_key=False,
        is_unique=False,
    )
    customer_id = _column(customers.id, name="id")
    db_session.add_all([order_customer_id, customer_id])
    db_session.flush()

    relation = DataSourceRelationship(
        source_table_id=orders.id,
        source_column_id=order_customer_id.id,
        target_table_id=customers.id,
        target_column_id=customer_id.id,
        relationship_type=DataSourceRelationshipType.MANY_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    assert relation.id is not None
    assert schema_a.data_source_id != schema_b.data_source_id


def test_relationship_type_enum_rejects_unknown_value(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    employees = _table(schema.id, name="employees")
    db_session.add(employees)
    db_session.flush()
    employee_id = _column(employees.id, name="id")
    manager_id = _column(
        employees.id,
        name="manager_id",
        ordinal_position=2,
        is_primary_key=False,
        is_unique=False,
    )
    db_session.add_all([employee_id, manager_id])
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=employees.id,
        source_column_id=manager_id.id,
        target_table_id=employees.id,
        target_column_id=employee_id.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_MANY,
    )
    db_session.add(relation)
    db_session.flush()

    with pytest.raises(DBAPIError):
        db_session.execute(
            text(
                "UPDATE data_source_relationships "
                "SET relationship_type = 'NOT_A_TYPE' WHERE id = :id"
            ),
            {"id": relation.id},
        )
        db_session.flush()


def test_deleting_data_source_cascades_metadata(db_session: Session) -> None:
    user, organization, workspace, data_source = _seed_data_source(db_session)
    connection = _connection(data_source.id)
    db_session.add(connection)
    db_session.flush()
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    user_id = user.id
    organization_id = organization.id
    workspace_id = workspace.id
    data_source_id = data_source.id
    schema_id = schema.id
    table_id = table.id
    column_id = column.id
    relationship_id = relation.id

    db_session.delete(data_source)
    db_session.flush()

    assert (
        db_session.scalar(select(DataSource).where(DataSource.id == data_source_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceSchema).where(DataSourceSchema.id == schema_id)
        )
        is None
    )
    assert (
        db_session.scalar(select(DataSourceTable).where(DataSourceTable.id == table_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceColumn).where(DataSourceColumn.id == column_id)
        )
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceRelationship).where(
                DataSourceRelationship.id == relationship_id
            )
        )
        is None
    )
    assert db_session.scalar(select(User).where(User.id == user_id)) is not None
    assert (
        db_session.scalar(
            select(Organization).where(Organization.id == organization_id)
        )
        is not None
    )
    remaining_workspace = db_session.scalar(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    assert remaining_workspace is not None
    assert remaining_workspace.id == workspace_id


def test_deleting_data_source_does_not_remove_other_data_source_metadata(
    db_session: Session,
) -> None:
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()

    schema_a = _schema(data_source_a.id, name="public")
    schema_b = _schema(data_source_b.id, name="public")
    db_session.add_all([schema_a, schema_b])
    db_session.flush()
    table_b = _table(schema_b.id)
    db_session.add(table_b)
    db_session.flush()

    schema_b_id = schema_b.id
    table_b_id = table_b.id
    db_session.delete(data_source_a)
    db_session.flush()

    assert (
        db_session.scalar(
            select(DataSourceSchema).where(DataSourceSchema.id == schema_b_id)
        )
        is not None
    )
    assert (
        db_session.scalar(
            select(DataSourceTable).where(DataSourceTable.id == table_b_id)
        )
        is not None
    )


def test_metadata_is_isolated_by_workspace_and_data_source(db_session: Session) -> None:
    user, organization, workspace_a, data_source_a = _seed_data_source(db_session)
    workspace_b = _workspace(organization.id, name="Finance", slug="finance")
    db_session.add(workspace_b)
    db_session.flush()
    data_source_b = _data_source(workspace_b.id, user.id, name="Finance DB")
    db_session.add(data_source_b)
    db_session.flush()

    schema_a = _schema(data_source_a.id, name="public")
    schema_b = _schema(data_source_b.id, name="public")
    db_session.add_all([schema_a, schema_b])
    db_session.flush()
    table_a = _table(schema_a.id, name="users")
    table_b = _table(schema_b.id, name="users")
    db_session.add_all([table_a, table_b])
    db_session.flush()

    assert table_a.id != table_b.id
    assert table_a.name == table_b.name
    assert schema_a.data_source_id == data_source_a.id
    assert schema_b.data_source_id == data_source_b.id
    assert data_source_a.workspace_id == workspace_a.id
    assert data_source_b.workspace_id == workspace_b.id
    assert table_a not in schema_b.tables
    assert table_b not in schema_a.tables


def test_sql_delete_of_data_source_cascades_metadata(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()

    data_source_id = data_source.id
    schema_id = schema.id
    table_id = table.id
    column_id = column.id
    db_session.execute(delete(DataSource).where(DataSource.id == data_source_id))
    db_session.flush()

    assert (
        db_session.scalar(
            select(DataSourceSchema).where(DataSourceSchema.id == schema_id)
        )
        is None
    )
    assert (
        db_session.scalar(select(DataSourceTable).where(DataSourceTable.id == table_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceColumn).where(DataSourceColumn.id == column_id)
        )
        is None
    )


def test_metadata_read_schemas_map_from_models(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id, name="public")
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id, name="users", description="Application users")
    db_session.add(table)
    db_session.flush()
    column = _column(table.id, default_value="nextval('users_id_seq')")
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
        constraint_name="fk_users_id",
    )
    db_session.add(relation)
    db_session.flush()

    schema_payload = DataSourceSchemaRead.model_validate(schema)
    table_payload = DataSourceTableRead.model_validate(table)
    column_payload = DataSourceColumnRead.model_validate(column)
    relation_payload = DataSourceRelationshipRead.model_validate(relation)

    assert schema_payload.data_source_id == data_source.id
    assert table_payload.schema_id == schema.id
    assert table_payload.table_type == DataSourceTableType.TABLE
    assert column_payload.ordinal_position == 1
    assert column_payload.data_type == "integer"
    assert column_payload.database_type == "int4"
    assert relation_payload.relationship_type == DataSourceRelationshipType.ONE_TO_ONE


def test_metadata_read_schemas_exclude_credentials(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    db_session.add(_connection(data_source.id))
    db_session.flush()
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    payloads = [
        DataSourceSchemaRead.model_validate(schema).model_dump(),
        DataSourceTableRead.model_validate(table).model_dump(),
        DataSourceColumnRead.model_validate(column).model_dump(),
        DataSourceRelationshipRead.model_validate(relation).model_dump(),
    ]
    credential_fields = {
        "password",
        "encrypted_password",
        "credentials",
        "connection",
        "host",
        "username",
        "database_name",
    }
    for payload in payloads:
        assert credential_fields.isdisjoint(payload)
        assert ENCRYPTED_PASSWORD not in str(payload)

    for schema_cls in (
        DataSourceSchemaRead,
        DataSourceTableRead,
        DataSourceColumnRead,
        DataSourceRelationshipRead,
    ):
        assert "password" not in schema_cls.model_fields
        assert "encrypted_password" not in schema_cls.model_fields
        assert "credentials" not in schema_cls.model_fields


def test_deleting_schema_cascades_tables_columns_and_relationships(
    db_session: Session,
) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    table_id = table.id
    column_id = column.id
    relationship_id = relation.id
    db_session.delete(schema)
    db_session.flush()

    assert (
        db_session.scalar(select(DataSourceTable).where(DataSourceTable.id == table_id))
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceColumn).where(DataSourceColumn.id == column_id)
        )
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceRelationship).where(
                DataSourceRelationship.id == relationship_id
            )
        )
        is None
    )
    assert (
        db_session.scalar(select(DataSource).where(DataSource.id == data_source.id))
        is not None
    )


def test_deleting_table_cascades_columns_and_relationships(
    db_session: Session,
) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    column_id = column.id
    relationship_id = relation.id
    db_session.delete(table)
    db_session.flush()

    assert (
        db_session.scalar(
            select(DataSourceColumn).where(DataSourceColumn.id == column_id)
        )
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceRelationship).where(
                DataSourceRelationship.id == relationship_id
            )
        )
        is None
    )
    assert (
        db_session.scalar(
            select(DataSourceSchema).where(DataSourceSchema.id == schema.id)
        )
        is not None
    )


def test_deleting_column_cascades_relationships(db_session: Session) -> None:
    _, _, _, data_source = _seed_data_source(db_session)
    schema = _schema(data_source.id)
    db_session.add(schema)
    db_session.flush()
    table = _table(schema.id)
    db_session.add(table)
    db_session.flush()
    column = _column(table.id)
    db_session.add(column)
    db_session.flush()
    relation = DataSourceRelationship(
        source_table_id=table.id,
        source_column_id=column.id,
        target_table_id=table.id,
        target_column_id=column.id,
        relationship_type=DataSourceRelationshipType.ONE_TO_ONE,
    )
    db_session.add(relation)
    db_session.flush()

    relationship_id = relation.id
    db_session.delete(column)
    db_session.flush()

    assert (
        db_session.scalar(
            select(DataSourceRelationship).where(
                DataSourceRelationship.id == relationship_id
            )
        )
        is None
    )
    assert (
        db_session.scalar(select(DataSourceTable).where(DataSourceTable.id == table.id))
        is not None
    )


def test_metadata_search_and_relationship_indexes_exist() -> None:
    schema_indexes = {index.name for index in DataSourceSchema.__table__.indexes}
    table_indexes = {index.name for index in DataSourceTable.__table__.indexes}
    column_indexes = {index.name for index in DataSourceColumn.__table__.indexes}
    relationship_indexes = {
        index.name for index in DataSourceRelationship.__table__.indexes
    }
    schema_constraints = {item.name for item in DataSourceSchema.__table__.constraints}
    table_constraints = {item.name for item in DataSourceTable.__table__.constraints}
    column_constraints = {item.name for item in DataSourceColumn.__table__.constraints}

    assert "ix_data_source_schemas_data_source_id" in schema_indexes
    assert "uq_data_source_schemas_data_source_id_name" in schema_constraints
    assert "ix_data_source_schemas_data_source_id_lower_name" in schema_indexes
    assert "ix_data_source_tables_schema_id" in table_indexes
    assert "uq_data_source_tables_schema_id_name" in table_constraints
    assert "ix_data_source_tables_schema_id_lower_name" in table_indexes
    assert "ix_data_source_columns_table_id" in column_indexes
    assert "uq_data_source_columns_table_id_name" in column_constraints
    assert "ix_data_source_columns_table_id_lower_name" in column_indexes
    assert "ix_data_source_relationships_source_table_id" in relationship_indexes
    assert "ix_data_source_relationships_target_table_id" in relationship_indexes
    assert "ix_data_source_relationships_source_column_id" in relationship_indexes
    assert "ix_data_source_relationships_target_column_id" in relationship_indexes


def test_search_indexes_compile_to_lower_name_expressions() -> None:
    expected = {
        DataSourceSchema: "ix_data_source_schemas_data_source_id_lower_name",
        DataSourceTable: "ix_data_source_tables_schema_id_lower_name",
        DataSourceColumn: "ix_data_source_columns_table_id_lower_name",
    }
    dialect = postgresql.dialect()
    for model, index_name in expected.items():
        index = next(
            item for item in model.__table__.indexes if item.name == index_name
        )
        compiled = str(CreateIndex(index).compile(dialect=dialect)).lower()
        assert "create index" in compiled
        assert "lower(name)" in compiled


def test_case_insensitive_schema_lookup_is_scoped_to_data_source(
    db_session: Session,
) -> None:
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()
    db_session.add_all(
        [
            _schema(data_source_a.id, name="Analytics"),
            _schema(data_source_b.id, name="analytics"),
        ]
    )
    db_session.flush()

    found = db_session.scalars(
        select(DataSourceSchema).where(
            DataSourceSchema.data_source_id == data_source_a.id,
            func.lower(DataSourceSchema.name) == "analytics",
        )
    ).all()

    assert [item.name for item in found] == ["Analytics"]
    assert all(item.data_source_id == data_source_a.id for item in found)


def test_schema_lookup_by_data_source_id_is_isolated(db_session: Session) -> None:
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()
    db_session.add_all(
        [
            _schema(data_source_a.id, name="analytics"),
            _schema(data_source_b.id, name="analytics"),
        ]
    )
    db_session.flush()

    names = db_session.scalars(
        select(DataSourceSchema.name).where(
            DataSourceSchema.data_source_id == data_source_a.id
        )
    ).all()

    assert names == ["analytics"]
    assert (
        db_session.scalar(
            select(func.count())
            .select_from(DataSourceSchema)
            .where(DataSourceSchema.data_source_id == data_source_a.id)
        )
        == 1
    )


def test_tables_and_columns_are_scoped_by_data_source_id(db_session: Session) -> None:
    user, _, workspace, data_source_a = _seed_data_source(db_session)
    data_source_b = _data_source(workspace.id, user.id, name="Warehouse")
    db_session.add(data_source_b)
    db_session.flush()
    schema_a = _schema(data_source_a.id, name="public")
    schema_b = _schema(data_source_b.id, name="public")
    db_session.add_all([schema_a, schema_b])
    db_session.flush()
    table_a = _table(schema_a.id, name="customers")
    table_b = _table(schema_b.id, name="customers")
    db_session.add_all([table_a, table_b])
    db_session.flush()
    db_session.add_all(
        [
            _column(table_a.id, name="email"),
            _column(table_b.id, name="email"),
        ]
    )
    db_session.flush()

    table_ids = db_session.scalars(
        select(DataSourceTable.id)
        .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
        .where(DataSourceSchema.data_source_id == data_source_a.id)
    ).all()
    column_names = db_session.scalars(
        select(DataSourceColumn.name)
        .join(DataSourceTable, DataSourceColumn.table_id == DataSourceTable.id)
        .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
        .where(DataSourceSchema.data_source_id == data_source_a.id)
    ).all()

    assert table_ids == [table_a.id]
    assert column_names == ["email"]
    assert table_b.id not in table_ids
