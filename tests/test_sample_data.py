from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.connectors import (
    ColumnInfo,
    ConnectionTestResult,
    ConnectorConfig,
    ConnectorConnectionError,
    ConnectorQueryError,
    QueryResult,
    SchemaInfo,
    TableInfo,
    UnsupportedConnectorError,
    UnsupportedOperationError,
    register_connector,
)
from app.core.logging import RedactingFilter
from app.db.models import (
    DataSource,
    DataSourceColumn,
    DataSourceConnection,
    DataSourceSchema,
    DataSourceTable,
    DataSourceTableType,
    DataSourceType,
    Organization,
    User,
    UserRole,
    Workspace,
)
from app.enums import ColumnSensitivity
from app.services.credentials import encrypt_secret
from app.services.data_masking import REDACTED
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceNotFoundError,
)
from app.services.sample_data import SampleDataService
from app.services.sample_data_exceptions import (
    SampleDataLimitError,
    SampleQueryError,
    SampleTableNotFoundError,
)
from app.services.sample_serialization import BINARY_PLACEHOLDER, TRUNCATED_SUFFIX
from tests.conftest import run_async

CUSTOMER_PASSWORD = "CustomerSampleSecret!@#"
CONNECTION_URI = (
    f"postgresql://readonly:{CUSTOMER_PASSWORD}@db.internal.example:5432/analytics"
)
RAW_EMAIL = "john.doe@example.com"
RAW_PHONE = "9876543210"
RAW_NAME = "John Doe"
RAW_PASSWORD = "hunter2-password"
RAW_TOKEN = "tok_live_secret_value"


class SampleConnector:
    """Test double. Not a production connector."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])
        self.calls: list[tuple[str, str, tuple[str, ...], int]] = []
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.connected = False
        self.config: ConnectorConfig | None = None
        self.fail_connect: Exception | None = None
        self.fail_error: Exception | None = None

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
    source_type: DataSourceType = DataSourceType.POSTGRESQL,
    with_connection: bool = True,
    password: str = CUSTOMER_PASSWORD,
    workspace: Workspace | None = None,
    name: str = "Production Analytics DB",
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
        type=source_type,
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
                encrypted_password=encrypt_secret(password),
                ssl_mode="prefer",
            )
        )
        db_session.flush()
    return data_source


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
) -> DataSourceTable:
    table = DataSourceTable(schema_id=schema_id, name=name, table_type=table_type)
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
    description: str | None = None,
) -> DataSourceColumn:
    column = DataSourceColumn(
        table_id=table_id,
        name=name,
        ordinal_position=position,
        data_type=data_type,
        database_type=database_type,
        is_nullable=nullable,
        is_primary_key=position == 1,
        is_unique=position == 1,
        description=description,
    )
    db_session.add(column)
    db_session.flush()
    return column


def _seed_customers_table(
    db_session: Session,
    data_source: DataSource,
    *,
    schema_name: str = "public",
    table_name: str = "customers",
    table_type: DataSourceTableType = DataSourceTableType.TABLE,
    extra_columns: int = 0,
) -> DataSourceTable:
    schema = _add_schema(db_session, data_source.id, schema_name)
    table = _add_table(db_session, schema.id, table_name, table_type=table_type)
    columns = [
        ("id", "integer", "int4"),
        ("first_name", "string", "text"),
        ("last_name", "string", "text"),
        ("email", "string", "text"),
        ("phone", "string", "text"),
        ("amount", "numeric", "numeric"),
        ("customer_uuid", "uuid", "uuid"),
        ("born_on", "date", "date"),
        ("created_at", "datetime", "timestamptz"),
        ("metadata", "json", "jsonb"),
        ("avatar", "binary", "bytea"),
        ("notes", "string", "text"),
        ("password", "string", "text"),
        ("api_token", "string", "text"),
        ("contact", "string", "text"),
    ]
    for position, (name, data_type, database_type) in enumerate(columns, start=1):
        _add_column(
            db_session,
            table.id,
            name,
            position=position,
            data_type=data_type,
            database_type=database_type,
        )
    for offset in range(extra_columns):
        _add_column(
            db_session,
            table.id,
            f"extra_{offset + 1}",
            position=len(columns) + offset + 1,
            data_type="string",
            database_type="text",
        )
    return table


def _sample_row() -> dict[str, Any]:
    return {
        "id": 12,
        "first_name": "John",
        "last_name": "Doe",
        "email": RAW_EMAIL,
        "phone": RAW_PHONE,
        "amount": Decimal("19.90"),
        "customer_uuid": UUID("550e8400-e29b-41d4-a716-446655440000"),
        "born_on": date(1990, 5, 1),
        "created_at": datetime(2024, 3, 15, 12, 30, tzinfo=UTC),
        "metadata": {"plan": "pro", "labels": ["vip"]},
        "avatar": b"\x00\xffsecret-bytes",
        "notes": "hello",
        "password": RAW_PASSWORD,
        "api_token": RAW_TOKEN,
        "contact": "ada@example.com",
    }


def _register_sample_connector(
    rows: list[dict[str, Any]] | None = None,
) -> list[SampleConnector]:
    created: list[SampleConnector] = []

    def _builder() -> SampleConnector:
        connector = SampleConnector(rows)
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    return created


def _get_sample(
    db_session: Session,
    data_source: DataSource,
    table: DataSourceTable,
    *,
    workspace_id: UUID | None = None,
    limit: int | None = None,
) -> Any:
    service = SampleDataService(db_session)
    return run_async(
        service.get_sample(
            data_source.id,
            table.id,
            workspace_id=workspace_id or data_source.workspace_id,
            limit=limit,
        )
    )


def test_default_row_limit(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row() for _ in range(50)])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table)

    assert result.row_limit == 10
    assert result.row_count == 10
    assert created[0].calls[0][3] == 10


def test_custom_valid_row_limit(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row() for _ in range(50)])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=5)

    assert result.row_limit == 5
    assert result.row_count == 5
    assert created[0].calls[0][3] == 5


def test_excessive_row_limit_is_capped_in_sql(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row() for _ in range(200)])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=10_000)

    assert result.row_limit == 100
    assert result.row_count == 100
    assert created[0].calls[0][3] == 100
    assert len(created[0].rows) == 200


@pytest.mark.parametrize("limit", [0, -1, True])
def test_zero_negative_and_bool_limits_are_rejected(
    db_session: Session,
    limit: object,
) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)
    service = SampleDataService(db_session)

    with pytest.raises(SampleDataLimitError):
        run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
                limit=limit,  # type: ignore[arg-type]
            )
        )


def test_column_order_follows_ordinal_position(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)

    names = [column.name for column in result.columns]
    assert names[:5] == ["id", "first_name", "last_name", "email", "phone"]
    assert list(result.rows[0].keys())[:5] == names[:5]
    assert created[0].calls[0][2][:5] == (
        "id",
        "first_name",
        "last_name",
        "email",
        "phone",
    )


def test_wide_table_column_limit(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.services.sample_data.settings.SAMPLE_DATA_MAX_COLUMNS", 20)
    created = _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source, extra_columns=30)

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.column_count == 20
    assert result.truncated_columns is True
    assert len(created[0].calls[0][2]) == 20
    assert "*" not in created[0].calls[0][2]


def test_normal_table_does_not_truncate_columns(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.truncated_columns is False
    assert result.column_count == 15


def test_valid_table_sample(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.data_source_id == data_source.id
    assert result.table_id == table.id
    assert result.schema_name == "public"
    assert result.table_name == "customers"
    assert result.rows[0]["id"] == 12
    assert created[0].connect_calls == 1
    assert created[0].disconnect_calls == 1
    assert created[0].connected is False


def test_missing_table_raises(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    service = SampleDataService(db_session)

    with pytest.raises(SampleTableNotFoundError, match="not found"):
        run_async(
            service.get_sample(
                data_source.id,
                uuid.uuid4(),
                workspace_id=data_source.workspace_id,
            )
        )


def test_table_from_another_data_source_is_isolated(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row()])
    source_a = _seed_data_source(db_session, name="Workspace A DB")
    source_b = _seed_data_source(
        db_session,
        workspace=source_a.workspace,
        name="Workspace B DB",
    )
    table_b = _seed_customers_table(db_session, source_b)
    service = SampleDataService(db_session)

    with pytest.raises(SampleTableNotFoundError, match="not found"):
        run_async(
            service.get_sample(
                source_a.id,
                table_b.id,
                workspace_id=source_a.workspace_id,
            )
        )
    assert created == []


def test_workspace_isolation_hides_other_data_sources(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    source_a = _seed_data_source(db_session, name="Workspace A DB")
    organization = source_a.workspace.organization
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Finance",
        slug=f"finance-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()
    source_b = _seed_data_source(
        db_session,
        workspace=workspace_b,
        name="Workspace B DB",
    )
    table_b = _seed_customers_table(db_session, source_b)
    service = SampleDataService(db_session)

    with pytest.raises(DataSourceNotFoundError, match="not found"):
        run_async(
            service.get_sample(
                source_b.id,
                table_b.id,
                workspace_id=source_a.workspace_id,
            )
        )


def test_empty_table_returns_empty_rows(db_session: Session) -> None:
    _register_sample_connector([])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table)

    assert result.rows == ()
    assert result.row_count == 0
    assert result.columns


def test_views_can_be_sampled(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(
        db_session,
        data_source,
        table_name="customer_view",
        table_type=DataSourceTableType.VIEW,
    )

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.table_type is DataSourceTableType.VIEW
    assert result.row_count == 1


def test_null_sample_values_stay_null(db_session: Session) -> None:
    row = _sample_row()
    row["email"] = None
    row["phone"] = None
    _register_sample_connector([row])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.rows[0]["email"] is None
    assert result.rows[0]["phone"] is None
    assert result.rows[0]["id"] == 12


def test_pii_is_masked_by_default(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)
    row = result.rows[0]

    assert row["email"] == "j***@example.com"
    assert row["phone"] == "******3210"
    assert row["first_name"] == "J***"
    assert row["last_name"] == "D**"
    assert row["password"] == REDACTED
    assert row["api_token"] == REDACTED
    assert RAW_EMAIL not in str(row)
    assert RAW_PHONE not in str(row)
    assert RAW_PASSWORD not in str(row)
    assert RAW_TOKEN not in str(row)
    assert "John Doe" not in str(row)


def test_value_level_detection_masks_public_email_column(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.rows[0]["contact"] == "a***@example.com"
    assert "ada@example.com" not in str(result.rows[0])


def test_serialization_of_special_types(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)
    row = result.rows[0]

    assert row["customer_uuid"] == "550e8400-e29b-41d4-a716-446655440000"
    assert row["created_at"] == datetime(2024, 3, 15, 12, 30, tzinfo=UTC).isoformat()
    assert row["amount"] == "19.90"
    assert row["metadata"] == {"plan": "pro", "labels": ["vip"]}
    assert row["avatar"] == BINARY_PLACEHOLDER
    assert row["born_on"] == "1990-05-01"


def test_large_text_is_bounded(db_session: Session) -> None:
    row = _sample_row()
    row["notes"] = "n" * 50_000
    _register_sample_connector([row])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)
    notes = result.rows[0]["notes"]

    assert isinstance(notes, str)
    assert notes.endswith(TRUNCATED_SUFFIX)
    assert len(notes) < 2_000
    assert "n" * 50_000 not in notes


def test_sql_injection_payload_is_treated_as_identifier(db_session: Session) -> None:
    created = _register_sample_connector([{"id": 1}])
    data_source = _seed_data_source(db_session)
    schema = _add_schema(db_session, data_source.id, 'public"; DROP TABLE users; --')
    table = _add_table(db_session, schema.id, "customers")
    _add_column(db_session, table.id, "id", position=1, data_type="integer")

    result = _get_sample(db_session, data_source, table, limit=1)

    assert result.row_count == 1
    schema_name, table_name, columns, limit = created[0].calls[0]
    assert schema_name == 'public"; DROP TABLE users; --'
    assert table_name == "customers"
    assert columns == ("id",)
    assert limit == 1


def test_select_star_is_not_used(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    _get_sample(db_session, data_source, table, limit=1)

    _schema, _table, columns, limit = created[0].calls[0]
    assert "*" not in columns
    assert "id" in columns
    assert limit == 1


def test_entire_table_is_not_loaded(db_session: Session) -> None:
    created = _register_sample_connector([_sample_row() for _ in range(500)])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=3)

    assert result.row_count == 3
    assert created[0].calls[0][3] == 3


def test_no_unmasked_bypass_parameter() -> None:
    signature = inspect.signature(SampleDataService.get_sample)
    assert "include_unmasked" not in signature.parameters
    assert "unmasked" not in signature.parameters
    source = inspect.getsource(SampleDataService)
    assert "include_unmasked" not in source
    assert "fastapi" not in source.lower()
    assert "HTTPException" not in source


def test_unsupported_connector_type_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, source_type=DataSourceType.MYSQL)
    table = _seed_customers_table(db_session, data_source)
    service = SampleDataService(db_session)

    with pytest.raises(UnsupportedConnectorError):
        run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
            )
        )


def test_missing_connection_configuration_raises(db_session: Session) -> None:
    data_source = _seed_data_source(db_session, with_connection=False)
    table = _seed_customers_table(db_session, data_source)
    service = SampleDataService(db_session)

    with pytest.raises(ConnectionConfigurationError, match="missing"):
        run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
            )
        )


def test_query_errors_and_logs_omit_credentials(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created: list[SampleConnector] = []

    def _builder() -> SampleConnector:
        connector = SampleConnector([_sample_row()])
        connector.fail_error = ConnectorQueryError(
            f"password={CUSTOMER_PASSWORD} connection_string={CONNECTION_URI}"
        )
        created.append(connector)
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    logger = logging.getLogger("app.services.sample_data")
    logger.addFilter(RedactingFilter())
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)
    assert data_source.connection is not None
    ciphertext = data_source.connection.encrypted_password
    service = SampleDataService(db_session)

    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        pytest.raises(SampleQueryError) as exc_info,
    ):
        run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
            )
        )

    message = str(exc_info.value)
    text = caplog.text
    assert CUSTOMER_PASSWORD not in message
    assert CONNECTION_URI not in message
    assert "OperationalError" not in message
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert ciphertext not in text
    assert RAW_EMAIL not in text
    assert RAW_PASSWORD not in text
    assert str(data_source.id) in text
    assert str(table.id) in text
    assert created[0].disconnect_calls == 1


def test_timeout_maps_to_safe_error(db_session: Session) -> None:
    def _builder() -> SampleConnector:
        connector = SampleConnector([_sample_row()])
        connector.fail_error = ConnectorConnectionError(
            "PostgreSQL connection timed out"
        )
        return connector

    register_connector(DataSourceType.POSTGRESQL, _builder)
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)
    service = SampleDataService(db_session)

    with pytest.raises(SampleQueryError, match="Unable to retrieve sample data"):
        run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
            )
        )


def test_successful_sample_logs_counts_without_secrets(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _register_sample_connector([_sample_row()])
    logger = logging.getLogger("app.services.sample_data")
    logger.addFilter(RedactingFilter())
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)
    service = SampleDataService(db_session)

    with caplog.at_level(logging.INFO, logger=logger.name):
        result = run_async(
            service.get_sample(
                data_source.id,
                table.id,
                workspace_id=data_source.workspace_id,
                limit=1,
            )
        )

    text = caplog.text
    assert "Sample data started" in text
    assert "Sample data completed" in text
    assert f"row_count={result.row_count}" in text
    assert f"column_count={result.column_count}" in text
    assert str(data_source.id) in text
    assert str(table.id) in text
    assert CUSTOMER_PASSWORD not in text
    assert CONNECTION_URI not in text
    assert RAW_EMAIL not in text
    assert RAW_PASSWORD not in text
    assert "encrypted_password" not in text


def test_openapi_sample_routes_have_no_unmasked_bypass(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    joined = " ".join(paths)
    assert "/pii" not in joined.lower()
    assert "include_unmasked" not in str(schema)
    sample_path = "/api/v1/data-sources/{data_source_id}/tables/{table_id}/sample"
    assert sample_path in paths
    properties = (
        schema.get("components", {})
        .get("schemas", {})
        .get("SampleDataRequest", {})
        .get("properties", {})
    )
    assert "include_unmasked" not in properties
    assert "unmasked" not in properties


def test_column_sensitivity_metadata(db_session: Session) -> None:
    _register_sample_connector([_sample_row()])
    data_source = _seed_data_source(db_session)
    table = _seed_customers_table(db_session, data_source)

    result = _get_sample(db_session, data_source, table, limit=1)
    by_name = {column.name: column for column in result.columns}

    assert by_name["id"].sensitivity is ColumnSensitivity.PUBLIC
    assert by_name["email"].sensitivity is ColumnSensitivity.PII
    assert by_name["password"].sensitivity is ColumnSensitivity.SECRET
    assert by_name["email"].masked is True
    assert by_name["password"].masked is True
    assert by_name["id"].masked is False
