from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Sequence

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

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
from app.enums import MetadataSearchType
from app.services.data_source_connections import DataSourceNotFoundError
from app.services.discovery_types import (
    DiscoveredColumn,
    DiscoveredSchema,
    DiscoveredTable,
    DiscoveryResult,
)
from app.services.metadata_persist import persist_discovered_metadata
from app.services.metadata_search import MetadataSearchService, escape_like_pattern
from app.services.metadata_search_exceptions import MetadataSearchLimitError
from app.services.metadata_search_types import MetadataSearchResult
from tests.test_metadata_sync import _seed_data_source, _sync

ENCRYPTED_PASSWORD = "enc:not-a-plaintext-password"
CUSTOMER_PASSWORD = "CustomerSearchSecret!@#"


def _user() -> User:
    return User(
        first_name="Ada",
        last_name="Lovelace",
        email=f"ada-{uuid.uuid4().hex[:8]}@example.com",
        password_hash="hashed-password",
        role=UserRole.USER,
    )


def _seed_search_source(
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
                encrypted_password=ENCRYPTED_PASSWORD,
                ssl_mode="prefer",
            )
        )
        db_session.flush()
    return data_source


def _add_schema(
    db_session: Session, data_source_id: uuid.UUID, name: str
) -> DataSourceSchema:
    schema = DataSourceSchema(data_source_id=data_source_id, name=name)
    db_session.add(schema)
    db_session.flush()
    return schema


def _add_table(
    db_session: Session,
    schema_id: uuid.UUID,
    name: str,
    *,
    description: str | None = None,
) -> DataSourceTable:
    table = DataSourceTable(
        schema_id=schema_id,
        name=name,
        table_type=DataSourceTableType.TABLE,
        description=description,
    )
    db_session.add(table)
    db_session.flush()
    return table


def _add_column(
    db_session: Session,
    table_id: uuid.UUID,
    name: str,
    *,
    position: int = 1,
    description: str | None = None,
) -> DataSourceColumn:
    column = DataSourceColumn(
        table_id=table_id,
        name=name,
        ordinal_position=position,
        data_type="integer",
        database_type="int4",
        is_nullable=False,
        is_primary_key=position == 1,
        is_unique=position == 1,
        description=description,
    )
    db_session.add(column)
    db_session.flush()
    return column


def _seed_customer_catalog(db_session: Session) -> DataSource:
    data_source = _seed_search_source(db_session)
    public_schema = _add_schema(db_session, data_source.id, "public")
    analytics_schema = _add_schema(db_session, data_source.id, "analytics")
    customers = _add_table(
        db_session,
        public_schema.id,
        "customers",
        description="Registered customers",
    )
    _add_table(db_session, public_schema.id, "customer_orders")
    _add_table(db_session, analytics_schema.id, "customer_summary")
    _add_table(db_session, public_schema.id, "my_customer_data")
    orders = _add_table(db_session, public_schema.id, "orders")
    _add_column(db_session, customers.id, "id", position=1)
    _add_column(
        db_session,
        customers.id,
        "email",
        position=2,
        description="Login email of the customer",
    )
    _add_column(
        db_session,
        orders.id,
        "order_id",
        position=1,
        description="Primary key of the order",
    )
    _add_column(
        db_session,
        orders.id,
        "customer_id",
        position=2,
        description="Identifier of the customer placing the order",
    )
    _add_column(db_session, orders.id, "created_at", position=3)
    return data_source


def _search(
    db_session: Session,
    data_source: DataSource,
    query: str,
    *,
    metadata_type: MetadataSearchType | None = None,
    limit: int | None = None,
    workspace_id: uuid.UUID | None = None,
):
    service = MetadataSearchService(db_session)
    return service.search_metadata(
        data_source.id,
        query,
        workspace_id=workspace_id or data_source.workspace_id,
        metadata_type=metadata_type,
        limit=limit,
    )


def _identities(
    results: Sequence[MetadataSearchResult],
) -> list[tuple[str, str, str | None, str | None]]:
    return [
        (
            item.metadata_type.value,
            item.schema_name,
            item.table_name,
            item.column_name,
        )
        for item in results
    ]


def test_schema_search_matches_name(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session,
        data_source,
        "analytics",
        metadata_type=MetadataSearchType.SCHEMA,
    )

    assert [(item.schema_name, item.match_rank) for item in page.results] == [
        ("analytics", 1)
    ]
    assert page.results[0].table_name is None
    assert page.results[0].column_name is None


def test_table_search_matches_name(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session,
        data_source,
        "orders",
        metadata_type=MetadataSearchType.TABLE,
    )

    identities = _identities(page.results)
    assert ("TABLE", "public", "orders", None) in identities
    assert page.results[0].matched_name == "orders"
    assert page.results[0].match_rank == 1


def test_column_search_matches_name(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session,
        data_source,
        "order_id",
        metadata_type=MetadataSearchType.COLUMN,
    )

    assert _identities(page.results) == [("COLUMN", "public", "orders", "order_id")]


def test_search_is_case_insensitive(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    lower_page = _search(
        db_session, data_source, "customer", metadata_type=MetadataSearchType.TABLE
    )
    upper_page = _search(
        db_session, data_source, "CUSTOMER", metadata_type=MetadataSearchType.TABLE
    )
    mixed_page = _search(
        db_session, data_source, "Customer", metadata_type=MetadataSearchType.TABLE
    )

    assert _identities(lower_page.results) == _identities(upper_page.results)
    assert _identities(lower_page.results) == _identities(mixed_page.results)
    assert {item.matched_name for item in lower_page.results} == {
        "customers",
        "customer_orders",
        "customer_summary",
        "my_customer_data",
    }


def test_search_preserves_original_identifier_case(db_session: Session) -> None:
    data_source = _seed_search_source(db_session)
    schema = _add_schema(db_session, data_source.id, "Public")
    _add_table(db_session, schema.id, "Customers")

    page = _search(
        db_session, data_source, "customers", metadata_type=MetadataSearchType.TABLE
    )

    assert page.results[0].schema_name == "Public"
    assert page.results[0].matched_name == "Customers"


def test_partial_and_prefix_matching(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session, data_source, "customer", metadata_type=MetadataSearchType.TABLE
    )
    names = [item.matched_name for item in page.results]

    assert "customers" in names
    assert "customer_orders" in names
    assert "customer_id" not in names
    assert "my_customer_data" in names


def test_prefix_match_customer_underscore(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session, data_source, "customer_", metadata_type=MetadataSearchType.TABLE
    )
    names = [item.matched_name for item in page.results]

    assert "customer_orders" in names
    assert "customer_summary" in names
    assert "customers" not in names
    prefix_ranks = [
        item.match_rank
        for item in page.results
        if item.matched_name == "customer_orders"
    ]
    assert prefix_ranks == [2]


def test_search_ranking_prefers_exact_then_prefix_then_contains(
    db_session: Session,
) -> None:
    data_source = _seed_search_source(db_session)
    schema = _add_schema(db_session, data_source.id, "public")
    _add_table(db_session, schema.id, "customer")
    _add_table(db_session, schema.id, "customer_orders")
    _add_table(db_session, schema.id, "customer_id")
    _add_table(db_session, schema.id, "my_customer_data")

    page = _search(
        db_session, data_source, "customer", metadata_type=MetadataSearchType.TABLE
    )

    assert [item.matched_name for item in page.results] == [
        "customer",
        "customer_id",
        "customer_orders",
        "my_customer_data",
    ]
    assert [item.match_rank for item in page.results] == [1, 2, 2, 3]


def test_description_search_matches_table_and_column(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    table_page = _search(
        db_session,
        data_source,
        "Registered customers",
        metadata_type=MetadataSearchType.TABLE,
    )
    column_page = _search(
        db_session,
        data_source,
        "placing the order",
        metadata_type=MetadataSearchType.COLUMN,
    )

    assert _identities(table_page.results) == [("TABLE", "public", "customers", None)]
    assert table_page.results[0].description == "Registered customers"
    assert table_page.results[0].match_rank == 4
    assert _identities(column_page.results) == [
        ("COLUMN", "public", "orders", "customer_id")
    ]
    assert column_page.results[0].description is not None
    assert "customer" in column_page.results[0].description.lower()


def test_description_search_for_customer_can_match_column(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(
        db_session,
        data_source,
        "customer",
        metadata_type=MetadataSearchType.COLUMN,
    )
    names = {item.matched_name for item in page.results}

    assert "customer_id" in names
    assert "email" in names


def test_no_result_search(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "definitely-not-a-match")

    assert page.results == ()
    assert page.truncated is False
    assert page.result_count == 0


def test_empty_query_returns_no_results(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "   ")

    assert page.results == ()
    assert page.limit == 50


def test_result_limit_default_and_custom(db_session: Session) -> None:
    data_source = _seed_search_source(db_session)
    schema = _add_schema(db_session, data_source.id, "public")
    for index in range(60):
        _add_table(db_session, schema.id, f"customer_{index:02d}")

    default_page = _search(db_session, data_source, "customer")
    custom_page = _search(db_session, data_source, "customer", limit=10)

    assert default_page.limit == 50
    assert default_page.result_count == 50
    assert default_page.truncated is True
    assert custom_page.limit == 10
    assert custom_page.result_count == 10
    assert custom_page.truncated is True


def test_result_limit_rejects_out_of_range_values(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)
    service = MetadataSearchService(db_session)

    with pytest.raises(MetadataSearchLimitError):
        service.search_metadata(
            data_source.id,
            "customer",
            workspace_id=data_source.workspace_id,
            limit=0,
        )
    with pytest.raises(MetadataSearchLimitError):
        service.search_metadata(
            data_source.id,
            "customer",
            workspace_id=data_source.workspace_id,
            limit=101,
        )
    with pytest.raises(MetadataSearchLimitError):
        service.search_metadata(
            data_source.id,
            "customer",
            workspace_id=data_source.workspace_id,
            limit=True,  # type: ignore[arg-type]
        )


def test_result_limit_accepts_configured_maximum(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "customer", limit=100)

    assert page.limit == 100
    assert page.truncated is False
    assert page.result_count < 100


def test_search_requires_workspace_id(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)
    service = MetadataSearchService(db_session)

    with pytest.raises(TypeError):
        service.search_metadata(data_source.id, "customer")  # type: ignore[call-arg]


def test_metadata_type_filtering(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    schemas = _search(
        db_session,
        data_source,
        "public",
        metadata_type=MetadataSearchType.SCHEMA,
    )
    tables = _search(
        db_session,
        data_source,
        "customer",
        metadata_type=MetadataSearchType.TABLE,
    )
    columns = _search(
        db_session,
        data_source,
        "customer",
        metadata_type=MetadataSearchType.COLUMN,
    )

    assert _identities(schemas.results) == [("SCHEMA", "public", None, None)]
    assert all(
        item.metadata_type is MetadataSearchType.TABLE for item in tables.results
    )
    assert all(
        item.metadata_type is MetadataSearchType.COLUMN for item in columns.results
    )
    assert all(item.column_name is None for item in tables.results)
    assert all(item.column_name is not None for item in columns.results)


def test_table_and_column_search_returns_distinct_objects(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "order")
    identities = _identities(page.results)

    assert ("TABLE", "public", "orders", None) in identities
    assert ("COLUMN", "public", "orders", "order_id") in identities
    assert len(identities) == len(set(identities))


def test_search_is_scoped_to_requested_data_source(db_session: Session) -> None:
    data_source_a = _seed_customer_catalog(db_session)
    data_source_b = _seed_search_source(
        db_session,
        workspace=data_source_a.workspace,
        name="Warehouse",
    )
    schema_b = _add_schema(db_session, data_source_b.id, "public")
    _add_table(db_session, schema_b.id, "customers")

    page = _search(
        db_session, data_source_a, "customers", metadata_type=MetadataSearchType.TABLE
    )

    assert _identities(page.results) == [("TABLE", "public", "customers", None)]
    other = _search(
        db_session,
        data_source_b,
        "customers",
        metadata_type=MetadataSearchType.TABLE,
    )
    assert _identities(other.results) == [("TABLE", "public", "customers", None)]
    assert data_source_a.id != data_source_b.id


def test_workspace_isolation_rejects_other_workspace(db_session: Session) -> None:
    data_source_a = _seed_customer_catalog(db_session)
    organization = data_source_a.workspace.organization
    workspace_b = Workspace(
        organization_id=organization.id,
        name="Finance",
        slug=f"finance-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace_b)
    db_session.flush()
    service = MetadataSearchService(db_session)

    with pytest.raises(DataSourceNotFoundError):
        service.search_metadata(
            data_source_a.id,
            "customer",
            workspace_id=workspace_b.id,
        )


def test_organization_isolation_rejects_other_organization(db_session: Session) -> None:
    data_source_a = _seed_customer_catalog(db_session)
    data_source_b = _seed_search_source(db_session, name="Other Org DB")
    schema_b = _add_schema(db_session, data_source_b.id, "public")
    _add_table(db_session, schema_b.id, "customers")
    service = MetadataSearchService(db_session)

    with pytest.raises(DataSourceNotFoundError):
        service.search_metadata(
            data_source_b.id,
            "customers",
            workspace_id=data_source_a.workspace_id,
        )

    page = service.search_metadata(
        data_source_a.id,
        "customers",
        workspace_id=data_source_a.workspace_id,
        metadata_type=MetadataSearchType.TABLE,
    )
    assert all(item.schema_name == "public" for item in page.results)
    assert (
        data_source_a.workspace.organization_id
        != data_source_b.workspace.organization_id
    )


def test_missing_data_source_is_not_found(db_session: Session) -> None:
    data_source = _seed_search_source(db_session)
    service = MetadataSearchService(db_session)

    with pytest.raises(DataSourceNotFoundError):
        service.search_metadata(
            uuid.uuid4(),
            "customer",
            workspace_id=data_source.workspace_id,
        )


def test_search_results_do_not_expose_credentials(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "customer")
    dumped = " ".join(str(item) for item in page.results)

    assert ENCRYPTED_PASSWORD not in dumped
    assert CUSTOMER_PASSWORD not in dumped
    assert "encrypted_password" not in dumped
    assert "password" not in dumped
    for item in page.results:
        assert not hasattr(item, "password")
        assert not hasattr(item, "encrypted_password")
        assert not hasattr(item, "host")
        assert not hasattr(item, "username")
        assert not hasattr(item, "database_name")
        assert not hasattr(item, "connection")


def test_search_does_not_require_connection_configuration(db_session: Session) -> None:
    data_source = _seed_search_source(db_session, with_connection=False)
    schema = _add_schema(db_session, data_source.id, "public")
    _add_table(db_session, schema.id, "customers")

    page = _search(
        db_session, data_source, "customers", metadata_type=MetadataSearchType.TABLE
    )

    assert _identities(page.results) == [("TABLE", "public", "customers", None)]


def test_like_metacharacters_are_literal(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    percent_page = _search(db_session, data_source, "%")
    underscore_page = _search(db_session, data_source, "customer_")
    injection_page = _search(db_session, data_source, "customer%' OR 1=1 --")

    assert percent_page.results == ()
    assert {item.matched_name for item in underscore_page.results} == {
        "customer_orders",
        "customer_summary",
        "customer_id",
        "my_customer_data",
    }
    assert injection_page.results == ()
    assert (
        db_session.scalar(select(func.count()).select_from(DataSourceSchema))
        is not None
    )


def test_sql_injection_payload_cannot_drop_metadata(db_session: Session) -> None:
    data_source = _seed_customer_catalog(db_session)

    page = _search(db_session, data_source, "'; DROP TABLE data_source_tables; --")

    assert page.results == ()
    assert db_session.scalar(select(func.count()).select_from(DataSourceTable)) >= 1


def test_escape_like_pattern_neutralizes_wildcards() -> None:
    assert escape_like_pattern("100%_off\\x") == "100\\%\\_off\\\\x"


def test_search_logs_do_not_include_query_or_credentials(
    db_session: Session,
    caplog: pytest.LogCaptureFixture,
) -> None:
    data_source = _seed_customer_catalog(db_session)

    with caplog.at_level(logging.INFO, logger="app.services.metadata_search"):
        page = _search(db_session, data_source, "customer secret terminology")

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Metadata search completed" in messages
    assert f"data_source_id={data_source.id}" in messages
    assert "search_type=all" in messages
    assert f"result_count={page.result_count}" in messages
    assert "duration_ms=" in messages
    assert "customer secret terminology" not in messages
    assert ENCRYPTED_PASSWORD not in messages
    assert "encrypted_password" not in messages


def test_search_runs_in_postgres_with_limit_and_data_source_scope(
    db_session: Session,
) -> None:
    data_source = _seed_search_source(db_session)
    schemas = [
        _add_schema(db_session, data_source.id, name)
        for name in ("public", "analytics", "reporting")
    ]
    tables: list[DataSourceTable] = []
    for schema in schemas:
        for index in range(80):
            tables.append(
                DataSourceTable(
                    schema_id=schema.id,
                    name=f"table_{index:03d}",
                    table_type=DataSourceTableType.TABLE,
                )
            )
    db_session.add_all(tables)
    db_session.flush()
    columns: list[DataSourceColumn] = []
    for table in tables:
        for position, name in enumerate(
            ("id", "name", "created_at", "updated_at"), start=1
        ):
            columns.append(
                DataSourceColumn(
                    table_id=table.id,
                    name=name,
                    ordinal_position=position,
                    data_type="integer" if name == "id" else "string",
                    database_type="int4" if name == "id" else "text",
                    is_nullable=name != "id",
                    is_primary_key=name == "id",
                    is_unique=name == "id",
                )
            )
    db_session.add_all(columns)
    db_session.flush()
    target = tables[0]
    target.name = "customers"
    db_session.flush()

    total_tables = db_session.scalar(select(func.count()).select_from(DataSourceTable))
    total_columns = db_session.scalar(
        select(func.count()).select_from(DataSourceColumn)
    )
    assert total_tables is not None and total_tables >= 240
    assert total_columns is not None and total_columns >= 960

    captured: list[tuple[str, object]] = []

    def _capture(
        _conn: object,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        captured.append((statement, parameters))

    bind = db_session.get_bind()
    event.listen(bind, "before_cursor_execute", _capture)
    try:
        page = _search(db_session, data_source, "customers", limit=20)
    finally:
        event.remove(bind, "before_cursor_execute", _capture)

    statements = " ".join(statement for statement, _ in captured).lower()
    assert page.result_count <= 20
    assert page.result_count < total_tables
    assert "limit" in statements
    assert "data_source_id" in statements or "data_source_schemas" in statements
    assert "encrypted_password" not in statements
    assert "data_source_connections" not in statements
    assert any(
        parameters is not None and parameters != {} for _, parameters in captured
    )


def test_sync_makes_metadata_searchable_and_removes_stale_rows(
    db_session: Session,
) -> None:
    data_source = _seed_data_source(db_session)
    initial = DiscoveryResult(
        schemas=(DiscoveredSchema(name="public"),),
        tables=(
            DiscoveredTable(
                schema_name="public",
                table_name="customers",
                table_type=DataSourceTableType.TABLE,
            ),
        ),
        columns=(
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="id",
                ordinal_position=1,
                data_type="integer",
                database_type="int4",
                is_nullable=False,
                default_value=None,
                is_primary_key=True,
                is_unique=True,
            ),
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="email",
                ordinal_position=2,
                data_type="string",
                database_type="text",
                is_nullable=False,
                default_value=None,
                is_primary_key=False,
                is_unique=True,
            ),
        ),
        relationships=(),
    )
    _sync(db_session, data_source, initial)

    page = _search(db_session, data_source, "customers")
    assert ("TABLE", "public", "customers", None) in _identities(page.results)
    assert ("COLUMN", "public", "customers", "email") in _identities(
        _search(db_session, data_source, "email").results
    )

    expanded = DiscoveryResult(
        schemas=(DiscoveredSchema(name="public"),),
        tables=(
            DiscoveredTable(
                schema_name="public",
                table_name="customers",
                table_type=DataSourceTableType.TABLE,
            ),
            DiscoveredTable(
                schema_name="public",
                table_name="orders",
                table_type=DataSourceTableType.TABLE,
            ),
        ),
        columns=(
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="id",
                ordinal_position=1,
                data_type="integer",
                database_type="int4",
                is_nullable=False,
                default_value=None,
                is_primary_key=True,
                is_unique=True,
            ),
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="email",
                ordinal_position=2,
                data_type="string",
                database_type="text",
                is_nullable=False,
                default_value=None,
                is_primary_key=False,
                is_unique=True,
            ),
            DiscoveredColumn(
                schema_name="public",
                table_name="orders",
                column_name="id",
                ordinal_position=1,
                data_type="integer",
                database_type="int4",
                is_nullable=False,
                default_value=None,
                is_primary_key=True,
                is_unique=True,
            ),
            DiscoveredColumn(
                schema_name="public",
                table_name="orders",
                column_name="customer_id",
                ordinal_position=2,
                data_type="integer",
                database_type="int4",
                is_nullable=False,
                default_value=None,
                is_primary_key=False,
                is_unique=False,
            ),
        ),
        relationships=(),
    )
    persist_discovered_metadata(db_session, data_source.id, expanded)
    db_session.flush()

    assert ("TABLE", "public", "orders", None) in _identities(
        _search(db_session, data_source, "orders").results
    )
    assert ("COLUMN", "public", "orders", "customer_id") in _identities(
        _search(db_session, data_source, "customer_id").results
    )

    reduced = DiscoveryResult(
        schemas=(DiscoveredSchema(name="public"),),
        tables=(
            DiscoveredTable(
                schema_name="public",
                table_name="customers",
                table_type=DataSourceTableType.TABLE,
            ),
        ),
        columns=(
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="id",
                ordinal_position=1,
                data_type="integer",
                database_type="int4",
                is_nullable=False,
                default_value=None,
                is_primary_key=True,
                is_unique=True,
            ),
        ),
        relationships=(),
    )
    persist_discovered_metadata(db_session, data_source.id, reduced)
    db_session.flush()

    assert _search(db_session, data_source, "orders").results == ()
    assert _search(db_session, data_source, "email").results == ()
    assert ("TABLE", "public", "customers", None) in _identities(
        _search(db_session, data_source, "customers").results
    )


def test_updated_description_becomes_searchable(db_session: Session) -> None:
    data_source = _seed_search_source(db_session)
    schema = _add_schema(db_session, data_source.id, "public")
    table = _add_table(db_session, schema.id, "orders")
    column = _add_column(db_session, table.id, "status", position=1)

    assert _search(db_session, data_source, "fulfillment").results == ()
    column.description = "Fulfillment status of the order"
    db_session.flush()

    page = _search(db_session, data_source, "fulfillment")
    assert _identities(page.results) == [("COLUMN", "public", "orders", "status")]


def test_renamed_metadata_does_not_leave_stale_search_results(
    db_session: Session,
) -> None:
    data_source = _seed_data_source(db_session)
    first = DiscoveryResult(
        schemas=(DiscoveredSchema(name="public"),),
        tables=(
            DiscoveredTable(
                schema_name="public",
                table_name="customers",
                table_type=DataSourceTableType.TABLE,
            ),
        ),
        columns=(
            DiscoveredColumn(
                schema_name="public",
                table_name="customers",
                column_name="email",
                ordinal_position=1,
                data_type="string",
                database_type="text",
                is_nullable=False,
                default_value=None,
                is_primary_key=False,
                is_unique=True,
            ),
        ),
        relationships=(),
    )
    persist_discovered_metadata(db_session, data_source.id, first)
    db_session.flush()
    renamed = DiscoveryResult(
        schemas=(DiscoveredSchema(name="public"),),
        tables=(
            DiscoveredTable(
                schema_name="public",
                table_name="clients",
                table_type=DataSourceTableType.TABLE,
            ),
        ),
        columns=(
            DiscoveredColumn(
                schema_name="public",
                table_name="clients",
                column_name="email_address",
                ordinal_position=1,
                data_type="string",
                database_type="text",
                is_nullable=False,
                default_value=None,
                is_primary_key=False,
                is_unique=True,
            ),
        ),
        relationships=(),
    )
    persist_discovered_metadata(db_session, data_source.id, renamed)
    db_session.flush()

    assert _search(db_session, data_source, "customers").results == ()
    identities = _identities(_search(db_session, data_source, "email").results)
    assert ("COLUMN", "public", "customers", "email") not in identities
    assert ("TABLE", "public", "clients", None) in _identities(
        _search(db_session, data_source, "clients").results
    )
    assert ("COLUMN", "public", "clients", "email_address") in _identities(
        _search(db_session, data_source, "email_address").results
    )


def test_search_service_is_independent_from_fastapi() -> None:
    source = inspect.getsource(MetadataSearchService)
    assert "fastapi" not in source.lower()
    assert "APIRouter" not in source
    assert "HTTPException" not in source
