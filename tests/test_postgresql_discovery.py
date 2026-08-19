from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.enums import DataSourceRelationshipType, DataSourceTableType
from app.services.discovery_exceptions import (
    MetadataDiscoveryLimitError,
    SchemaDiscoveryError,
)
from app.services.discovery_types import DiscoveryLimits, SchemaFilter
from app.services.postgresql_discovery import (
    _COLUMNS_SQL,
    _FOREIGN_KEYS_SQL,
    _KEY_CONSTRAINTS_SQL,
    _LIKE_ESCAPE,
    _PG_SYSTEM_SCHEMA_LIKE,
    _SCHEMAS_SQL,
    _SCHEMAS_SQL_INCLUDED,
    _TABLES_SQL,
    PostgreSQLMetadataDiscoveryService,
)
from app.services.schema_filters import EXCLUDED_SYSTEM_SCHEMAS, is_system_schema
from tests.conftest import run_async

_DISCOVERY_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "services" / "postgresql_discovery.py"
)
_WRITE_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
)


def _schema(name: str) -> dict[str, Any]:
    return {"schema_name": name}


def _table(schema: str, name: str, table_type: str = "BASE TABLE") -> dict[str, Any]:
    return {
        "table_schema": schema,
        "table_name": name,
        "table_type": table_type,
    }


def _column(
    schema: str,
    table: str,
    name: str,
    *,
    position: int,
    data_type: str,
    udt_name: str | None = None,
    nullable: str = "YES",
    default: str | None = None,
) -> dict[str, Any]:
    return {
        "table_schema": schema,
        "table_name": table,
        "column_name": name,
        "ordinal_position": position,
        "data_type": data_type,
        "udt_name": udt_name or data_type,
        "is_nullable": nullable,
        "column_default": default,
    }


def _key(
    schema: str,
    table: str,
    constraint: str,
    column: str,
    *,
    constraint_type: str,
    position: int = 1,
) -> dict[str, Any]:
    return {
        "table_schema": schema,
        "table_name": table,
        "constraint_name": constraint,
        "constraint_type": constraint_type,
        "column_name": column,
        "ordinal_position": position,
    }


def _fk(
    constraint: str,
    source_schema: str,
    source_table: str,
    source_column: str,
    target_schema: str,
    target_table: str,
    target_column: str,
    *,
    position: int = 1,
) -> dict[str, Any]:
    return {
        "constraint_name": constraint,
        "source_schema": source_schema,
        "source_table": source_table,
        "source_column": source_column,
        "ordinal_position": position,
        "target_schema": target_schema,
        "target_table": target_table,
        "target_column": target_column,
    }


def _sample_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        "schemas": [
            _schema("analytics"),
            _schema("empty_schema"),
            _schema("information_schema"),
            _schema("pg_catalog"),
            _schema("pg_temp_1"),
            _schema("pg_toast"),
            _schema("public"),
            _schema("reporting"),
        ],
        "tables": [
            _table("analytics", "users"),
            _table("public", "customers"),
            _table("public", "employees"),
            _table("public", "order_items"),
            _table("public", "order_summaries", "VIEW"),
            _table("public", "orders"),
            _table("public", "products"),
            _table("public", "tenant_customers"),
            _table("public", "tenant_orders"),
            _table("reporting", "users"),
        ],
        "columns": [
            _column(
                "analytics",
                "users",
                "id",
                position=1,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "analytics",
                "users",
                "email",
                position=2,
                data_type="character varying",
                udt_name="varchar",
                nullable="NO",
            ),
            _column(
                "public",
                "customers",
                "id",
                position=1,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "customers",
                "email",
                position=2,
                data_type="character varying",
                nullable="NO",
            ),
            _column(
                "public",
                "customers",
                "country_code",
                position=3,
                data_type="character varying",
                nullable="NO",
            ),
            _column(
                "public",
                "customers",
                "phone_number",
                position=4,
                data_type="character varying",
                nullable="NO",
            ),
            _column(
                "public",
                "employees",
                "id",
                position=1,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "employees",
                "manager_id",
                position=2,
                data_type="integer",
                nullable="YES",
            ),
            _column(
                "public",
                "employees",
                "email",
                position=3,
                data_type="text",
                nullable="NO",
            ),
            _column(
                "public",
                "order_items",
                "order_id",
                position=1,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "order_items",
                "product_id",
                position=2,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "order_items",
                "quantity",
                position=3,
                data_type="integer",
                nullable="NO",
                default="1",
            ),
            _column(
                "public",
                "order_summaries",
                "order_id",
                position=1,
                data_type="integer",
                nullable="YES",
            ),
            _column(
                "public", "orders", "id", position=1, data_type="integer", nullable="NO"
            ),
            _column(
                "public",
                "orders",
                "customer_id",
                position=2,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "orders",
                "notes",
                position=3,
                data_type="text",
                nullable="YES",
                default="pg_sleep(1)",
            ),
            _column(
                "public", "products", "id", position=1, data_type="uuid", nullable="NO"
            ),
            _column(
                "public",
                "products",
                "name",
                position=2,
                data_type="text",
                nullable="NO",
            ),
            _column(
                "public",
                "tenant_customers",
                "tenant_id",
                position=1,
                data_type="uuid",
                nullable="NO",
            ),
            _column(
                "public",
                "tenant_customers",
                "id",
                position=2,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "public",
                "tenant_orders",
                "tenant_id",
                position=1,
                data_type="uuid",
                nullable="NO",
            ),
            _column(
                "public",
                "tenant_orders",
                "customer_id",
                position=2,
                data_type="integer",
                nullable="NO",
            ),
            _column(
                "reporting",
                "users",
                "id",
                position=1,
                data_type="bigint",
                nullable="NO",
            ),
            _column(
                "reporting",
                "users",
                "label",
                position=2,
                data_type="text",
                nullable="YES",
            ),
        ],
        "keys": [
            _key(
                "analytics", "users", "users_pkey", "id", constraint_type="PRIMARY KEY"
            ),
            _key(
                "public",
                "customers",
                "customers_pkey",
                "id",
                constraint_type="PRIMARY KEY",
            ),
            _key(
                "public",
                "customers",
                "customers_email_key",
                "email",
                constraint_type="UNIQUE",
            ),
            _key(
                "public",
                "customers",
                "customers_phone_key",
                "country_code",
                constraint_type="UNIQUE",
                position=1,
            ),
            _key(
                "public",
                "customers",
                "customers_phone_key",
                "phone_number",
                constraint_type="UNIQUE",
                position=2,
            ),
            _key(
                "public",
                "employees",
                "employees_pkey",
                "id",
                constraint_type="PRIMARY KEY",
            ),
            _key(
                "public",
                "employees",
                "employees_email_key",
                "email",
                constraint_type="UNIQUE",
            ),
            _key(
                "public",
                "order_items",
                "order_items_pkey",
                "order_id",
                constraint_type="PRIMARY KEY",
                position=1,
            ),
            _key(
                "public",
                "order_items",
                "order_items_pkey",
                "product_id",
                constraint_type="PRIMARY KEY",
                position=2,
            ),
            _key(
                "public", "orders", "orders_pkey", "id", constraint_type="PRIMARY KEY"
            ),
            _key(
                "public",
                "products",
                "products_pkey",
                "id",
                constraint_type="PRIMARY KEY",
            ),
            _key(
                "public",
                "tenant_customers",
                "tenant_customers_pkey",
                "tenant_id",
                constraint_type="PRIMARY KEY",
                position=1,
            ),
            _key(
                "public",
                "tenant_customers",
                "tenant_customers_pkey",
                "id",
                constraint_type="PRIMARY KEY",
                position=2,
            ),
        ],
        "foreign_keys": [
            _fk(
                "fk_orders_customer_id",
                "public",
                "orders",
                "customer_id",
                "public",
                "customers",
                "id",
            ),
            _fk(
                "fk_order_items_order_id",
                "public",
                "order_items",
                "order_id",
                "public",
                "orders",
                "id",
            ),
            _fk(
                "fk_order_items_product_id",
                "public",
                "order_items",
                "product_id",
                "public",
                "products",
                "id",
            ),
            _fk(
                "fk_employees_manager_id",
                "public",
                "employees",
                "manager_id",
                "public",
                "employees",
                "id",
            ),
            _fk(
                "fk_tenant_orders_customer",
                "public",
                "tenant_orders",
                "tenant_id",
                "public",
                "tenant_customers",
                "tenant_id",
                position=1,
            ),
            _fk(
                "fk_tenant_orders_customer",
                "public",
                "tenant_orders",
                "customer_id",
                "public",
                "tenant_customers",
                "id",
                position=2,
            ),
        ],
    }


class FakeCatalogConnector:
    """Test double. Not a production connector."""

    def __init__(self, catalog: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.catalog = catalog or _sample_catalog()
        self.calls: list[tuple[str, Mapping[str, object] | None]] = []
        self.fail_error: Exception | None = None

    async def _fetch_all(
        self,
        query: str,
        params: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        if self.fail_error is not None:
            raise self.fail_error
        if "information_schema.schemata" in query:
            rows = list(self.catalog["schemas"])
            excluded = _excluded_schema_param(params)
            included = _included_schema_param(params)
            rows = [
                row
                for row in rows
                if _as_schema_name(row) not in excluded
                and not is_system_schema(_as_schema_name(row))
                and (included is None or _as_schema_name(row) in included)
            ]
            rows.sort(key=lambda row: _as_schema_name(row))
            return _apply_row_limit(rows, params)
        allowed = _schema_param(params)
        table_ids = _table_id_param(params)
        if "information_schema.tables" in query:
            rows = _rows_in_schemas(self.catalog["tables"], "table_schema", allowed)
            return _apply_row_limit(rows, params)
        if "information_schema.columns" in query:
            rows = _rows_in_schemas(self.catalog["columns"], "table_schema", allowed)
            rows = _rows_in_tables(rows, "table_schema", "table_name", table_ids)
            return _apply_row_limit(rows, params)
        if "pg_catalog.pg_constraint" in query and "confrelid" in query:
            rows = _rows_in_schemas(
                self.catalog["foreign_keys"], "source_schema", allowed
            )
            rows = _rows_in_tables(rows, "source_schema", "source_table", table_ids)
            rows = _rows_in_tables(rows, "target_schema", "target_table", table_ids)
            return _apply_row_limit(rows, params)
        if "pg_catalog.pg_constraint" in query:
            rows = _rows_in_schemas(self.catalog["keys"], "table_schema", allowed)
            rows = _rows_in_tables(rows, "table_schema", "table_name", table_ids)
            return _apply_row_limit(rows, params)
        raise AssertionError(f"unexpected discovery query: {query}")


def _as_schema_name(row: dict[str, Any]) -> str:
    return str(row.get("schema_name", "")).strip()


def _included_schema_param(params: Mapping[str, object] | None) -> set[str] | None:
    if params is None or "included" not in params:
        return None
    raw = params["included"]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(item) for item in raw}
    return {str(raw)}


def _table_id_param(
    params: Mapping[str, object] | None,
) -> set[tuple[str, str]] | None:
    if params is None or "table_schemas" not in params or "table_names" not in params:
        return None
    schemas = params["table_schemas"]
    names = params["table_names"]
    if not isinstance(schemas, (list, tuple)) or not isinstance(names, (list, tuple)):
        return None
    return {
        (str(schema), str(name)) for schema, name in zip(schemas, names, strict=True)
    }


def _rows_in_tables(
    rows: list[dict[str, Any]],
    schema_key: str,
    table_key: str,
    table_ids: set[tuple[str, str]] | None,
) -> list[dict[str, Any]]:
    if table_ids is None:
        return list(rows)
    return [
        row
        for row in rows
        if (str(row.get(schema_key, "")), str(row.get(table_key, ""))) in table_ids
    ]


def _excluded_schema_param(params: Mapping[str, object] | None) -> set[str]:
    if params is None or "excluded" not in params:
        return set(EXCLUDED_SYSTEM_SCHEMAS)
    raw = params["excluded"]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(item) for item in raw}
    return {str(raw)}


def _apply_row_limit(
    rows: list[dict[str, Any]],
    params: Mapping[str, object] | None,
) -> list[dict[str, Any]]:
    if params is None or "row_limit" not in params:
        return list(rows)
    limit = params["row_limit"]
    if not isinstance(limit, int):
        return list(rows)
    return list(rows[:limit])


def _schema_param(params: Mapping[str, object] | None) -> set[str] | None:
    if params is None or "schemas" not in params:
        return None
    raw = params["schemas"]
    if isinstance(raw, (list, tuple, set, frozenset)):
        return {str(item) for item in raw}
    return {str(raw)}


def _rows_in_schemas(
    rows: list[dict[str, Any]],
    key: str,
    allowed: set[str] | None,
) -> list[dict[str, Any]]:
    if allowed is None:
        return list(rows)
    return [row for row in rows if str(row.get(key, "")) in allowed]


def _service(
    catalog: dict[str, list[dict[str, Any]]] | None = None,
    *,
    limits: DiscoveryLimits | None = None,
    schema_filter: SchemaFilter | None = None,
) -> tuple[PostgreSQLMetadataDiscoveryService, FakeCatalogConnector]:
    connector = FakeCatalogConnector(catalog)
    service = PostgreSQLMetadataDiscoveryService(
        connector,
        limits=limits,
        schema_filter=schema_filter,
    )
    return service, connector


def test_public_and_user_schemas_are_discovered() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    names = [schema.name for schema in result.schemas]
    assert names == ["analytics", "empty_schema", "public", "reporting"]


def test_system_schemas_are_filtered() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    names = {schema.name for schema in result.schemas}
    assert names.isdisjoint(
        {"pg_catalog", "information_schema", "pg_toast", "pg_temp_1"}
    )


def test_empty_schema_is_included_without_tables() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    assert any(schema.name == "empty_schema" for schema in result.schemas)
    assert all(table.schema_name != "empty_schema" for table in result.tables)


def test_tables_and_views_are_discovered() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    by_name = {
        (table.schema_name, table.table_name): table.table_type
        for table in result.tables
    }
    assert by_name[("public", "orders")] is DataSourceTableType.TABLE
    assert by_name[("public", "order_summaries")] is DataSourceTableType.VIEW
    assert ("public", "customers") in by_name
    assert ("public", "employees") in by_name


def test_duplicate_table_names_are_schema_qualified() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    users = [table for table in result.tables if table.table_name == "users"]
    assert {(table.schema_name, table.table_name) for table in users} == {
        ("analytics", "users"),
        ("reporting", "users"),
    }


def test_system_tables_are_not_returned() -> None:
    catalog = _sample_catalog()
    catalog["tables"].append(_table("pg_catalog", "pg_class"))
    service, _ = _service(catalog)
    result = run_async(service.discover())
    assert all(table.schema_name != "pg_catalog" for table in result.tables)


def test_column_names_ordinal_nullable_default_and_types() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    notes = next(
        column
        for column in result.columns
        if column.table_name == "orders" and column.column_name == "notes"
    )
    quantity = next(
        column
        for column in result.columns
        if column.table_name == "order_items" and column.column_name == "quantity"
    )
    product_id = next(
        column
        for column in result.columns
        if column.table_name == "products" and column.column_name == "id"
    )
    email = next(
        column
        for column in result.columns
        if column.schema_name == "public"
        and column.table_name == "customers"
        and column.column_name == "email"
    )
    assert notes.ordinal_position == 3
    assert notes.is_nullable is True
    assert notes.default_value == "pg_sleep(1)"
    assert notes.data_type == "string"
    assert notes.database_type == "text"
    assert quantity.default_value == "1"
    assert quantity.is_nullable is False
    assert product_id.data_type == "uuid"
    assert product_id.database_type == "uuid"
    assert email.data_type == "string"
    assert email.database_type == "character varying"
    assert email.is_nullable is False


def test_columns_are_ordered_by_ordinal_position() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    order_columns = [
        column.column_name
        for column in result.columns
        if column.schema_name == "public" and column.table_name == "orders"
    ]
    assert order_columns == ["id", "customer_id", "notes"]


def test_single_primary_key() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    customers = [
        column
        for column in result.columns
        if column.schema_name == "public" and column.table_name == "customers"
    ]
    by_name = {column.column_name: column for column in customers}
    assert by_name["id"].is_primary_key is True
    assert by_name["email"].is_primary_key is False


def test_composite_primary_key() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    items = {
        column.column_name: column
        for column in result.columns
        if column.table_name == "order_items"
    }
    assert items["order_id"].is_primary_key is True
    assert items["product_id"].is_primary_key is True
    assert items["quantity"].is_primary_key is False
    assert items["order_id"].is_unique is False
    assert items["product_id"].is_unique is False


def test_table_without_primary_key() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    reporting_users = [
        column
        for column in result.columns
        if column.schema_name == "reporting" and column.table_name == "users"
    ]
    assert reporting_users
    assert all(column.is_primary_key is False for column in reporting_users)


def test_single_column_unique_and_non_unique() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    customers = {
        column.column_name: column
        for column in result.columns
        if column.schema_name == "public" and column.table_name == "customers"
    }
    assert customers["email"].is_unique is True
    assert customers["id"].is_unique is True
    assert customers["country_code"].is_unique is False
    assert customers["phone_number"].is_unique is False


def test_composite_unique_is_not_marked_globally_unique() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    customers = {
        column.column_name: column
        for column in result.columns
        if column.schema_name == "public" and column.table_name == "customers"
    }
    assert customers["country_code"].is_unique is False
    assert customers["phone_number"].is_unique is False


def test_foreign_key_and_multiple_foreign_keys() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    relations = {
        (item.source_table, item.source_column, item.target_table, item.target_column)
        for item in result.relationships
    }
    assert ("orders", "customer_id", "customers", "id") in relations
    assert ("order_items", "order_id", "orders", "id") in relations
    assert ("order_items", "product_id", "products", "id") in relations
    customer_fk = next(
        item
        for item in result.relationships
        if item.source_table == "orders" and item.source_column == "customer_id"
    )
    assert customer_fk.constraint_name == "fk_orders_customer_id"
    assert customer_fk.relationship_type is DataSourceRelationshipType.MANY_TO_ONE
    assert customer_fk.source_schema == "public"
    assert customer_fk.target_schema == "public"


def test_composite_foreign_key_keeps_all_columns() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    composite = [
        item
        for item in result.relationships
        if item.constraint_name == "fk_tenant_orders_customer"
    ]
    assert len(composite) == 2
    assert {item.source_column for item in composite} == {"tenant_id", "customer_id"}
    assert {item.target_column for item in composite} == {"tenant_id", "id"}
    assert all(item.constraint_column_count == 2 for item in composite)
    assert [item.ordinal_position for item in composite] == [1, 2]
    assert all(
        item.relationship_type is DataSourceRelationshipType.MANY_TO_ONE
        for item in composite
    )


def test_self_referencing_foreign_key_is_kept() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    manager = next(
        item
        for item in result.relationships
        if item.constraint_name == "fk_employees_manager_id"
    )
    assert manager.source_table == "employees"
    assert manager.target_table == "employees"
    assert manager.source_column == "manager_id"
    assert manager.target_column == "id"
    assert manager.relationship_type is DataSourceRelationshipType.MANY_TO_ONE


def test_unique_foreign_key_is_one_to_one() -> None:
    catalog = _sample_catalog()
    catalog["tables"].append(_table("public", "employee_profiles"))
    catalog["columns"].extend(
        [
            _column(
                "public",
                "employee_profiles",
                "employee_id",
                position=1,
                data_type="integer",
                nullable="NO",
            )
        ]
    )
    catalog["keys"].extend(
        [
            _key(
                "public",
                "employee_profiles",
                "employee_profiles_pkey",
                "employee_id",
                constraint_type="PRIMARY KEY",
            )
        ]
    )
    catalog["foreign_keys"].append(
        _fk(
            "fk_employee_profiles_employee_id",
            "public",
            "employee_profiles",
            "employee_id",
            "public",
            "employees",
            "id",
        )
    )
    service, _ = _service(catalog)
    result = run_async(service.discover())
    profile = next(
        item
        for item in result.relationships
        if item.source_table == "employee_profiles"
    )
    assert profile.relationship_type is DataSourceRelationshipType.ONE_TO_ONE


def test_discovery_uses_a_bounded_number_of_queries() -> None:
    service, connector = _service()
    run_async(service.discover())
    assert len(connector.calls) == 5
    catalogs = [
        "information_schema.schemata",
        "information_schema.tables",
        "pg_catalog.pg_constraint",
        "information_schema.columns",
    ]
    joined = " ".join(query for query, _params in connector.calls)
    for name in catalogs:
        assert name in joined


def test_schema_limit_is_not_silently_truncated() -> None:
    catalog = _sample_catalog()
    service, _ = _service(catalog, limits=DiscoveryLimits(max_schemas=1))
    with pytest.raises(MetadataDiscoveryLimitError, match="Schema discovery limit"):
        run_async(service.discover())


def test_system_schemas_do_not_consume_schema_fetch_window() -> None:
    catalog = {
        "schemas": [_schema(f"pg_temp_{index:03d}") for index in range(1, 51)]
        + [_schema("z_analytics"), _schema("z_public")],
        "tables": [],
        "columns": [],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(
        catalog,
        limits=DiscoveryLimits(max_schemas=2),
        schema_filter=SchemaFilter(
            include_schemas=frozenset({"z_analytics", "z_public"})
        ),
    )
    result = run_async(service.discover())
    assert [schema.name for schema in result.schemas] == ["z_analytics", "z_public"]


def test_schema_overflow_raises_when_user_schemas_exceed_limit() -> None:
    catalog = {
        "schemas": [_schema(f"z_schema_{index:02d}") for index in range(1, 6)],
        "tables": [],
        "columns": [],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(catalog, limits=DiscoveryLimits(max_schemas=2))
    with pytest.raises(MetadataDiscoveryLimitError, match="Schema discovery limit"):
        run_async(service.discover())


def test_key_constraint_overflow_is_not_silently_truncated() -> None:
    catalog = _sample_catalog()
    catalog["keys"] = [
        _key(
            "public",
            "customers",
            f"customers_uq_{index}",
            f"col_{index}",
            constraint_type="UNIQUE",
        )
        for index in range(1, 8)
    ]
    service, _ = _service(catalog, limits=DiscoveryLimits(max_columns=5))
    with pytest.raises(MetadataDiscoveryLimitError, match="Column discovery limit"):
        run_async(service.discover())


def test_column_overflow_is_not_silently_truncated() -> None:
    catalog = {
        "schemas": [_schema("public")],
        "tables": [_table("public", "wide")],
        "columns": [
            _column(
                "public", "wide", f"col_{index}", position=index, data_type="integer"
            )
            for index in range(1, 6)
        ],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(catalog, limits=DiscoveryLimits(max_columns=3))
    with pytest.raises(MetadataDiscoveryLimitError, match="Column discovery limit"):
        run_async(service.discover())


def test_foreign_table_columns_do_not_consume_column_limit() -> None:
    catalog = {
        "schemas": [_schema("public")],
        "tables": [
            _table("public", "users"),
            _table("public", "external_x", "FOREIGN TABLE"),
        ],
        "columns": [
            _column("public", "users", "id", position=1, data_type="integer"),
            _column("public", "users", "email", position=2, data_type="text"),
        ]
        + [
            _column(
                "public",
                "external_x",
                f"remote_{index}",
                position=index,
                data_type="integer",
            )
            for index in range(1, 11)
        ],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(catalog, limits=DiscoveryLimits(max_columns=2))
    result = run_async(service.discover())
    assert [column.column_name for column in result.columns] == ["id", "email"]


def test_include_schemas_does_not_overflow_on_other_user_schemas() -> None:
    catalog = {
        "schemas": [_schema(f"other_{index:02d}") for index in range(1, 8)]
        + [_schema("public")],
        "tables": [],
        "columns": [],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(
        catalog,
        limits=DiscoveryLimits(max_schemas=1),
        schema_filter=SchemaFilter(include_schemas=frozenset({"public"})),
    )
    result = run_async(service.discover())
    assert [schema.name for schema in result.schemas] == ["public"]


def test_duplicate_constraint_names_are_not_mixed() -> None:
    catalog = {
        "schemas": [_schema("public")],
        "tables": [
            _table("public", "accounts"),
            _table("public", "orders"),
            _table("public", "account_notes"),
            _table("public", "order_notes"),
        ],
        "columns": [
            _column("public", "accounts", "user_id", position=1, data_type="integer"),
            _column("public", "orders", "order_id", position=1, data_type="integer"),
            _column(
                "public",
                "account_notes",
                "account_id",
                position=1,
                data_type="integer",
            ),
            _column(
                "public", "order_notes", "order_id", position=1, data_type="integer"
            ),
        ],
        "keys": [
            _key(
                "public", "accounts", "pkey", "user_id", constraint_type="PRIMARY KEY"
            ),
            _key("public", "orders", "pkey", "order_id", constraint_type="PRIMARY KEY"),
        ],
        "foreign_keys": [
            _fk(
                "fk",
                "public",
                "account_notes",
                "account_id",
                "public",
                "accounts",
                "user_id",
            ),
            _fk(
                "fk",
                "public",
                "order_notes",
                "order_id",
                "public",
                "orders",
                "order_id",
            ),
        ],
    }
    service, _ = _service(catalog)
    result = run_async(service.discover())
    by_column = {
        (column.table_name, column.column_name): column for column in result.columns
    }
    assert by_column[("accounts", "user_id")].is_primary_key is True
    assert by_column[("orders", "order_id")].is_primary_key is True
    assert by_column[("accounts", "user_id")].is_unique is True
    assert by_column[("order_notes", "order_id")].is_primary_key is False
    relations = {
        (item.source_table, item.source_column, item.target_table, item.target_column)
        for item in result.relationships
    }
    assert relations == {
        ("account_notes", "account_id", "accounts", "user_id"),
        ("order_notes", "order_id", "orders", "order_id"),
    }
    assert ("account_notes", "account_id", "orders", "order_id") not in relations
    assert ("order_notes", "order_id", "accounts", "user_id") not in relations


def test_cross_schema_foreign_keys_are_discovered() -> None:
    catalog = {
        "schemas": [_schema("analytics"), _schema("public")],
        "tables": [
            _table("analytics", "users"),
            _table("public", "events"),
        ],
        "columns": [
            _column("analytics", "users", "id", position=1, data_type="integer"),
            _column("public", "events", "id", position=1, data_type="integer"),
            _column(
                "public",
                "events",
                "user_id",
                position=2,
                data_type="integer",
            ),
        ],
        "keys": [
            _key(
                "analytics",
                "users",
                "users_pkey",
                "id",
                constraint_type="PRIMARY KEY",
            )
        ],
        "foreign_keys": [
            _fk(
                "fk_events_user_id",
                "public",
                "events",
                "user_id",
                "analytics",
                "users",
                "id",
            )
        ],
    }
    service, _ = _service(catalog)
    result = run_async(service.discover())
    relation = next(iter(result.relationships))
    assert relation.source_schema == "public"
    assert relation.source_table == "events"
    assert relation.source_column == "user_id"
    assert relation.target_schema == "analytics"
    assert relation.target_table == "users"
    assert relation.target_column == "id"


def test_foreign_table_columns_and_keys_are_ignored() -> None:
    catalog = _sample_catalog()
    catalog["tables"].append(_table("public", "external_x", "FOREIGN TABLE"))
    catalog["columns"].extend(
        [
            _column(
                "public",
                "customers",
                "legacy_id",
                position=5,
                data_type="integer",
                nullable="YES",
            ),
            _column(
                "public",
                "external_x",
                "remote_id",
                position=1,
                data_type="integer",
                nullable="NO",
            ),
        ]
    )
    catalog["keys"].extend(
        [
            _key(
                "public",
                "customers",
                "customers_legacy_id_key",
                "legacy_id",
                constraint_type="UNIQUE",
            ),
            _key(
                "public",
                "external_x",
                "external_x_pkey",
                "remote_id",
                constraint_type="PRIMARY KEY",
            ),
        ]
    )
    service, _ = _service(catalog)
    result = run_async(service.discover())
    assert ("public", "external_x") not in {
        (table.schema_name, table.table_name) for table in result.tables
    }
    assert all(
        not (column.schema_name == "public" and column.table_name == "external_x")
        for column in result.columns
    )
    legacy = next(
        column
        for column in result.columns
        if column.schema_name == "public"
        and column.table_name == "customers"
        and column.column_name == "legacy_id"
    )
    assert legacy.is_unique is True
    assert all(
        not (item.source_table == "external_x" or item.target_table == "external_x")
        for item in result.relationships
    )


@pytest.mark.parametrize(
    ("position", "expected_name"),
    [
        (None, None),
        (0, None),
        (-1, None),
        ("invalid", None),
        (2, "valid_col"),
    ],
)
def test_invalid_ordinal_positions_are_skipped(
    position: object,
    expected_name: str | None,
) -> None:
    catalog = {
        "schemas": [_schema("public")],
        "tables": [_table("public", "ordinal_probe")],
        "columns": [
            {
                "table_schema": "public",
                "table_name": "ordinal_probe",
                "column_name": "valid_col",
                "ordinal_position": position,
                "data_type": "integer",
                "udt_name": "int4",
                "is_nullable": "YES",
                "column_default": None,
            }
        ],
        "keys": [],
        "foreign_keys": [],
    }
    service, _ = _service(catalog)
    result = run_async(service.discover())
    matches = [
        column for column in result.columns if column.table_name == "ordinal_probe"
    ]
    if expected_name is None:
        assert matches == []
    else:
        assert len(matches) == 1
        assert matches[0].column_name == expected_name
        assert matches[0].ordinal_position > 0


def test_table_limit_is_not_silently_truncated() -> None:
    catalog = _sample_catalog()
    service, _ = _service(catalog, limits=DiscoveryLimits(max_tables=2))
    with pytest.raises(MetadataDiscoveryLimitError, match="Table discovery limit"):
        run_async(service.discover())


def test_include_and_exclude_schema_filters() -> None:
    service, _ = _service(
        schema_filter=SchemaFilter(
            include_schemas=frozenset({"public", "analytics"}),
            exclude_schemas=frozenset({"analytics"}),
        )
    )
    result = run_async(service.discover())
    assert [schema.name for schema in result.schemas] == ["public"]
    assert all(table.schema_name == "public" for table in result.tables)


def test_sql_injection_payload_is_passed_as_a_parameter() -> None:
    payload = "public'; DROP TABLE users; --"
    service, connector = _service(
        schema_filter=SchemaFilter(include_schemas=frozenset({payload, "public"}))
    )
    result = run_async(service.discover())
    assert [schema.name for schema in result.schemas] == ["public"]
    for query, params in connector.calls:
        assert payload not in query
        assert "DROP TABLE" not in query
        assert (
            "%(schemas)s" in query
            or "%(row_limit)s" in query
            or "%(excluded)s" in query
            or "%(included)s" in query
            or "%(table_schemas)s" in query
        )
        if params is not None and "schemas" in params:
            assert isinstance(params["schemas"], list)


def test_schemas_sql_excludes_system_catalogs_in_query() -> None:
    assert "%(excluded)s" in _SCHEMAS_SQL
    assert "%(row_limit)s" in _SCHEMAS_SQL
    assert "%(pg_prefix)s" in _SCHEMAS_SQL
    assert "%(like_escape)s" in _SCHEMAS_SQL
    assert "NOT LIKE %(pg_prefix)s ESCAPE %(like_escape)s" in _SCHEMAS_SQL
    assert "NOT LIKE %(pg_prefix)s ESCAPE %(like_escape)s" in _SCHEMAS_SQL_INCLUDED
    for excluded in EXCLUDED_SYSTEM_SCHEMAS:
        assert f"'{excluded}'" not in _SCHEMAS_SQL
    assert "%(included)s" in _SCHEMAS_SQL_INCLUDED
    assert "%(included)s" not in _SCHEMAS_SQL


def test_schema_query_binds_pg_prefix_like_pattern() -> None:
    service, connector = _service()
    run_async(service.discover_schemas())
    query, params = connector.calls[0]
    assert params is not None
    assert params["pg_prefix"] == _PG_SYSTEM_SCHEMA_LIKE
    assert params["like_escape"] == _LIKE_ESCAPE
    leftover = re.sub(r"%\([^)]+\)s", "", query)
    assert "%" not in leftover


def test_catalog_sql_identifies_constraints_by_table_oid() -> None:
    assert "rel.oid = con.conrelid" in _KEY_CONSTRAINTS_SQL
    assert "src_rel.oid = con.conrelid" in _FOREIGN_KEYS_SQL
    assert "tgt_rel.oid = con.confrelid" in _FOREIGN_KEYS_SQL
    assert "CAST(%(table_schemas)s AS text[])" in _COLUMNS_SQL
    assert "CAST(%(table_names)s AS text[])" in _COLUMNS_SQL
    assert "CAST(%(table_schemas)s AS text[])" in _KEY_CONSTRAINTS_SQL
    assert "CAST(%(table_names)s AS text[])" in _KEY_CONSTRAINTS_SQL
    assert "CAST(%(table_schemas)s AS text[])" in _FOREIGN_KEYS_SQL
    assert "CAST(%(table_names)s AS text[])" in _FOREIGN_KEYS_SQL
    assert _COLUMNS_SQL.count("CAST(%(table_schemas)s AS text[])") == 1
    assert _FOREIGN_KEYS_SQL.count("CAST(%(table_schemas)s AS text[])") == 2
    assert "information_schema.key_column_usage" not in _KEY_CONSTRAINTS_SQL
    assert "information_schema.referential_constraints" not in _FOREIGN_KEYS_SQL


def test_discovery_sql_is_static_and_read_only() -> None:
    source = _DISCOVERY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            pytest.fail("discovery SQL must not use f-strings")
    for sql in (
        _SCHEMAS_SQL,
        _SCHEMAS_SQL_INCLUDED,
        _TABLES_SQL,
        _COLUMNS_SQL,
        _KEY_CONSTRAINTS_SQL,
        _FOREIGN_KEYS_SQL,
    ):
        upper = sql.upper()
        assert upper.strip().startswith("SELECT")
        catalog_stripped = upper.replace("INFORMATION_SCHEMA", "").replace(
            "PG_CATALOG", ""
        )
        for keyword in _WRITE_KEYWORDS:
            assert re.search(rf"\b{keyword}\b", catalog_stripped) is None
        assert "%(" in sql
        leftover = re.sub(r"%\([^)]+\)s", "", sql)
        assert "%" not in leftover, sql


def test_connector_errors_are_mapped_without_raw_driver_details() -> None:
    from app.connectors import ConnectorQueryError

    service, connector = _service()
    connector.fail_error = ConnectorQueryError(
        "password=secret OperationalError: relation does not exist"
    )
    with pytest.raises(
        SchemaDiscoveryError, match="Unable to discover database schemas"
    ):
        run_async(service.discover())


def test_discovery_result_omits_credentials() -> None:
    service, _ = _service()
    result = run_async(service.discover())
    rendered = repr(result)
    assert "password" not in rendered.lower()
    assert "credential" not in rendered.lower()
    assert "encrypted_password" not in rendered
