from __future__ import annotations

import uuid
from typing import Any

import psycopg
import pytest
from sqlalchemy.orm import Session

from app.connectors import (
    ConnectorConfig,
    PostgreSQLConnector,
    connector_lifecycle,
)
from app.enums import DataSourceRelationshipType, DataSourceTableType
from app.services.data_source_discovery import DataSourceDiscoveryService
from app.services.discovery_types import DiscoveryResult, SchemaFilter
from app.services.postgresql_discovery import PostgreSQLMetadataDiscoveryService
from tests.conftest import run_async
from tests.integration.test_credential_connector_live import _seed_encrypted_connection
from tests.integration.test_postgresql_connector_live import _integration_config

_SKIP_REASON = "PostgreSQL metadata integration tests: SKIPPED"


def _require_config() -> ConnectorConfig:
    config = _integration_config()
    if config is None:
        pytest.skip(_SKIP_REASON)
    return config


def _writable_connection(config: ConnectorConfig) -> psycopg.Connection[Any]:
    return psycopg.connect(
        host=config.host,
        port=config.port,
        dbname=config.database_name,
        user=config.username,
        password=config.credential,
        sslmode=config.ssl_mode or "prefer",
        autocommit=True,
    )


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _install_fixture(
    connection: psycopg.Connection[Any],
    schema: str,
    analytics: str,
    empty: str,
) -> None:
    s = _quote_ident(schema)
    a = _quote_ident(analytics)
    e = _quote_ident(empty)
    statements = [
        f"CREATE SCHEMA {s}",
        f"CREATE SCHEMA {a}",
        f"CREATE SCHEMA {e}",
        f"""
        CREATE TABLE {s}.customers (
            id integer PRIMARY KEY,
            email character varying(255) NOT NULL UNIQUE,
            country_code character varying(8) NOT NULL,
            phone_number character varying(32) NOT NULL,
            UNIQUE (country_code, phone_number)
        )
        """,
        f"""
        CREATE TABLE {s}.products (
            id integer PRIMARY KEY,
            name text NOT NULL
        )
        """,
        f"""
        CREATE TABLE {s}.orders (
            id integer PRIMARY KEY,
            customer_id integer NOT NULL REFERENCES {s}.customers (id),
            notes text
        )
        """,
        f"""
        CREATE TABLE {s}.order_items (
            order_id integer NOT NULL REFERENCES {s}.orders (id),
            product_id integer NOT NULL REFERENCES {s}.products (id),
            quantity integer NOT NULL DEFAULT 1,
            PRIMARY KEY (order_id, product_id)
        )
        """,
        f"""
        CREATE TABLE {s}.employees (
            id integer PRIMARY KEY,
            manager_id integer REFERENCES {s}.employees (id),
            email text NOT NULL UNIQUE
        )
        """,
        f"""
        CREATE TABLE {s}.tenant_customers (
            tenant_id uuid NOT NULL,
            id integer NOT NULL,
            PRIMARY KEY (tenant_id, id)
        )
        """,
        f"""
        CREATE TABLE {s}.tenant_orders (
            tenant_id uuid NOT NULL,
            customer_id integer NOT NULL,
            FOREIGN KEY (tenant_id, customer_id)
                REFERENCES {s}.tenant_customers (tenant_id, id)
        )
        """,
        f"""
        CREATE TABLE {s}.type_samples (
            id integer PRIMARY KEY,
            small_val smallint,
            int_val integer,
            big_val bigint,
            num_val numeric,
            dec_val decimal,
            real_val real,
            double_val double precision,
            bool_val boolean,
            char_val character,
            varchar_val character varying,
            text_val text,
            date_val date,
            ts_val timestamp,
            tstz_val timestamp with time zone,
            ts_naive timestamp without time zone,
            time_val time,
            timetz_val time with time zone,
            time_naive time without time zone,
            uuid_val uuid,
            json_val json,
            jsonb_val jsonb,
            bin_val bytea
        )
        """,
        f"""
        CREATE VIEW {s}.order_summaries AS
        SELECT id, customer_id FROM {s}.orders
        """,
        f"""
        CREATE TABLE {a}.users (
            id integer PRIMARY KEY,
            email text NOT NULL
        )
        """,
        f"""
        CREATE TABLE {s}.accounts (
            user_id integer,
            CONSTRAINT accounts_pkey PRIMARY KEY (user_id)
        )
        """,
        f"""
        CREATE TABLE {s}.named_orders (
            order_id integer,
            CONSTRAINT named_orders_pkey PRIMARY KEY (order_id)
        )
        """,
        f"""
        CREATE TABLE {s}.account_notes (
            account_id integer,
            CONSTRAINT account_notes_account_id_fkey
                FOREIGN KEY (account_id) REFERENCES {s}.accounts (user_id)
        )
        """,
        f"""
        CREATE TABLE {s}.order_notes (
            order_id integer,
            CONSTRAINT order_notes_order_id_fkey
                FOREIGN KEY (order_id) REFERENCES {s}.named_orders (order_id)
        )
        """,
        f"""
        CREATE TABLE {s}.events (
            id integer PRIMARY KEY,
            user_id integer REFERENCES {a}.users (id)
        )
        """,
    ]
    for statement in statements:
        connection.execute(statement)


def _drop_fixture(
    connection: psycopg.Connection[Any],
    schema: str,
    analytics: str,
    empty: str,
) -> None:
    for name in (schema, analytics, empty):
        connection.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(name)} CASCADE")


def test_live_postgresql_metadata_discovery() -> None:
    config = _require_config()
    schema = f"ac_disc_{uuid.uuid4().hex[:10]}"
    analytics = f"{schema}_an"
    empty = f"{schema}_empty"
    try:
        setup = _writable_connection(config)
    except (psycopg.Error, OSError) as exc:
        pytest.skip(f"{_SKIP_REASON} — {type(exc).__name__}")

    try:
        try:
            _install_fixture(setup, schema, analytics, empty)
        except psycopg.Error:
            pytest.skip(f"{_SKIP_REASON} — test role cannot create schemas")

        async def _discover() -> DiscoveryResult:
            connector = PostgreSQLConnector(connect_timeout=5)
            async with connector_lifecycle(connector, config) as active:
                service = PostgreSQLMetadataDiscoveryService(
                    active,
                    schema_filter=SchemaFilter(
                        include_schemas=frozenset({schema, analytics, empty})
                    ),
                )
                return await service.discover()

        result = run_async(_discover())
    finally:
        try:
            _drop_fixture(setup, schema, analytics, empty)
        finally:
            setup.close()

    names = [item.name for item in result.schemas]
    assert names == sorted([analytics, empty, schema])
    tables = {
        (item.schema_name, item.table_name): item.table_type for item in result.tables
    }
    assert tables[(schema, "customers")] is DataSourceTableType.TABLE
    assert tables[(schema, "order_summaries")] is DataSourceTableType.VIEW
    assert tables[(analytics, "users")] is DataSourceTableType.TABLE
    assert (empty, "users") not in tables
    assert (schema, "users") not in tables

    customers_id = next(
        column
        for column in result.columns
        if column.schema_name == schema
        and column.table_name == "customers"
        and column.column_name == "id"
    )
    assert customers_id.is_primary_key is True
    assert customers_id.is_unique is True
    assert customers_id.is_nullable is False
    assert customers_id.data_type == "integer"

    email = next(
        column
        for column in result.columns
        if column.schema_name == schema
        and column.table_name == "customers"
        and column.column_name == "email"
    )
    assert email.is_unique is True
    assert email.data_type == "string"
    assert email.database_type == "character varying"

    country = next(
        column
        for column in result.columns
        if column.table_name == "customers" and column.column_name == "country_code"
    )
    phone = next(
        column
        for column in result.columns
        if column.table_name == "customers" and column.column_name == "phone_number"
    )
    assert country.is_unique is False
    assert phone.is_unique is False

    items = {
        column.column_name: column
        for column in result.columns
        if column.table_name == "order_items"
    }
    assert items["order_id"].is_primary_key is True
    assert items["product_id"].is_primary_key is True
    assert items["quantity"].is_primary_key is False
    assert items["quantity"].default_value is not None
    assert items["quantity"].is_unique is False

    reporting_missing_pk = [
        column for column in result.columns if column.table_name == "tenant_orders"
    ]
    assert reporting_missing_pk
    assert all(column.is_primary_key is False for column in reporting_missing_pk)

    relations = {
        (item.source_table, item.source_column, item.target_table, item.target_column)
        for item in result.relationships
    }
    assert ("orders", "customer_id", "customers", "id") in relations
    assert ("order_items", "order_id", "orders", "id") in relations
    assert ("order_items", "product_id", "products", "id") in relations
    manager = next(
        item
        for item in result.relationships
        if item.source_table == "employees" and item.source_column == "manager_id"
    )
    assert manager.target_table == "employees"
    assert manager.target_column == "id"
    assert manager.relationship_type is DataSourceRelationshipType.MANY_TO_ONE

    composite = [
        item for item in result.relationships if item.source_table == "tenant_orders"
    ]
    assert {item.source_column for item in composite} == {"tenant_id", "customer_id"}
    assert {item.target_column for item in composite} == {"tenant_id", "id"}
    assert all(item.constraint_column_count == 2 for item in composite)

    accounts_pk = next(
        column
        for column in result.columns
        if column.table_name == "accounts" and column.column_name == "user_id"
    )
    named_orders_pk = next(
        column
        for column in result.columns
        if column.table_name == "named_orders" and column.column_name == "order_id"
    )
    assert accounts_pk.is_primary_key is True
    assert named_orders_pk.is_primary_key is True
    assert (
        "account_notes",
        "account_id",
        "accounts",
        "user_id",
    ) in relations
    assert (
        "order_notes",
        "order_id",
        "named_orders",
        "order_id",
    ) in relations
    assert ("account_notes", "account_id", "named_orders", "order_id") not in relations
    assert ("order_notes", "order_id", "accounts", "user_id") not in relations
    assert ("events", "user_id", "users", "id") in relations
    event_fk = next(
        item
        for item in result.relationships
        if item.source_table == "events" and item.source_column == "user_id"
    )
    assert event_fk.source_schema == schema
    assert event_fk.target_schema == analytics

    types = {
        column.column_name: (column.data_type, column.database_type)
        for column in result.columns
        if column.table_name == "type_samples"
    }
    assert types["small_val"][0] == "integer"
    assert types["int_val"][0] == "integer"
    assert types["big_val"][0] == "integer"
    assert types["num_val"][0] == "numeric"
    assert types["dec_val"][0] == "numeric"
    assert types["real_val"][0] == "float"
    assert types["double_val"][0] == "float"
    assert types["bool_val"] == ("boolean", "boolean")
    assert types["char_val"][0] == "string"
    assert types["varchar_val"][0] == "string"
    assert types["text_val"] == ("string", "text")
    assert types["date_val"] == ("date", "date")
    assert types["ts_val"][0] == "datetime"
    assert types["tstz_val"][0] == "datetime"
    assert types["ts_naive"][0] == "datetime"
    assert types["time_val"][0] == "time"
    assert types["timetz_val"][0] == "time"
    assert types["time_naive"][0] == "time"
    assert types["uuid_val"] == ("uuid", "uuid")
    assert types["json_val"][0] == "json"
    assert types["jsonb_val"][0] == "json"
    assert types["bin_val"] == ("binary", "bytea")


def test_live_discovery_through_data_source_service(db_session: Session) -> None:
    config = _require_config()
    schema = f"ac_svc_{uuid.uuid4().hex[:10]}"
    analytics = f"{schema}_an"
    empty = f"{schema}_empty"
    try:
        setup = _writable_connection(config)
    except (psycopg.Error, OSError) as exc:
        pytest.skip(f"{_SKIP_REASON} — {type(exc).__name__}")

    try:
        try:
            _install_fixture(setup, schema, analytics, empty)
        except psycopg.Error:
            pytest.skip(f"{_SKIP_REASON} — test role cannot create schemas")

        connection = _seed_encrypted_connection(
            db_session,
            host=config.host,
            port=config.port,
            database_name=config.database_name,
            username=config.username,
            password=config.credential,
            ssl_mode=config.ssl_mode or "prefer",
        )
        ciphertext = connection.encrypted_password
        service = DataSourceDiscoveryService(db_session)
        result = run_async(
            service.discover(
                connection.data_source_id,
                workspace_id=connection.data_source.workspace_id,
                schema_filter=SchemaFilter(
                    include_schemas=frozenset({schema, analytics, empty})
                ),
            )
        )
        db_session.refresh(connection)
        assert connection.encrypted_password == ciphertext
        assert config.credential not in repr(result)
        assert ("orders", "customer_id", "customers", "id") in {
            (
                item.source_table,
                item.source_column,
                item.target_table,
                item.target_column,
            )
            for item in result.relationships
        }
    finally:
        try:
            _drop_fixture(setup, schema, analytics, empty)
        finally:
            setup.close()
