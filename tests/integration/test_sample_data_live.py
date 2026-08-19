from __future__ import annotations

import uuid
from typing import Any
from uuid import UUID

import psycopg
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors import PostgreSQLConnector
from app.connectors.registry import register_connector
from app.db.models import DataSourceSchema, DataSourceTable
from app.enums import DataSourceTableType, DataSourceType, MetadataSyncStatus
from app.services.data_masking import REDACTED
from app.services.data_source_discovery import DataSourceDiscoveryService
from app.services.discovery_types import SchemaFilter
from app.services.metadata_sync import MetadataSyncService
from app.services.sample_data import SampleDataService
from app.services.sample_serialization import BINARY_PLACEHOLDER, TRUNCATED_SUFFIX
from tests.conftest import run_async
from tests.integration.test_credential_connector_live import _seed_encrypted_connection
from tests.integration.test_postgresql_connector_live import _integration_config
from tests.integration.test_postgresql_discovery_live import (
    _quote_ident,
    _writable_connection,
)

_SKIP_REASON = "PostgreSQL sample-data integration tests: SKIPPED"

_CUSTOMER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")


class _RecordingConnector(PostgreSQLConnector):
    def __init__(self) -> None:
        super().__init__(connect_timeout=5)
        self.queries: list[tuple[str, object]] = []

    async def _fetch_sample_rows(
        self,
        schema_name: str,
        table_name: str,
        column_names: Any,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        from psycopg import sql

        composed = sql.SQL("SELECT {fields} FROM {table} LIMIT %(row_limit)s").format(
            fields=sql.SQL(", ").join(sql.Identifier(name) for name in column_names),
            table=sql.Identifier(schema_name, table_name),
        )
        self.queries.append((composed.as_string(None), {"row_limit": limit}))
        return await super()._fetch_sample_rows(
            schema_name,
            table_name,
            column_names,
            limit=limit,
        )


def _require_config() -> Any:
    config = _integration_config()
    if config is None:
        pytest.skip(_SKIP_REASON)
    return config


def _install_sample_fixture(connection: psycopg.Connection[Any], schema: str) -> None:
    s = _quote_ident(schema)
    statements = [
        f"CREATE SCHEMA {s}",
        f"""
        CREATE TABLE {s}.customers (
            id integer PRIMARY KEY,
            first_name text,
            last_name text,
            email text,
            phone text,
            amount numeric,
            customer_uuid uuid,
            born_on date,
            created_at timestamptz,
            metadata jsonb,
            avatar bytea,
            notes text,
            password text,
            api_token text,
            contact text
        )
        """,
        f"""
        INSERT INTO {s}.customers VALUES (
            12,
            'John',
            'Doe',
            'john.doe@example.com',
            '9876543210',
            19.90,
            '{_CUSTOMER_ID}',
            DATE '1990-05-01',
            TIMESTAMPTZ '2024-03-15 12:30:00+00',
            '{{"plan": "pro"}}'::jsonb,
            decode('0001', 'hex'),
            '{"n" * 20_000}',
            'hunter2-password',
            'tok_live_secret_value',
            'ada@example.com'
        )
        """,
        f"CREATE TABLE {s}.empty_customers (id integer PRIMARY KEY, email text)",
        f"CREATE VIEW {s}.customer_emails AS SELECT id, email FROM {s}.customers",
    ]
    for statement in statements:
        connection.execute(statement)


def _drop_sample_fixture(connection: psycopg.Connection[Any], schema: str) -> None:
    connection.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")


def _table_id(
    db_session: Session,
    data_source_id: UUID,
    schema_name: str,
    table_name: str,
) -> UUID:
    table = db_session.scalars(
        select(DataSourceTable)
        .join(DataSourceSchema, DataSourceTable.schema_id == DataSourceSchema.id)
        .where(
            DataSourceSchema.data_source_id == data_source_id,
            DataSourceSchema.name == schema_name,
            DataSourceTable.name == table_name,
        )
    ).one()
    return table.id


def test_live_sample_masks_and_serializes_rows(db_session: Session) -> None:
    config = _require_config()
    schema = f"ac_sample_{uuid.uuid4().hex[:10]}"
    try:
        setup = _writable_connection(config)
    except (psycopg.Error, OSError) as exc:
        pytest.skip(f"{_SKIP_REASON} — {type(exc).__name__}")

    try:
        try:
            _install_sample_fixture(setup, schema)
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
        schema_filter = SchemaFilter(include_schemas=frozenset({schema}))
        sync_service = MetadataSyncService(
            db_session,
            discovery_service=DataSourceDiscoveryService(db_session),
        )
        outcome = run_async(
            sync_service.synchronize(
                connection.data_source_id,
                workspace_id=connection.data_source.workspace_id,
                schema_filter=schema_filter,
            )
        )
        assert outcome.status is MetadataSyncStatus.SUCCESS

        created: list[_RecordingConnector] = []

        def _builder() -> _RecordingConnector:
            connector = _RecordingConnector()
            created.append(connector)
            return connector

        register_connector(DataSourceType.POSTGRESQL, _builder)

        table_id = _table_id(db_session, connection.data_source_id, schema, "customers")
        service = SampleDataService(db_session)
        result = run_async(
            service.get_sample(
                connection.data_source_id,
                table_id,
                workspace_id=connection.data_source.workspace_id,
                limit=5,
            )
        )

        assert result.row_count == 1
        assert result.row_limit == 5
        row = result.rows[0]
        assert row["id"] == 12
        assert row["email"] == "j***@example.com"
        assert row["phone"] == "******3210"
        assert row["first_name"] == "J***"
        assert row["last_name"] == "D**"
        assert row["password"] == REDACTED
        assert row["api_token"] == REDACTED
        assert row["contact"] == "a***@example.com"
        assert row["customer_uuid"] == str(_CUSTOMER_ID)
        assert row["amount"] == "19.90"
        assert row["born_on"] == "1990-05-01"
        assert isinstance(row["created_at"], str)
        assert row["metadata"] == {"plan": "pro"}
        assert row["avatar"] == BINARY_PLACEHOLDER
        assert isinstance(row["notes"], str)
        assert row["notes"].endswith(TRUNCATED_SUFFIX)
        assert "john.doe@example.com" not in str(row)
        assert "hunter2-password" not in str(row)
        assert "tok_live_secret_value" not in str(row)
        assert config.credential not in str(row)

        assert created
        query, params = created[0].queries[0]
        assert "LIMIT" in query.upper()
        assert params == {"row_limit": 5}
        assert "*" not in query.split("FROM", 1)[0]
        assert "ORDER BY RANDOM" not in query.upper()

        empty_id = _table_id(
            db_session, connection.data_source_id, schema, "empty_customers"
        )
        empty = run_async(
            service.get_sample(
                connection.data_source_id,
                empty_id,
                workspace_id=connection.data_source.workspace_id,
            )
        )
        assert empty.rows == ()
        assert empty.row_count == 0

        view_id = _table_id(
            db_session, connection.data_source_id, schema, "customer_emails"
        )
        view = run_async(
            service.get_sample(
                connection.data_source_id,
                view_id,
                workspace_id=connection.data_source.workspace_id,
                limit=1,
            )
        )
        assert view.table_type is DataSourceTableType.VIEW
        assert view.row_count == 1
        assert view.rows[0]["email"] == "j***@example.com"
    finally:
        try:
            _drop_sample_fixture(setup, schema)
        finally:
            setup.close()
