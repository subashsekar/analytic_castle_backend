from __future__ import annotations

import uuid

import psycopg
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.connectors import PostgreSQLConnector, connector_lifecycle
from app.db.models import (
    DataSourceColumn,
    DataSourceRelationship,
    DataSourceSchema,
    DataSourceTable,
)
from app.enums import MetadataSyncStatus
from app.services.data_source_discovery import DataSourceDiscoveryService
from app.services.discovery_types import SchemaFilter
from app.services.metadata_sync import MetadataSyncService
from app.services.postgresql_discovery import PostgreSQLMetadataDiscoveryService
from tests.conftest import run_async
from tests.integration.test_credential_connector_live import _seed_encrypted_connection
from tests.integration.test_postgresql_connector_live import _integration_config
from tests.integration.test_postgresql_discovery_live import (
    _drop_fixture,
    _install_fixture,
    _quote_ident,
    _require_config,
    _writable_connection,
)

_SKIP_REASON = "PostgreSQL metadata sync integration tests: SKIPPED"


def _discovered_keys(
    db_session: Session,
    data_source_id: uuid.UUID,
) -> tuple[
    list[str],
    list[tuple[str, str]],
    list[tuple[str, str, str]],
    list[tuple[str, str, str, str, str, str, str | None]],
]:
    schemas = list(
        db_session.scalars(
            select(DataSourceSchema.name)
            .where(DataSourceSchema.data_source_id == data_source_id)
            .order_by(DataSourceSchema.name)
        ).all()
    )
    tables = [
        (str(schema), str(table))
        for schema, table in db_session.execute(
            select(DataSourceSchema.name, DataSourceTable.name)
            .join(DataSourceTable, DataSourceTable.schema_id == DataSourceSchema.id)
            .where(DataSourceSchema.data_source_id == data_source_id)
            .order_by(DataSourceSchema.name, DataSourceTable.name)
        ).all()
    ]
    columns = [
        (str(schema), str(table), str(column))
        for schema, table, column in db_session.execute(
            select(DataSourceSchema.name, DataSourceTable.name, DataSourceColumn.name)
            .join(DataSourceTable, DataSourceTable.schema_id == DataSourceSchema.id)
            .join(DataSourceColumn, DataSourceColumn.table_id == DataSourceTable.id)
            .where(DataSourceSchema.data_source_id == data_source_id)
            .order_by(
                DataSourceSchema.name,
                DataSourceTable.name,
                DataSourceColumn.name,
            )
        ).all()
    ]
    relationships = (
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
    relation_keys = sorted(
        (
            relation.source_table.schema.name,
            relation.source_table.name,
            relation.source_column.name,
            relation.target_table.schema.name,
            relation.target_table.name,
            relation.target_column.name,
            relation.constraint_name,
        )
        for relation in relationships
    )
    return schemas, tables, columns, relation_keys


def test_live_metadata_sync_matches_discovery(db_session: Session) -> None:
    config = _require_config()
    schema = f"ac_sync_{uuid.uuid4().hex[:10]}"
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
        schema_filter = SchemaFilter(
            include_schemas=frozenset({schema, analytics, empty})
        )

        async def _discover_direct() -> tuple[int, int, int, int]:
            connector = PostgreSQLConnector(connect_timeout=5)
            async with connector_lifecycle(connector, config) as active:
                service = PostgreSQLMetadataDiscoveryService(
                    active,
                    schema_filter=schema_filter,
                )
                result = await service.discover()
            return (
                len(result.schemas),
                len(result.tables),
                len(result.columns),
                len(result.relationships),
            )

        discovery_counts = run_async(_discover_direct())
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
        assert (
            outcome.schema_count,
            outcome.table_count,
            outcome.column_count,
            outcome.relationship_count,
        ) == discovery_counts

        schemas, tables, columns, relations = _discovered_keys(
            db_session,
            connection.data_source_id,
        )
        assert len(schemas) == discovery_counts[0]
        assert len(tables) == discovery_counts[1]
        assert len(columns) == discovery_counts[2]
        assert len(relations) == discovery_counts[3]
        assert (schema, "orders") in tables
        assert (schema, "order_summaries") in tables
        assert (schema, "orders", "customer_id") in columns
        assert (
            schema,
            "orders",
            "customer_id",
            schema,
            "customers",
            "id",
            "customers_customer_id_fkey",
        ) in relations or any(
            item[0] == schema and item[1] == "orders" and item[2] == "customer_id"
            for item in relations
        )
    finally:
        try:
            _drop_fixture(setup, schema, analytics, empty)
        finally:
            setup.close()


def test_live_metadata_sync_is_idempotent(db_session: Session) -> None:
    config = _integration_config()
    if config is None:
        pytest.skip(_SKIP_REASON)
    schema = f"ac_sync_idem_{uuid.uuid4().hex[:10]}"
    analytics = f"{schema}_an"
    empty = f"{schema}_empty"
    try:
        setup = psycopg.connect(
            host=config.host,
            port=config.port,
            dbname=config.database_name,
            user=config.username,
            password=config.credential,
            sslmode=config.ssl_mode or "prefer",
            autocommit=True,
        )
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
        schema_filter = SchemaFilter(
            include_schemas=frozenset({schema, analytics, empty})
        )
        sync_service = MetadataSyncService(
            db_session,
            discovery_service=DataSourceDiscoveryService(db_session),
        )
        first = run_async(
            sync_service.synchronize(
                connection.data_source_id,
                workspace_id=connection.data_source.workspace_id,
                schema_filter=schema_filter,
            )
        )
        schema_ids = set(
            db_session.scalars(
                select(DataSourceSchema.id).where(
                    DataSourceSchema.data_source_id == connection.data_source_id
                )
            ).all()
        )
        second = run_async(
            sync_service.synchronize(
                connection.data_source_id,
                workspace_id=connection.data_source.workspace_id,
                schema_filter=schema_filter,
            )
        )
        assert first.status is MetadataSyncStatus.SUCCESS
        assert second.status is MetadataSyncStatus.SUCCESS
        assert (
            set(
                db_session.scalars(
                    select(DataSourceSchema.id).where(
                        DataSourceSchema.data_source_id == connection.data_source_id
                    )
                ).all()
            )
            == schema_ids
        )
    finally:
        try:
            setup.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(schema)} CASCADE")
            setup.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(analytics)} CASCADE")
            setup.execute(f"DROP SCHEMA IF EXISTS {_quote_ident(empty)} CASCADE")
        finally:
            setup.close()
