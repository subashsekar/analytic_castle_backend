"""Integration coverage for Phase 7.3 SQL execution through MCP."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.orm import Session

from app.ai.sql_execution import (
    SQLExecuteParams,
    SQLExecutionService,
    SQLExecutionStatus,
    SQLExecutionValidationError,
)
from app.connectors.types import QueryResult
from app.db.models import DataSourceConnection
from app.enums import WorkspaceRole
from app.mcp import POSTGRES_QUERY_TOOL_NAME, build_postgres_mcp
from app.services.credentials import encrypt_secret
from tests.conftest import run_async
from tests.test_sql_execution import _seed_member
from tests.test_sql_generation import _seed_orders_catalog
from tests.test_supervisor import _seed_data_source, _seed_workspace


def test_sql_execution_validate_then_mcp_end_to_end(db_session: Session) -> None:
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.OWNER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    db_session.add(
        DataSourceConnection(
            data_source_id=data_source.id,
            host="db.internal.example",
            port=5432,
            database_name="analytics",
            username="readonly",
            encrypted_password=encrypt_secret("IntegrationSecret!42"),
            ssl_mode="prefer",
        )
    )
    db_session.flush()
    metadata = _seed_orders_catalog(db_session, data_source)

    calls: list[tuple[str, int]] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config
        calls.append((sql, limit))
        return QueryResult(
            columns=("region", "amount"),
            rows=(("east", 42),),
            truncated=False,
        )

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    service = SQLExecutionService(db_session, mcp_client=client, mcp_registry=registry)

    async def _run() -> None:
        outcome = await service.execute(
            SQLExecuteParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql="SELECT region, amount FROM public.orders",
                metadata=metadata,
                limit=25,
            )
        )
        assert outcome.result.status is SQLExecutionStatus.SUCCEEDED
        assert outcome.result.rows == [["east", 42]]
        assert outcome.result.applied_row_limit == 25

        with pytest.raises(SQLExecutionValidationError):
            await service.execute(
                SQLExecuteParams(
                    workspace_id=workspace.id,
                    organization_id=organization.id,
                    user_id=user.id,
                    data_source_id=data_source.id,
                    sql="DROP TABLE public.orders",
                    metadata=metadata,
                )
            )

    run_async(_run())
    assert len(calls) == 1
    assert "DROP" not in calls[0][0].upper()
    assert calls[0][1] == 25


def test_sql_execution_concurrent_requests_do_not_cross_workspaces(
    db_session: Session,
) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    for source in (source_a, source_b):
        db_session.add(
            DataSourceConnection(
                data_source_id=source.id,
                host="db.internal.example",
                port=5432,
                database_name="analytics",
                username="readonly",
                encrypted_password=encrypt_secret("IntegrationSecret!42"),
                ssl_mode="prefer",
            )
        )
    db_session.flush()
    metadata_a = _seed_orders_catalog(db_session, source_a)
    metadata_b = _seed_orders_catalog(db_session, source_b)

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        await asyncio.sleep(0)
        tag = "A" if "amount" in sql else "B"
        return QueryResult(columns=("tag",), rows=((tag,),))

    registry, client = build_postgres_mcp(db_session)
    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    service = SQLExecutionService(db_session, mcp_client=client, mcp_registry=registry)

    async def _run() -> None:
        first, second = await asyncio.gather(
            service.execute(
                SQLExecuteParams(
                    workspace_id=workspace_a.id,
                    organization_id=organization_a.id,
                    user_id=user_a.id,
                    data_source_id=source_a.id,
                    sql="SELECT region, amount FROM public.orders",
                    metadata=metadata_a,
                )
            ),
            service.execute(
                SQLExecuteParams(
                    workspace_id=workspace_b.id,
                    organization_id=organization_b.id,
                    user_id=user_b.id,
                    data_source_id=source_b.id,
                    sql="SELECT region FROM public.orders",
                    metadata=metadata_b,
                )
            ),
        )
        assert first.result.rows == [["A"]]
        assert second.result.rows == [["B"]]
        assert first.workspace_id == workspace_a.id
        assert second.workspace_id == workspace_b.id

    run_async(_run())
