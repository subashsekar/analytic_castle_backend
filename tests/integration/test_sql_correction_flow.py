"""Integration coverage for Phase 7.4 SQL correction and re-validation."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig
from app.ai.sql_correction import (
    SQLCorrectionService,
    SQLCorrectionStatus,
    SQLCorrectionValidationError,
    SQLCorrectParams,
)
from app.ai.sql_validation.models import (
    SQLValidationViolation,
    SQLValidationViolationCode,
)
from app.connectors.types import QueryResult
from app.db.models import DataSourceConnection
from app.enums import WorkspaceRole
from app.mcp import POSTGRES_QUERY_TOOL_NAME, build_postgres_mcp
from app.services.credentials import encrypt_secret
from tests.conftest import run_async
from tests.test_sql_correction import VALID_SQL, _sql_body, _success_response
from tests.test_sql_execution import _seed_member
from tests.test_sql_generation import _seed_orders_catalog
from tests.test_supervisor import _seed_data_source, _seed_workspace

FAKE_API_KEY = "sk-fake-sql-correction-integration"


def _mock_client(handler: object) -> AsyncLLMClient:
    return AsyncLLMClient(
        LLMClientConfig(
            api_key=FAKE_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            timeout=5.0,
            temperature=0.2,
            max_tokens=512,
            max_retries=0,
            retry_base_backoff=0.1,
            retry_max_backoff=1.0,
        ),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


def test_sql_correction_validate_then_optional_execute(db_session: Session) -> None:
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
    service = SQLCorrectionService(
        db_session,
        llm_client=_mock_client(
            lambda _: httpx.Response(200, json=_success_response(_sql_body()))
        ),
    )
    registry, client = build_postgres_mcp(db_session)
    calls: list[str] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        calls.append(sql)
        return QueryResult(columns=("region", "amount"), rows=(("east", 42),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    params = SQLCorrectParams(
        workspace_id=workspace.id,
        organization_id=organization.id,
        user_id=user.id,
        data_source_id=data_source.id,
        sql="SELECT region, revenue FROM public.orders",
        metadata=metadata,
        violations=(
            SQLValidationViolation(
                code=SQLValidationViolationCode.UNKNOWN_COLUMN,
                message="Unknown column revenue",
                identifier="revenue",
            ),
        ),
    )

    async def _run() -> None:
        correction, execution = await service.correct_then_execute(
            params, mcp_client=client, limit=25
        )
        assert correction.outcome.status is SQLCorrectionStatus.CORRECTED
        assert correction.outcome.validated is not None
        assert VALID_SQL.split()[0].lower() == "select"
        assert execution.rows == [["east", 42]]
        assert execution.applied_row_limit == 25

        dangerous = SQLCorrectionService(
            db_session,
            llm_client=_mock_client(
                lambda _: httpx.Response(
                    200,
                    json=_success_response(_sql_body(sql="DROP TABLE public.orders")),
                )
            ),
        )
        with pytest.raises(SQLCorrectionValidationError):
            await dangerous.correct_then_execute(params, mcp_client=client)

    run_async(_run())
    assert len(calls) == 1
    assert "DROP" not in calls[0].upper()


def test_sql_correction_concurrent_requests_do_not_cross_workspaces(
    db_session: Session,
) -> None:
    user_a, workspace_a, organization_a = _seed_workspace(db_session)
    user_b, workspace_b, organization_b = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace_a, user=user_a)
    _seed_member(db_session, workspace=workspace_b, user=user_b)
    source_a = _seed_data_source(db_session, workspace=workspace_a, user=user_a)
    source_b = _seed_data_source(db_session, workspace=workspace_b, user=user_b)
    metadata_a = _seed_orders_catalog(db_session, source_a)
    metadata_b = _seed_orders_catalog(db_session, source_b)

    def handler_for(tag: str):
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            return httpx.Response(
                200,
                json=_success_response(
                    _sql_body(
                        sql=(
                            "SELECT region, amount FROM public.orders"
                            if tag == "A"
                            else "SELECT region FROM public.orders"
                        )
                    )
                ),
            )

        return handler

    service_a = SQLCorrectionService(
        db_session, llm_client=_mock_client(handler_for("A"))
    )
    service_b = SQLCorrectionService(
        db_session, llm_client=_mock_client(handler_for("B"))
    )

    async def _run() -> None:
        first, second = await asyncio.gather(
            service_a.correct(
                SQLCorrectParams(
                    workspace_id=workspace_a.id,
                    organization_id=organization_a.id,
                    user_id=user_a.id,
                    data_source_id=source_a.id,
                    sql="SELECT region, revenue FROM public.orders",
                    metadata=metadata_a,
                    violations=(
                        SQLValidationViolation(
                            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
                            message="Unknown column revenue",
                            identifier="revenue",
                        ),
                    ),
                )
            ),
            service_b.correct(
                SQLCorrectParams(
                    workspace_id=workspace_b.id,
                    organization_id=organization_b.id,
                    user_id=user_b.id,
                    data_source_id=source_b.id,
                    sql="SELECT missing FROM public.orders",
                    metadata=metadata_b,
                    violations=(
                        SQLValidationViolation(
                            code=SQLValidationViolationCode.UNKNOWN_COLUMN,
                            message="Unknown column missing",
                            identifier="missing",
                        ),
                    ),
                )
            ),
        )
        assert first.workspace_id == workspace_a.id
        assert second.workspace_id == workspace_b.id
        assert first.data_source_id == source_a.id
        assert second.data_source_id == source_b.id
        assert first.outcome.status is SQLCorrectionStatus.CORRECTED
        assert second.outcome.status is SQLCorrectionStatus.CORRECTED

    run_async(_run())
