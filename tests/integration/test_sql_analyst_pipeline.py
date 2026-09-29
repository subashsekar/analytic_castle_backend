"""Integration coverage for the Phase 7 SQL analyst pipeline.

Chains generation (7.1), validation (7.2), execution through MCP (7.3),
correction retry (7.4), and history recording (7.5) in one authorized flow.
"""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy.orm import Session

from app.ai.sql_correction import (
    SQLCorrectionService,
    SQLCorrectionStatus,
    SQLCorrectParams,
)
from app.ai.sql_execution import (
    SQLExecuteParams,
    SQLExecutionDatabaseError,
    SQLExecutionService,
    SQLExecutionStatus,
)
from app.ai.sql_generation import SQLGenerateParams, SQLGenerationService
from app.ai.sql_validation import SQLValidateParams, SQLValidationService
from app.connectors.exceptions import ConnectorQueryError
from app.connectors.types import QueryResult
from app.enums import QueryHistoryStatus, WorkspaceRole
from app.mcp import POSTGRES_QUERY_TOOL_NAME, build_postgres_mcp
from app.schemas.query_history import QueryHistoryFilter
from app.services.query_history import QueryHistoryService
from tests.conftest import run_async
from tests.test_sql_execution import _attach_connection, _seed_member
from tests.test_sql_generation import (
    _mock_client,
    _seed_orders_catalog,
    _sql_body,
    _success_response,
)
from tests.test_supervisor import _seed_data_source, _seed_workspace

VALID_SQL = "SELECT region, amount FROM public.orders"


def _llm_returning(*sql_statements: str):
    """Mock LLM client that answers each call with the next SQL draft."""
    queue = list(sql_statements)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        sql = queue.pop(0) if len(queue) > 1 else queue[0]
        return httpx.Response(200, json=_success_response(_sql_body(sql=sql)))

    return _mock_client(handler)


def _seed_authorized_data_source(db_session: Session):
    user, workspace, organization = _seed_workspace(db_session)
    _seed_member(db_session, workspace=workspace, user=user, role=WorkspaceRole.OWNER)
    data_source = _seed_data_source(db_session, workspace=workspace, user=user)
    _attach_connection(db_session, data_source)
    metadata = _seed_orders_catalog(db_session, data_source)
    return user, workspace, organization, data_source, metadata


def test_generate_validate_execute_then_record_history(db_session: Session) -> None:
    user, workspace, organization, data_source, metadata = _seed_authorized_data_source(
        db_session
    )
    generation = SQLGenerationService(db_session, llm_client=_llm_returning(VALID_SQL))
    validation = SQLValidationService(db_session)
    registry, client = build_postgres_mcp(db_session)
    calls: list[tuple[str, int]] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config
        calls.append((sql, limit))
        return QueryResult(columns=("region", "amount"), rows=(("east", 42),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    execution = SQLExecutionService(
        db_session,
        mcp_client=client,
        mcp_registry=registry,
    )
    history = QueryHistoryService(db_session)

    async def _run() -> None:
        generated = await generation.generate(
            SQLGenerateParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                message="Show revenue by region for customer 'alice@example.com'",
                metadata=metadata,
            )
        )
        assert generated.outcome.generated is not None
        draft_sql = generated.outcome.generated.sql

        validated = validation.validate(
            SQLValidateParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql=draft_sql,
                metadata=metadata,
            )
        )
        assert validated.result.is_valid is True
        assert validated.result.validated is not None

        executed = await execution.execute(
            SQLExecuteParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql=validated.result.validated.sql,
                metadata=metadata,
                limit=25,
            )
        )
        assert executed.result.status is SQLExecutionStatus.SUCCEEDED

        recorded = history.record_from_execution_result(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=draft_sql,
            validated_sql=validated.result.validated.sql,
            result=executed.result,
        )
        assert recorded.status is QueryHistoryStatus.SUCCEEDED
        assert recorded.result_metadata is not None
        assert recorded.result_metadata["applied_row_limit"] == 25

    run_async(_run())
    assert len(calls) == 1
    assert calls[0][1] == 25
    listed = history.list_for_user(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
    )
    assert listed.total == 1
    assert listed.items[0].status is QueryHistoryStatus.SUCCEEDED


def test_unsafe_generated_sql_is_rejected_before_mcp_and_recorded(
    db_session: Session,
) -> None:
    user, workspace, organization, data_source, metadata = _seed_authorized_data_source(
        db_session
    )
    generation = SQLGenerationService(
        db_session,
        llm_client=_llm_returning("SELECT region, revenue FROM public.orders"),
    )
    validation = SQLValidationService(db_session)
    registry, _client = build_postgres_mcp(db_session)
    calls: list[str] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        calls.append(sql)
        return QueryResult(columns=("region",), rows=(("east",),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    history = QueryHistoryService(db_session)

    async def _run() -> None:
        generated = await generation.generate(
            SQLGenerateParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                message="Show revenue by region",
                metadata=metadata,
            )
        )
        assert generated.outcome.generated is not None
        result = validation.validate(
            SQLValidateParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql=generated.outcome.generated.sql,
                metadata=metadata,
            )
        )
        assert result.result.is_valid is False
        recorded = history.record_from_validation_result(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=generated.outcome.generated.sql,
            validation=result.result,
        )
        assert recorded is not None
        assert recorded.status is QueryHistoryStatus.REJECTED
        assert recorded.error_metadata is not None
        assert "UNKNOWN_COLUMN" in recorded.error_metadata["violation_codes"]

    run_async(_run())
    assert calls == []


def test_execution_failure_then_correction_is_recorded(db_session: Session) -> None:
    user, workspace, organization, data_source, metadata = _seed_authorized_data_source(
        db_session
    )
    registry, client = build_postgres_mcp(db_session)
    attempts: list[str] = []

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, limit
        attempts.append(sql)
        if len(attempts) == 1:
            raise ConnectorQueryError("column revenue does not exist")
        return QueryResult(columns=("region", "amount"), rows=(("east", 42),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    execution = SQLExecutionService(
        db_session,
        mcp_client=client,
        mcp_registry=registry,
    )
    correction = SQLCorrectionService(
        db_session,
        llm_client=_llm_returning(VALID_SQL),
    )
    history = QueryHistoryService(db_session)
    failing_sql = "SELECT region, amount FROM public.orders WHERE amount > 0"

    async def _run() -> None:
        with pytest.raises(SQLExecutionDatabaseError) as exc_info:
            await execution.execute(
                SQLExecuteParams(
                    workspace_id=workspace.id,
                    organization_id=organization.id,
                    user_id=user.id,
                    data_source_id=data_source.id,
                    sql=failing_sql,
                    metadata=metadata,
                )
            )
        history.record_failed(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=failing_sql,
            validated_sql=failing_sql,
            error_metadata={"code": exc_info.value.code.value},
        )

        corrected, executed = await correction.correct_then_execute(
            SQLCorrectParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql=failing_sql,
                metadata=metadata,
                execution_error_code=exc_info.value.code.value,
                execution_error_message="column revenue does not exist",
            ),
            mcp_client=client,
        )
        assert corrected.outcome.status is SQLCorrectionStatus.CORRECTED
        assert executed.status is SQLExecutionStatus.SUCCEEDED

        recorded = history.record_from_correction_outcome(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=failing_sql,
            outcome=corrected.outcome,
        )
        assert recorded is not None
        assert recorded.status is QueryHistoryStatus.CORRECTED

    run_async(_run())
    assert len(attempts) == 2
    failed = history.list_for_data_source(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        filter_params=QueryHistoryFilter(status=QueryHistoryStatus.FAILED),
    )
    assert failed.total == 1
    corrected_entries = history.list_for_data_source(
        user_id=user.id,
        workspace_id=workspace.id,
        organization_id=organization.id,
        data_source_id=data_source.id,
        filter_params=QueryHistoryFilter(status=QueryHistoryStatus.CORRECTED),
    )
    assert corrected_entries.total == 1


def test_pipeline_history_never_persists_literals_or_secrets(
    db_session: Session,
) -> None:
    user, workspace, organization, data_source, metadata = _seed_authorized_data_source(
        db_session
    )
    sql_with_literal = (
        "SELECT region, amount FROM public.orders WHERE region = 'east-secret'"
    )
    registry, client = build_postgres_mcp(db_session)

    async def _executor(config: object, sql: str, limit: int) -> QueryResult:
        del config, sql, limit
        return QueryResult(columns=("region", "amount"), rows=(("east", 42),))

    registry.get(POSTGRES_QUERY_TOOL_NAME)._executor = _executor
    execution = SQLExecutionService(
        db_session,
        mcp_client=client,
        mcp_registry=registry,
    )
    history = QueryHistoryService(db_session)

    async def _run() -> None:
        executed = await execution.execute(
            SQLExecuteParams(
                workspace_id=workspace.id,
                organization_id=organization.id,
                user_id=user.id,
                data_source_id=data_source.id,
                sql=sql_with_literal,
                metadata=metadata,
            )
        )
        recorded = history.record_from_execution_result(
            user_id=user.id,
            workspace_id=workspace.id,
            organization_id=organization.id,
            data_source_id=data_source.id,
            generated_sql=sql_with_literal,
            validated_sql=sql_with_literal,
            result=executed.result,
        )
        assert "east-secret" not in recorded.generated_sql
        assert recorded.validated_sql is not None
        assert "east-secret" not in recorded.validated_sql
        assert "[REDACTED]" in recorded.generated_sql
        # Row values are never persisted in history metadata.
        assert "42" not in str(recorded.result_metadata)

    run_async(_run())
