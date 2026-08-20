from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.enums import ColumnSensitivity, DataSourceTableType
from app.mcp import MCPError, MCPToolContext, build_postgres_mcp
from app.mcp.exceptions import MCPToolValidationError
from app.mcp.schemas import MCPSampleRowsRequest
from app.mcp.servers.postgres.tools.sample_rows import (
    POSTGRES_SAMPLE_ROWS_TOOL_NAME,
    PostgresSampleRowsTool,
)
from app.services.data_masking import REDACTED
from app.services.sample_data_exceptions import SampleDataLimitError
from app.services.sample_data_types import SampleColumn, SampleDataResult
from app.db.models import User, Workspace
from tests.conftest import run_async
from tests.test_ai_metadata import _source


def test_sample_rows_reuses_service_and_preserves_masking(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    workspace_member: object,
) -> None:
    source = _source(db_session, workspace, test_user)
    source_id = source.id
    table_id = uuid.uuid4()
    result = SampleDataResult(
        data_source_id=source_id,
        table_id=table_id,
        schema_name="public",
        table_name="customers",
        table_type=DataSourceTableType.TABLE,
        columns=(
            SampleColumn(
                name="email",
                data_type="string",
                sensitivity=ColumnSensitivity.PII,
                masked=True,
            ),
        ),
        rows=({"email": REDACTED},),
        row_limit=10,
        truncated_columns=False,
    )
    sampler = _FakeSampler(result)

    async def _run() -> None:
        payload = await PostgresSampleRowsTool(db_session, sampler=sampler).invoke(
            MCPSampleRowsRequest(data_source_id=source_id, table_id=table_id, limit=5),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert payload.rows == [{"email": REDACTED}]
        assert payload.columns[0].masked is True
        assert payload.row_limit == 10
        assert sampler.calls == [(source_id, table_id, workspace.id, 5)]
        dumped = payload.model_dump()
        assert "password" not in dumped
        assert "connection" not in str(dumped)

    run_async(_run())


def test_sample_rows_maps_limit_errors(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    sampler = _FakeSampler(error=SampleDataLimitError("Sample row limit is invalid"))
    source = _source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(MCPToolValidationError, match="Sample row limit"):
            await PostgresSampleRowsTool(db_session, sampler=sampler).invoke(
                MCPSampleRowsRequest(
                    data_source_id=source.id,
                    table_id=uuid.uuid4(),
                    limit=1,
                ),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )

    run_async(_run())


def test_sample_rows_rejects_credentials() -> None:
    with pytest.raises(ValidationError):
        MCPSampleRowsRequest.model_validate(
            {
                "data_source_id": str(uuid.uuid4()),
                "table_id": str(uuid.uuid4()),
                "sql": "SELECT * FROM customers",
            }
        )


def test_sample_rows_rejects_foreign_workspace(
    db_session: Session, workspace: Workspace, test_user: User, admin_user: User
) -> None:
    source = _source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(MCPError, match="Data source not found"):
            await PostgresSampleRowsTool(db_session).invoke(
                MCPSampleRowsRequest(
                    data_source_id=source.id,
                    table_id=uuid.uuid4(),
                ),
                MCPToolContext(workspace_id=uuid.uuid4(), user_id=admin_user.id),
            )

    run_async(_run())


def test_client_resolves_sample_rows_tool(db_session: Session) -> None:
    registry, _client = build_postgres_mcp(db_session)
    tool = registry.get(POSTGRES_SAMPLE_ROWS_TOOL_NAME)
    assert tool.name == POSTGRES_SAMPLE_ROWS_TOOL_NAME


class _FakeSampler:
    def __init__(
        self,
        result: SampleDataResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, ...]] = []

    async def get_sample(
        self,
        data_source_id: uuid.UUID,
        table_id: uuid.UUID,
        *,
        workspace_id: uuid.UUID,
        limit: int | None = None,
    ) -> SampleDataResult:
        self.calls.append((data_source_id, table_id, workspace_id, limit))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result
