"""Phase 5.6 — sample_rows masking bypass, failure, and limit tests."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.connectors.exceptions import ConnectorConnectionError, ConnectorQueryError
from app.db.models import User, Workspace
from app.enums import ColumnSensitivity, DataSourceTableType
from app.mcp import MCPErrorCode, MCPToolContext, build_postgres_mcp
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPDatabaseUnavailableError,
    MCPQueryTimeoutError,
    MCPSampleDataError,
)
from app.mcp.schemas import MCPSampleRowsRequest
from app.mcp.servers.postgres.tools.sample_rows import (
    POSTGRES_SAMPLE_ROWS_TOOL_NAME,
    PostgresSampleRowsTool,
)
from app.services.data_masking import REDACTED
from app.services.sample_data_exceptions import (
    SampleSerializationError,
    SampleTableNotFoundError,
)
from app.services.sample_data_types import SampleColumn, SampleDataResult
from tests.conftest import run_async
from tests.test_ai_metadata import _schema, _source, _table


class _FakeSampler:
    def __init__(
        self,
        result: SampleDataResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls = 0

    async def get_sample(self, *args: object, **kwargs: object) -> SampleDataResult:
        del args, kwargs
        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _masked_result(source_id: uuid.UUID, table_id: uuid.UUID) -> SampleDataResult:
    return SampleDataResult(
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
            SampleColumn(
                name="api_key",
                data_type="string",
                sensitivity=ColumnSensitivity.SECRET,
                masked=True,
            ),
        ),
        rows=({"email": REDACTED, "api_key": REDACTED},),
        row_limit=10,
        truncated_columns=False,
    )


@pytest.mark.parametrize(
    "payload_extra",
    [
        {"raw": True},
        {"mask": False},
        {"include_unmasked": True},
        {"raw": True, "mask": False, "include_unmasked": True},
    ],
)
def test_sample_rows_rejects_masking_bypass_flags(
    payload_extra: dict[str, object],
) -> None:
    body = {
        "data_source_id": str(uuid.uuid4()),
        "table_id": str(uuid.uuid4()),
        **payload_extra,
    }
    with pytest.raises(ValidationError):
        MCPSampleRowsRequest.model_validate(body)


def test_sample_rows_masks_pii_and_secrets(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    table_id = uuid.uuid4()
    sampler = _FakeSampler(_masked_result(source.id, table_id))

    async def _run() -> None:
        result = await PostgresSampleRowsTool(db_session, sampler=sampler).invoke(
            MCPSampleRowsRequest(data_source_id=source.id, table_id=table_id, limit=3),
            MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
        )
        assert result.rows == [{"email": REDACTED, "api_key": REDACTED}]
        assert all(column.masked for column in result.columns)
        dumped = str(result.model_dump())
        assert "secret@" not in dumped
        assert "sk-live" not in dumped

    run_async(_run())


def test_sample_rows_timeout_and_unavailable_mapping(
    db_session: Session, workspace: Workspace, test_user: User, workspace_member: object
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)

    async def _run() -> None:
        with pytest.raises(MCPQueryTimeoutError):
            await PostgresSampleRowsTool(
                db_session,
                sampler=_FakeSampler(error=ConnectorQueryError("query timed out")),
            ).invoke(
                MCPSampleRowsRequest(data_source_id=source.id, table_id=uuid.uuid4()),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        with pytest.raises(MCPDatabaseUnavailableError):
            await PostgresSampleRowsTool(
                db_session,
                sampler=_FakeSampler(
                    error=ConnectorConnectionError("database unavailable")
                ),
            ).invoke(
                MCPSampleRowsRequest(data_source_id=source.id, table_id=uuid.uuid4()),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        with pytest.raises(MCPSampleDataError) as exc_info:
            await PostgresSampleRowsTool(
                db_session,
                sampler=_FakeSampler(error=SampleSerializationError("bad value")),
            ).invoke(
                MCPSampleRowsRequest(data_source_id=source.id, table_id=uuid.uuid4()),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert "bad value" not in str(exc_info.value)
        with pytest.raises(MCPSampleDataError) as missing:
            await PostgresSampleRowsTool(
                db_session,
                sampler=_FakeSampler(error=SampleTableNotFoundError("Table not found")),
            ).invoke(
                MCPSampleRowsRequest(data_source_id=source.id, table_id=uuid.uuid4()),
                MCPToolContext(workspace_id=workspace.id, user_id=test_user.id),
            )
        assert missing.value.code is MCPErrorCode.MCP_RESOURCE_NOT_FOUND

    run_async(_run())


def test_sample_rows_unauthorized_and_cross_workspace(
    db_session: Session,
    workspace: Workspace,
    test_user: User,
    admin_user: User,
    workspace_member: object,
) -> None:
    del workspace_member
    source = _source(db_session, workspace, test_user)
    schema = _schema(db_session, source, "public")
    table = _table(db_session, schema, "customers")
    sampler = _FakeSampler(_masked_result(source.id, table.id))
    _, client = build_postgres_mcp(db_session)

    async def _run() -> None:
        with pytest.raises(MCPAccessDeniedError):
            await PostgresSampleRowsTool(db_session, sampler=sampler).invoke(
                MCPSampleRowsRequest(data_source_id=source.id, table_id=table.id),
                MCPToolContext(workspace_id=workspace.id, user_id=admin_user.id),
            )
        with pytest.raises(MCPAccessDeniedError):
            await client.call_tool(
                POSTGRES_SAMPLE_ROWS_TOOL_NAME,
                {"data_source_id": str(source.id), "table_id": str(table.id)},
                MCPToolContext(workspace_id=uuid.uuid4(), user_id=admin_user.id),
            )
        assert sampler.calls == 0

    run_async(_run())
