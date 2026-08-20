"""PostgreSQL MCP tool: return a masked sample of table rows."""

from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.mcp.schemas import MCPSampleRowsRequest, MCPToolContext
from app.mcp.security import MCPToolPermission, resolve_authorized_workspace_id
from app.mcp.servers.postgres.tools.common import parse_arguments, run_sample
from app.schemas.metadata import SampleColumnResponse, SampleDataResponse
from app.services.sample_data import SampleDataService
from app.services.sample_data_types import SampleDataResult

POSTGRES_SAMPLE_ROWS_TOOL_NAME = "postgres.sample_rows"


class PostgresSampleRowsTool:
    name = POSTGRES_SAMPLE_ROWS_TOOL_NAME
    permission: MCPToolPermission = MCPToolPermission.SAMPLE_DATA_READ
    description = (
        "Return a small, masked sample of rows from a discovered table on an "
        "authorized PostgreSQL data source. Uses existing sample limits, PII "
        "masking, and read-only retrieval."
    )
    input_model: type[BaseModel] = MCPSampleRowsRequest
    output_model: type[BaseModel] = SampleDataResponse

    def __init__(
        self,
        session: Session,
        *,
        sampler: SampleDataService | None = None,
    ) -> None:
        self._session = session
        self._sampler = sampler

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> SampleDataResponse:
        request = parse_arguments(MCPSampleRowsRequest, arguments)
        return await self.execute(request, context)

    async def execute(
        self, request: MCPSampleRowsRequest, context: MCPToolContext
    ) -> SampleDataResponse:
        sampler = self._sampler or SampleDataService(self._session)
        resolved_workspace_id = resolve_authorized_workspace_id(
            session=self._session,
            context=context,
            data_source_id=request.data_source_id,
            permission=self.permission,
        )

        async def _load() -> SampleDataResult:
            return await sampler.get_sample(
                request.data_source_id,
                request.table_id,
                workspace_id=resolved_workspace_id,
                limit=request.limit,
            )

        result = await run_sample(_load)
        return SampleDataResponse(
            data_source_id=result.data_source_id,
            table_id=result.table_id,
            schema_name=result.schema_name,
            table_name=result.table_name,
            table_type=result.table_type,
            columns=[
                SampleColumnResponse.model_validate(column) for column in result.columns
            ],
            rows=list(result.rows),
            row_count=result.row_count,
            row_limit=result.row_limit,
            truncated_columns=result.truncated_columns,
        )
