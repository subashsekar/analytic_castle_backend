"""PostgreSQL read-only query MCP tool."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from app.connectors import ConnectorConfig, ConnectorError, ConnectorQueryError
from app.connectors.postgresql import PostgreSQLConnector
from app.connectors.readonly_sql import validate_readonly_sql
from app.connectors.types import QueryResult
from app.connectors.base import connector_lifecycle
from app.core.config import settings
from app.enums import DataSourceType
from app.mcp.exceptions import (
    MCPQueryError,
    MCPQueryRejectedError,
    MCPQueryResultError,
    MCPQueryTimeoutError,
    MCPToolValidationError,
)
from app.mcp.types import MCPQueryRequest, MCPQueryResult, MCPToolContext
from app.services.credentials import CredentialError, connector_config_from_connection
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceNotFoundError,
    load_configured_data_source,
)
from app.services.sample_serialization import serialize_sample_value

logger = logging.getLogger(__name__)

POSTGRES_QUERY_TOOL_NAME = "postgres.query"

QueryExecutor = Callable[[ConnectorConfig, str, int], Awaitable[QueryResult]]


class PostgresQueryTool:
    name = POSTGRES_QUERY_TOOL_NAME
    description = (
        "Run a single read-only analytical SQL query against an authorized "
        "PostgreSQL data source. Does not modify data."
    )
    input_model = MCPQueryRequest
    output_model = MCPQueryResult

    def __init__(
        self,
        session: Session,
        *,
        executor: QueryExecutor | None = None,
    ) -> None:
        self._session = session
        self._executor = executor or _execute_with_connector

    async def invoke(
        self, arguments: BaseModel, context: MCPToolContext
    ) -> MCPQueryResult:
        try:
            request = (
                arguments
                if isinstance(arguments, MCPQueryRequest)
                else MCPQueryRequest.model_validate(arguments)
            )
        except ValidationError as exc:
            raise MCPToolValidationError("MCP tool arguments are invalid") from exc
        return await self.execute(request, context)

    async def execute(
        self, request: MCPQueryRequest, context: MCPToolContext
    ) -> MCPQueryResult:
        started = time.perf_counter()
        try:
            validate_readonly_sql(request.sql)
        except ConnectorQueryError as exc:
            raise MCPQueryRejectedError(
                "Only a single read-only query is allowed"
            ) from exc
        row_limit = _resolve_limit(request.limit)
        config = self._load_config(request.data_source_id, context.workspace_id)
        try:
            raw = await self._executor(config, request.sql, row_limit)
        except ConnectorQueryError as exc:
            message = str(exc).lower()
            if "timed out" in message:
                raise MCPQueryTimeoutError("The query timed out") from exc
            raise MCPQueryError("Read-only query failed") from exc
        except TimeoutError as exc:
            raise MCPQueryTimeoutError("The query timed out") from exc
        except ConnectorError as exc:
            raise MCPQueryError("Read-only query failed") from exc

        result = _serialize_result(raw)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "MCP query completed data_source_id=%s row_count=%s truncated=%s "
            "duration_ms=%.0f",
            request.data_source_id,
            result.row_count,
            result.truncated,
            duration_ms,
        )
        return result

    def _load_config(
        self, data_source_id: UUID, workspace_id: UUID
    ) -> ConnectorConfig:
        try:
            data_source, connection = load_configured_data_source(
                self._session,
                data_source_id,
                workspace_id=workspace_id,
            )
        except DataSourceNotFoundError as exc:
            raise MCPQueryError("Data source not found") from exc
        except ConnectionConfigurationError as exc:
            raise MCPQueryError("Data source connection configuration is invalid") from exc
        if data_source.type is not DataSourceType.POSTGRESQL:
            raise MCPQueryError("Read-only query is not supported for this connector")
        try:
            return connector_config_from_connection(connection)
        except CredentialError as exc:
            raise MCPQueryError(
                "Unable to load data source credentials"
            ) from exc


async def _execute_with_connector(
    config: ConnectorConfig,
    sql: str,
    limit: int,
) -> QueryResult:
    connector = PostgreSQLConnector(connect_timeout=settings.MCP_QUERY_TIMEOUT_SECONDS)
    async with connector_lifecycle(connector, config) as active:
        return await active.execute_query(sql, limit=limit)


def _resolve_limit(limit: int | None) -> int:
    maximum = settings.MCP_QUERY_MAX_LIMIT
    if limit is None:
        return min(settings.MCP_QUERY_DEFAULT_LIMIT, maximum)
    return min(limit, maximum)


def _serialize_result(raw: QueryResult) -> MCPQueryResult:
    rows: list[list[object]] = []
    encoded_size = 2
    truncated = raw.truncated
    for row in raw.rows:
        serialized = [
            serialize_sample_value(
                value,
                max_value_chars=settings.MCP_QUERY_MAX_VALUE_CHARS,
                max_json_chars=settings.MCP_QUERY_MAX_JSON_CHARS,
            )
            for value in row
        ]
        encoded_size += _encoded_size(serialized)
        if encoded_size > settings.MCP_QUERY_MAX_RESULT_CHARS:
            if not rows:
                raise MCPQueryResultError("The query result is too large")
            truncated = True
            break
        rows.append(serialized)
    return MCPQueryResult(
        columns=list(raw.columns),
        rows=rows,
        row_count=len(rows),
        truncated=truncated,
    )


def _encoded_size(value: object) -> int:
    try:
        return len(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise MCPQueryResultError("Unable to serialize query result") from exc
