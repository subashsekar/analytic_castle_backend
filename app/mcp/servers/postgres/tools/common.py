"""Shared helpers for PostgreSQL MCP catalog and sample adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.connectors import (
    ConnectorAuthenticationError,
    ConnectorConnectionError,
    ConnectorError,
    ConnectorQueryError,
    UnsupportedOperationError,
)
from app.mcp.error_codes import MCPErrorCategory, MCPErrorCode
from app.mcp.exceptions import (
    MCPAccessDeniedError,
    MCPDatabaseError,
    MCPDatabaseUnavailableError,
    MCPError,
    MCPQueryTimeoutError,
    MCPSampleDataError,
    MCPToolValidationError,
)
from app.schemas.metadata import MetadataPageResponse
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionError,
    DataSourceNotFoundError,
)
from app.services.metadata_catalog_exceptions import (
    MetadataCatalogError,
    MetadataNotFoundError,
    MetadataPageError,
)
from app.services.metadata_catalog_types import MetadataPage
from app.services.sample_data_exceptions import (
    SampleDataError,
    SampleDataLimitError,
    SampleSerializationError,
    SampleTableNotFoundError,
)

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
P = TypeVar("P", bound=MetadataPageResponse)


def parse_arguments(model: type[M], arguments: BaseModel) -> M:
    try:
        if isinstance(arguments, model):
            return arguments
        return model.model_validate(arguments)
    except ValidationError as exc:
        raise MCPToolValidationError("MCP tool arguments are invalid") from exc


def run_catalog(operation: Callable[[], T]) -> T:
    try:
        return operation()
    except DataSourceNotFoundError as exc:
        raise MCPAccessDeniedError() from exc
    except MetadataNotFoundError as exc:
        raise MCPError(
            str(exc),
            code=MCPErrorCode.MCP_RESOURCE_NOT_FOUND,
            category=MCPErrorCategory.NOT_FOUND,
            http_status=404,
        ) from exc
    except MetadataPageError as exc:
        raise MCPToolValidationError(str(exc)) from exc
    except MetadataCatalogError as exc:
        raise MCPError(
            "Unable to load metadata",
            code=MCPErrorCode.MCP_DATABASE_ERROR,
            category=MCPErrorCategory.DATA_SOURCE_ERROR,
            http_status=502,
        ) from exc


async def run_sample(operation: Callable[[], Awaitable[T]]) -> T:
    try:
        return await operation()
    except DataSourceNotFoundError as exc:
        raise MCPAccessDeniedError() from exc
    except SampleTableNotFoundError as exc:
        raise MCPSampleDataError(
            str(exc),
            code=MCPErrorCode.MCP_RESOURCE_NOT_FOUND,
            category=MCPErrorCategory.NOT_FOUND,
            http_status=404,
        ) from exc
    except SampleDataLimitError as exc:
        raise MCPToolValidationError(str(exc)) from exc
    except SampleSerializationError as exc:
        raise MCPSampleDataError("Unable to serialize sample data") from exc
    except ConnectionConfigurationError as exc:
        raise MCPSampleDataError(
            "Data source connection configuration is invalid",
            code=MCPErrorCode.MCP_CONFIGURATION_ERROR,
            category=MCPErrorCategory.DATA_SOURCE_ERROR,
        ) from exc
    except DataSourceConnectionError as exc:
        raise _map_connection_failure(exc) from exc
    except UnsupportedOperationError as exc:
        raise MCPSampleDataError(
            "Sample data is not supported for this connector",
        ) from exc
    except ConnectorQueryError as exc:
        raise _map_connector_query_failure(exc) from exc
    except ConnectorAuthenticationError as exc:
        raise MCPDatabaseError(
            "Unable to authenticate with the data source",
        ) from exc
    except ConnectorConnectionError as exc:
        raise _map_connector_connection_failure(exc) from exc
    except ConnectorError as exc:
        raise MCPDatabaseError() from exc
    except TimeoutError as exc:
        raise MCPQueryTimeoutError("The query timed out") from exc
    except SampleDataError as exc:
        raise MCPSampleDataError("Unable to retrieve sample data") from exc


def catalog_page(
    page: MetadataPage[T],
    list_model: type[P],
    item_model: type[BaseModel],
) -> P:
    return list_model.model_validate(
        {
            "items": [item_model.model_validate(item) for item in page.items],
            "page": page.page,
            "page_size": page.page_size,
            "total": page.total,
        }
    )


def _map_connection_failure(exc: DataSourceConnectionError) -> MCPError:
    message = str(exc).lower()
    if "unavailable" in message or "timed out" in message:
        return MCPDatabaseUnavailableError()
    return MCPDatabaseError()


def _map_connector_connection_failure(exc: ConnectorConnectionError) -> MCPError:
    message = str(exc).lower()
    if "unavailable" in message or "timed out" in message:
        return MCPDatabaseUnavailableError()
    return MCPDatabaseError()


def _map_connector_query_failure(exc: ConnectorQueryError) -> MCPError:
    message = str(exc).lower()
    if "timed out" in message:
        return MCPQueryTimeoutError("The query timed out")
    return MCPSampleDataError("Unable to retrieve sample data")
