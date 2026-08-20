"""Shared helpers for PostgreSQL MCP catalog and sample adapters."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.connectors import UnsupportedOperationError
from app.mcp.exceptions import MCPError, MCPToolValidationError
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
        raise MCPError("Data source not found") from exc
    except MetadataNotFoundError as exc:
        raise MCPError(str(exc)) from exc
    except MetadataPageError as exc:
        raise MCPToolValidationError(str(exc)) from exc
    except MetadataCatalogError as exc:
        raise MCPError("Unable to load metadata") from exc


async def run_sample(operation: Callable[[], Awaitable[T]]) -> T:
    try:
        return await operation()
    except DataSourceNotFoundError as exc:
        raise MCPError("Data source not found") from exc
    except SampleTableNotFoundError as exc:
        raise MCPError(str(exc)) from exc
    except SampleDataLimitError as exc:
        raise MCPToolValidationError(str(exc)) from exc
    except ConnectionConfigurationError as exc:
        raise MCPError(str(exc)) from exc
    except DataSourceConnectionError as exc:
        raise MCPError(str(exc)) from exc
    except UnsupportedOperationError as exc:
        raise MCPError(str(exc)) from exc
    except SampleDataError as exc:
        raise MCPError("Unable to retrieve sample data") from exc


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
