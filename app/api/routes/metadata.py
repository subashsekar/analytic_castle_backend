from typing import Annotated, Any, NoReturn
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import require_data_source_permission, require_rate_limit
from app.connectors import UnsupportedConnectorError, UnsupportedOperationError
from app.core.authorization import DataSourceAccess
from app.core.config import settings
from app.db.session import get_db
from app.enums import DataSourceTableType, MetadataSearchType, WorkspacePermission
from app.schemas.metadata import (
    MetadataColumnListResponse,
    MetadataColumnResponse,
    MetadataRelationshipListResponse,
    MetadataRelationshipResponse,
    MetadataSchemaListResponse,
    MetadataSchemaResponse,
    MetadataSearchItemResponse,
    MetadataSearchResponse,
    MetadataSyncResponse,
    MetadataSyncStatusResponse,
    MetadataTableListResponse,
    MetadataTableResponse,
    SampleColumnResponse,
    SampleDataRequest,
    SampleDataResponse,
)
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceNotFoundError,
)
from app.services.metadata_catalog import MetadataCatalogService
from app.services.metadata_catalog_exceptions import (
    MetadataNotFoundError,
    MetadataPageError,
)
from app.services.metadata_catalog_types import (
    ColumnRecord,
    MetadataPage,
    RelationshipRecord,
    SchemaRecord,
    TableRecord,
)
from app.services.metadata_search import MetadataSearchService
from app.services.metadata_search_exceptions import (
    MetadataSearchError,
    MetadataSearchLimitError,
)
from app.services.metadata_sync import MetadataSyncService
from app.services.metadata_sync_exceptions import (
    ConcurrentMetadataSyncError,
    MetadataSyncError,
    MetadataSyncPersistenceError,
)
from app.services.metadata_sync_types import MetadataSyncResult
from app.services.sample_data import SampleDataService
from app.services.sample_data_exceptions import (
    SampleDataError,
    SampleDataLimitError,
    SampleIdentifierError,
    SampleQueryError,
    SampleTableNotFoundError,
)
from app.services.sample_data_types import SampleDataResult

router = APIRouter(prefix="/api/v1/data-sources", tags=["metadata"])

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "Resource not found"}
}
_RATE_LIMITED: dict[int | str, dict[str, Any]] = {
    status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"}
}

TABLE_NOT_FOUND_DETAIL = "Table not found"
DATA_SOURCE_NOT_FOUND_DETAIL = "Data source not found"
SEARCH_FAILED_DETAIL = "Unable to search metadata"
SYNC_FAILED_DETAIL = "Metadata synchronization failed"
SAMPLE_FAILED_DETAIL = "Unable to retrieve sample data"
CONNECTION_CONFIG_DETAIL = "Data source connection configuration is invalid"
UNSUPPORTED_CONNECTOR_DETAIL = "Unsupported connector type"
UNSUPPORTED_SAMPLE_DETAIL = "Sample data is not supported for this connector"

_MAX_PAGE_SIZE = settings.METADATA_API_MAX_PAGE_SIZE
_DEFAULT_PAGE_SIZE = settings.METADATA_API_DEFAULT_PAGE_SIZE
_MAX_SEARCH_QUERY = settings.METADATA_SEARCH_MAX_QUERY_LENGTH
_MAX_SEARCH_LIMIT = settings.METADATA_SEARCH_MAX_LIMIT


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def _bad_gateway(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail)


def _workspace_id(access: DataSourceAccess) -> UUID:
    return access.data_source.workspace_id


def _schema_response(record: SchemaRecord) -> MetadataSchemaResponse:
    return MetadataSchemaResponse.model_validate(record)


def _table_response(record: TableRecord) -> MetadataTableResponse:
    return MetadataTableResponse.model_validate(record)


def _column_response(record: ColumnRecord) -> MetadataColumnResponse:
    return MetadataColumnResponse.model_validate(record)


def _relationship_response(
    record: RelationshipRecord,
) -> MetadataRelationshipResponse:
    return MetadataRelationshipResponse.model_validate(record)


def _schema_list(page: MetadataPage[SchemaRecord]) -> MetadataSchemaListResponse:
    return MetadataSchemaListResponse(
        items=[_schema_response(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
    )


def _table_list(page: MetadataPage[TableRecord]) -> MetadataTableListResponse:
    return MetadataTableListResponse(
        items=[_table_response(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
    )


def _column_list(page: MetadataPage[ColumnRecord]) -> MetadataColumnListResponse:
    return MetadataColumnListResponse(
        items=[_column_response(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
    )


def _relationship_list(
    page: MetadataPage[RelationshipRecord],
) -> MetadataRelationshipListResponse:
    return MetadataRelationshipListResponse(
        items=[_relationship_response(item) for item in page.items],
        page=page.page,
        page_size=page.page_size,
        total=page.total,
    )


def _sync_response(result: MetadataSyncResult) -> MetadataSyncResponse:
    return MetadataSyncResponse(
        status=result.status,
        started_at=result.started_at,
        completed_at=result.completed_at,
        schemas=result.schema_count,
        tables=result.table_count,
        columns=result.column_count,
        relationships=result.relationship_count,
        error_message=result.error_message,
    )


def _sample_response(result: SampleDataResult) -> SampleDataResponse:
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


def _raise_catalog_error(exc: Exception) -> NoReturn:
    if isinstance(exc, DataSourceNotFoundError):
        raise _not_found(DATA_SOURCE_NOT_FOUND_DETAIL) from None
    if isinstance(exc, MetadataNotFoundError):
        raise _not_found(str(exc)) from None
    if isinstance(exc, MetadataPageError):
        raise _bad_request(str(exc)) from None
    raise exc


_DEFAULT_SAMPLE_REQUEST = SampleDataRequest()

PageQuery = Annotated[
    int,
    Query(ge=1, description="1-based page number."),
]
PageSizeQuery = Annotated[
    int,
    Query(
        ge=1,
        le=_MAX_PAGE_SIZE,
        description=(
            f"Items per page. Default {_DEFAULT_PAGE_SIZE}, maximum {_MAX_PAGE_SIZE}."
        ),
    ),
]


@router.get(
    "/{data_source_id}/schemas",
    response_model=MetadataSchemaListResponse,
    summary="List metadata schemas",
    description=(
        "Return persisted schemas for a data source. Results are paginated. "
        "Does not return credentials or connection details."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def list_schemas(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    page: PageQuery = 1,
    page_size: PageSizeQuery = _DEFAULT_PAGE_SIZE,
) -> MetadataSchemaListResponse:
    try:
        result = MetadataCatalogService(db).list_schemas(
            access.data_source.id,
            workspace_id=_workspace_id(access),
            page=page,
            page_size=page_size,
        )
    except (DataSourceNotFoundError, MetadataNotFoundError, MetadataPageError) as exc:
        _raise_catalog_error(exc)
    return _schema_list(result)


@router.get(
    "/{data_source_id}/schemas/{schema_id}",
    response_model=MetadataSchemaResponse,
    summary="Get a metadata schema",
    description="Return one persisted schema, including a table count summary.",
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def get_schema(
    schema_id: UUID,
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MetadataSchemaResponse:
    try:
        record = MetadataCatalogService(db).get_schema(
            access.data_source.id,
            schema_id,
            workspace_id=_workspace_id(access),
        )
    except (DataSourceNotFoundError, MetadataNotFoundError) as exc:
        _raise_catalog_error(exc)
    return _schema_response(record)


@router.get(
    "/{data_source_id}/tables",
    response_model=MetadataTableListResponse,
    summary="List metadata tables",
    description=(
        "Return persisted tables for a data source. Filter by schema_id, "
        "table_type, or a case-insensitive name/description search. "
        "Search uses the same LIKE escaping as metadata search."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def list_tables(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    schema_id: UUID | None = None,
    search: Annotated[
        str | None,
        Query(
            max_length=_MAX_SEARCH_QUERY,
            description="Optional case-insensitive table name or description filter.",
        ),
    ] = None,
    table_type: DataSourceTableType | None = None,
    page: PageQuery = 1,
    page_size: PageSizeQuery = _DEFAULT_PAGE_SIZE,
) -> MetadataTableListResponse:
    try:
        result = MetadataCatalogService(db).list_tables(
            access.data_source.id,
            workspace_id=_workspace_id(access),
            schema_id=schema_id,
            search=search,
            table_type=table_type,
            page=page,
            page_size=page_size,
        )
    except (DataSourceNotFoundError, MetadataNotFoundError, MetadataPageError) as exc:
        _raise_catalog_error(exc)
    return _table_list(result)


@router.get(
    "/{data_source_id}/tables/{table_id}",
    response_model=MetadataTableResponse,
    summary="Get table metadata",
    description=(
        "Return persisted metadata for one table. Reads the metadata database "
        "and does not query the customer database."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def get_table(
    table_id: UUID,
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MetadataTableResponse:
    try:
        record = MetadataCatalogService(db).get_table(
            access.data_source.id,
            table_id,
            workspace_id=_workspace_id(access),
        )
    except (DataSourceNotFoundError, MetadataNotFoundError) as exc:
        _raise_catalog_error(exc)
    return _table_response(record)


@router.get(
    "/{data_source_id}/tables/{table_id}/columns",
    response_model=MetadataColumnListResponse,
    summary="List table columns",
    description=(
        "Return persisted columns for a table in ordinal_position order. "
        "Default expressions are metadata only and are never executed."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def list_columns(
    table_id: UUID,
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    page: PageQuery = 1,
    page_size: PageSizeQuery = _DEFAULT_PAGE_SIZE,
) -> MetadataColumnListResponse:
    try:
        result = MetadataCatalogService(db).list_columns(
            access.data_source.id,
            table_id,
            workspace_id=_workspace_id(access),
            page=page,
            page_size=page_size,
        )
    except (DataSourceNotFoundError, MetadataNotFoundError, MetadataPageError) as exc:
        _raise_catalog_error(exc)
    return _column_list(result)


@router.get(
    "/{data_source_id}/relationships",
    response_model=MetadataRelationshipListResponse,
    summary="List metadata relationships",
    description=(
        "Return persisted foreign-key relationships for a data source, "
        "including composite, self-referencing, and cross-schema keys."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def list_relationships(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    page: PageQuery = 1,
    page_size: PageSizeQuery = _DEFAULT_PAGE_SIZE,
) -> MetadataRelationshipListResponse:
    try:
        result = MetadataCatalogService(db).list_relationships(
            access.data_source.id,
            workspace_id=_workspace_id(access),
            page=page,
            page_size=page_size,
        )
    except (DataSourceNotFoundError, MetadataNotFoundError, MetadataPageError) as exc:
        _raise_catalog_error(exc)
    return _relationship_list(result)


@router.get(
    "/{data_source_id}/metadata/search",
    response_model=MetadataSearchResponse,
    summary="Search metadata",
    description=(
        "Search persisted schema, table, and column names for one data source. "
        "Uses parameterized queries. Query length and result limits are enforced."
    ),
    responses={
        **_AUTH_ERRORS,
        **_NOT_FOUND,
        **_RATE_LIMITED,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid search request"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
    dependencies=[Depends(require_rate_limit("metadata-search"))],
)
def search_metadata(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=_MAX_SEARCH_QUERY,
            description="Case-insensitive metadata search query.",
            examples=["customer"],
        ),
    ],
    metadata_type: MetadataSearchType | None = None,
    limit: Annotated[
        int | None,
        Query(
            ge=1,
            le=_MAX_SEARCH_LIMIT,
            description=(
                "Maximum results to return. Default "
                f"{settings.METADATA_SEARCH_DEFAULT_LIMIT}, maximum {_MAX_SEARCH_LIMIT}."
            ),
        ),
    ] = None,
) -> MetadataSearchResponse:
    query = q.strip()
    if not query:
        raise _bad_request("Search query is required")
    try:
        page = MetadataSearchService(db).search_metadata(
            access.data_source.id,
            query,
            workspace_id=_workspace_id(access),
            metadata_type=metadata_type,
            limit=limit,
        )
    except DataSourceNotFoundError:
        raise _not_found(DATA_SOURCE_NOT_FOUND_DETAIL) from None
    except MetadataSearchLimitError as exc:
        raise _bad_request(str(exc)) from None
    except MetadataSearchError:
        raise _bad_request(SEARCH_FAILED_DETAIL) from None
    return MetadataSearchResponse(
        items=[
            MetadataSearchItemResponse(
                metadata_type=item.metadata_type,
                schema_name=item.schema_name,
                table_name=item.table_name,
                column_name=item.column_name,
                description=item.description,
            )
            for item in page.results
        ],
        total=page.result_count,
        limit=page.limit,
        truncated=page.truncated,
    )


@router.post(
    "/{data_source_id}/metadata/sync",
    response_model=MetadataSyncResponse,
    summary="Synchronize metadata",
    description=(
        "Run metadata discovery against the customer database and persist the "
        "result. Synchronization is synchronous. Credentials are never returned."
    ),
    responses={
        **_AUTH_ERRORS,
        **_NOT_FOUND,
        **_RATE_LIMITED,
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported connector or invalid configuration"
        },
        status.HTTP_409_CONFLICT: {
            "description": "Metadata synchronization is already running"
        },
        status.HTTP_502_BAD_GATEWAY: {
            "description": "Customer database synchronization failed"
        },
    },
    dependencies=[Depends(require_rate_limit("metadata-sync"))],
)
async def synchronize_metadata(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_UPDATE,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MetadataSyncResponse:
    try:
        result = await MetadataSyncService(db).synchronize(
            access.data_source.id,
            workspace_id=_workspace_id(access),
        )
    except DataSourceNotFoundError:
        raise _not_found(DATA_SOURCE_NOT_FOUND_DETAIL) from None
    except ConcurrentMetadataSyncError as exc:
        raise _conflict(str(exc)) from None
    except UnsupportedConnectorError:
        raise _bad_request(UNSUPPORTED_CONNECTOR_DETAIL) from None
    except ConnectionConfigurationError:
        raise _bad_request(CONNECTION_CONFIG_DETAIL) from None
    except MetadataSyncPersistenceError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save metadata",
        ) from None
    except MetadataSyncError:
        raise _bad_gateway(SYNC_FAILED_DETAIL) from None
    return _sync_response(result)


@router.get(
    "/{data_source_id}/metadata/sync-status",
    response_model=MetadataSyncStatusResponse,
    summary="Get metadata synchronization status",
    description=(
        "Return the current metadata sync status and safe counts. "
        "Does not include stack traces, credentials, or driver errors."
    ),
    responses={**_AUTH_ERRORS, **_NOT_FOUND},
)
def get_sync_status(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MetadataSyncStatusResponse:
    try:
        result = MetadataSyncService(db).get_status(
            access.data_source.id,
            workspace_id=_workspace_id(access),
        )
    except DataSourceNotFoundError:
        raise _not_found(DATA_SOURCE_NOT_FOUND_DETAIL) from None
    return MetadataSyncStatusResponse.model_validate(_sync_response(result))


@router.post(
    "/{data_source_id}/tables/{table_id}/sample",
    response_model=SampleDataResponse,
    summary="Get safe sample rows",
    description=(
        "Return a small, masked sample of rows from a discovered table. "
        "PII and secrets are masked. Unmasked data cannot be requested."
    ),
    responses={
        **_AUTH_ERRORS,
        **_NOT_FOUND,
        **_RATE_LIMITED,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid sample request"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
        status.HTTP_502_BAD_GATEWAY: {"description": "Unable to retrieve sample data"},
    },
    dependencies=[Depends(require_rate_limit("sample-data"))],
)
async def sample_table(
    table_id: UUID,
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
    payload: Annotated[SampleDataRequest, Body()] = _DEFAULT_SAMPLE_REQUEST,
) -> SampleDataResponse:
    try:
        result = await SampleDataService(db).get_sample(
            access.data_source.id,
            table_id,
            workspace_id=_workspace_id(access),
            limit=payload.limit,
        )
    except DataSourceNotFoundError:
        raise _not_found(DATA_SOURCE_NOT_FOUND_DETAIL) from None
    except SampleTableNotFoundError:
        raise _not_found(TABLE_NOT_FOUND_DETAIL) from None
    except SampleDataLimitError as exc:
        raise _bad_request(str(exc)) from None
    except SampleIdentifierError:
        raise _bad_request("Invalid identifier") from None
    except UnsupportedConnectorError:
        raise _bad_request(UNSUPPORTED_CONNECTOR_DETAIL) from None
    except UnsupportedOperationError:
        raise _bad_request(UNSUPPORTED_SAMPLE_DETAIL) from None
    except ConnectionConfigurationError:
        raise _bad_request(CONNECTION_CONFIG_DETAIL) from None
    except (SampleQueryError, SampleDataError):
        raise _bad_gateway(SAMPLE_FAILED_DETAIL) from None
    return _sample_response(result)
