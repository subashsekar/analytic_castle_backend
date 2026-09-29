from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.ai.glossary import parse_glossary
from app.api.deps import (
    get_current_user,
    load_workspace_access,
    require_data_source_permission,
    require_rate_limit,
)
from app.connectors import UnsupportedConnectorError
from app.core.authorization import DataSourceAccess, is_super_admin
from app.db.models import DataSource, User
from app.db.session import get_db
from app.enums import WorkspacePermission
from app.schemas.auth import MessageResponse
from app.schemas.data_source import (
    ConnectionTestResponse,
    DataSourceCreate,
    DataSourceGlossary,
    DataSourceRead,
    DataSourceUpdate,
)
from app.services.credentials import CredentialError
from app.services.data_source_connections import (
    ConnectionConfigurationError,
    DataSourceConnectionService,
    DataSourceNotFoundError,
)
from app.services.data_sources import DataSourceService, DataSourceWriteError

router = APIRouter(prefix="/api/v1/data-sources", tags=["data-sources"])

UNSUPPORTED_CONNECTOR_DETAIL = "Unsupported connector type"
CREDENTIAL_STORE_DETAIL = "Unable to store data source credentials"
DATA_SOURCE_WRITE_DETAIL = "Unable to save data source"
CONNECTION_CONFIG_DETAIL = "Data source connection configuration is invalid"
CONNECTION_SUCCESS_MESSAGE = "Connection successful."
CONNECTION_FAILURE_MESSAGE = "Unable to connect to the data source."

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


@router.post(
    "",
    response_model=DataSourceRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a data source",
    description=(
        "Create a data source and encrypted connection configuration. "
        "Does not test the external database."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Unsupported connector type"},
        status.HTTP_404_NOT_FOUND: {"description": "Workspace not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to save data source"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
def create_data_source(
    payload: DataSourceCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> DataSource:
    load_workspace_access(
        db,
        current_user,
        payload.workspace_id,
        permission=WorkspacePermission.DATA_SOURCE_CREATE,
        allow_super_admin=True,
    )
    try:
        return DataSourceService(db).create(
            created_by=current_user.id,
            workspace_id=payload.workspace_id,
            name=payload.name,
            source_type=payload.type,
            host=payload.connection.host,
            port=payload.connection.port,
            database_name=payload.connection.database_name,
            username=payload.connection.username,
            password=payload.connection.password,
            ssl_mode=payload.connection.ssl_mode,
        )
    except UnsupportedConnectorError:
        raise _bad_request(UNSUPPORTED_CONNECTOR_DETAIL) from None
    except CredentialError:
        raise _bad_request(CREDENTIAL_STORE_DETAIL) from None
    except DataSourceWriteError:
        raise _conflict(DATA_SOURCE_WRITE_DETAIL) from None


@router.get(
    "",
    response_model=list[DataSourceRead],
    summary="List data sources",
    description=(
        "Return data sources in workspaces the current user can access. "
        "An optional workspace_id filter is authorized before results are returned."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Workspace not found"},
    },
)
def list_data_sources(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    workspace_id: UUID | None = None,
) -> list[DataSource]:
    if workspace_id is not None:
        load_workspace_access(
            db,
            current_user,
            workspace_id,
            permission=WorkspacePermission.DATA_SOURCE_READ,
            allow_super_admin=True,
        )
    member_user_id = None if is_super_admin(current_user.role) else current_user.id
    return DataSourceService(db).list_accessible(
        workspace_id=workspace_id,
        member_user_id=member_user_id,
    )


@router.get(
    "/{data_source_id}",
    response_model=DataSourceRead,
    summary="Get a data source",
    description="Return non-secret metadata for a data source the user can access.",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
    },
)
def get_data_source(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> DataSource:
    return access.data_source


@router.patch(
    "/{data_source_id}",
    response_model=DataSourceRead,
    summary="Update a data source",
    description="Update allowed data source fields. Credentials and status cannot be set.",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to save data source"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
def update_data_source(
    payload: DataSourceUpdate,
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
) -> DataSource:
    try:
        return DataSourceService(db).update(access.data_source, name=payload.name)
    except DataSourceWriteError:
        raise _conflict(DATA_SOURCE_WRITE_DETAIL) from None


@router.get(
    "/{data_source_id}/glossary",
    response_model=DataSourceGlossary,
    summary="Get the data source business glossary",
    description="Metric synonyms and definitions the AI analyst uses to map questions to columns.",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
    },
)
def get_data_source_glossary(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> DataSourceGlossary:
    return DataSourceGlossary(entries=parse_glossary(access.data_source.business_glossary))


@router.put(
    "/{data_source_id}/glossary",
    response_model=DataSourceGlossary,
    summary="Replace the data source business glossary",
    description=(
        "Replace glossary entries. Entries describe business meaning only; "
        "they never contain SQL and only take effect for existing catalog columns."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to save data source"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
)
def replace_data_source_glossary(
    payload: DataSourceGlossary,
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
) -> DataSourceGlossary:
    source = access.data_source
    source.business_glossary = [
        entry.model_dump(mode="json", exclude_none=True) for entry in payload.entries
    ]
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise _conflict(DATA_SOURCE_WRITE_DETAIL) from None
    return DataSourceGlossary(entries=parse_glossary(source.business_glossary))


@router.delete(
    "/{data_source_id}",
    response_model=MessageResponse,
    summary="Delete a data source",
    description="Delete a data source and its encrypted connection configuration.",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to delete data source"},
    },
)
def delete_data_source(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_DELETE,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    try:
        DataSourceService(db).delete(access.data_source)
    except DataSourceWriteError:
        raise _conflict("Unable to delete data source") from None
    return MessageResponse(detail="Data source deleted")


@router.post(
    "/{data_source_id}/test-connection",
    response_model=ConnectionTestResponse,
    summary="Test a data source connection",
    description=(
        "Test the stored data source configuration against the customer database. "
        "The client cannot supply connection settings on this endpoint."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {
            "description": "Unsupported connector or invalid configuration"
        },
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"},
    },
    dependencies=[Depends(require_rate_limit("test-connection"))],
)
async def test_data_source_connection(
    access: Annotated[
        DataSourceAccess,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_TEST,
                allow_super_admin=True,
            )
        ),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> ConnectionTestResponse:
    try:
        result = await DataSourceConnectionService(db).test_connection(
            access.data_source.id
        )
    except DataSourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        ) from None
    except UnsupportedConnectorError:
        raise _bad_request(UNSUPPORTED_CONNECTOR_DETAIL) from None
    except ConnectionConfigurationError:
        raise _bad_request(CONNECTION_CONFIG_DETAIL) from None

    if result.success:
        return ConnectionTestResponse(
            success=True,
            message=CONNECTION_SUCCESS_MESSAGE,
        )
    return ConnectionTestResponse(
        success=False,
        message=CONNECTION_FAILURE_MESSAGE,
    )
