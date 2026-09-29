"""Query history API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, load_workspace_access, require_rate_limit
from app.core.authorization import WorkspaceAccess
from app.db.models import DataSource, User
from app.db.session import get_db
from app.enums import QueryHistoryStatus
from app.schemas.query_history import (
    QueryHistoryFilter,
    QueryHistoryList,
    QueryHistoryRead,
)
from app.services.query_history import (
    QueryHistoryAuthorizationError,
    QueryHistoryPaginationError,
    QueryHistoryReadError,
    QueryHistoryService,
)

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/query-history",
    tags=["query-history"],
)

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}


def _forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized",
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Query history entry not found",
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _require_workspace_access(
    db: Session,
    user: User,
    workspace_id: UUID,
) -> WorkspaceAccess:
    return load_workspace_access(db, user, workspace_id)


def _organization_id(access: WorkspaceAccess) -> UUID:
    return access.workspace.organization_id


@router.get(
    "",
    response_model=QueryHistoryList,
    summary="List query history for the current user in a workspace",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pagination or filters"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def list_query_history(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    status_filter: Annotated[
        QueryHistoryStatus | None,
        Query(alias="status"),
    ] = None,
    data_source_id: UUID | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_duration_ms: float | None = None,
    max_duration_ms: float | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> QueryHistoryList:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    if data_source_id is not None:
        data_source = db.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise _not_found()

    service = QueryHistoryService(db)
    try:
        return service.list_for_user(
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            filter_params=QueryHistoryFilter(
                status=status_filter,
                data_source_id=data_source_id,
                start_date=start_date,
                end_date=end_date,
                min_duration_ms=min_duration_ms,
                max_duration_ms=max_duration_ms,
            ),
            page=page,
            page_size=page_size,
        )
    except QueryHistoryPaginationError as exc:
        raise _bad_request(str(exc)) from exc
    except QueryHistoryAuthorizationError as exc:
        raise _forbidden() from exc


@router.get(
    "/data-sources/{data_source_id}",
    response_model=QueryHistoryList,
    summary="List query history for a workspace data source",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pagination or filters"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def list_query_history_for_data_source(
    workspace_id: UUID,
    data_source_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    status_filter: Annotated[
        QueryHistoryStatus | None,
        Query(alias="status"),
    ] = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    min_duration_ms: float | None = None,
    max_duration_ms: float | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> QueryHistoryList:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    data_source = db.get(DataSource, data_source_id)
    if data_source is None or data_source.workspace_id != workspace_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data source not found",
        )

    service = QueryHistoryService(db)
    try:
        return service.list_for_data_source(
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            data_source_id=data_source_id,
            filter_params=QueryHistoryFilter(
                status=status_filter,
                start_date=start_date,
                end_date=end_date,
                min_duration_ms=min_duration_ms,
                max_duration_ms=max_duration_ms,
            ),
            page=page,
            page_size=page_size,
        )
    except QueryHistoryPaginationError as exc:
        raise _bad_request(str(exc)) from exc
    except QueryHistoryAuthorizationError as exc:
        raise _forbidden() from exc


@router.get(
    "/{history_id}",
    response_model=QueryHistoryRead,
    summary="Get a query history entry",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Query history entry not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def get_query_history(
    workspace_id: UUID,
    history_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> QueryHistoryRead:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = QueryHistoryService(db)
    try:
        return service.get(
            history_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except QueryHistoryReadError as exc:
        raise _not_found() from exc
