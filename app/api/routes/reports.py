"""User report API routes (Phase 9.4)."""

from __future__ import annotations

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    load_workspace_access,
    require_data_source_permission,
    require_rate_limit,
)
from app.core.authorization import WorkspaceAccess, WorkspacePermission
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import MessageResponse
from app.schemas.reports import (
    ReportCreate,
    ReportList,
    ReportRead,
)
from app.services.reports import (
    ReportAuthorizationError,
    ReportPaginationError,
    ReportReadError,
    ReportService,
    ReportWriteError,
)
from app.schemas.reports import ReportFilter
from app.services.report_exports import (
    build_report_pdf_bytes,
    export_report_charts_csv,
    export_report_results_csv,
)

router = APIRouter(prefix="/api/v1/workspaces/{workspace_id}", tags=["reports"])

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}


def _forbidden() -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized")


def _not_found(detail: str = "Report not found") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _workspace_not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Workspace not found",
    )


def _organization_id(access: WorkspaceAccess) -> UUID:
    return access.workspace.organization_id


def _require_workspace_access(
    db: Session,
    user: User,
    workspace_id: UUID,
) -> WorkspaceAccess:
    return load_workspace_access(db, user, workspace_id)


@router.post(
    "/data-sources/{data_source_id}/reports",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
    summary="Generate and persist a report",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Empty report or invalid input"},
        status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
        status.HTTP_404_NOT_FOUND: {"description": "Workspace or data source not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to save report"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def create_report(
    workspace_id: UUID,
    data_source_id: UUID,
    payload: ReportCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    _access: Annotated[
        Any,
        Depends(
            require_data_source_permission(
                WorkspacePermission.DATA_SOURCE_READ,
                allow_super_admin=True,
            )
        ),
    ],
) -> ReportRead:
    # Avoid leaking existence of other workspaces.
    if _access.workspace.id != workspace_id:
        raise _workspace_not_found()

    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)

    try:
        return service.create(
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            data_source_id=data_source_id,
            payload=payload,
        )
    except ReportWriteError as exc:
        detail = str(exc)
        if detail.lower().strip() == "report content is empty":
            raise _bad_request("Report content is empty") from None
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Unable to save report",
        ) from None
    except ReportAuthorizationError as exc:
        if "not found" in str(exc).lower():
            # Invalid references should not show up as 403.
            raise _not_found(str(exc)) from None
        raise _forbidden() from None
    except Exception:
        raise _bad_request("Invalid report") from None


@router.get(
    "/reports",
    response_model=ReportList,
    summary="List persisted reports for the current user",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid pagination parameters"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def list_reports(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    data_source_id: UUID | None = Query(default=None),
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ReportList:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)

    filter_params = ReportFilter(data_source_id=data_source_id)
    try:
        return service.list_for_user(
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            filter_params=filter_params,
            page=page,
            page_size=page_size,
        )
    except ReportPaginationError as exc:
        raise _bad_request(str(exc)) from None
    except ReportAuthorizationError as exc:
        if "not found" in str(exc).lower():
            raise _not_found("Data source not found") from None
        raise _forbidden() from None


@router.get(
    "/reports/{report_id}",
    response_model=ReportRead,
    summary="Get a persisted report",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Report not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def get_report(
    workspace_id: UUID,
    report_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ReportRead:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)
    try:
        return service.get(
            report_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except ReportReadError:
        raise _not_found() from None
    except ReportAuthorizationError:
        raise _forbidden() from None


@router.delete(
    "/reports/{report_id}",
    response_model=MessageResponse,
    summary="Delete a persisted report",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Report not found"},
        status.HTTP_409_CONFLICT: {"description": "Unable to delete report"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def delete_report(
    workspace_id: UUID,
    report_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)
    try:
        service.delete(
            report_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except ReportReadError:
        raise _not_found() from None
    except ReportWriteError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Unable to delete report",
        ) from None
    return MessageResponse(detail="Report deleted")


@router.get(
    "/reports/{report_id}/export/csv",
    response_class=Response,
    summary="Export a report as CSV",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Report not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def export_report_csv(
    workspace_id: UUID,
    report_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)
    try:
        report = service.get(
            report_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except ReportReadError:
        raise _not_found() from None

    csv_text = export_report_results_csv(report)
    filename = f"report_{report.id}_results.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
    }
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@router.get(
    "/reports/{report_id}/export/charts/csv",
    response_class=Response,
    summary="Export report chart data as CSV",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Report not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def export_report_charts_csv(
    workspace_id: UUID,
    report_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)
    try:
        report = service.get(
            report_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except ReportReadError:
        raise _not_found() from None

    csv_text = export_report_charts_csv(report)
    filename = f"report_{report.id}_charts.csv"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
    }
    return Response(
        content=csv_text.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@router.get(
    "/reports/{report_id}/export/pdf",
    response_class=Response,
    summary="Export a generated report as PDF",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Report not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def export_report_pdf(
    workspace_id: UUID,
    report_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    access = _require_workspace_access(db, current_user, workspace_id)
    organization_id = _organization_id(access)
    service = ReportService(db)
    try:
        report = service.get(
            report_id,
            user_id=current_user.id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
    except ReportReadError:
        raise _not_found() from None

    pdf_bytes = build_report_pdf_bytes(report)
    filename = f"report_{report.id}.pdf"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
    }
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers=headers,
    )

