"""Report persistence + tenant-safe CRUD for Phase 9.4."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.logging import redact_secret
from app.db.models import DataSource, Report, User, Workspace, WorkspaceMember
from app.schemas.reports import ReportCreate, ReportFilter, ReportList, ReportRead

logger = logging.getLogger(__name__)

_MAX_PAGE_SIZE = 100


class ReportWriteError(Exception):
    def __init__(self, message: str = "Unable to save report") -> None:
        super().__init__(message)


class ReportReadError(Exception):
    def __init__(self, message: str = "Report not found") -> None:
        super().__init__(message)


class ReportAuthorizationError(Exception):
    def __init__(self, message: str = "Not authorized to access report") -> None:
        super().__init__(message)


class ReportPaginationError(Exception):
    def __init__(self, message: str = "Invalid pagination parameters") -> None:
        super().__init__(message)


def _sanitize_metadata(value: Any) -> Any:
    """Recursively redact secrets from user-supplied report payload."""
    if value is None:
        return None
    if isinstance(value, str):
        return redact_secret(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, (bool, int, float)):
        return value
    return redact_secret(str(value))


def _sanitize_str(value: str) -> str:
    return redact_secret((value or "").strip())


class ReportService:
    """Flush-and-commit repository for report persistence.

    Callers must pass correct user/workspace/org ids for tenant isolation.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _ensure_workspace_access(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
    ) -> Workspace:
        workspace = self._session.get(Workspace, workspace_id)
        if workspace is None or workspace.organization_id != organization_id:
            raise ReportAuthorizationError("Workspace is not accessible for organization")
        member = self._session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == user_id,
            )
        )
        if member is None:
            raise ReportAuthorizationError("User is not a member of this workspace")
        return workspace

    def _ensure_user_active(self, user_id: UUID) -> None:
        user = self._session.get(User, user_id)
        if user is None or not user.is_active:
            raise ReportAuthorizationError("User is not authorized")

    def _ensure_data_source_reference(
        self,
        *,
        workspace_id: UUID,
        data_source_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise ReportAuthorizationError("Data source not found")
        return data_source

    def _normalize_page(self, page: int, page_size: int) -> tuple[int, int]:
        if page < 1:
            raise ReportPaginationError("Page must be at least 1")
        if page_size < 1:
            raise ReportPaginationError("Page size must be at least 1")
        if page_size > _MAX_PAGE_SIZE:
            raise ReportPaginationError(f"Page size must not exceed {_MAX_PAGE_SIZE}")
        return page, page_size

    def create(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID,
        payload: ReportCreate,
    ) -> ReportRead:
        self._ensure_user_active(user_id)
        self._ensure_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
        self._ensure_data_source_reference(
            workspace_id=workspace_id,
            data_source_id=data_source_id,
        )

        has_chart_specs = bool(payload.chart_specifications)
        has_key_findings = bool(payload.key_findings)
        has_supporting_data = bool(payload.supporting_data)
        if not (has_chart_specs or has_key_findings or has_supporting_data):
            raise ReportWriteError("Report content is empty")

        question = _sanitize_str(payload.question)
        summary = _sanitize_str(payload.summary)
        if not question or not summary:
            # Treat whitespace-only input as "empty report" to keep the API
            # response consistent and client-actionable (400, not DB-driven 409).
            raise ReportWriteError("Report content is empty")

        report = Report(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            data_source_id=data_source_id,
            question=question,
            summary=summary,
            key_findings=[_sanitize_str(item) for item in payload.key_findings],
            chart_specifications=[
                item.model_dump(mode="json", exclude_none=True) for item in payload.chart_specifications
            ],
            supporting_data=_sanitize_metadata(payload.supporting_data),
        )
        self._session.add(report)
        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            raise ReportWriteError("Unable to save report") from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise ReportWriteError("Unable to save report") from exc

        self._session.refresh(report)
        logger.info(
            "Created report report_id=%s user_id=%s workspace_id=%s",
            report.id,
            user_id,
            workspace_id,
        )
        return ReportRead.model_validate(report)

    def get(
        self,
        report_id: UUID,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
    ) -> ReportRead:
        report = self._session.scalar(
            select(Report).where(
                Report.id == report_id,
                Report.user_id == user_id,
                Report.workspace_id == workspace_id,
                Report.organization_id == organization_id,
            )
        )
        if report is None:
            raise ReportReadError()
        return ReportRead.model_validate(report)

    def delete(
        self,
        report_id: UUID,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
    ) -> None:
        report = self._session.scalar(
            select(Report).where(
                Report.id == report_id,
                Report.user_id == user_id,
                Report.workspace_id == workspace_id,
                Report.organization_id == organization_id,
            )
        )
        if report is None:
            raise ReportReadError()
        self._session.delete(report)
        try:
            self._session.commit()
        except IntegrityError as exc:
            self._session.rollback()
            raise ReportWriteError("Unable to delete report") from exc
        except SQLAlchemyError as exc:
            self._session.rollback()
            raise ReportWriteError("Unable to delete report") from exc
        logger.info(
            "Deleted report report_id=%s user_id=%s workspace_id=%s",
            report_id,
            user_id,
            workspace_id,
        )

    def list_for_user(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        filter_params: ReportFilter | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> ReportList:
        self._ensure_workspace_access(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
        )
        page_num, size = self._normalize_page(page, page_size)
        data_source_id = filter_params.data_source_id if filter_params else None

        filters: list[Any] = [
            Report.user_id == user_id,
            Report.workspace_id == workspace_id,
            Report.organization_id == organization_id,
        ]
        if data_source_id is not None:
            self._ensure_data_source_reference(
                workspace_id=workspace_id,
                data_source_id=data_source_id,
            )
            filters.append(Report.data_source_id == data_source_id)

        total = self._session.scalar(
            select(func.count()).select_from(Report).where(*filters)
        )
        total_count = int(total or 0)
        if total_count == 0:
            return ReportList(items=[], total=0, page=page_num, page_size=size)

        start = (page_num - 1) * size
        if start >= total_count:
            raise ReportPaginationError("Page exceeds available results")

        rows = self._session.scalars(
            select(Report)
            .where(*filters)
            .order_by(Report.created_at.desc(), Report.id.desc())
            .offset(start)
            .limit(size)
        ).all()

        # ReportSummary has the subset of fields needed for listings.
        from app.schemas.reports import ReportSummary

        items = [ReportSummary.model_validate(row) for row in rows]
        return ReportList(items=items, total=total_count, page=page_num, page_size=size)

