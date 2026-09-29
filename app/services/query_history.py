"""Query history service/repository for SQL lifecycle persistence.

Persistence contract: methods flush changes to the SQLAlchemy session but never
commit. Callers (API routes or higher-level workflows) must commit after success.
History recording stays separate from SQL execution / MCP logic.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.sql_correction.models import SQLCorrectionOutcome, SQLCorrectionStatus
from app.ai.sql_execution.models import SQLExecutionResult, SQLExecutionStatus
from app.ai.sql_validation.models import SQLValidationResult
from app.core.logging import redact_secret
from app.db.models import DataSource, QueryHistory, User, Workspace, WorkspaceMember
from app.enums import QueryHistoryStatus
from app.schemas.query_history import (
    QueryHistoryCreate,
    QueryHistoryFilter,
    QueryHistoryList,
    QueryHistoryRead,
    QueryHistorySummary,
)

logger = logging.getLogger(__name__)

_DEFAULT_PAGE_SIZE = 20
_MAX_PAGE_SIZE = 100
_MAX_SQL_LENGTH = 4000
_SQL_SINGLE_QUOTED = re.compile(r"'(?:''|[^'])*'")
_SQL_DOLLAR_QUOTED = re.compile(
    r"\$([A-Za-z_][A-Za-z0-9_]*)\$.*?\$\1\$",
    re.DOTALL,
)
_SQL_SIMPLE_DOLLAR = re.compile(r"\$\$.*?\$\$", re.DOTALL)


class QueryHistoryWriteError(Exception):
    """A query history entry could not be written."""

    def __init__(self, message: str = "Unable to save query history") -> None:
        super().__init__(message)


class QueryHistoryReadError(Exception):
    """A query history entry could not be read."""

    def __init__(self, message: str = "Query history entry not found") -> None:
        super().__init__(message)


class QueryHistoryPaginationError(Exception):
    """Invalid pagination parameters for query history listing."""

    def __init__(self, message: str = "Invalid pagination parameters") -> None:
        super().__init__(message)


class QueryHistoryAuthorizationError(Exception):
    """Caller is not allowed to access the requested query history scope."""

    def __init__(self, message: str = "Query history is not accessible") -> None:
        super().__init__(message)


def _truncate_sql(sql: str) -> str:
    text = sql.strip()
    if len(text) > _MAX_SQL_LENGTH:
        return text[:_MAX_SQL_LENGTH]
    return text


def _scrub_sql_for_history(sql: str) -> str:
    """Strip string literals and credential-shaped tokens before persistence."""
    text = _SQL_DOLLAR_QUOTED.sub("$$[REDACTED]$$", sql)
    text = _SQL_SIMPLE_DOLLAR.sub("$$[REDACTED]$$", text)
    text = _SQL_SINGLE_QUOTED.sub("'[REDACTED]'", text)
    return _truncate_sql(redact_secret(text))


def _sanitize_metadata(value: Any) -> Any:
    """Recursively redact secrets from metadata before persistence."""
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


def _execution_status_to_history(
    status: SQLExecutionStatus,
) -> QueryHistoryStatus:
    if status is SQLExecutionStatus.SUCCEEDED:
        return QueryHistoryStatus.SUCCEEDED
    if status is SQLExecutionStatus.REJECTED:
        return QueryHistoryStatus.REJECTED
    return QueryHistoryStatus.FAILED


class QueryHistoryService:
    """Flush-only repository for query history persistence and retrieval."""

    def __init__(
        self,
        session: Session,
        *,
        max_page_size: int = _MAX_PAGE_SIZE,
    ) -> None:
        self._session = session
        self._max_page_size = max_page_size

    def create(self, params: QueryHistoryCreate) -> QueryHistoryRead:
        """Create a query history entry and flush it to the session."""
        self._ensure_write_authorized(params)
        history = QueryHistory(
            user_id=params.user_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
            data_source_id=params.data_source_id,
            generated_sql=_scrub_sql_for_history(params.generated_sql),
            validated_sql=(
                _scrub_sql_for_history(params.validated_sql)
                if params.validated_sql is not None
                else None
            ),
            corrected_sql=(
                _scrub_sql_for_history(params.corrected_sql)
                if params.corrected_sql is not None
                else None
            ),
            status=params.status,
            duration_ms=float(params.duration_ms),
            result_metadata=_sanitize_metadata(params.result_metadata),
            error_metadata=_sanitize_metadata(params.error_metadata),
        )
        self._session.add(history)
        try:
            self._session.flush()
        except IntegrityError as exc:
            raise QueryHistoryWriteError("Unable to save query history") from exc
        self._session.refresh(history)
        logger.info(
            "Created query history entry",
            extra={
                "history_id": str(history.id),
                "user_id": str(params.user_id),
                "workspace_id": str(params.workspace_id),
                "organization_id": str(params.organization_id),
                "data_source_id": (
                    str(params.data_source_id) if params.data_source_id else None
                ),
                "status": params.status.value,
                "duration_ms": float(params.duration_ms),
            },
        )
        return QueryHistoryRead.model_validate(history)

    def record_succeeded(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        validated_sql: str | None = None,
        duration_ms: float = 0.0,
        result_metadata: dict[str, Any] | None = None,
    ) -> QueryHistoryRead:
        return self.create(
            QueryHistoryCreate(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                data_source_id=data_source_id,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                status=QueryHistoryStatus.SUCCEEDED,
                duration_ms=duration_ms,
                result_metadata=result_metadata,
            )
        )

    def record_failed(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        validated_sql: str | None = None,
        duration_ms: float = 0.0,
        error_metadata: dict[str, Any] | None = None,
    ) -> QueryHistoryRead:
        return self.create(
            QueryHistoryCreate(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                data_source_id=data_source_id,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                status=QueryHistoryStatus.FAILED,
                duration_ms=duration_ms,
                error_metadata=error_metadata,
            )
        )

    def record_rejected(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        validated_sql: str | None = None,
        duration_ms: float = 0.0,
        error_metadata: dict[str, Any] | None = None,
    ) -> QueryHistoryRead:
        return self.create(
            QueryHistoryCreate(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                data_source_id=data_source_id,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                status=QueryHistoryStatus.REJECTED,
                duration_ms=duration_ms,
                error_metadata=error_metadata,
            )
        )

    def record_corrected(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        corrected_sql: str,
        validated_sql: str | None = None,
        duration_ms: float = 0.0,
        result_metadata: dict[str, Any] | None = None,
        error_metadata: dict[str, Any] | None = None,
    ) -> QueryHistoryRead:
        return self.create(
            QueryHistoryCreate(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                data_source_id=data_source_id,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                corrected_sql=corrected_sql,
                status=QueryHistoryStatus.CORRECTED,
                duration_ms=duration_ms,
                result_metadata=result_metadata,
                error_metadata=error_metadata,
            )
        )

    def record_from_execution_result(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        validated_sql: str | None,
        result: SQLExecutionResult,
    ) -> QueryHistoryRead:
        """Map a Chapter 7.3 execution result into history without coupling MCP."""
        status = _execution_status_to_history(result.status)
        result_metadata = None
        error_metadata = None
        if status is QueryHistoryStatus.SUCCEEDED:
            result_metadata = {
                "row_count": result.row_count,
                "column_count": len(result.columns),
                "truncated": result.truncated,
                "applied_row_limit": result.applied_row_limit,
            }
        else:
            error_metadata = {
                "execution_status": result.status.value,
                "sql_char_count": result.sql_char_count,
            }
        return self.create(
            QueryHistoryCreate(
                user_id=user_id,
                workspace_id=workspace_id,
                organization_id=organization_id,
                data_source_id=data_source_id,
                generated_sql=generated_sql,
                validated_sql=validated_sql,
                status=status,
                duration_ms=float(result.duration_ms),
                result_metadata=result_metadata,
                error_metadata=error_metadata,
            )
        )

    def record_from_validation_result(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        validation: SQLValidationResult,
        duration_ms: float = 0.0,
    ) -> QueryHistoryRead | None:
        """Record rejected SQL when Chapter 7.2 validation fails."""
        if validation.is_valid:
            return None
        return self.record_rejected(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            data_source_id=data_source_id,
            generated_sql=generated_sql,
            duration_ms=duration_ms,
            error_metadata={
                "source": "VALIDATION",
                "violation_codes": [item.code.value for item in validation.violations],
                "violation_count": len(validation.violations),
            },
        )

    def record_from_correction_outcome(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID | None,
        generated_sql: str,
        outcome: SQLCorrectionOutcome,
        duration_ms: float = 0.0,
    ) -> QueryHistoryRead | None:
        """Integrate Chapter 7.4 correction outcomes into history when corrected."""
        if outcome.status is not SQLCorrectionStatus.CORRECTED:
            return None
        corrected = None
        if outcome.validated is not None:
            corrected = outcome.validated.sql
        elif outcome.generated is not None:
            corrected = outcome.generated.sql
        if corrected is None:
            return None
        validated_sql = outcome.validated.sql if outcome.validated is not None else None
        return self.record_corrected(
            user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            data_source_id=data_source_id,
            generated_sql=generated_sql,
            corrected_sql=corrected,
            validated_sql=validated_sql,
            duration_ms=duration_ms,
            result_metadata={
                "attempt_count": outcome.attempt_count,
                "max_attempts": outcome.max_attempts,
                "schema_truncated": outcome.schema_truncated,
            },
        )

    def get(
        self,
        history_id: UUID,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
    ) -> QueryHistoryRead:
        """Fetch one history entry with strict user/workspace/org isolation."""
        history = self._session.scalar(
            select(QueryHistory).where(
                QueryHistory.id == history_id,
                QueryHistory.user_id == user_id,
                QueryHistory.workspace_id == workspace_id,
                QueryHistory.organization_id == organization_id,
            )
        )
        if history is None:
            raise QueryHistoryReadError()
        return QueryHistoryRead.model_validate(history)

    def list_for_user(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        filter_params: QueryHistoryFilter | None = None,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> QueryHistoryList:
        """List the caller's history within one workspace/organization."""
        self._ensure_workspace_organization(workspace_id, organization_id)
        page_num, size = self._normalize_page(page, page_size)
        filters = [
            QueryHistory.user_id == user_id,
            QueryHistory.workspace_id == workspace_id,
            QueryHistory.organization_id == organization_id,
        ]
        filters.extend(self._filter_clauses(filter_params or QueryHistoryFilter()))
        return self._paginate(filters, page=page_num, page_size=size)

    def list_for_data_source(
        self,
        *,
        user_id: UUID,
        workspace_id: UUID,
        organization_id: UUID,
        data_source_id: UUID,
        filter_params: QueryHistoryFilter | None = None,
        page: int = 1,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> QueryHistoryList:
        """List the caller's history for one data source in the workspace."""
        self._ensure_workspace_organization(workspace_id, organization_id)
        self._ensure_workspace_data_source(data_source_id, workspace_id)
        page_num, size = self._normalize_page(page, page_size)
        scoped = filter_params or QueryHistoryFilter()
        filters = [
            QueryHistory.user_id == user_id,
            QueryHistory.workspace_id == workspace_id,
            QueryHistory.organization_id == organization_id,
            QueryHistory.data_source_id == data_source_id,
            *self._filter_clauses(
                QueryHistoryFilter(
                    status=scoped.status,
                    start_date=scoped.start_date,
                    end_date=scoped.end_date,
                    min_duration_ms=scoped.min_duration_ms,
                    max_duration_ms=scoped.max_duration_ms,
                )
            ),
        ]
        return self._paginate(filters, page=page_num, page_size=size)

    def _paginate(
        self,
        filters: list[Any],
        *,
        page: int,
        page_size: int,
    ) -> QueryHistoryList:
        total = self._session.scalar(
            select(func.count()).select_from(QueryHistory).where(*filters)
        )
        total_count = int(total or 0)
        if total_count == 0:
            return QueryHistoryList(items=[], total=0, page=page, page_size=page_size)

        start = (page - 1) * page_size
        if start >= total_count:
            raise QueryHistoryPaginationError("Page exceeds available results")

        rows = self._session.scalars(
            select(QueryHistory)
            .where(*filters)
            .order_by(QueryHistory.created_at.desc(), QueryHistory.id.desc())
            .offset(start)
            .limit(page_size)
        ).all()
        items = [QueryHistorySummary.model_validate(row) for row in rows]
        return QueryHistoryList(
            items=items,
            total=total_count,
            page=page,
            page_size=page_size,
        )

    def _filter_clauses(self, filter_params: QueryHistoryFilter) -> list[Any]:
        clauses: list[Any] = []
        if filter_params.status is not None:
            clauses.append(QueryHistory.status == filter_params.status)
        if filter_params.data_source_id is not None:
            clauses.append(QueryHistory.data_source_id == filter_params.data_source_id)
        if filter_params.start_date is not None:
            start = filter_params.start_date
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            clauses.append(QueryHistory.created_at >= start)
        if filter_params.end_date is not None:
            end = filter_params.end_date
            if end.tzinfo is None:
                end = end.replace(tzinfo=UTC)
            clauses.append(QueryHistory.created_at <= end)
        if filter_params.min_duration_ms is not None:
            clauses.append(QueryHistory.duration_ms >= filter_params.min_duration_ms)
        if filter_params.max_duration_ms is not None:
            clauses.append(QueryHistory.duration_ms <= filter_params.max_duration_ms)
        return clauses

    def _normalize_page(self, page: int, page_size: int) -> tuple[int, int]:
        if page < 1:
            raise QueryHistoryPaginationError("Page must be at least 1")
        if page_size < 1:
            raise QueryHistoryPaginationError("Page size must be at least 1")
        if page_size > self._max_page_size:
            raise QueryHistoryPaginationError(
                f"Page size must not exceed {self._max_page_size}"
            )
        return page, page_size

    def _ensure_workspace_organization(
        self,
        workspace_id: UUID,
        organization_id: UUID,
    ) -> Workspace:
        workspace = self._session.get(Workspace, workspace_id)
        if workspace is None or workspace.organization_id != organization_id:
            raise QueryHistoryAuthorizationError(
                "Workspace is not accessible for organization"
            )
        return workspace

    def _ensure_workspace_data_source(
        self,
        data_source_id: UUID,
        workspace_id: UUID,
    ) -> DataSource:
        data_source = self._session.get(DataSource, data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise QueryHistoryAuthorizationError("Data source is not accessible")
        return data_source

    def _ensure_write_authorized(self, params: QueryHistoryCreate) -> None:
        user = self._session.get(User, params.user_id)
        if user is None or not user.is_active:
            raise QueryHistoryAuthorizationError("User is not authorized")
        self._ensure_workspace_organization(
            params.workspace_id,
            params.organization_id,
        )
        if params.data_source_id is not None:
            self._ensure_workspace_data_source(
                params.data_source_id,
                params.workspace_id,
            )
        member = self._session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == params.workspace_id,
                WorkspaceMember.user_id == params.user_id,
            )
        )
        if member is None:
            raise QueryHistoryAuthorizationError(
                "User is not a member of this workspace"
            )
