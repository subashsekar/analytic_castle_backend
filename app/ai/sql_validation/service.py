"""Authorized SQL validation service.

Validates untrusted generated SQL against Phase 5 read-only rules and Phase 4
catalog metadata. Never executes SQL and never calls MCP query tools.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.sql_generation.catalog import ensure_metadata_catalog_authorized
from app.ai.sql_generation.errors import SQLGenerationAuthorizationError
from app.ai.sql_validation.errors import SQLValidationAuthorizationError
from app.ai.sql_validation.logging_helpers import validation_log_context
from app.ai.sql_validation.models import (
    SQLValidateParams,
    SQLValidationServiceResult,
)
from app.ai.sql_validation.validation import validate_generated_sql
from app.ai.state import AgentStateService, AnalysisSessionNotFoundError
from app.core.authorization import is_super_admin, workspace_role_has_permission
from app.db.models import DataSource, User, Workspace, WorkspaceMember
from app.enums import WorkspacePermission

logger = logging.getLogger(__name__)


class SQLValidationService:
    """Schema-aware SQL safety validation for an authorized workspace data source."""

    def __init__(
        self,
        session: Session,
        *,
        state_service: AgentStateService | None = None,
        allow_super_admin: bool = True,
        required_permission: WorkspacePermission = WorkspacePermission.DATA_SOURCE_READ,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._allow_super_admin = allow_super_admin
        self._required_permission = required_permission

    def validate(self, params: SQLValidateParams) -> SQLValidationServiceResult:
        self._ensure_request_authorized(params)
        try:
            ensure_metadata_catalog_authorized(
                self._session,
                data_source_id=params.data_source_id,
                metadata=params.metadata,
            )
        except SQLGenerationAuthorizationError as exc:
            raise SQLValidationAuthorizationError(str(exc)) from exc
        if params.session_id is not None:
            self._ensure_session_access(params)

        result = validate_generated_sql(params.sql, params.metadata)
        logger.info(
            "sql validation service completed",
            extra={
                **validation_log_context(result),
                "workspace_id": str(params.workspace_id),
                "organization_id": str(params.organization_id),
                "data_source_id": str(params.data_source_id),
            },
        )
        return SQLValidationServiceResult(
            result=result,
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )

    def _ensure_request_authorized(self, params: SQLValidateParams) -> DataSource:
        user = self._session.get(User, params.user_id)
        if user is None or not user.is_active:
            raise SQLValidationAuthorizationError("User is not authorized")

        workspace = self._session.get(Workspace, params.workspace_id)
        if workspace is None or workspace.organization_id != params.organization_id:
            raise SQLValidationAuthorizationError(
                "Workspace is not accessible for organization"
            )

        data_source = self._session.get(DataSource, params.data_source_id)
        if data_source is None or data_source.workspace_id != params.workspace_id:
            raise SQLValidationAuthorizationError("Data source is not accessible")

        if params.metadata.data_source_id != params.data_source_id:
            raise SQLValidationAuthorizationError(
                "Metadata context does not match the authorized data source"
            )

        member = self._session.scalar(
            select(WorkspaceMember).where(
                WorkspaceMember.workspace_id == params.workspace_id,
                WorkspaceMember.user_id == params.user_id,
            )
        )
        if self._allow_super_admin and is_super_admin(user.role):
            return data_source
        if member is None:
            raise SQLValidationAuthorizationError(
                "User is not a member of the workspace"
            )
        if not workspace_role_has_permission(member.role, self._required_permission):
            raise SQLValidationAuthorizationError(
                "User lacks permission for SQL validation"
            )
        return data_source

    def _ensure_session_access(self, params: SQLValidateParams) -> None:
        assert params.session_id is not None
        try:
            snapshot = self._state_service.get_session(
                params.session_id,
                workspace_id=params.workspace_id,
                user_id=params.user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise SQLValidationAuthorizationError(
                "Analysis session is not accessible"
            ) from exc
        if snapshot.organization_id != params.organization_id:
            raise SQLValidationAuthorizationError(
                "Analysis session organization does not match"
            )
        if snapshot.data_source_id is None:
            raise SQLValidationAuthorizationError(
                "Analysis session has no authorized data source"
            )
        if snapshot.data_source_id != params.data_source_id:
            raise SQLValidationAuthorizationError(
                "Data source does not match the analysis session"
            )
