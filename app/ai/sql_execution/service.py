"""Authorized SQL execution service.

Validates untrusted SQL with Chapter 7.2, then executes only through the
existing PostgreSQL MCP `postgres.query` tool. Does not correct SQL (7.4) or
persist query history (7.5).
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.sql_execution.errors import (
    SQLExecutionAuthorizationError,
    SQLExecutionCancelledError,
    SQLExecutionValidationError,
)
from app.ai.sql_execution.execution import (
    ensure_requested_row_limit,
    execute_validated_sql,
)
from app.ai.sql_execution.logging_helpers import execution_log_context
from app.ai.sql_execution.models import (
    SQLExecuteParams,
    SQLExecutionServiceResult,
)
from app.ai.sql_generation.catalog import ensure_metadata_catalog_authorized
from app.ai.sql_generation.errors import SQLGenerationAuthorizationError
from app.ai.sql_validation.validation import validate_generated_sql
from app.ai.state import AgentStateService, AnalysisSessionNotFoundError
from app.core.authorization import is_super_admin, workspace_role_has_permission
from app.db.models import DataSource, User, Workspace, WorkspaceMember
from app.enums import WorkspacePermission
from app.mcp import MCPClient, MCPRegistry, build_postgres_mcp

logger = logging.getLogger(__name__)


class SQLExecutionService:
    """Execute validated analytical SQL for an authorized workspace data source."""

    def __init__(
        self,
        session: Session,
        *,
        mcp_client: MCPClient | None = None,
        mcp_registry: MCPRegistry | None = None,
        state_service: AgentStateService | None = None,
        allow_super_admin: bool = True,
        required_permission: WorkspacePermission = WorkspacePermission.DATA_SOURCE_QUERY,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._allow_super_admin = allow_super_admin
        self._required_permission = required_permission
        if mcp_client is not None and mcp_registry is not None:
            self._mcp_client = mcp_client
            self._mcp_registry = mcp_registry
        elif mcp_client is None and mcp_registry is None:
            self._mcp_registry, self._mcp_client = build_postgres_mcp(session)
        else:
            raise ValueError("mcp_client and mcp_registry must be provided together")

    async def execute(self, params: SQLExecuteParams) -> SQLExecutionServiceResult:
        self._ensure_request_authorized(params)
        try:
            ensure_metadata_catalog_authorized(
                self._session,
                data_source_id=params.data_source_id,
                metadata=params.metadata,
            )
        except SQLGenerationAuthorizationError as exc:
            raise SQLExecutionAuthorizationError(str(exc)) from exc
        if params.session_id is not None:
            self._ensure_session_access(params)

        ensure_requested_row_limit(params.limit)
        validation = validate_generated_sql(params.sql, params.metadata)
        if not validation.is_valid or validation.validated is None:
            raise SQLExecutionValidationError(
                "SQL failed validation and was not executed",
                violations=validation.violations,
            )

        validated = validation.validated
        try:
            result = await execute_validated_sql(
                client=self._mcp_client,
                data_source_id=params.data_source_id,
                workspace_id=params.workspace_id,
                user_id=params.user_id,
                validated=validated,
                metadata=params.metadata,
                limit=params.limit,
            )
        except asyncio.CancelledError as exc:
            raise SQLExecutionCancelledError("SQL execution was cancelled") from exc
        except SQLExecutionCancelledError as exc:
            logger.info(
                "sql execution cancelled",
                extra={
                    "workspace_id": str(params.workspace_id),
                    "organization_id": str(params.organization_id),
                    "data_source_id": str(params.data_source_id),
                    "sql_char_count": len(validated.sql),
                    "duration_ms": round(exc.duration_ms, 3),
                },
            )
            raise

        logger.info(
            "sql execution service completed",
            extra={
                **execution_log_context(result),
                "workspace_id": str(params.workspace_id),
                "organization_id": str(params.organization_id),
                "data_source_id": str(params.data_source_id),
            },
        )
        return SQLExecutionServiceResult(
            result=result,
            validated=validated,
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )

    def _ensure_request_authorized(self, params: SQLExecuteParams) -> DataSource:
        user = self._session.get(User, params.user_id)
        if user is None or not user.is_active:
            raise SQLExecutionAuthorizationError("User is not authorized")

        workspace = self._session.get(Workspace, params.workspace_id)
        if workspace is None or workspace.organization_id != params.organization_id:
            raise SQLExecutionAuthorizationError(
                "Workspace is not accessible for organization"
            )

        data_source = self._session.get(DataSource, params.data_source_id)
        if data_source is None or data_source.workspace_id != params.workspace_id:
            raise SQLExecutionAuthorizationError("Data source is not accessible")

        if params.metadata.data_source_id != params.data_source_id:
            raise SQLExecutionAuthorizationError(
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
            raise SQLExecutionAuthorizationError(
                "User is not a member of the workspace"
            )
        if not workspace_role_has_permission(member.role, self._required_permission):
            raise SQLExecutionAuthorizationError(
                "User lacks permission for SQL execution"
            )
        return data_source

    def _ensure_session_access(self, params: SQLExecuteParams) -> None:
        assert params.session_id is not None
        try:
            snapshot = self._state_service.get_session(
                params.session_id,
                workspace_id=params.workspace_id,
                user_id=params.user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise SQLExecutionAuthorizationError(
                "Analysis session is not accessible"
            ) from exc
        if snapshot.organization_id != params.organization_id:
            raise SQLExecutionAuthorizationError(
                "Analysis session organization does not match"
            )
        if snapshot.data_source_id is None:
            raise SQLExecutionAuthorizationError(
                "Analysis session has no authorized data source"
            )
        if snapshot.data_source_id != params.data_source_id:
            raise SQLExecutionAuthorizationError(
                "Data source does not match the analysis session"
            )
