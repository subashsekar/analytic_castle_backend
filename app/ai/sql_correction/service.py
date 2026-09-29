"""Authorized SQL correction service.

Corrects failed generated or executed SQL for an authorized workspace data
source. Never executes a draft until Chapter 7.2 validation succeeds, and
never calls MCP itself except through the optional 7.3 execution helper.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings
from app.ai.prompt import PromptRegistry
from app.ai.sql_correction.errors import (
    SQLCorrectionAuthorizationError,
    SQLCorrectionConfigurationError,
)
from app.ai.sql_correction.logging_helpers import correction_log_context
from app.ai.sql_correction.models import (
    SQLCorrectionServiceResult,
    SQLCorrectParams,
)
from app.ai.sql_correction.prompts import build_sql_correction_prompt_registry
from app.ai.sql_correction.workflow import (
    correct_failed_sql,
    execute_corrected_sql,
)
from app.ai.sql_execution.models import SQLExecutionResult
from app.ai.sql_generation.catalog import ensure_metadata_catalog_authorized
from app.ai.sql_generation.errors import SQLGenerationAuthorizationError
from app.ai.state import AgentStateService, AnalysisSessionNotFoundError
from app.core.authorization import is_super_admin, workspace_role_has_permission
from app.db.models import DataSource, User, Workspace, WorkspaceMember
from app.enums import WorkspacePermission
from app.mcp import MCPClient

logger = logging.getLogger(__name__)


class SQLCorrectionService:
    """Schema-aware SQL correction for an authorized workspace data source."""

    def __init__(
        self,
        session: Session,
        *,
        llm_client: AsyncLLMClient | None = None,
        llm_config: LLMClientConfig | None = None,
        prompt_registry: PromptRegistry | None = None,
        state_service: AgentStateService | None = None,
        allow_super_admin: bool = True,
        required_permission: WorkspacePermission = WorkspacePermission.DATA_SOURCE_READ,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._prompt_registry = (
            prompt_registry or build_sql_correction_prompt_registry()
        )
        self._allow_super_admin = allow_super_admin
        self._required_permission = required_permission
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise SQLCorrectionConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def correct(self, params: SQLCorrectParams) -> SQLCorrectionServiceResult:
        self._ensure_request_authorized(params)
        try:
            ensure_metadata_catalog_authorized(
                self._session,
                data_source_id=params.data_source_id,
                metadata=params.metadata,
            )
        except SQLGenerationAuthorizationError as exc:
            raise SQLCorrectionAuthorizationError(str(exc)) from exc
        if params.session_id is not None:
            self._ensure_session_access(params)

        outcome = await correct_failed_sql(
            client=self._llm_client,
            registry=self._prompt_registry,
            params=params,
        )
        logger.info(
            "sql correction service completed",
            extra={
                **correction_log_context(outcome),
                "workspace_id": str(params.workspace_id),
                "organization_id": str(params.organization_id),
                "data_source_id": str(params.data_source_id),
            },
        )
        return SQLCorrectionServiceResult(
            outcome=outcome,
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )

    async def correct_then_execute(
        self,
        params: SQLCorrectParams,
        *,
        mcp_client: MCPClient,
        limit: int | None = None,
    ) -> tuple[SQLCorrectionServiceResult, SQLExecutionResult]:
        """Correct, re-validate, then execute only through Chapter 7.3.

        Execution requires ``DATA_SOURCE_QUERY`` even if this service was
        constructed with a weaker permission for correction-only use.
        """
        previous = self._required_permission
        self._required_permission = WorkspacePermission.DATA_SOURCE_QUERY
        try:
            result = await self.correct(params)
            execution = await execute_corrected_sql(
                client=mcp_client,
                params=params,
                outcome=result.outcome,
                limit=limit,
            )
        finally:
            self._required_permission = previous
        return result, execution

    def _ensure_request_authorized(self, params: SQLCorrectParams) -> DataSource:
        user = self._session.get(User, params.user_id)
        if user is None or not user.is_active:
            raise SQLCorrectionAuthorizationError("User is not authorized")

        workspace = self._session.get(Workspace, params.workspace_id)
        if workspace is None or workspace.organization_id != params.organization_id:
            raise SQLCorrectionAuthorizationError(
                "Workspace is not accessible for organization"
            )

        data_source = self._session.get(DataSource, params.data_source_id)
        if data_source is None or data_source.workspace_id != params.workspace_id:
            raise SQLCorrectionAuthorizationError("Data source is not accessible")

        if params.metadata.data_source_id != params.data_source_id:
            raise SQLCorrectionAuthorizationError(
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
            raise SQLCorrectionAuthorizationError(
                "User is not a member of the workspace"
            )
        if not workspace_role_has_permission(member.role, self._required_permission):
            raise SQLCorrectionAuthorizationError(
                "User lacks permission for SQL correction"
            )
        return data_source

    def _ensure_session_access(self, params: SQLCorrectParams) -> None:
        assert params.session_id is not None
        try:
            snapshot = self._state_service.get_session(
                params.session_id,
                workspace_id=params.workspace_id,
                user_id=params.user_id,
            )
        except AnalysisSessionNotFoundError as exc:
            raise SQLCorrectionAuthorizationError(
                "Analysis session is not accessible"
            ) from exc
        if snapshot.organization_id != params.organization_id:
            raise SQLCorrectionAuthorizationError(
                "Analysis session organization does not match"
            )
        if snapshot.data_source_id is None:
            raise SQLCorrectionAuthorizationError(
                "Analysis session has no authorized data source"
            )
        if snapshot.data_source_id != params.data_source_id:
            raise SQLCorrectionAuthorizationError(
                "Data source does not match the analysis session"
            )
