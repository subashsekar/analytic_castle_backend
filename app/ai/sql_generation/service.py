"""Authorized SQL generation service.

Produces structured SQL drafts from Phase 4 catalog metadata. Never executes
SQL and never calls MCP query tools.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings
from app.ai.prompt import PromptRegistry
from app.ai.sql_generation.catalog import ensure_metadata_catalog_authorized
from app.ai.sql_generation.errors import (
    SQLGenerationAuthorizationError,
    SQLGenerationConfigurationError,
    SQLGenerationSchemaError,
)
from app.ai.sql_generation.generation import generate_sql
from app.ai.sql_generation.logging_helpers import generation_log_context
from app.ai.sql_generation.models import SQLGenerateParams, SQLGenerationResult
from app.ai.sql_generation.prompts import build_sql_generation_prompt_registry
from app.ai.sql_generation.schema_cache import cached_schema_prompt_context
from app.ai.sql_generation.schema_context import has_usable_schema
from app.ai.state import AgentStateService
from app.core.authorization import is_super_admin, workspace_role_has_permission
from app.db.models import DataSource, User, Workspace, WorkspaceMember
from app.enums import WorkspacePermission

logger = logging.getLogger(__name__)


class SQLGenerationService:
    """Schema-aware SQL generation for an authorized workspace data source."""

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
            prompt_registry or build_sql_generation_prompt_registry()
        )
        self._allow_super_admin = allow_super_admin
        self._required_permission = required_permission
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            # Intent-like drafting uses the fast/cheap model when configured.
            config = llm_config or llm_client_config_from_settings(fast=True)
            if not config.api_key:
                raise SQLGenerationConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def generate(self, params: SQLGenerateParams) -> SQLGenerationResult:
        self._ensure_request_authorized(params)
        ensure_metadata_catalog_authorized(
            self._session,
            data_source_id=params.data_source_id,
            metadata=params.metadata,
        )
        if params.session_id is not None:
            self._ensure_session_access(params)

        if not has_usable_schema(params.metadata):
            raise SQLGenerationSchemaError(
                "Schema context is required for SQL generation"
            )

        schema = cached_schema_prompt_context(params.metadata)
        outcome = await generate_sql(
            client=self._llm_client,
            registry=self._prompt_registry,
            message=params.message,
            metadata=params.metadata,
            intent=params.intent,
            plan_summary=params.plan_summary,
            data_source_name=params.data_source_name,
            schema_context=schema,
            conversation_context=params.conversation_context,
        )
        logger.info(
            "sql generation service completed",
            extra={
                **generation_log_context(outcome),
                "workspace_id": str(params.workspace_id),
                "organization_id": str(params.organization_id),
                "data_source_id": str(params.data_source_id),
            },
        )
        return SQLGenerationResult(
            outcome=outcome,
            data_source_id=params.data_source_id,
            workspace_id=params.workspace_id,
            organization_id=params.organization_id,
        )

    def _ensure_request_authorized(self, params: SQLGenerateParams) -> DataSource:
        user = self._session.get(User, params.user_id)
        if user is None or not user.is_active:
            raise SQLGenerationAuthorizationError("User is not authorized")

        workspace = self._session.get(Workspace, params.workspace_id)
        if workspace is None or workspace.organization_id != params.organization_id:
            raise SQLGenerationAuthorizationError(
                "Workspace is not accessible for organization"
            )

        data_source = self._session.get(DataSource, params.data_source_id)
        if data_source is None or data_source.workspace_id != params.workspace_id:
            raise SQLGenerationAuthorizationError("Data source is not accessible")

        if params.metadata.data_source_id != params.data_source_id:
            raise SQLGenerationAuthorizationError(
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
            raise SQLGenerationAuthorizationError(
                "User is not a member of the workspace"
            )
        if not workspace_role_has_permission(member.role, self._required_permission):
            raise SQLGenerationAuthorizationError(
                "User lacks permission for SQL generation"
            )
        return data_source

    def _ensure_session_access(self, params: SQLGenerateParams) -> None:
        assert params.session_id is not None
        snapshot = self._state_service.get_session(
            params.session_id,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
        )
        if snapshot.organization_id != params.organization_id:
            raise SQLGenerationAuthorizationError(
                "Analysis session organization does not match"
            )
        if snapshot.data_source_id is None:
            raise SQLGenerationAuthorizationError(
                "Analysis session has no authorized data source"
            )
        if snapshot.data_source_id != params.data_source_id:
            raise SQLGenerationAuthorizationError(
                "Data source does not match the analysis session"
            )
