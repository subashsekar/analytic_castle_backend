"""Shared fixtures and helpers for MCP Phase 5.6 tests."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.connectors.types import QueryResult
from app.db.models import (
    DataSource,
    DataSourceConnection,
    Organization,
    User,
    Workspace,
    WorkspaceMember,
)
from app.enums import WorkspaceRole
from app.mcp import MCPToolContext, build_postgres_mcp
from app.mcp.client import MCPClient
from app.mcp.registry import MCPRegistry
from app.mcp.security import MCPToolPermission
from app.services.credentials import encrypt_secret
from tests.test_ai_metadata import _source

MCP_TEST_PASSWORD = "McpPhase56Secret!42"


@pytest.fixture
def mcp_context(
    workspace: Workspace,
    test_user: User,
    workspace_member: WorkspaceMember,
) -> MCPToolContext:
    del workspace_member
    return MCPToolContext(workspace_id=workspace.id, user_id=test_user.id)


@pytest.fixture
def mcp_stack(db_session: Session) -> tuple[MCPRegistry, MCPClient]:
    return build_postgres_mcp(db_session)


def connected_source(
    db_session: Session,
    workspace: Workspace,
    owner: User,
    *,
    name: str = "Analytics",
    password: str = MCP_TEST_PASSWORD,
) -> DataSource:
    source = _source(db_session, workspace, owner, name=name)
    db_session.add(
        DataSourceConnection(
            data_source_id=source.id,
            host="db.internal.example",
            port=5432,
            database_name="analytics",
            username="readonly",
            encrypted_password=encrypt_secret(password),
            ssl_mode="prefer",
        )
    )
    db_session.flush()
    return source


def add_workspace_member(
    db_session: Session,
    *,
    workspace: Workspace,
    user: User,
    role: WorkspaceRole,
) -> WorkspaceMember:
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=role,
    )
    db_session.add(member)
    db_session.flush()
    return member


def create_workspace(
    db_session: Session,
    organization: Organization,
    *,
    name: str = "Other Workspace",
) -> Workspace:
    workspace = Workspace(
        organization_id=organization.id,
        name=name,
        slug=f"ws-{uuid.uuid4().hex[:8]}",
    )
    db_session.add(workspace)
    db_session.flush()
    return workspace


def create_organization(
    db_session: Session, *, name: str = "Other Org"
) -> Organization:
    org = Organization(name=name, slug=f"org-{uuid.uuid4().hex[:8]}")
    db_session.add(org)
    db_session.flush()
    return org


async def fake_query_ok(config: object, sql: str, limit: int) -> QueryResult:
    del config, sql
    rows = tuple((index,) for index in range(min(3, limit)))
    return QueryResult(columns=("id",), rows=rows, truncated=False)


def tracking_executor(
    *,
    calls: list[tuple[object, ...]] | None = None,
    result: QueryResult | None = None,
    error: Exception | None = None,
) -> Callable[[object, str, int], Awaitable[QueryResult]]:
    recorded = calls if calls is not None else []

    async def _execute(config: object, sql: str, limit: int) -> QueryResult:
        recorded.append((config, sql, limit))
        if error is not None:
            raise error
        if result is not None:
            return result
        return await fake_query_ok(config, sql, limit)

    return _execute


class PermissionProbeTool:
    """Minimal registered tool used to assert central client authorization."""

    def __init__(self, *, permission: MCPToolPermission) -> None:
        self.name = "probe.action"
        self.permission = permission
        self.description = "Permission probe"
        self.invoked: list[str] = []

        class _In(BaseModel):
            model_config = ConfigDict(extra="forbid")
            data_source_id: UUID

        class _Out(BaseModel):
            model_config = ConfigDict(extra="forbid")
            ok: bool = True

        self.input_model: type[BaseModel] = _In
        self.output_model: type[BaseModel] = _Out

    async def invoke(self, arguments: BaseModel, context: MCPToolContext) -> BaseModel:
        del arguments, context
        self.invoked.append("called")
        return self.output_model()


class StaticServer:
    name = "probe"

    def __init__(self, tools: list[Any]) -> None:
        self._tools = tools

    def register(self, registry: MCPRegistry) -> None:
        registry.register_server(self)
        for tool in self._tools:
            registry.register(tool, server_name=self.name)
