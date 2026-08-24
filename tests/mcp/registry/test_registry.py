"""Phase 5.6 — MCP registry registration and lookup tests."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.mcp import build_postgres_mcp
from app.mcp.exceptions import MCPServerNotFoundError, MCPToolNotFoundError
from app.mcp.registry import MCPRegistry
from app.mcp.schemas import MCPToolContext
from app.mcp.security import MCPToolPermission
from app.mcp.servers.postgres.server import PostgreSQLMCPServer


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_source_id: UUID


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool = True


class _Tool:
    def __init__(self, name: str, permission: MCPToolPermission) -> None:
        self.name = name
        self.permission = permission
        self.description = "test"
        self.input_model = _In
        self.output_model = _Out

    async def invoke(self, arguments: BaseModel, context: MCPToolContext) -> BaseModel:
        del arguments, context
        return _Out(ok=True)


class _Server:
    def __init__(self, name: str = "alpha") -> None:
        self.name = name

    def register(self, registry: MCPRegistry) -> None:
        registry.register_server(self)


def test_server_registration_and_lookup() -> None:
    registry = MCPRegistry()
    server = _Server("alpha")
    registry.register_server(server)
    assert registry.get_server("alpha") is server
    assert registry.list_servers() == (server,)


def test_duplicate_server_registration_rejected() -> None:
    registry = MCPRegistry()
    registry.register_server(_Server("alpha"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register_server(_Server("alpha"))


def test_tool_registration_and_lookup() -> None:
    registry = MCPRegistry()
    registry.register_server(_Server("alpha"))
    tool = _Tool("alpha.ping", MCPToolPermission.METADATA_READ)
    registry.register(tool, server_name="alpha")
    assert registry.get("alpha.ping") is tool
    record = registry.get_record("alpha.ping")
    assert record.permission is MCPToolPermission.METADATA_READ
    assert record.server_name == "alpha"
    assert registry.list_tools() == (tool,)


def test_duplicate_tool_registration_rejected() -> None:
    registry = MCPRegistry()
    registry.register_server(_Server("alpha"))
    tool = _Tool("alpha.ping", MCPToolPermission.METADATA_READ)
    registry.register(tool, server_name="alpha")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool, server_name="alpha")


def test_unknown_server_and_tool_errors() -> None:
    registry = MCPRegistry()
    registry.register_server(_Server("alpha"))
    with pytest.raises(MCPServerNotFoundError):
        registry.get_server("missing")
    with pytest.raises(MCPServerNotFoundError):
        registry.get("missing.ping")
    with pytest.raises(MCPToolNotFoundError):
        registry.get("alpha.missing")


def test_tool_without_permission_cannot_register() -> None:
    registry = MCPRegistry()
    registry.register_server(_Server("alpha"))

    class _NoPerm:
        name = "alpha.noperm"
        description = "x"
        input_model = BaseModel
        output_model = BaseModel

        async def invoke(
            self, arguments: BaseModel, context: MCPToolContext
        ) -> BaseModel:
            del arguments, context
            return BaseModel()

    with pytest.raises(TypeError, match="permission"):
        registry.register(_NoPerm(), server_name="alpha")


def test_deterministic_registration_order(db_session: Session) -> None:
    registry_a, _ = build_postgres_mcp(db_session)
    registry_b = MCPRegistry()
    PostgreSQLMCPServer(db_session).register(registry_b)
    assert [t.name for t in registry_a.list_tools()] == [
        t.name for t in registry_b.list_tools()
    ]
    assert [s.name for s in registry_a.list_servers()] == [
        s.name for s in registry_b.list_servers()
    ]


def test_registry_isolation_between_builds(db_session: Session) -> None:
    registry_a, _ = build_postgres_mcp(db_session)
    registry_b, _ = build_postgres_mcp(db_session)
    assert registry_a is not registry_b
    assert registry_a.list_tools()[0] is not registry_b.list_tools()[0]


def test_list_schemas_excludes_credentials(db_session: Session) -> None:
    registry, _ = build_postgres_mcp(db_session)
    rendered = str(registry.list_schemas())
    assert "password" not in rendered.lower()
    assert "encrypted" not in rendered.lower()
    assert "jwt" not in rendered.lower()
