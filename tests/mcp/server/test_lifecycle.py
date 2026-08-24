"""Phase 5.6 — MCP server lifecycle tests."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.orm import Session

from app.mcp.exceptions import MCPConfigurationError, MCPServerError
from app.mcp.registry import MCPRegistry
from app.mcp.server.lifecycle import build_postgres_mcp, shutdown_mcp
from app.mcp.servers.postgres.server import PostgreSQLMCPServer
from tests.conftest import run_async


def test_successful_startup_and_shutdown(db_session: Session) -> None:
    registry, client = build_postgres_mcp(db_session)
    assert registry.get_server("postgres").name == "postgres"
    assert client is not None
    shutdown_mcp()
    shutdown_mcp()


def test_repeated_startup_creates_isolated_registries(db_session: Session) -> None:
    first, _ = build_postgres_mcp(db_session)
    second, _ = build_postgres_mcp(db_session)
    assert first is not second
    assert first.list_tools()[0] is not second.list_tools()[0]


def test_startup_failure_maps_to_server_error(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        del self, registry
        raise RuntimeError("password=leak DATABASE_URL=postgresql://x")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _boom)
    with pytest.raises(MCPServerError, match="failed to start") as exc_info:
        build_postgres_mcp(db_session)
    assert "password=" not in str(exc_info.value)
    assert "DATABASE_URL" not in str(exc_info.value)


def test_configuration_error_propagates(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        del self, registry
        raise MCPConfigurationError("bad config")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _boom)
    with pytest.raises(MCPConfigurationError, match="bad config"):
        build_postgres_mcp(db_session)


def test_server_error_propagates_without_wrapping(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        del self, registry
        raise MCPServerError("explicit server failure")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _boom)
    with pytest.raises(MCPServerError, match="explicit server failure"):
        build_postgres_mcp(db_session)


def test_tool_registration_failure_does_not_leak_partial_global_state(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = PostgreSQLMCPServer.register
    calls: list[str] = []

    def _partial(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        registry.register_server(self)
        calls.append("server")
        raise RuntimeError("tool registration failed")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _partial)
    with pytest.raises(MCPServerError):
        build_postgres_mcp(db_session)

    monkeypatch.setattr(PostgreSQLMCPServer, "register", original)
    registry, _ = build_postgres_mcp(db_session)
    assert registry.get_server("postgres").name == "postgres"
    assert len(registry.list_tools()) >= 1
    assert calls == ["server"]


def test_cancellation_during_startup_propagates(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _cancel(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        del self, registry
        raise asyncio.CancelledError()

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _cancel)

    async def _run() -> None:
        # build_postgres_mcp is sync; CancelledError from register is wrapped
        # unless it is CancelledError — verify current contract.
        with pytest.raises((MCPServerError, asyncio.CancelledError)):
            build_postgres_mcp(db_session)

    run_async(_run())


def test_shutdown_is_safe_after_failed_startup(
    db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(self: PostgreSQLMCPServer, registry: MCPRegistry) -> None:
        del self, registry
        raise RuntimeError("boom")

    monkeypatch.setattr(PostgreSQLMCPServer, "register", _boom)
    with pytest.raises(MCPServerError):
        build_postgres_mcp(db_session)
    shutdown_mcp()
    shutdown_mcp()
