"""MCP settings accessors.

Values are loaded by the application Settings object. This module does not
read `.env` itself and does not store credentials.
"""

from __future__ import annotations

from app.core.config import settings as app_settings


class MCPSettings:
    @property
    def MCP_QUERY_DEFAULT_LIMIT(self) -> int:
        return app_settings.MCP_QUERY_DEFAULT_LIMIT

    @property
    def MCP_QUERY_MAX_LIMIT(self) -> int:
        return app_settings.MCP_QUERY_MAX_LIMIT

    @property
    def MCP_QUERY_TIMEOUT_SECONDS(self) -> float:
        return app_settings.MCP_QUERY_TIMEOUT_SECONDS

    @property
    def MCP_QUERY_MAX_SQL_CHARS(self) -> int:
        return app_settings.MCP_QUERY_MAX_SQL_CHARS

    @property
    def MCP_QUERY_MAX_RESULT_CHARS(self) -> int:
        return app_settings.MCP_QUERY_MAX_RESULT_CHARS

    @property
    def MCP_QUERY_MAX_VALUE_CHARS(self) -> int:
        return app_settings.MCP_QUERY_MAX_VALUE_CHARS

    @property
    def MCP_QUERY_MAX_JSON_CHARS(self) -> int:
        return app_settings.MCP_QUERY_MAX_JSON_CHARS


mcp_settings = MCPSettings()
