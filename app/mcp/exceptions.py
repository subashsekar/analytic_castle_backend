"""Safe MCP errors. Messages never include credentials or stack traces."""

from __future__ import annotations

from app.core.logging import redact_secret


class MCPError(Exception):
    def __init__(self, message: str = "MCP request failed") -> None:
        super().__init__(redact_secret(message))


class MCPToolNotFoundError(MCPError):
    def __init__(self, message: str = "MCP tool was not found") -> None:
        super().__init__(message)


class MCPToolValidationError(MCPError):
    def __init__(self, message: str = "MCP tool arguments are invalid") -> None:
        super().__init__(message)


class MCPQueryError(MCPError):
    def __init__(self, message: str = "Read-only query failed") -> None:
        super().__init__(message)


class MCPQueryRejectedError(MCPQueryError):
    def __init__(
        self, message: str = "Only a single read-only query is allowed"
    ) -> None:
        super().__init__(message)


class MCPQueryTimeoutError(MCPQueryError):
    def __init__(self, message: str = "The query timed out") -> None:
        super().__init__(message)


class MCPQueryResultError(MCPQueryError):
    def __init__(self, message: str = "The query result is too large") -> None:
        super().__init__(message)


class MCPUnauthorizedError(MCPError):
    def __init__(self, message: str = "Not authenticated") -> None:
        super().__init__(message)


class MCPAccessDeniedError(MCPError):
    """Denial used for workspace/data-source authorization.

    Messages intentionally avoid leaking whether the target resource exists.
    """

    def __init__(self, message: str = "Data source not found") -> None:
        super().__init__(message)


class MCPServerNotFoundError(MCPError):
    def __init__(self, message: str = "MCP server was not found") -> None:
        super().__init__(message)


class MCPToolNameValidationError(MCPToolValidationError):
    def __init__(self, message: str = "MCP tool name is invalid") -> None:
        super().__init__(message)


class MCPRateLimitError(MCPError):
    def __init__(self, message: str = "Too many MCP requests") -> None:
        super().__init__(message)
