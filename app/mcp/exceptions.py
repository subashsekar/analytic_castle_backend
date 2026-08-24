"""Safe MCP errors. Messages never include credentials or stack traces."""

from __future__ import annotations

from app.core.logging import redact_secret
from app.mcp.error_codes import MCPErrorCategory, MCPErrorCode


class MCPError(Exception):
    """Base MCP failure. Client responses must use ``to_error_response``."""

    code: MCPErrorCode = MCPErrorCode.MCP_INTERNAL_ERROR
    category: MCPErrorCategory = MCPErrorCategory.INTERNAL_ERROR
    http_status: int = 500

    def __init__(
        self,
        message: str = "MCP request failed",
        *,
        request_id: str | None = None,
        retry_after: int | None = None,
        code: MCPErrorCode | None = None,
        category: MCPErrorCategory | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(redact_secret(message))
        self.request_id = request_id
        self.retry_after = retry_after
        if code is not None:
            self.code = code
        if category is not None:
            self.category = category
        if http_status is not None:
            self.http_status = http_status


class MCPConfigurationError(MCPError):
    code = MCPErrorCode.MCP_CONFIGURATION_ERROR
    category = MCPErrorCategory.SERVER_ERROR
    http_status = 500

    def __init__(
        self,
        message: str = "MCP configuration is invalid",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPServerError(MCPError):
    code = MCPErrorCode.MCP_SERVER_ERROR
    category = MCPErrorCategory.SERVER_ERROR
    http_status = 502

    def __init__(
        self,
        message: str = "MCP server failed",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPServerNotFoundError(MCPError):
    code = MCPErrorCode.MCP_SERVER_NOT_FOUND
    category = MCPErrorCategory.NOT_FOUND
    http_status = 404

    def __init__(
        self,
        message: str = "MCP server was not found",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPToolError(MCPError):
    code = MCPErrorCode.MCP_INTERNAL_ERROR
    category = MCPErrorCategory.INTERNAL_ERROR
    http_status = 500

    def __init__(
        self,
        message: str = "MCP tool execution failed",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPToolNotFoundError(MCPError):
    code = MCPErrorCode.MCP_TOOL_NOT_FOUND
    category = MCPErrorCategory.NOT_FOUND
    http_status = 404

    def __init__(
        self,
        message: str = "MCP tool was not found",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPToolValidationError(MCPError):
    code = MCPErrorCode.MCP_TOOL_VALIDATION_FAILED
    category = MCPErrorCategory.VALIDATION_ERROR
    http_status = 422

    def __init__(
        self,
        message: str = "MCP tool arguments are invalid",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPToolNameValidationError(MCPToolValidationError):
    def __init__(
        self,
        message: str = "MCP tool name is invalid",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPUnauthorizedError(MCPError):
    code = MCPErrorCode.MCP_UNAUTHORIZED
    category = MCPErrorCategory.AUTHENTICATION_ERROR
    http_status = 401

    def __init__(
        self,
        message: str = "Not authenticated",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPAuthorizationError(MCPError):
    """Authorization failure that is safe to distinguish from authentication."""

    code = MCPErrorCode.MCP_FORBIDDEN
    category = MCPErrorCategory.AUTHORIZATION_ERROR
    http_status = 403

    def __init__(
        self,
        message: str = "Not authorized",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPAccessDeniedError(MCPAuthorizationError):
    """Denial used for workspace/data-source authorization.

    Messages intentionally avoid leaking whether the target resource exists.
    """

    code = MCPErrorCode.MCP_DATA_SOURCE_NOT_FOUND
    category = MCPErrorCategory.NOT_FOUND
    http_status = 404

    def __init__(
        self,
        message: str = "Data source not found",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPWorkspaceAccessError(MCPAccessDeniedError):
    """Workspace isolation denial (anti-enumeration)."""


class MCPDataSourceAccessError(MCPAccessDeniedError):
    """Data-source isolation denial (anti-enumeration)."""

    code = MCPErrorCode.MCP_DATA_SOURCE_ACCESS_DENIED


class MCPRateLimitError(MCPError):
    code = MCPErrorCode.MCP_RATE_LIMITED
    category = MCPErrorCategory.RATE_LIMITED
    http_status = 429

    def __init__(
        self,
        message: str = "Too many MCP requests",
        *,
        request_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id, retry_after=retry_after)


class MCPQueryError(MCPError):
    code = MCPErrorCode.MCP_DATABASE_ERROR
    category = MCPErrorCategory.QUERY_ERROR
    http_status = 400

    def __init__(
        self,
        message: str = "Read-only query failed",
        *,
        request_id: str | None = None,
        code: MCPErrorCode | None = None,
        category: MCPErrorCategory | None = None,
    ) -> None:
        super().__init__(
            message,
            request_id=request_id,
            code=code,
            category=category,
        )


class MCPQueryRejectedError(MCPQueryError):
    code = MCPErrorCode.MCP_QUERY_INVALID
    category = MCPErrorCategory.QUERY_ERROR
    http_status = 400

    def __init__(
        self,
        message: str = "Only a single read-only query is allowed",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPQueryTimeoutError(MCPQueryError):
    code = MCPErrorCode.MCP_QUERY_TIMEOUT
    category = MCPErrorCategory.QUERY_TIMEOUT
    http_status = 504

    def __init__(
        self,
        message: str = "The query timed out",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPQueryLimitError(MCPQueryError):
    code = MCPErrorCode.MCP_QUERY_LIMIT_EXCEEDED
    category = MCPErrorCategory.LIMIT_EXCEEDED
    http_status = 400

    def __init__(
        self,
        message: str = "The query row limit was exceeded",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPQueryResultError(MCPQueryError):
    code = MCPErrorCode.MCP_RESULT_LIMIT_EXCEEDED
    category = MCPErrorCategory.LIMIT_EXCEEDED
    http_status = 400

    def __init__(
        self,
        message: str = "The query result is too large",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPResultLimitError(MCPQueryResultError):
    """Alias for result-size violations."""


class MCPDatabaseUnavailableError(MCPError):
    code = MCPErrorCode.MCP_DATABASE_UNAVAILABLE
    category = MCPErrorCategory.DATA_SOURCE_ERROR
    http_status = 503

    def __init__(
        self,
        message: str = "The database is unavailable",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPDatabaseError(MCPError):
    code = MCPErrorCode.MCP_DATABASE_ERROR
    category = MCPErrorCategory.DATA_SOURCE_ERROR
    http_status = 502

    def __init__(
        self,
        message: str = "The requested database operation could not be completed",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)


class MCPSampleDataError(MCPError):
    code = MCPErrorCode.MCP_SAMPLE_DATA_ERROR
    category = MCPErrorCategory.DATA_SOURCE_ERROR
    http_status = 502

    def __init__(
        self,
        message: str = "Unable to retrieve sample data",
        *,
        request_id: str | None = None,
        code: MCPErrorCode | None = None,
        category: MCPErrorCategory | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(
            message,
            request_id=request_id,
            code=code,
            category=category,
            http_status=http_status,
        )


class MCPInternalError(MCPError):
    code = MCPErrorCode.MCP_INTERNAL_ERROR
    category = MCPErrorCategory.INTERNAL_ERROR
    http_status = 500

    def __init__(
        self,
        message: str = "An unexpected MCP error occurred",
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, request_id=request_id)
