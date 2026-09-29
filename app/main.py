import asyncio
import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.ai import router as ai_router
from app.api.routes.auth import router as auth_router
from app.api.routes.conversations import router as conversations_router
from app.api.routes.data_sources import router as data_sources_router
from app.api.routes.health import router as health_router
from app.api.routes.metadata import router as metadata_router
from app.api.routes.organizations import router as organizations_router
from app.api.routes.query_history import router as query_history_router
from app.api.routes.workspaces import router as workspaces_router
from app.core.config import settings
from app.core.logging import configure_logging, redact_secret
from app.core.middleware import (
    RequestContextMiddleware,
    RequestSizeLimitMiddleware,
    SecurityHeadersMiddleware,
)
from app.core.request_id import REQUEST_ID_HEADER, get_request_id, new_request_id
from app.mcp.errors import error_response_dict, http_status_for
from app.mcp.exceptions import MCPError, MCPRateLimitError

# psycopg async requires SelectorEventLoop on Windows. Uvicorn sets this too;
# keep it here so FastAPI TestClient and other ASGI servers also work.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

configure_logging(debug=settings.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.expose_debug_details,
)
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(organizations_router)
app.include_router(workspaces_router)
app.include_router(data_sources_router)
app.include_router(metadata_router)
app.include_router(ai_router)
app.include_router(conversations_router)
app.include_router(query_history_router)

app.add_middleware(RequestSizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.cors_allowed_methods,
    allow_headers=settings.cors_allowed_headers,
    expose_headers=[REQUEST_ID_HEADER],
)
app.add_middleware(RequestContextMiddleware)


def _request_id_for(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if isinstance(existing, str) and existing:
        return existing
    contextual = get_request_id()
    if contextual and contextual != "-":
        return contextual
    return new_request_id()


@app.exception_handler(MCPError)
async def mcp_exception_handler(request: Request, exc: MCPError) -> JSONResponse:
    """Map MCP failures to safe structured JSON without leaking internals."""

    request_id = exc.request_id if exc.request_id else _request_id_for(request)
    content = error_response_dict(exc, request_id=request_id)
    headers: dict[str, str] = {REQUEST_ID_HEADER: request_id}
    if isinstance(exc, MCPRateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    return JSONResponse(
        status_code=http_status_for(exc),
        content=content,
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id_for(request)
    logger.exception(
        "Unhandled error method=%s path=%s request_id=%s error_type=%s",
        request.method,
        request.url.path,
        request_id,
        type(exc).__name__,
    )
    detail = (
        redact_secret(str(exc))
        if settings.expose_debug_details
        else "Internal server error"
    )
    return JSONResponse(
        status_code=500,
        content={"detail": detail, "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )
