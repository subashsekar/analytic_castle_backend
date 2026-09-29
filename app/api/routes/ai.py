import asyncio
import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.ai.context import MetadataSearchContextProvider, build_ai_context
from app.ai.exceptions import (
    AIConfigurationError,
    AIContextError,
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderRateLimitError,
    AIProviderTimeoutError,
    AIRequestValidationError,
    AIResponseValidationError,
)
from app.ai.metadata_resolver import MetadataContextResolver
from app.ai.metadata_types import empty_resolved_context
from app.ai.orchestrator import AIAnalystOrchestrator
from app.ai.providers import LLMProvider, create_llm_provider
from app.ai.types import AIRequest, AIResponse, TokenUsage
from app.api.deps import (
    get_current_user,
    load_workspace_access,
    require_rate_limit,
)
from app.core.authorization import DataSourceAccess
from app.core.request_id import get_request_id, new_request_id
from app.db.models import DataSource, User
from app.db.session import get_db
from app.enums import WorkspacePermission
from app.schemas.ai import (
    AIChatRequest,
    AIChatResponse,
    AIUsageResponse,
    intent_response,
    phase8_analysis_response,
    plan_response,
)

router = APIRouter(prefix="/api/v1/ai", tags=["ai"])

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}

DATA_SOURCE_NOT_FOUND_DETAIL = "Data source not found"
PROVIDER_UNAVAILABLE_DETAIL = "AI provider is unavailable"
PROVIDER_NOT_CONFIGURED_DETAIL = "AI provider is not configured"
PROVIDER_TIMEOUT_DETAIL = "AI provider timed out"
PROVIDER_RATE_LIMIT_DETAIL = "Too many requests"
INVALID_RESPONSE_DETAIL = "AI provider returned an invalid response"
INVALID_REQUEST_DETAIL = "AI request is invalid"


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _usage_response(usage: TokenUsage) -> AIUsageResponse | None:
    if not usage.available:
        return None
    payload: dict[str, int] = {}
    if usage.input_tokens is not None:
        payload["input_tokens"] = usage.input_tokens
    if usage.output_tokens is not None:
        payload["output_tokens"] = usage.output_tokens
    if usage.total_tokens is not None:
        payload["total_tokens"] = usage.total_tokens
    return AIUsageResponse(**payload)


def require_llm_provider(
    current_user: Annotated[User, Depends(get_current_user)],
) -> LLMProvider:
    del current_user
    try:
        # Intent/planner/supervisor use the fast model when LLM_FAST_MODEL is set.
        return create_llm_provider(fast=True)
    except AIConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PROVIDER_NOT_CONFIGURED_DETAIL,
        ) from None


def _chat_response(result: AIResponse, data_source_id: object) -> AIChatResponse:
    if result.intent is None or result.plan is None:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=INVALID_RESPONSE_DETAIL,
        )
    return AIChatResponse(
        request_id=result.request_id,
        response=result.response,
        model=result.model,
        usage=_usage_response(result.usage),
        intent=intent_response(result.intent),
        plan=plan_response(result.plan),
        metadata_context=result.metadata_context
        or empty_resolved_context(data_source_id),  # type: ignore[arg-type]
        conversation_id=result.conversation_id,
        conversation_version=result.conversation_version,
        analysis=phase8_analysis_response(result.phase8_analysis),
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _authorize_data_source(
    db: Session,
    user: User,
    data_source_id: object,
) -> DataSourceAccess:
    data_source = db.get(DataSource, data_source_id)
    if data_source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DATA_SOURCE_NOT_FOUND_DETAIL,
        )
    access = load_workspace_access(
        db,
        user,
        data_source.workspace_id,
        permission=WorkspacePermission.DATA_SOURCE_READ,
        allow_super_admin=True,
    )
    return DataSourceAccess(
        user=access.user,
        workspace=access.workspace,
        member=access.member,
        data_source=data_source,
        via_super_admin=access.via_super_admin,
    )


@router.post(
    "/chat",
    response_model=AIChatResponse,
    response_model_exclude_none=True,
    summary="Send an AI analyst chat message",
    description=(
        "AI analyst endpoint. Authenticates the user, authorizes the data source, "
        "detects analytical intent, resolves catalog metadata, generates and executes "
        "read-only SQL when the plan requires data access, runs Phase 8 analysis "
        "agents, and returns a structured response with optional analysis panels."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid AI request"},
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"},
        status.HTTP_502_BAD_GATEWAY: {"description": "AI provider failed"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "AI provider is not configured"
        },
        status.HTTP_504_GATEWAY_TIMEOUT: {"description": "AI provider timed out"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
async def chat(
    payload: AIChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    provider: Annotated[LLMProvider, Depends(require_llm_provider)],
) -> AIChatResponse:
    access = _authorize_data_source(db, current_user, payload.data_source_id)
    context = build_ai_context(access)
    request_id = get_request_id()
    if not request_id or request_id == "-":
        request_id = new_request_id()
    orchestrator = AIAnalystOrchestrator(
        provider,
        metadata=MetadataSearchContextProvider(db),
        resolver=MetadataContextResolver(db),
    )
    try:
        result = await orchestrator.chat(
            AIRequest(
                message=payload.message,
                data_source_id=payload.data_source_id,
                request_id=request_id,
                conversation_id=payload.conversation_id,
                conversation_version=payload.conversation_version,
            ),
            context,
            db=db,
        )
        db.commit()
    except AIRequestValidationError as exc:
        db.rollback()
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None
    except AIContextError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized",
        ) from None
    except AIConfigurationError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PROVIDER_NOT_CONFIGURED_DETAIL,
        ) from None
    except AIProviderTimeoutError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=PROVIDER_TIMEOUT_DETAIL,
        ) from None
    except AIProviderRateLimitError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=PROVIDER_RATE_LIMIT_DETAIL,
        ) from None
    except AIProviderAuthenticationError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=PROVIDER_UNAVAILABLE_DETAIL,
        ) from None
    except AIResponseValidationError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=INVALID_RESPONSE_DETAIL,
        ) from None
    except AIProviderError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=PROVIDER_UNAVAILABLE_DETAIL,
        ) from None

    return _chat_response(result, payload.data_source_id)


@router.post(
    "/chat/stream",
    summary="Stream an AI analyst chat response",
    description=(
        "Same authorization and pipeline as /chat, but emits Server-Sent Events so "
        "clients can show measured query facts before Phase 8 agents finish. "
        "Events: partial, final, error."
    ),
    responses={
        **_AUTH_ERRORS,
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid AI request"},
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Too many requests"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "AI provider is not configured"
        },
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
async def chat_stream(
    payload: AIChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    provider: Annotated[LLMProvider, Depends(require_llm_provider)],
) -> StreamingResponse:
    access = _authorize_data_source(db, current_user, payload.data_source_id)
    context = build_ai_context(access)
    request_id = get_request_id()
    if not request_id or request_id == "-":
        request_id = new_request_id()
    orchestrator = AIAnalystOrchestrator(
        provider,
        metadata=MetadataSearchContextProvider(db),
        resolver=MetadataContextResolver(db),
    )
    queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

    async def on_partial(event: dict[str, Any]) -> None:
        await queue.put(("partial", event))

    async def run() -> None:
        try:
            result = await orchestrator.chat(
                AIRequest(
                    message=payload.message,
                    data_source_id=payload.data_source_id,
                    request_id=request_id,
                    conversation_id=payload.conversation_id,
                    conversation_version=payload.conversation_version,
                ),
                context,
                db=db,
                on_partial=on_partial,
            )
            db.commit()
            response = _chat_response(result, payload.data_source_id)
            await queue.put(
                ("final", response.model_dump(mode="json", exclude_none=True))
            )
        except AIRequestValidationError as exc:
            db.rollback()
            await queue.put(("error", {"detail": str(exc) or INVALID_REQUEST_DETAIL}))
        except AIContextError:
            db.rollback()
            await queue.put(("error", {"detail": "Not authorized"}))
        except AIConfigurationError:
            db.rollback()
            await queue.put(("error", {"detail": PROVIDER_NOT_CONFIGURED_DETAIL}))
        except AIProviderTimeoutError:
            db.rollback()
            await queue.put(("error", {"detail": PROVIDER_TIMEOUT_DETAIL}))
        except AIProviderRateLimitError:
            db.rollback()
            await queue.put(("error", {"detail": PROVIDER_RATE_LIMIT_DETAIL}))
        except (
            AIProviderAuthenticationError,
            AIProviderError,
            AIResponseValidationError,
        ):
            db.rollback()
            await queue.put(("error", {"detail": PROVIDER_UNAVAILABLE_DETAIL}))
        except Exception:
            db.rollback()
            await queue.put(("error", {"detail": PROVIDER_UNAVAILABLE_DETAIL}))
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event, data = item
                yield _sse(event, data)
        finally:
            await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
