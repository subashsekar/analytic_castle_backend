from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.ai.memory import (
    AgentStateDataSourceNotFoundError,
    ContextLimitExceededError,
    ConversationMemoryService,
    ConversationNotActiveError,
    ConversationNotFoundError,
    CreateConversationParams,
    InvalidPaginationError,
    MessageLimitExceededError,
    StateVersionConflictError,
    UpdateConversationParams,
)
from app.ai.state.models import ConversationMessage
from app.api.deps import get_current_user, load_workspace_access, require_rate_limit
from app.core.authorization import WorkspaceAccess
from app.db.models import DataSource, User
from app.db.session import get_db
from app.enums import AnalysisSessionStatus
from app.schemas.conversation import (
    AppendMessageRequest,
    ConversationCreateRequest,
    ConversationListResponse,
    ConversationMessageListResponse,
    ConversationResponse,
    ConversationUpdateRequest,
    conversation_response,
    message_response,
)

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/conversations",
    tags=["conversations"],
)

_AUTH_ERRORS: dict[int | str, dict[str, Any]] = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated"},
    status.HTTP_403_FORBIDDEN: {"description": "Not authorized"},
}

CONVERSATION_NOT_FOUND_DETAIL = "Conversation not found"
DATA_SOURCE_NOT_FOUND_DETAIL = "Data source not found"
INVALID_REQUEST_DETAIL = "Invalid request"


def _forbidden() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized",
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=CONVERSATION_NOT_FOUND_DETAIL,
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _conflict(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def _require_workspace_access(
    db: Session,
    user: User,
    workspace_id: UUID,
) -> WorkspaceAccess:
    return load_workspace_access(db, user, workspace_id)


def _organization_id(access: WorkspaceAccess) -> UUID:
    return access.workspace.organization_id


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a conversation",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Data source not found"},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Validation error"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def create_conversation(
    workspace_id: UUID,
    payload: ConversationCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ConversationResponse:
    access = _require_workspace_access(db, current_user, workspace_id)
    if payload.data_source_id is not None:
        data_source = db.get(DataSource, payload.data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=DATA_SOURCE_NOT_FOUND_DETAIL,
            )

    initial_message = None
    if payload.initial_message is not None:
        initial_message = ConversationMessage(
            role="user", content=payload.initial_message
        )

    service = ConversationMemoryService(db)
    try:
        record = service.create_conversation(
            CreateConversationParams(
                user_id=current_user.id,
                workspace_id=workspace_id,
                organization_id=_organization_id(access),
                data_source_id=payload.data_source_id,
                initial_message=initial_message,
            )
        )
        db.commit()
    except AgentStateDataSourceNotFoundError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DATA_SOURCE_NOT_FOUND_DETAIL,
        ) from None
    except (ContextLimitExceededError, MessageLimitExceededError) as exc:
        db.rollback()
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None
    except ConversationNotFoundError:
        db.rollback()
        raise _not_found() from None

    return conversation_response(record)


@router.get(
    "",
    response_model=ConversationListResponse,
    summary="List conversations for the current user",
    responses={**_AUTH_ERRORS},
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def list_conversations(
    workspace_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(alias="page_size")] = None,
    status_filter: Annotated[
        AnalysisSessionStatus | None, Query(alias="status")
    ] = None,
) -> ConversationListResponse:
    _require_workspace_access(db, current_user, workspace_id)
    service = ConversationMemoryService(db)
    try:
        result = service.list_conversations(
            workspace_id=workspace_id,
            user_id=current_user.id,
            page=page,
            page_size=page_size,
            status=status_filter,
        )
    except InvalidPaginationError as exc:
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None
    except ConversationNotFoundError:
        raise _not_found() from None

    return ConversationListResponse(
        items=[conversation_response(item) for item in result.items],
        page=result.page,
        page_size=result.page_size,
        total=result.total,
    )


@router.get(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Get one conversation",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def get_conversation(
    workspace_id: UUID,
    conversation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ConversationResponse:
    _require_workspace_access(db, current_user, workspace_id)
    service = ConversationMemoryService(db)
    try:
        record = service.get_conversation(
            conversation_id,
            workspace_id=workspace_id,
            user_id=current_user.id,
        )
    except ConversationNotFoundError:
        raise _not_found() from None
    return conversation_response(record)


@router.patch(
    "/{conversation_id}",
    response_model=ConversationResponse,
    summary="Update a conversation",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
        status.HTTP_409_CONFLICT: {"description": "Version conflict or inactive"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def update_conversation(
    workspace_id: UUID,
    conversation_id: UUID,
    payload: ConversationUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ConversationResponse:
    _require_workspace_access(db, current_user, workspace_id)
    service = ConversationMemoryService(db)
    try:
        record = service.update_conversation(
            conversation_id,
            UpdateConversationParams(status=payload.status),
            workspace_id=workspace_id,
            user_id=current_user.id,
            expected_agent_version=payload.expected_agent_version,
        )
        db.commit()
    except ConversationNotFoundError:
        db.rollback()
        raise _not_found() from None
    except ConversationNotActiveError as exc:
        db.rollback()
        raise _conflict(str(exc) or INVALID_REQUEST_DETAIL) from None
    except StateVersionConflictError as exc:
        db.rollback()
        raise _conflict(str(exc) or INVALID_REQUEST_DETAIL) from None
    except InvalidPaginationError as exc:
        db.rollback()
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None

    return conversation_response(record)


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationMessageListResponse,
    summary="List conversation messages",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def list_messages(
    workspace_id: UUID,
    conversation_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(alias="page_size")] = None,
) -> ConversationMessageListResponse:
    _require_workspace_access(db, current_user, workspace_id)
    service = ConversationMemoryService(db)
    try:
        result = service.get_messages(
            conversation_id,
            workspace_id=workspace_id,
            user_id=current_user.id,
            page=page,
            page_size=page_size,
        )
    except ConversationNotFoundError:
        raise _not_found() from None
    except InvalidPaginationError as exc:
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None

    return ConversationMessageListResponse(
        items=[message_response(item) for item in result.items],
        page=result.page,
        page_size=result.page_size,
        total=result.total,
    )


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationResponse,
    summary="Append a message to a conversation",
    responses={
        **_AUTH_ERRORS,
        status.HTTP_404_NOT_FOUND: {"description": "Conversation not found"},
        status.HTTP_409_CONFLICT: {"description": "Version conflict or inactive"},
    },
    dependencies=[Depends(require_rate_limit("ai-chat"))],
)
def append_message(
    workspace_id: UUID,
    conversation_id: UUID,
    payload: AppendMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ConversationResponse:
    _require_workspace_access(db, current_user, workspace_id)
    service = ConversationMemoryService(db)
    try:
        record = service.append_message(
            conversation_id,
            ConversationMessage(role="user", content=payload.content),
            workspace_id=workspace_id,
            user_id=current_user.id,
            expected_context_version=payload.expected_context_version,
            trim_if_needed=payload.trim_if_needed,
        )
        db.commit()
    except ConversationNotFoundError:
        db.rollback()
        raise _not_found() from None
    except ConversationNotActiveError as exc:
        db.rollback()
        raise _conflict(str(exc) or INVALID_REQUEST_DETAIL) from None
    except StateVersionConflictError as exc:
        db.rollback()
        raise _conflict(str(exc) or INVALID_REQUEST_DETAIL) from None
    except (ContextLimitExceededError, MessageLimitExceededError) as exc:
        db.rollback()
        raise _bad_request(str(exc) or INVALID_REQUEST_DETAIL) from None

    return conversation_response(record)
