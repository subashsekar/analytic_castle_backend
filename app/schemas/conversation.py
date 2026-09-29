from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ai.state.models import ConversationMessage
from app.enums import AnalysisSessionStatus


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    message_id: UUID


class ConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation_id: UUID
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None = None
    status: str
    message_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    conversation_version: int = Field(ge=1)
    agent_version: int = Field(ge=1)
    started_at: datetime
    updated_at: datetime
    error_message: str | None = None


class ConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ConversationResponse]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)


class ConversationMessageListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ConversationMessageResponse]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total: int = Field(ge=0)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_source_id: UUID | None = None
    initial_message: str | None = Field(default=None, min_length=1)


class ConversationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: AnalysisSessionStatus | None = None
    expected_agent_version: int = Field(ge=1)


class AppendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    expected_context_version: int = Field(ge=1)
    trim_if_needed: bool = False
    # Role is server-controlled: public API may only append user messages.
    role: Literal["user"] = "user"


def conversation_response(record) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=record.conversation_id,
        user_id=record.user_id,
        workspace_id=record.workspace_id,
        organization_id=record.organization_id,
        data_source_id=record.data_source_id,
        status=record.status,
        message_count=record.message_count,
        char_count=record.char_count,
        conversation_version=record.conversation_version,
        agent_version=record.agent_version,
        started_at=record.started_at,
        updated_at=record.updated_at,
        error_message=record.error_message,
    )


def message_response(message: ConversationMessage) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        role=message.role,
        content=message.content,
        message_id=message.message_id,
    )
