"""Domain models for conversation memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from app.ai.llm.models import LLMMessage
from app.ai.state.models import ConversationMessage

T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    page: int
    page_size: int
    total: int


@dataclass(frozen=True)
class ConversationRecord:
    """Safe read model for one persisted conversation."""

    conversation_id: UUID
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None
    status: str
    message_count: int
    char_count: int
    conversation_version: int
    agent_version: int
    started_at: datetime
    updated_at: datetime
    error_message: str | None = None


MessageRecord = ConversationMessage


@dataclass(frozen=True)
class LLMContextSelection:
    """Result of selecting messages that fit LLM context limits."""

    messages: tuple[ConversationMessage, ...]
    total_message_count: int
    trimmed: bool
    dropped_message_count: int
    summary_text: str | None = None


@dataclass(frozen=True)
class LLMContextBuildResult:
    """LLM-ready messages plus safe trimming metadata."""

    messages: tuple[LLMMessage, ...]
    included_message_count: int
    total_message_count: int
    trimmed: bool
    dropped_message_count: int
    summary_injected: bool
