"""Typed Pydantic models for agent state and conversation context."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.llm.models import LLMMessage, LLMMessageRole
from app.enums import AgentPhase

ConversationMessageRole = LLMMessageRole

_MAX_PAYLOAD_KEYS = 50
_MAX_PAYLOAD_STRING_CHARS = 4_096
_MAX_METADATA_REFS = 20
_MAX_CUSTOM_ENTRIES = 20


class ConversationMessage(BaseModel):
    """One conversation turn. Content is never logged by helpers."""

    model_config = ConfigDict(extra="forbid")

    role: ConversationMessageRole
    content: str = Field(min_length=1)
    message_id: UUID = Field(default_factory=uuid4)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message content must not be empty")
        return stripped

    @classmethod
    def from_llm(
        cls,
        message: LLMMessage,
        *,
        message_id: UUID | None = None,
    ) -> ConversationMessage:
        if message_id is None:
            return cls(role=message.role, content=message.content)
        return cls(role=message.role, content=message.content, message_id=message_id)

    def to_llm(self) -> LLMMessage:
        return LLMMessage(role=self.role, content=self.content)


class ConversationContextData(BaseModel):
    """Validated conversation context payload."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ConversationMessage] = Field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(message.content) for message in self.messages)

    @property
    def message_count(self) -> int:
        return len(self.messages)


class AgentStatePayload(BaseModel):
    """Structured agent state. Does not store prompts or credentials."""

    model_config = ConfigDict(extra="forbid")

    intent: str | None = Field(default=None, max_length=64)
    plan_version: str | None = Field(default=None, max_length=64)
    metadata_refs: list[str] = Field(
        default_factory=list, max_length=_MAX_METADATA_REFS
    )
    custom: dict[str, str] = Field(default_factory=dict, max_length=_MAX_CUSTOM_ENTRIES)
    notes: str | None = Field(default=None, max_length=_MAX_PAYLOAD_STRING_CHARS)

    @field_validator("metadata_refs")
    @classmethod
    def validate_metadata_refs(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("custom")
    @classmethod
    def validate_custom(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > _MAX_CUSTOM_ENTRIES:
            raise ValueError("too many custom entries")
        validated: dict[str, str] = {}
        for key, item in value.items():
            stripped_key = key.strip()
            if not stripped_key:
                continue
            if len(item) > _MAX_PAYLOAD_STRING_CHARS:
                raise ValueError("custom value exceeds size limit")
            validated[stripped_key] = item
        return validated


class AgentStateSnapshot(BaseModel):
    """In-memory view of persisted agent state."""

    model_config = ConfigDict(extra="forbid")

    phase: AgentPhase
    payload: AgentStatePayload
    version: int = Field(ge=1)


class AnalysisSessionSnapshot(BaseModel):
    """Safe read model for one analysis session."""

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    user_id: UUID
    workspace_id: UUID
    organization_id: UUID
    data_source_id: UUID | None
    status: str
    agent_state: AgentStateSnapshot | None = None
    conversation: ConversationContextData | None = None
    conversation_version: int | None = Field(default=None, ge=1)
    error_message: str | None = None


class SetPhaseTransition(BaseModel):
    kind: Literal["set_phase"] = "set_phase"
    phase: AgentPhase


class MergePayloadTransition(BaseModel):
    kind: Literal["merge_payload"] = "merge_payload"
    updates: dict[str, Any] = Field(default_factory=dict, max_length=_MAX_PAYLOAD_KEYS)


class ReplacePayloadTransition(BaseModel):
    kind: Literal["replace_payload"] = "replace_payload"
    payload: AgentStatePayload


StateTransition = SetPhaseTransition | MergePayloadTransition | ReplacePayloadTransition
