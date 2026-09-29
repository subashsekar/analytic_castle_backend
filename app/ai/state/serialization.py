"""JSON serialization and deserialization for agent state payloads."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.ai.state.errors import StateSerializationError
from app.ai.state.models import (
    AgentStatePayload,
    ConversationContextData,
    ConversationMessage,
)
from app.enums import AgentPhase


def serialize_payload(payload: AgentStatePayload) -> dict[str, Any]:
    return payload.model_dump(mode="json")


def deserialize_payload(data: dict[str, Any] | None) -> AgentStatePayload:
    try:
        return AgentStatePayload.model_validate(data or {})
    except ValidationError as exc:
        raise StateSerializationError("Agent state payload is invalid") from exc


def serialize_context(context: ConversationContextData) -> list[dict[str, Any]]:
    return [message.model_dump(mode="json") for message in context.messages]


def deserialize_context(
    messages: list[dict[str, Any]] | None,
) -> ConversationContextData:
    try:
        parsed = [ConversationMessage.model_validate(item) for item in (messages or [])]
        return ConversationContextData(messages=parsed)
    except ValidationError as exc:
        raise StateSerializationError("Conversation context is invalid") from exc


def phase_from_value(value: str | AgentPhase) -> AgentPhase:
    if isinstance(value, AgentPhase):
        return value
    try:
        return AgentPhase(value)
    except ValueError as exc:
        raise StateSerializationError("Agent phase is invalid") from exc
