"""Context and message size limits for agent state."""

from __future__ import annotations

from dataclasses import dataclass

from app.ai.state.errors import ContextLimitExceededError, MessageLimitExceededError
from app.ai.state.models import ConversationContextData, ConversationMessage


@dataclass(frozen=True)
class ContextLimits:
    max_context_chars: int
    max_message_chars: int
    max_context_messages: int

    @classmethod
    def from_settings(cls) -> ContextLimits:
        from app.core.config import settings

        return cls(
            max_context_chars=settings.AI_MAX_CONTEXT_CHARS,
            max_message_chars=settings.AI_MAX_MESSAGE_CHARS,
            max_context_messages=settings.AI_MAX_CONTEXT_MESSAGES,
        )

    @classmethod
    def for_llm(cls) -> ContextLimits:
        """Limits for LLM prompt construction (may be tighter than storage)."""
        from app.core.config import settings

        storage = cls.from_settings()
        return cls(
            max_context_chars=settings.AI_LLM_CONTEXT_CHARS,
            max_message_chars=storage.max_message_chars,
            max_context_messages=storage.max_context_messages,
        )


def validate_message_size(
    message: ConversationMessage, *, limits: ContextLimits
) -> None:
    if len(message.content) > limits.max_message_chars:
        raise MessageLimitExceededError(
            f"Message exceeds {limits.max_message_chars} character limit"
        )


def validate_context_size(
    context: ConversationContextData, *, limits: ContextLimits
) -> None:
    if context.message_count > limits.max_context_messages:
        raise ContextLimitExceededError(
            f"Conversation context exceeds {limits.max_context_messages} message limit"
        )
    if context.char_count > limits.max_context_chars:
        raise ContextLimitExceededError(
            f"Conversation context exceeds {limits.max_context_chars} character limit"
        )


def trim_context_for_storage(
    context: ConversationContextData,
    incoming: ConversationMessage,
    *,
    limits: ContextLimits,
) -> ConversationContextData:
    """Drop oldest messages until the incoming message fits storage limits."""
    messages = list(context.messages)
    while True:
        candidate = ConversationContextData(messages=[*messages, incoming])
        if candidate.message_count <= limits.max_context_messages and (
            candidate.char_count <= limits.max_context_chars
        ):
            return candidate
        if not messages:
            return candidate
        messages.pop(0)
