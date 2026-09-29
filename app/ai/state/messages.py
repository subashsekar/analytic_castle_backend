"""Bridge persisted conversation messages to the 6.1 LLM message model."""

from __future__ import annotations

from uuid import UUID

from app.ai.llm.models import LLMMessage
from app.ai.state.models import ConversationContextData, ConversationMessage


def conversation_message_to_llm(message: ConversationMessage) -> LLMMessage:
    return LLMMessage(role=message.role, content=message.content)


def llm_message_to_conversation(
    message: LLMMessage,
    *,
    message_id: UUID | None = None,
) -> ConversationMessage:
    if message_id is None:
        return ConversationMessage(role=message.role, content=message.content)
    return ConversationMessage(
        role=message.role,
        content=message.content,
        message_id=message_id,
    )


def conversation_to_llm_messages(
    context: ConversationContextData,
) -> list[LLMMessage]:
    return [conversation_message_to_llm(message) for message in context.messages]
