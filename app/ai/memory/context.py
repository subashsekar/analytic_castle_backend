"""LLM context construction, trimming, and summarization hooks."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.ai.llm.models import LLMMessage
from app.ai.memory.models import LLMContextBuildResult, LLMContextSelection
from app.ai.state.limits import ContextLimits
from app.ai.state.messages import conversation_message_to_llm
from app.ai.state.models import ConversationContextData, ConversationMessage

logger = logging.getLogger(__name__)

_TRIM_NOTICE = "[Earlier conversation trimmed for context limits]"


@runtime_checkable
class SummarizationHook(Protocol):
    """Optional hook to summarize dropped messages instead of omitting them."""

    async def summarize(self, messages: Sequence[ConversationMessage]) -> str: ...


class PlaceholderSummarizationHook:
    """Default hook that injects a safe placeholder instead of raw history."""

    async def summarize(self, messages: Sequence[ConversationMessage]) -> str:
        del messages
        return _TRIM_NOTICE


def select_messages_for_llm(
    messages: Sequence[ConversationMessage],
    *,
    limits: ContextLimits,
    reserved_chars: int = 0,
    summary_text: str | None = None,
) -> LLMContextSelection:
    """Select the most recent messages that fit configured context limits."""
    if not messages:
        return LLMContextSelection(
            messages=(),
            total_message_count=0,
            trimmed=False,
            dropped_message_count=0,
            summary_text=summary_text,
        )

    budget = limits.max_context_chars - max(reserved_chars, 0)
    if summary_text is not None:
        budget -= len(summary_text)
    budget = max(budget, 1)

    selected_reversed: list[ConversationMessage] = []
    used_chars = 0
    for message in reversed(messages):
        if len(selected_reversed) >= limits.max_context_messages:
            break
        message_chars = len(message.content)
        if selected_reversed and used_chars + message_chars > budget:
            break
        if not selected_reversed and message_chars > budget:
            selected_reversed.append(message)
            used_chars = message_chars
            break
        selected_reversed.append(message)
        used_chars += message_chars

    selected = tuple(reversed(selected_reversed))
    dropped = len(messages) - len(selected)
    return LLMContextSelection(
        messages=selected,
        total_message_count=len(messages),
        trimmed=dropped > 0,
        dropped_message_count=dropped,
        summary_text=summary_text,
    )


async def build_llm_context(
    context: ConversationContextData,
    *,
    limits: ContextLimits,
    reserved_chars: int = 0,
    summarization_hook: SummarizationHook | None = None,
) -> LLMContextBuildResult:
    """Build LLM messages without sending the full conversation when over limits."""
    summary_text: str | None = None
    selection = select_messages_for_llm(
        context.messages,
        limits=limits,
        reserved_chars=reserved_chars,
    )
    if selection.trimmed:
        dropped = context.messages[: selection.dropped_message_count]
        hook = summarization_hook or PlaceholderSummarizationHook()
        summary_text = await hook.summarize(dropped)
        selection = select_messages_for_llm(
            context.messages,
            limits=limits,
            reserved_chars=reserved_chars,
            summary_text=summary_text,
        )

    llm_messages: list[LLMMessage] = []
    summary_injected = False
    if summary_text:
        llm_messages.append(LLMMessage(role="system", content=summary_text))
        summary_injected = True
    llm_messages.extend(
        conversation_message_to_llm(message) for message in selection.messages
    )

    if selection.trimmed:
        logger.info(
            "conversation context trimmed for llm",
            extra={
                "conversation_message_count": selection.total_message_count,
                "llm_message_count": len(selection.messages),
                "dropped_message_count": selection.dropped_message_count,
                "summary_injected": summary_injected,
            },
        )

    return LLMContextBuildResult(
        messages=tuple(llm_messages),
        included_message_count=len(selection.messages),
        total_message_count=selection.total_message_count,
        trimmed=selection.trimmed,
        dropped_message_count=selection.dropped_message_count,
        summary_injected=summary_injected,
    )
