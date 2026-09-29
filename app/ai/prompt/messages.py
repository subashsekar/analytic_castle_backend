"""Bridge rendered prompts to LLM message models without importing the client."""

from __future__ import annotations

from app.ai.llm.models import LLMMessage
from app.ai.prompt.models import PromptBundle, RenderedPrompt


def rendered_to_llm_message(rendered: RenderedPrompt) -> LLMMessage:
    return LLMMessage(role=rendered.role, content=rendered.content)


def bundle_to_llm_messages(bundle: PromptBundle) -> list[LLMMessage]:
    messages: list[LLMMessage] = []
    for rendered in (bundle.system, bundle.user, bundle.assistant):
        if rendered is not None:
            messages.append(rendered_to_llm_message(rendered))
    return messages
