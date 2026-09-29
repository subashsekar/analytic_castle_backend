"""LLM-backed intent detection. Planning concepts only; never SQL."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from app.ai.exceptions import AIResponseValidationError
from app.ai.intent_types import (
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIOperationType,
    LLMIntentDetection,
)
from app.ai.prompts import INTENT_SYSTEM_PROMPT_V1, USER_PROMPT_TEMPLATE_V1
from app.ai.providers.base import LLMMessage, LLMProvider
from app.ai.safety import (
    classify_unsupported,
    policy_model_name,
    unsupported_intent,
)
from app.ai.state.models import ConversationContextData
from app.ai.types import AIContext, MetadataSnippet, TokenUsage
from app.core.config import settings

logger = logging.getLogger(__name__)

_COUNT_LIKE = {AIOperationType.COUNT, AIOperationType.DISTINCT, AIOperationType.SELECT}
_ANALYTICAL_INTENTS = {
    AIIntentType.AGGREGATION,
    AIIntentType.RANKING,
    AIIntentType.TREND_ANALYSIS,
    AIIntentType.COMPARISON,
    AIIntentType.SUMMARY,
    AIIntentType.ANALYTICAL_QUERY,
}

DEFAULT_AMBIGUOUS_QUESTION = (
    "What would you like to know — a total, a count, a ranking, or a breakdown?"
)


@dataclass(frozen=True)
class IntentDetectionResult:
    intent: AIIntent
    model: str
    usage: TokenUsage
    skipped_llm: bool = False


class AIIntentService:
    """Detect a structured analytical intent from a user message."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def detect(
        self,
        message: str,
        context: AIContext,
        *,
        conversation: ConversationContextData | None = None,
    ) -> IntentDetectionResult:
        reason = classify_unsupported(message)
        if reason is not None:
            return IntentDetectionResult(
                intent=unsupported_intent(reason),
                model=policy_model_name(),
                usage=TokenUsage(),
                skipped_llm=True,
            )
        if conversation is not None and conversation.messages:
            messages = await build_intent_prompt_messages_with_memory(
                message,
                context,
                conversation=conversation,
            )
        else:
            messages = build_intent_prompt_messages(message, context)
        try:
            structured = await self._provider.generate_structured(
                messages,
                LLMIntentDetection,
            )
        except AIResponseValidationError as exc:
            logger.warning(
                "Intent structured response invalid; asking for clarification "
                "raw_content_chars=%s",
                len(exc.raw_content or ""),
            )
            return IntentDetectionResult(
                intent=AIIntent(
                    intent=AIIntentType.UNKNOWN,
                    confidence=AIConfidence.LOW,
                    requires_clarification=True,
                    clarification_question=DEFAULT_AMBIGUOUS_QUESTION,
                ),
                model=policy_model_name(),
                usage=TokenUsage(),
                skipped_llm=False,
            )
        try:
            intent = normalize_intent(structured.result)
        except (ValidationError, ValueError) as exc:
            raise AIResponseValidationError(
                raw_content=structured.result.model_dump_json()
            ) from exc
        override = classify_unsupported(message)
        if override is not None:
            intent = unsupported_intent(override)
        return IntentDetectionResult(
            intent=intent,
            model=structured.model,
            usage=structured.usage,
            skipped_llm=False,
        )


def build_intent_prompt_messages(message: str, context: AIContext) -> list[LLMMessage]:
    user_prompt = _build_user_prompt(message, context)
    return [
        LLMMessage(role="system", content=INTENT_SYSTEM_PROMPT_V1),
        LLMMessage(role="user", content=user_prompt),
    ]


async def build_intent_prompt_messages_with_memory(
    message: str,
    context: AIContext,
    *,
    conversation: ConversationContextData,
) -> list[LLMMessage]:
    """Build intent-detection messages including trimmed prior conversation turns."""
    from app.ai.memory.context import build_llm_context
    from app.ai.state.limits import ContextLimits

    user_prompt = _build_user_prompt(message, context)
    reserved = len(INTENT_SYSTEM_PROMPT_V1) + len(user_prompt)
    limits = ContextLimits.for_llm()

    prior = list(conversation.messages)
    stripped = message.strip()
    if prior and prior[-1].role == "user" and prior[-1].content == stripped:
        prior = prior[:-1]

    messages: list[LLMMessage] = [
        LLMMessage(role="system", content=INTENT_SYSTEM_PROMPT_V1),
    ]
    if prior:
        llm_context = await build_llm_context(
            ConversationContextData(messages=prior),
            limits=limits,
            reserved_chars=reserved,
        )
        messages.extend(
            LLMMessage(role=message.role, content=message.content)
            for message in llm_context.messages
        )
    messages.append(LLMMessage(role="user", content=user_prompt))
    return messages


def _build_user_prompt(message: str, context: AIContext) -> str:
    from app.ai.prompt.render import render_template_text

    metadata_block = _format_metadata(context.metadata)
    capabilities = ", ".join(sorted(context.allowed_capabilities)) or "none"
    variables = {
        "workspace_name": context.workspace_name or "",
        "data_source_name": context.data_source_name or "",
        "data_source_type": context.data_source_type or "",
        "capabilities": capabilities,
        "metadata": metadata_block,
        "message": message.strip(),
    }
    user_prompt = render_template_text(USER_PROMPT_TEMPLATE_V1, variables)
    max_context = settings.AI_LLM_CONTEXT_CHARS
    if len(user_prompt) > max_context:
        overflow = len(user_prompt) - max_context
        if len(metadata_block) > overflow:
            metadata_block = metadata_block[: max(0, len(metadata_block) - overflow)]
            variables["metadata"] = metadata_block
            user_prompt = render_template_text(USER_PROMPT_TEMPLATE_V1, variables)
        user_prompt = user_prompt[:max_context]
    return user_prompt


def normalize_intent(detected: LLMIntentDetection) -> AIIntent:
    if len(detected.metrics) > settings.AI_MAX_PLAN_METRICS:
        raise ValueError("Too many metrics")
    if len(detected.dimensions) > settings.AI_MAX_PLAN_DIMENSIONS:
        raise ValueError("Too many dimensions")
    if len(detected.filters) > settings.AI_MAX_PLAN_FILTERS:
        raise ValueError("Too many filters")
    for item in detected.filters:
        if item.values is not None and len(item.values) > settings.AI_MAX_FILTER_VALUES:
            raise ValueError("Too many filter values")

    requested_limit = detected.requested_limit
    safe_limit: int | None = None
    exceeds_limit = False
    if requested_limit is not None:
        safe_limit = min(requested_limit, settings.AI_MAX_RESULT_LIMIT)
        exceeds_limit = requested_limit > settings.AI_MAX_RESULT_LIMIT

    intent_type = detected.intent
    if intent_type is AIIntentType.UNSUPPORTED:
        return unsupported_intent(detected.unsupported_reason or "unsupported_request")

    requires_clarification = detected.requires_clarification
    clarification = detected.clarification_question
    confidence = detected.confidence
    if _missing_analytical_detail(detected):
        requires_clarification = True
        if confidence is AIConfidence.HIGH:
            confidence = AIConfidence.LOW
        if not clarification:
            clarification = _clarification_for(detected)
    if intent_type is AIIntentType.UNKNOWN:
        requires_clarification = True
        if not clarification:
            clarification = _clarification_for(detected)
        if confidence is AIConfidence.HIGH:
            confidence = AIConfidence.LOW

    return AIIntent(
        intent=intent_type,
        operation=detected.operation,
        subject=detected.subject,
        metrics=list(detected.metrics),
        dimensions=list(detected.dimensions),
        filters=list(detected.filters),
        time_range=detected.time_range,
        sort=detected.sort,
        requested_limit=requested_limit,
        safe_limit=safe_limit,
        exceeds_limit=exceeds_limit,
        requires_data_access=detected.requires_data_access,
        requires_metadata=detected.requires_metadata,
        requires_relationships=detected.requires_relationships,
        confidence=confidence,
        requires_clarification=requires_clarification,
        clarification_question=clarification,
        unsupported_reason=detected.unsupported_reason,
    )


def _missing_analytical_detail(detected: LLMIntentDetection) -> bool:
    if detected.intent in {
        AIIntentType.UNSUPPORTED,
        AIIntentType.SCHEMA_QUESTION,
        AIIntentType.UNKNOWN,
    }:
        return False
    if detected.operation in _COUNT_LIKE:
        return not detected.subject and not detected.metrics
    if detected.intent is AIIntentType.DATA_LOOKUP:
        return not detected.subject
    if detected.intent in _ANALYTICAL_INTENTS:
        if detected.operation is AIOperationType.COUNT:
            return not detected.subject and not detected.metrics
        return not detected.metrics
    return False


def _clarification_for(detected: LLMIntentDetection) -> str:
    subject = detected.subject or "this"
    if detected.intent is AIIntentType.UNKNOWN or not detected.metrics:
        return (
            f"What would you like to know about {subject} — total revenue, "
            "number of orders, or a breakdown by category?"
        )
    return DEFAULT_AMBIGUOUS_QUESTION


def _format_metadata(snippets: Sequence[MetadataSnippet]) -> str:
    if not snippets:
        return "(none)"
    return "\n".join(f"- {snippet.display_name()}" for snippet in snippets)
