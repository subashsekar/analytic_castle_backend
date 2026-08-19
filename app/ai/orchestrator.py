from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from app.ai.context import MetadataContextProvider
from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIRequestValidationError,
)
from app.ai.intent import AIIntentService, build_intent_prompt_messages
from app.ai.intent_types import AIIntent, AIIntentType, AIRequestPlan
from app.ai.metadata_resolver import MetadataContextResolver
from app.ai.metadata_types import ResolvedMetadataContext, empty_resolved_context
from app.ai.planner import AIRequestPlanner, foundation_response
from app.ai.prompts import PROMPT_VERSION
from app.ai.providers.base import LLMMessage, LLMProvider
from app.ai.types import (
    AI_CAPABILITY_CHAT,
    AIAnalysisResult,
    AIContext,
    AIRequest,
    AIResponse,
    MetadataSnippet,
)
from app.core.config import settings
from app.core.request_id import get_request_id, new_request_id

logger = logging.getLogger(__name__)


class AIAnalystOrchestrator:
    """Validate context, detect intent, and return a structured plan."""

    def __init__(
        self,
        provider: LLMProvider,
        metadata: MetadataContextProvider | None = None,
        intent_service: AIIntentService | None = None,
        planner: AIRequestPlanner | None = None,
        resolver: MetadataContextResolver | None = None,
    ) -> None:
        self._provider = provider
        self._metadata = metadata
        self._intent = intent_service or AIIntentService(provider)
        self._planner = planner or AIRequestPlanner()
        self._resolver = resolver

    async def chat(self, request: AIRequest, context: AIContext) -> AIResponse:
        started = time.perf_counter()
        request_id = request.request_id or get_request_id()
        if not request_id or request_id == "-":
            request_id = new_request_id()
        self._validate(request, context)
        metadata = await self._load_metadata(request, context)
        prompt_context = AIContext(
            user_id=context.user_id,
            workspace_id=context.workspace_id,
            organization_id=context.organization_id,
            data_source_id=context.data_source_id,
            data_source_name=context.data_source_name,
            data_source_type=context.data_source_type,
            workspace_name=context.workspace_name,
            workspace_role=context.workspace_role,
            allowed_capabilities=context.allowed_capabilities,
            metadata=metadata,
            conversation_id=request.conversation_id or context.conversation_id,
        )
        try:
            detection = await self._intent.detect(request.message, prompt_context)
        except AIError as exc:
            self._log_failure(request_id, context, started, exc)
            raise
        plan = self._planner.plan(detection.intent)
        metadata_context = self._resolve_metadata(detection.intent, prompt_context)
        plan = _merge_metadata_plan(plan, metadata_context)
        answer = _bound_answer(foundation_response(detection.intent, plan))
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "AI chat completed request_id=%s data_source_id=%s provider=%s "
            "model=%s prompt_version=%s input_tokens=%s output_tokens=%s "
            "total_tokens=%s table_count=%s column_count=%s "
            "duration_ms=%.0f success=true",
            request_id,
            context.data_source_id,
            self._provider.name,
            detection.model,
            PROMPT_VERSION,
            detection.usage.input_tokens,
            detection.usage.output_tokens,
            detection.usage.total_tokens,
            len(metadata_context.tables),
            len(metadata_context.columns),
            duration_ms,
        )
        return AIResponse(
            response=answer,
            model=detection.model,
            request_id=request_id,
            usage=detection.usage,
            analysis=_analysis_from_intent(
                detection.intent, answer, metadata, metadata_context
            ),
            intent=detection.intent,
            plan=plan,
            metadata_context=metadata_context,
        )

    def _log_failure(
        self,
        request_id: str,
        context: AIContext,
        started: float,
        exc: Exception,
    ) -> None:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.warning(
            "AI chat failed request_id=%s data_source_id=%s provider=%s "
            "duration_ms=%.0f success=false error_type=%s",
            request_id,
            context.data_source_id,
            self._provider.name,
            duration_ms,
            type(exc).__name__,
        )

    def _validate(self, request: AIRequest, context: AIContext) -> None:
        message = request.message.strip()
        if not message:
            raise AIRequestValidationError("Message is required")
        max_message = settings.AI_MAX_MESSAGE_CHARS
        if len(message) > max_message:
            raise AIRequestValidationError(
                f"Message exceeds maximum length of {max_message} characters"
            )
        if request.data_source_id != context.data_source_id:
            raise AIContextError("AI request context does not match the data source")
        if context.workspace_id is None or context.user_id is None:
            raise AIContextError()
        if AI_CAPABILITY_CHAT not in context.allowed_capabilities:
            raise AIContextError("AI chat is not an allowed capability")

    async def _load_metadata(
        self,
        request: AIRequest,
        context: AIContext,
    ) -> tuple[MetadataSnippet, ...]:
        if self._metadata is None:
            return context.metadata
        try:
            return self._metadata.get_relevant_metadata(
                data_source_id=context.data_source_id,
                workspace_id=context.workspace_id,
                query=request.message,
            )
        except Exception as exc:
            logger.warning(
                "AI metadata retrieval failed data_source_id=%s error_type=%s",
                context.data_source_id,
                type(exc).__name__,
            )
            return ()

    def _resolve_metadata(
        self,
        intent: AIIntent,
        context: AIContext,
    ) -> ResolvedMetadataContext:
        if self._resolver is None:
            return empty_resolved_context(context.data_source_id)
        return self._resolver.resolve(intent, context)


def build_prompt_messages(message: str, context: AIContext) -> list[LLMMessage]:
    return build_intent_prompt_messages(message, context)


def _bound_answer(answer: str) -> str:
    stripped = answer.strip()
    max_output = settings.AI_MAX_OUTPUT_CHARS
    if len(stripped) > max_output:
        return stripped[:max_output]
    return stripped


def _merge_metadata_plan(
    plan: AIRequestPlan,
    metadata_context: ResolvedMetadataContext,
) -> AIRequestPlan:
    if plan.unsupported or plan.requires_clarification:
        return plan
    if not metadata_context.requires_clarification:
        return plan
    return plan.model_copy(
        update={
            "requires_clarification": True,
            "clarification_question": metadata_context.clarification_question,
        }
    )


def _analysis_from_intent(
    intent: AIIntent,
    answer: str,
    metadata: Sequence[MetadataSnippet],
    metadata_context: ResolvedMetadataContext,
) -> AIAnalysisResult:
    warnings: list[str] = []
    if intent.requires_clarification or metadata_context.requires_clarification:
        warnings.append("Clarification is required before this request can be planned.")
    if intent.exceeds_limit:
        warnings.append("The requested row limit exceeds the configured safe limit.")
    if intent.intent is AIIntentType.UNSUPPORTED:
        warnings.append("The request is outside the read-only analyst scope.")
    names = [
        f"{item.schema_name}.{item.table_name}" for item in metadata_context.tables
    ]
    if not names:
        names = [item.display_name() for item in metadata]
    return AIAnalysisResult(
        answer=answer,
        intent=intent.intent.value,
        requires_data_access=intent.requires_data_access,
        metadata_context=names[:20],
        warnings=warnings[:20],
    )
