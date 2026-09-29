from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.analysis_pipeline import (
    AnalysisPipelineParams,
    run_analysis_pipeline,
    should_run_analysis,
)
from app.ai.analysis_profile import build_conversation_context
from app.ai.context import MetadataContextProvider
from app.ai.exceptions import (
    AIContextError,
    AIError,
    AIRequestValidationError,
)
from app.ai.intent import AIIntentService, build_intent_prompt_messages
from app.ai.intent_types import (
    AIConfidence,
    AIIntent,
    AIIntentType,
    AIRequestPlan,
)
from app.ai.memory import ConversationMemoryService
from app.ai.metadata_resolver import MetadataContextResolver
from app.ai.metadata_types import ResolvedMetadataContext, empty_resolved_context
from app.ai.planner import AIRequestPlanner, foundation_response
from app.ai.planner_agent import (
    PlannerAgent,
    advance_with_planning,
    apply_planner_intent,
)
from app.ai.planner_agent.errors_mapping import map_planner_error
from app.ai.prompts import PROMPT_VERSION
from app.ai.providers.base import LLMMessage, LLMProvider
from app.ai.safety import unsupported_intent
from app.ai.state import (
    AgentStateService,
    ConversationMessage,
    CreateSessionParams,
    MergePayloadTransition,
    SetPhaseTransition,
)
from app.ai.supervisor import (
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
    SupervisorAgent,
)
from app.ai.supervisor.errors_mapping import map_supervisor_error
from app.ai.types import (
    AI_CAPABILITY_CHAT,
    AIAnalysisResult,
    AIContext,
    AIRequest,
    AIResponse,
    MetadataSnippet,
    TokenUsage,
)
from app.core.config import settings
from app.core.request_id import get_request_id, new_request_id
from app.enums import AgentPhase

logger = logging.getLogger(__name__)


class AIAnalystOrchestrator:
    """Validate context, detect intent, and return a structured plan."""

    def __init__(
        self,
        provider: LLMProvider,
        metadata: MetadataContextProvider | None = None,
        intent_service: AIIntentService | None = None,
        planner: AIRequestPlanner | None = None,
        planner_agent: PlannerAgent | None = None,
        resolver: MetadataContextResolver | None = None,
        supervisor: SupervisorAgent | None = None,
    ) -> None:
        self._provider = provider
        self._metadata = metadata
        self._intent = intent_service or AIIntentService(provider)
        self._planner = planner or AIRequestPlanner()
        self._planner_agent = planner_agent
        self._resolver = resolver
        self._supervisor = supervisor

    async def chat(
        self,
        request: AIRequest,
        context: AIContext,
        *,
        db: Session | None = None,
    ) -> AIResponse:
        started = time.perf_counter()
        request_id = request.request_id or get_request_id()
        if not request_id or request_id == "-":
            request_id = new_request_id()
        self._validate(request, context)
        if db is not None:
            try:
                return await self._supervised_chat(
                    request,
                    context,
                    db,
                    request_id=request_id,
                    started=started,
                )
            except AIError:
                raise
            except Exception as exc:
                from app.ai.planner_agent.errors import PlannerError
                from app.ai.supervisor.errors import SupervisorError

                if isinstance(exc, SupervisorError):
                    raise map_supervisor_error(exc) from exc
                if isinstance(exc, PlannerError):
                    raise map_planner_error(exc) from exc
                raise
        return await self._legacy_chat(
            request,
            context,
            request_id=request_id,
            started=started,
        )

    async def _legacy_chat(
        self,
        request: AIRequest,
        context: AIContext,
        *,
        request_id: str,
        started: float,
    ) -> AIResponse:
        metadata = await self._load_metadata(request, context)
        prompt_context = _prompt_context(context, metadata, request)
        try:
            detection = await self._intent.detect(request.message, prompt_context)
        except AIError as exc:
            self._log_failure(request_id, context, started, exc)
            raise
        plan = self._planner.plan(detection.intent)
        metadata_context = self._resolve_metadata(
            detection.intent,
            prompt_context,
            message=request.message,
        )
        plan = _merge_metadata_plan(plan, metadata_context)
        answer = _bound_answer(
            _answer_with_metadata(
                foundation_response(detection.intent, plan),
                metadata_context,
            )
        )
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

    async def _supervised_chat(
        self,
        request: AIRequest,
        context: AIContext,
        db: Session,
        *,
        request_id: str,
        started: float,
    ) -> AIResponse:
        if request.conversation_id is not None:
            return await self._resume_supervised_chat(
                request,
                context,
                db,
                request_id=request_id,
                started=started,
            )
        return await self._start_supervised_chat(
            request,
            context,
            db,
            request_id=request_id,
            started=started,
        )

    async def _start_supervised_chat(
        self,
        request: AIRequest,
        context: AIContext,
        db: Session,
        *,
        request_id: str,
        started: float,
    ) -> AIResponse:
        state_service = AgentStateService(db)
        memory_service = ConversationMemoryService(db)
        session = state_service.create_session(
            CreateSessionParams(
                user_id=_require_uuid(context.user_id, "user_id"),
                workspace_id=_require_uuid(context.workspace_id, "workspace_id"),
                organization_id=_require_uuid(
                    context.organization_id, "organization_id"
                ),
                data_source_id=context.data_source_id,
            )
        )
        supervisor = self._supervisor or SupervisorAgent(db)
        agent_state = session.agent_state
        if agent_state is None:
            raise AIContextError("Analysis session state is missing")

        try:
            supervised = await supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=session.session_id,
                    workspace_id=context.workspace_id,
                    user_id=context.user_id,
                    message=request.message,
                    expected_agent_version=agent_state.version,
                    expected_context_version=session.conversation_version or 1,
                )
            )
        except Exception as exc:
            from app.ai.supervisor.errors import SupervisorError

            if isinstance(exc, SupervisorError):
                raise map_supervisor_error(exc) from exc
            raise

        return await self._finish_supervised_chat(
            request,
            context,
            db,
            request_id=request_id,
            started=started,
            session=supervised.session,
            decision=supervised.decision,
            memory_service=memory_service,
        )

    async def _resume_supervised_chat(
        self,
        request: AIRequest,
        context: AIContext,
        db: Session,
        *,
        request_id: str,
        started: float,
    ) -> AIResponse:
        from app.ai.memory.errors import (
            ConversationNotActiveError,
            ConversationNotFoundError,
            StateVersionConflictError,
        )
        from app.ai.state.errors import (
            AnalysisSessionNotActiveError,
            AnalysisSessionNotFoundError,
        )
        from app.ai.state.errors import (
            StateVersionConflictError as AgentStateVersionConflictError,
        )
        from app.ai.supervisor.errors import InvalidRoutingError, SupervisorError

        memory_service = ConversationMemoryService(db)
        state_service = AgentStateService(db)
        user_uuid = _require_uuid(context.user_id, "user_id")
        workspace_uuid = _require_uuid(context.workspace_id, "workspace_id")
        conversation_id = request.conversation_id
        assert conversation_id is not None

        try:
            session = state_service.get_session(
                conversation_id,
                workspace_id=workspace_uuid,
                user_id=user_uuid,
            )
        except AnalysisSessionNotFoundError as exc:
            raise AIContextError("Conversation not found") from exc

        if session.data_source_id is not None and (
            session.data_source_id != context.data_source_id
        ):
            raise AIContextError("Conversation data source mismatch")
        agent_state = session.agent_state
        if agent_state is None:
            raise AIContextError("Analysis session state is missing")

        supervisor = self._supervisor or SupervisorAgent(db)
        try:
            supervised = await supervisor.handle_message(
                SuperviseMessageParams(
                    session_id=conversation_id,
                    workspace_id=workspace_uuid,
                    user_id=user_uuid,
                    message=request.message,
                    expected_agent_version=agent_state.version,
                    expected_context_version=request.conversation_version
                    or session.conversation_version
                    or 1,
                )
            )
        except (
            ConversationNotFoundError,
            AnalysisSessionNotFoundError,
        ) as exc:
            raise AIContextError("Conversation not found") from exc
        except (
            ConversationNotActiveError,
            AnalysisSessionNotActiveError,
        ) as exc:
            raise AIRequestValidationError("Conversation is not active") from exc
        except (
            StateVersionConflictError,
            AgentStateVersionConflictError,
        ) as exc:
            raise AIRequestValidationError("Conversation version conflict") from exc
        except InvalidRoutingError as exc:
            raise AIRequestValidationError(str(exc) or "Invalid resume state") from exc
        except SupervisorError as exc:
            raise map_supervisor_error(exc) from exc

        return await self._finish_supervised_chat(
            request,
            context,
            db,
            request_id=request_id,
            started=started,
            session=supervised.session,
            decision=supervised.decision,
            memory_service=memory_service,
        )

    async def _finish_supervised_chat(
        self,
        request: AIRequest,
        context: AIContext,
        db: Session,
        *,
        request_id: str,
        started: float,
        session,
        decision,
        memory_service: ConversationMemoryService,
    ) -> AIResponse:
        state_service = AgentStateService(db)

        if decision.action is SupervisorAction.RESPOND_UNSUPPORTED:
            response = self._response_from_supervisor_terminal(
                request_id=request_id,
                context=context,
                decision_message=decision.message,
                unsupported_reason=decision.classification.reason,
                started=started,
            )
            return self._persist_assistant_turn(
                memory_service,
                session=session,
                context=context,
                answer=response.response,
                response=response,
            )
        if decision.action is SupervisorAction.REQUEST_CLARIFICATION:
            response = self._response_from_supervisor_clarification(
                request_id=request_id,
                context=context,
                question=decision.message,
                started=started,
            )
            return self._persist_assistant_turn(
                memory_service,
                session=session,
                context=context,
                answer=response.response,
                response=response,
            )
        if decision.action is not SupervisorAction.RUN_INTENT_AGENT:
            raise AIRequestValidationError(
                f"Unexpected supervisor action: {decision.action.value}"
            )

        metadata = await self._load_metadata(request, context)
        prompt_context = _prompt_context(context, metadata, request)
        conversation_data = memory_service.get_context_data(
            session.session_id,
            workspace_id=_require_uuid(context.workspace_id, "workspace_id"),
            user_id=_require_uuid(context.user_id, "user_id"),
        )
        try:
            detection = await self._intent.detect(
                request.message,
                prompt_context,
                conversation=conversation_data,
            )
        except AIError as exc:
            self._log_failure(request_id, context, started, exc)
            raise

        updated_state = session.agent_state
        if updated_state is not None:
            session = state_service.update_agent_state(
                session.session_id,
                MergePayloadTransition(
                    updates={"intent": detection.intent.intent.value}
                ),
                workspace_id=context.workspace_id,
                user_id=context.user_id,
                expected_version=updated_state.version,
            )

        updated_state = session.agent_state
        if updated_state is None:
            raise AIContextError("Analysis session state is missing")
        supervisor = self._supervisor or SupervisorAgent(db)
        planner_agent = self._planner_agent or PlannerAgent(db)
        analysis_plan = None
        try:
            session, analysis_plan = await advance_with_planning(
                supervisor,
                planner_agent,
                session_id=session.session_id,
                workspace_id=context.workspace_id,
                user_id=context.user_id,
                expected_agent_version=updated_state.version,
                intent=detection.intent,
                message=request.message,
                data_source_name=context.data_source_name,
            )
        except Exception as exc:
            from app.ai.planner_agent.errors import PlannerError
            from app.ai.supervisor.errors import SupervisorError

            if isinstance(exc, SupervisorError):
                raise map_supervisor_error(exc) from exc
            if isinstance(exc, PlannerError):
                raise map_planner_error(exc) from exc
            raise

        plan = (
            analysis_plan.request_plan
            if analysis_plan is not None
            else self._planner.plan(detection.intent)
        )
        resolved_intent = (
            apply_planner_intent(detection.intent, analysis_plan)
            if analysis_plan is not None
            else detection.intent
        )
        metadata_context = self._resolve_metadata(
            resolved_intent,
            prompt_context,
            message=request.message,
        )
        # Soften clarification when catalog already matched tables/metrics.
        resolved_intent, plan, metadata_context = _apply_clarification_policy(
            resolved_intent, plan, metadata_context
        )
        plan = _merge_metadata_plan(plan, metadata_context)

        phase8_payload = None
        if should_run_analysis(
            plan=plan, intent=resolved_intent, metadata=metadata_context
        ):
            session, answer, phase8_payload, plan = await self._execute_analysis(
                request=request,
                context=context,
                db=db,
                session=session,
                state_service=state_service,
                supervisor=supervisor,
                resolved_intent=resolved_intent,
                plan=plan,
                metadata_context=metadata_context,
                conversation_context=build_conversation_context(
                    conversation_data, current_message=request.message
                ),
            )
        else:
            answer = _bound_answer(
                _answer_with_metadata(
                    foundation_response(resolved_intent, plan),
                    metadata_context,
                )
            )

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "AI supervised chat completed request_id=%s session_id=%s "
            "data_source_id=%s provider=%s model=%s prompt_version=%s "
            "supervisor_action=%s request_category=%s input_tokens=%s "
            "output_tokens=%s total_tokens=%s table_count=%s column_count=%s "
            "duration_ms=%.0f success=true",
            request_id,
            session.session_id,
            context.data_source_id,
            self._provider.name,
            detection.model,
            PROMPT_VERSION,
            decision.action.value,
            decision.classification.category.value,
            detection.usage.input_tokens,
            detection.usage.output_tokens,
            detection.usage.total_tokens,
            len(metadata_context.tables),
            len(metadata_context.columns),
            duration_ms,
        )
        response = AIResponse(
            response=answer,
            model=detection.model,
            request_id=request_id,
            usage=detection.usage,
            analysis=_analysis_from_intent(
                resolved_intent, answer, metadata, metadata_context
            ),
            intent=resolved_intent,
            plan=plan,
            metadata_context=metadata_context,
            phase8_analysis=phase8_payload,
        )
        return self._persist_assistant_turn(
            memory_service,
            session=session,
            context=context,
            answer=answer,
            response=response,
        )

    async def _execute_analysis(
        self,
        *,
        request: AIRequest,
        context: AIContext,
        db: Session,
        session,
        state_service: AgentStateService,
        supervisor: SupervisorAgent,
        resolved_intent: AIIntent,
        plan: AIRequestPlan,
        metadata_context: ResolvedMetadataContext,
        conversation_context: str | None = None,
    ):
        """Advance PLANNING→EXECUTING, run SQL+Phase8, return to PLANNING for multi-turn."""
        agent_state = session.agent_state
        if agent_state is None:
            raise AIContextError("Analysis session state is missing")
        agent_version = agent_state.version

        if agent_state.phase is AgentPhase.PLANNING:
            advanced = await supervisor.advance_workflow(
                SuperviseAdvanceParams(
                    session_id=session.session_id,
                    workspace_id=context.workspace_id,
                    user_id=context.user_id,
                    expected_agent_version=agent_version,
                )
            )
            session = advanced.session
            if session.agent_state is not None:
                agent_version = session.agent_state.version

        plan_summary = None
        if plan.operations:
            plan_summary = ", ".join(op.value for op in plan.operations[:8])

        pipeline = await run_analysis_pipeline(
            AnalysisPipelineParams(
                db=db,
                session_id=session.session_id,
                workspace_id=_require_uuid(context.workspace_id, "workspace_id"),
                organization_id=_require_uuid(
                    context.organization_id, "organization_id"
                ),
                user_id=_require_uuid(context.user_id, "user_id"),
                data_source_id=context.data_source_id,
                message=request.message,
                metadata=metadata_context,
                intent=resolved_intent,
                plan=plan,
                data_source_name=context.data_source_name,
                plan_summary=plan_summary,
                # Agents share one session and may run concurrently; skip
                # optimistic version writes so their updates cannot conflict.
                expected_agent_version=None,
                conversation_context=conversation_context,
            )
        )
        answer = _bound_answer(pipeline.answer)

        # Refresh session after pipeline (SQL/history may have committed side state)
        try:
            session = state_service.get_session(
                session.session_id,
                workspace_id=_require_uuid(context.workspace_id, "workspace_id"),
                user_id=_require_uuid(context.user_id, "user_id"),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "session refresh after analysis failed session_id=%s",
                session.session_id,
            )

        # EXECUTING → RESPONDING → PLANNING (stay ACTIVE for multi-turn resume)
        if session.agent_state is not None and session.agent_state.phase is AgentPhase.EXECUTING:
            try:
                advanced = await supervisor.advance_workflow(
                    SuperviseAdvanceParams(
                        session_id=session.session_id,
                        workspace_id=context.workspace_id,
                        user_id=context.user_id,
                        expected_agent_version=session.agent_state.version,
                    )
                )
                session = advanced.session
            except Exception:  # noqa: BLE001 — phase bookkeeping must not fail the answer
                logger.warning(
                    "phase advance EXECUTING→RESPONDING failed session_id=%s",
                    session.session_id,
                )

        if session.agent_state is not None and session.agent_state.phase is AgentPhase.RESPONDING:
            try:
                session = state_service.update_agent_state(
                    session.session_id,
                    SetPhaseTransition(phase=AgentPhase.PLANNING),
                    workspace_id=context.workspace_id,
                    user_id=context.user_id,
                    expected_version=session.agent_state.version,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "phase reset RESPONDING→PLANNING failed session_id=%s",
                    session.session_id,
                )

        if pipeline.requires_clarification and pipeline.clarification_question:
            plan = plan.model_copy(
                update={
                    "requires_clarification": True,
                    "clarification_question": pipeline.clarification_question,
                }
            )

        return session, answer, pipeline.payload, plan

    def _persist_assistant_turn(
        self,
        memory_service: ConversationMemoryService,
        *,
        session,
        context: AIContext,
        answer: str,
        response: AIResponse,
    ) -> AIResponse:
        from dataclasses import replace

        from app.enums import AnalysisSessionStatus

        allow_inactive = session.status != AnalysisSessionStatus.ACTIVE.value
        try:
            updated = memory_service.append_message(
                session.session_id,
                ConversationMessage(role="assistant", content=answer),
                workspace_id=_require_uuid(context.workspace_id, "workspace_id"),
                user_id=_require_uuid(context.user_id, "user_id"),
                expected_context_version=session.conversation_version or 1,
                trim_if_needed=True,
                allow_inactive=allow_inactive,
            )
        except Exception:  # noqa: BLE001 — best-effort persistence must not fail the chat
            logger.warning(
                "assistant message persistence failed session_id=%s",
                session.session_id,
            )
            return replace(
                response,
                conversation_id=session.session_id,
                conversation_version=session.conversation_version,
            )
        return replace(
            response,
            conversation_id=updated.conversation_id,
            conversation_version=updated.conversation_version,
        )

    def _response_from_supervisor_terminal(
        self,
        *,
        request_id: str,
        context: AIContext,
        decision_message: str | None,
        unsupported_reason: str | None,
        started: float,
    ) -> AIResponse:
        intent = unsupported_intent(unsupported_reason or "unsupported_request")
        plan = self._planner.plan(intent)
        answer = _bound_answer(decision_message or foundation_response(intent, plan))
        metadata_context = empty_resolved_context(context.data_source_id)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "AI supervised chat completed request_id=%s data_source_id=%s "
            "provider=%s supervisor_action=%s duration_ms=%.0f success=true",
            request_id,
            context.data_source_id,
            self._provider.name,
            SupervisorAction.RESPOND_UNSUPPORTED.value,
            duration_ms,
        )
        return AIResponse(
            response=answer,
            model="policy",
            request_id=request_id,
            usage=TokenUsage(),
            analysis=_analysis_from_intent(intent, answer, (), metadata_context),
            intent=intent,
            plan=plan,
            metadata_context=metadata_context,
        )

    def _response_from_supervisor_clarification(
        self,
        *,
        request_id: str,
        context: AIContext,
        question: str | None,
        started: float,
    ) -> AIResponse:
        clarification = question or "Could you clarify what you would like to analyze?"
        intent = AIIntent(
            intent=AIIntentType.UNKNOWN,
            confidence=AIConfidence.MEDIUM,
            requires_clarification=True,
            clarification_question=clarification,
        )
        plan = self._planner.plan(intent)
        answer = _bound_answer(foundation_response(intent, plan))
        metadata_context = empty_resolved_context(context.data_source_id)
        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "AI supervised chat completed request_id=%s data_source_id=%s "
            "provider=%s supervisor_action=%s duration_ms=%.0f success=true",
            request_id,
            context.data_source_id,
            self._provider.name,
            SupervisorAction.REQUEST_CLARIFICATION.value,
            duration_ms,
        )
        return AIResponse(
            response=answer,
            model="policy",
            request_id=request_id,
            usage=TokenUsage(),
            analysis=_analysis_from_intent(intent, answer, (), metadata_context),
            intent=intent,
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
        detail = str(exc)[:240]
        raw_chars = 0
        if isinstance(exc, AIError) and getattr(exc, "raw_content", None):
            raw_chars = len(exc.raw_content or "")
        logger.warning(
            "AI chat failed request_id=%s data_source_id=%s provider=%s "
            "duration_ms=%.0f success=false error_type=%s detail=%s "
            "raw_content_chars=%s",
            request_id,
            context.data_source_id,
            self._provider.name,
            duration_ms,
            type(exc).__name__,
            detail,
            raw_chars,
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
        except Exception as exc:  # noqa: BLE001 — metadata lookup is best-effort
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
        *,
        message: str | None = None,
    ) -> ResolvedMetadataContext:
        if self._resolver is None:
            return empty_resolved_context(context.data_source_id)
        return self._resolver.resolve(intent, context, message=message)


def build_prompt_messages(message: str, context: AIContext) -> list[LLMMessage]:
    return build_intent_prompt_messages(message, context)


def _prompt_context(
    context: AIContext,
    metadata: tuple[MetadataSnippet, ...],
    request: AIRequest,
) -> AIContext:
    return AIContext(
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


def _require_uuid(value: UUID | None, field_name: str) -> UUID:
    if value is None:
        raise AIContextError(f"Missing required context field: {field_name}")
    return value


def _bound_answer(answer: str) -> str:
    stripped = answer.strip()
    max_output = settings.AI_MAX_OUTPUT_CHARS
    if len(stripped) > max_output:
        return stripped[:max_output]
    return stripped


def _answer_with_metadata(
    answer: str,
    metadata_context: ResolvedMetadataContext,
) -> str:
    """Append compact catalog facts so schema answers are useful after sync."""
    parts = [answer.strip()]
    if metadata_context.tables:
        table_names = ", ".join(
            f"{item.schema_name}.{item.table_name}"
            for item in metadata_context.tables[:10]
        )
        parts.append(f"Relevant tables: {table_names}.")
    if metadata_context.columns:
        column_names = ", ".join(
            f"{item.schema_name}.{item.table_name}.{item.column_name}"
            for item in metadata_context.columns[:20]
        )
        parts.append(f"Relevant columns: {column_names}.")
    elif metadata_context.unresolved_concepts:
        unresolved = ", ".join(metadata_context.unresolved_concepts[:8])
        parts.append(f"I could not find metadata for: {unresolved}.")
    return " ".join(part for part in parts if part)


def _merge_metadata_plan(
    plan: AIRequestPlan,
    metadata_context: ResolvedMetadataContext,
) -> AIRequestPlan:
    if plan.unsupported or plan.requires_clarification:
        return plan
    if not metadata_context.requires_clarification:
        return plan
    from app.ai.analysis_pipeline import has_sufficient_metadata

    # Already matched tables/metrics — do not block SQL on residual unresolved labels.
    if has_sufficient_metadata(metadata_context):
        return plan
    return plan.model_copy(
        update={
            "requires_clarification": True,
            "clarification_question": metadata_context.clarification_question,
        }
    )


def _apply_clarification_policy(
    intent: AIIntent,
    plan: AIRequestPlan,
    metadata_context: ResolvedMetadataContext,
) -> tuple[AIIntent, AIRequestPlan, ResolvedMetadataContext]:
    """Clear soft clarification when catalog already matched tables/columns.

    The planner often invents column-name questions (revenue vs amount) even after
    sync resolved public.sales.revenue. Override those so SQL can run.
    """
    from app.ai.analysis_pipeline import has_sufficient_metadata

    if plan.unsupported or not has_sufficient_metadata(metadata_context):
        return intent, plan, metadata_context

    question = _genuine_clarification(intent, plan, metadata_context)
    if question is not None:
        plan = plan.model_copy(
            update={"requires_clarification": True, "clarification_question": question}
        )
        return intent, plan, metadata_context

    if intent.requires_clarification:
        intent = intent.model_copy(
            update={"requires_clarification": False, "clarification_question": None}
        )
    if metadata_context.requires_clarification:
        metadata_context = metadata_context.model_copy(
            update={"requires_clarification": False, "clarification_question": None}
        )
    if plan.requires_clarification or not plan.requires_database:
        # Clarification-only planner stubs omit requires_database — rebuild.
        rebuilt = AIRequestPlanner().plan(intent)
        if plan.operations:
            rebuilt = rebuilt.model_copy(
                update={
                    "operations": list(plan.operations),
                    "required_capabilities": list(plan.required_capabilities)
                    or list(rebuilt.required_capabilities),
                }
            )
        plan = rebuilt
    return intent, plan, metadata_context


def _genuine_clarification(
    intent: AIIntent,
    plan: AIRequestPlan,
    metadata_context: ResolvedMetadataContext,
) -> str | None:
    """Clarification that must be kept even when the catalog matched something.

    Soft planner questions are cleared elsewhere; these cases cannot be answered
    faithfully without the user: no identifiable metric/subject, a requested
    metric with no catalog match, or a metric the model itself flagged as
    ambiguous between several catalog columns.
    """
    existing = (
        plan.clarification_question
        or intent.clarification_question
        or metadata_context.clarification_question
    )
    if (
        intent.intent is AIIntentType.UNKNOWN
        and not intent.metrics
        and not intent.dimensions
        and not intent.subject
    ):
        return existing or (
            "What would you like to know — for example a total, a trend over time, "
            "a comparison, or a ranking — and for which metric?"
        )
    metrics = metadata_context.resolved_metrics
    if metrics and not any(item.resolved for item in metrics):
        missing = ", ".join(item.requested for item in metrics[:5])
        return (
            f"I could not find a field for {missing} in this data source. "
            "Which available field should I use?"
        )
    asked = intent.requires_clarification or plan.requires_clarification
    ambiguous = [item for item in metrics if item.ambiguous and len(item.candidates) > 1]
    if asked and ambiguous:
        item = ambiguous[0]
        options = ", ".join(
            f"{c.table_name}.{c.column_name}" for c in item.candidates[:4]
        )
        return f"'{item.requested}' matches several fields ({options}). Which one should I use?"
    return None


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
