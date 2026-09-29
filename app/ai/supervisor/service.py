"""Supervisor agent workflow control and routing."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings
from app.ai.prompt import PromptRegistry
from app.ai.safety import classify_unsupported
from app.ai.state import (
    AgentStateService,
    ConversationMessage,
    MergePayloadTransition,
    SetPhaseTransition,
)
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.state.transitions import RESUMABLE_MESSAGE_PHASES
from app.ai.supervisor.classification import classify_request
from app.ai.supervisor.errors import (
    InvalidRoutingError,
    SupervisorAuthorizationError,
    SupervisorConfigurationError,
)
from app.ai.supervisor.logging_helpers import decision_log_context
from app.ai.supervisor.models import (
    ClassificationConfidence,
    RequestCategory,
    RequestClassification,
    SuperviseAdvanceParams,
    SuperviseMessageParams,
    SupervisorAction,
    SupervisorDecision,
    SupervisorResult,
)
from app.ai.supervisor.prompts import build_supervisor_prompt_registry
from app.ai.supervisor.routing import resolve_advance_routing
from app.ai.supervisor.validation import (
    build_decision,
    classification_from_policy,
    ensure_routing_allowed,
)
from app.db.models import DataSource
from app.enums import AgentPhase

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Classify requests, route workflow actions, and manage agent-state transitions.

    The supervisor never executes MCP tools. It only decides the next action and
    updates persisted session state through ``AgentStateService``.
    """

    def __init__(
        self,
        session: Session,
        *,
        llm_client: AsyncLLMClient | None = None,
        llm_config: LLMClientConfig | None = None,
        prompt_registry: PromptRegistry | None = None,
        state_service: AgentStateService | None = None,
    ) -> None:
        self._session = session
        self._state_service = state_service or AgentStateService(session)
        self._prompt_registry = prompt_registry or build_supervisor_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise SupervisorConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def handle_message(self, params: SuperviseMessageParams) -> SupervisorResult:
        snapshot = self._state_service.get_session(
            params.session_id,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
        )
        self._ensure_data_source_access(snapshot, params.workspace_id)

        current_phase = self._require_phase(snapshot)
        if current_phase not in RESUMABLE_MESSAGE_PHASES:
            raise InvalidRoutingError(
                f"Supervisor cannot accept new user messages in phase "
                f"{current_phase.value}"
            )

        agent_version = params.expected_agent_version
        if current_phase is not AgentPhase.INITIAL:
            # Reset waiting/planned turns to INITIAL before reclassification.
            snapshot = self._state_service.update_agent_state(
                params.session_id,
                SetPhaseTransition(phase=AgentPhase.INITIAL),
                workspace_id=params.workspace_id,
                user_id=params.user_id,
                expected_version=agent_version,
            )
            agent_version = (
                snapshot.agent_state.version
                if snapshot.agent_state is not None
                else agent_version
            )

        snapshot = self._state_service.append_message(
            params.session_id,
            ConversationMessage(role="user", content=params.message),
            workspace_id=params.workspace_id,
            user_id=params.user_id,
            expected_context_version=params.expected_context_version,
        )
        current_phase = self._require_phase(snapshot)
        if current_phase is not AgentPhase.INITIAL:
            raise InvalidRoutingError(
                f"Supervisor classification requires INITIAL phase, "
                f"got {current_phase.value}"
            )

        policy_reason = classify_unsupported(params.message)
        if policy_reason is not None:
            classification = classification_from_policy(policy_reason)
        else:
            classification = await classify_request(
                client=self._llm_client,
                registry=self._prompt_registry,
                message=params.message,
                current_phase=current_phase,
                has_data_source=snapshot.data_source_id is not None,
            )

        decision = build_decision(
            classification=classification,
            current_phase=current_phase,
            has_data_source=snapshot.data_source_id is not None,
        )
        updated = self._apply_decision(
            snapshot=snapshot,
            decision=decision,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
            expected_agent_version=agent_version,
        )
        logger.info(
            "supervisor handled message",
            extra=decision_log_context(decision),
        )
        return SupervisorResult(decision=decision, session=updated)

    async def advance_workflow(
        self, params: SuperviseAdvanceParams
    ) -> SupervisorResult:
        snapshot = self._state_service.get_session(
            params.session_id,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
        )
        self._ensure_data_source_access(snapshot, params.workspace_id)
        current_phase = self._require_phase(snapshot)

        action, next_phase, message = resolve_advance_routing(current_phase)
        ensure_routing_allowed(
            action=action,
            current_phase=current_phase,
            next_phase=next_phase,
        )
        decision = SupervisorDecision(
            action=action,
            classification=RequestClassification(
                category=RequestCategory.GENERAL,
                confidence=ClassificationConfidence.HIGH,
                source="workflow",
            ),
            current_phase=current_phase,
            next_phase=next_phase,
            message=message,
        )
        updated = self._apply_decision(
            snapshot=snapshot,
            decision=decision,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
            expected_agent_version=params.expected_agent_version,
        )
        logger.info(
            "supervisor advanced workflow",
            extra=decision_log_context(decision),
        )
        return SupervisorResult(decision=decision, session=updated)

    def _apply_decision(
        self,
        *,
        snapshot: AnalysisSessionSnapshot,
        decision: SupervisorDecision,
        workspace_id: UUID,
        user_id: UUID,
        expected_agent_version: int,
    ) -> AnalysisSessionSnapshot:
        current = snapshot.agent_state
        if current is None:
            raise SupervisorAuthorizationError("Session agent state is missing")

        agent_version = expected_agent_version
        updated = snapshot

        active_state = current
        payload_custom = dict(active_state.payload.custom)
        payload_custom["supervisor_action"] = decision.action.value
        payload_custom["request_category"] = decision.classification.category.value
        if decision.classification.reason:
            payload_custom["unsupported_reason"] = decision.classification.reason
        if decision.message:
            payload_custom["supervisor_message"] = decision.message

        updated = self._state_service.update_agent_state(
            snapshot.session_id,
            MergePayloadTransition(updates={"custom": payload_custom}),
            workspace_id=workspace_id,
            user_id=user_id,
            expected_version=agent_version,
        )
        agent_version = (
            updated.agent_state.version if updated.agent_state else agent_version
        )

        if decision.next_phase is not None:
            updated = self._state_service.update_agent_state(
                snapshot.session_id,
                SetPhaseTransition(phase=decision.next_phase),
                workspace_id=workspace_id,
                user_id=user_id,
                expected_version=agent_version,
            )
            agent_version = (
                updated.agent_state.version if updated.agent_state else agent_version
            )

        if (
            decision.action is SupervisorAction.RESPOND_UNSUPPORTED
            or decision.action is SupervisorAction.COMPLETE
        ):
            updated = self._state_service.complete_session(
                snapshot.session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                expected_agent_version=agent_version,
            )
        elif decision.action is SupervisorAction.FAIL:
            updated = self._state_service.fail_session(
                snapshot.session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                expected_agent_version=agent_version,
            )

        return updated

    @staticmethod
    def _require_phase(snapshot: AnalysisSessionSnapshot) -> AgentPhase:
        if snapshot.agent_state is None:
            raise SupervisorAuthorizationError("Session agent state is missing")
        return snapshot.agent_state.phase

    def _ensure_data_source_access(
        self,
        snapshot: AnalysisSessionSnapshot,
        workspace_id: UUID,
    ) -> None:
        if snapshot.data_source_id is None:
            return
        data_source = self._session.get(DataSource, snapshot.data_source_id)
        if data_source is None or data_source.workspace_id != workspace_id:
            raise SupervisorAuthorizationError("Data source is not accessible")
