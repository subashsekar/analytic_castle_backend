"""Planner agent service for analysis planning and state integration."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.orm import Session

from app.ai.intent_types import AIIntentType
from app.ai.llm import AsyncLLMClient, LLMClientConfig, llm_client_config_from_settings
from app.ai.planner_agent.errors import (
    PlannerAuthorizationError,
    PlannerConfigurationError,
)
from app.ai.planner_agent.logging_helpers import plan_log_context
from app.ai.planner_agent.models import AnalysisPlan, PlanCreateParams, PlannerResult
from app.ai.planner_agent.planning import create_analysis_plan
from app.ai.planner_agent.prompts import build_planner_prompt_registry
from app.ai.planner_agent.serialization import (
    metadata_refs_from_plan,
    plan_custom_entries,
)
from app.ai.planner_agent.validation import plan_from_intent
from app.ai.prompt import PromptRegistry
from app.ai.state import AgentStateService, MergePayloadTransition
from app.ai.state.models import AnalysisSessionSnapshot
from app.db.models import DataSource
from app.enums import AgentPhase

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Produce structured analysis plans without executing MCP tools."""

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
        self._prompt_registry = prompt_registry or build_planner_prompt_registry()
        if llm_client is not None:
            self._llm_client = llm_client
        else:
            config = llm_config or llm_client_config_from_settings()
            if not config.api_key:
                raise PlannerConfigurationError("LLM API key is not configured")
            self._llm_client = AsyncLLMClient(config)

    async def create_plan(self, params: PlanCreateParams) -> PlannerResult:
        snapshot = self._state_service.get_session(
            params.session_id,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
        )
        self._ensure_data_source_access(snapshot, params.workspace_id)
        current_phase = self._require_phase(snapshot)
        if current_phase is not AgentPhase.PLANNING:
            raise PlannerAuthorizationError(
                f"Planner only accepts sessions in PLANNING phase, got "
                f"{current_phase.value}"
            )

        if (
            params.intent.intent is AIIntentType.UNSUPPORTED
            or params.intent.requires_clarification
            or params.intent.intent is AIIntentType.UNKNOWN
        ):
            plan = plan_from_intent(params.intent)
        else:
            plan = await create_analysis_plan(
                client=self._llm_client,
                registry=self._prompt_registry,
                intent=params.intent,
                message=params.message,
                has_data_source=snapshot.data_source_id is not None,
                data_source_name=params.data_source_name,
            )

        updated = self._persist_plan(
            snapshot=snapshot,
            plan=plan,
            workspace_id=params.workspace_id,
            user_id=params.user_id,
            expected_agent_version=params.expected_agent_version,
        )
        logger.info("planner created analysis plan", extra=plan_log_context(plan))
        return PlannerResult(plan=plan, session=updated)

    def _persist_plan(
        self,
        *,
        snapshot: AnalysisSessionSnapshot,
        plan: AnalysisPlan,
        workspace_id: UUID,
        user_id: UUID,
        expected_agent_version: int,
    ) -> AnalysisSessionSnapshot:
        current = snapshot.agent_state
        if current is None:
            raise PlannerAuthorizationError("Session agent state is missing")

        metadata_refs = metadata_refs_from_plan(plan)
        payload_custom = dict(current.payload.custom)
        payload_custom["planner_action"] = "PLAN_CREATED"
        payload_custom["detected_intent"] = plan.detected_intent.value
        payload_custom["action_step_count"] = str(len(plan.action_steps))
        payload_custom["missing_information_count"] = str(len(plan.missing_information))
        payload_custom.update(plan_custom_entries(plan))
        if plan.request_plan.requires_clarification:
            payload_custom["requires_clarification"] = "true"
        if plan.request_plan.unsupported:
            payload_custom["unsupported"] = "true"

        return self._state_service.update_agent_state(
            snapshot.session_id,
            MergePayloadTransition(
                updates={
                    "intent": plan.detected_intent.value,
                    "plan_version": plan.plan_version,
                    "metadata_refs": metadata_refs,
                    "custom": payload_custom,
                }
            ),
            workspace_id=workspace_id,
            user_id=user_id,
            expected_version=expected_agent_version,
        )

    @staticmethod
    def _require_phase(snapshot: AnalysisSessionSnapshot) -> AgentPhase:
        if snapshot.agent_state is None:
            raise PlannerAuthorizationError("Session agent state is missing")
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
            raise PlannerAuthorizationError("Data source is not accessible")
