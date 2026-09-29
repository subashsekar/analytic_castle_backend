"""Planner workflow helpers for orchestrator integration."""

from __future__ import annotations

from uuid import UUID

from app.ai.intent_types import AIIntent
from app.ai.planner_agent.models import AnalysisPlan, PlanCreateParams, PlannerResult
from app.ai.planner_agent.service import PlannerAgent
from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.supervisor.errors import InvalidRoutingError
from app.ai.supervisor.models import SuperviseAdvanceParams, SupervisorAction
from app.ai.supervisor.service import SupervisorAgent
from app.ai.supervisor.workflow import advance_until_complete
from app.enums import AgentPhase, AnalysisSessionStatus

_MAX_ADVANCE_STEPS = 8


async def advance_with_planning(
    supervisor: SupervisorAgent,
    planner: PlannerAgent,
    *,
    session_id: UUID,
    workspace_id: UUID,
    user_id: UUID,
    expected_agent_version: int,
    intent: AIIntent,
    message: str,
    data_source_name: str | None = None,
) -> tuple[AnalysisSessionSnapshot, AnalysisPlan | None]:
    """Advance through INTENT→PLANNING and stop after the plan is created.

    Execution (SQL + Phase 8) is owned by the orchestrator after this returns.
    The session stays ACTIVE in PLANNING until the orchestrator advances it.
    """
    agent_version = expected_agent_version
    snapshot: AnalysisSessionSnapshot | None = None
    analysis_plan: AnalysisPlan | None = None

    for _ in range(_MAX_ADVANCE_STEPS):
        if snapshot is not None:
            if snapshot.status != AnalysisSessionStatus.ACTIVE.value:
                return snapshot, analysis_plan
            if snapshot.agent_state is not None and snapshot.agent_state.phase in {
                AgentPhase.COMPLETED,
                AgentPhase.FAILED,
            }:
                return snapshot, analysis_plan
            if (
                analysis_plan is not None
                and snapshot.agent_state is not None
                and snapshot.agent_state.phase is AgentPhase.PLANNING
            ):
                return snapshot, analysis_plan

        result = await supervisor.advance_workflow(
            SuperviseAdvanceParams(
                session_id=session_id,
                workspace_id=workspace_id,
                user_id=user_id,
                expected_agent_version=agent_version,
            )
        )
        snapshot = result.session
        if snapshot.agent_state is not None:
            agent_version = snapshot.agent_state.version

        if result.decision.action is SupervisorAction.RUN_PLANNER:
            planner_result: PlannerResult = await planner.create_plan(
                PlanCreateParams(
                    session_id=session_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    expected_agent_version=agent_version,
                    intent=intent,
                    message=message,
                    data_source_name=data_source_name,
                )
            )
            snapshot = planner_result.session
            analysis_plan = planner_result.plan
            if snapshot.agent_state is not None:
                agent_version = snapshot.agent_state.version
            # Planning-only stop: do not advance into EXECUTING.
            return snapshot, analysis_plan

        if snapshot.status != AnalysisSessionStatus.ACTIVE.value:
            return snapshot, analysis_plan
        if result.decision.action in {
            SupervisorAction.COMPLETE,
            SupervisorAction.FAIL,
            SupervisorAction.RESPOND_UNSUPPORTED,
        }:
            return snapshot, analysis_plan
        if snapshot.agent_state is not None and snapshot.agent_state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
            AgentPhase.PLANNING,
        }:
            return snapshot, analysis_plan

    raise InvalidRoutingError("Planner workflow did not reach a planning stop state")


__all__ = ["advance_until_complete", "advance_with_planning"]
