"""Supervisor workflow helpers for orchestrator integration."""

from __future__ import annotations

from uuid import UUID

from app.ai.state.models import AnalysisSessionSnapshot
from app.ai.supervisor.errors import InvalidRoutingError
from app.ai.supervisor.models import SuperviseAdvanceParams, SupervisorAction
from app.ai.supervisor.service import SupervisorAgent
from app.enums import AgentPhase, AnalysisSessionStatus

_MAX_ADVANCE_STEPS = 8


async def advance_until_complete(
    supervisor: SupervisorAgent,
    *,
    session_id: UUID,
    workspace_id: UUID,
    user_id: UUID,
    expected_agent_version: int,
) -> AnalysisSessionSnapshot:
    """Advance the supervisor workflow until the session is terminal."""
    agent_version = expected_agent_version
    snapshot: AnalysisSessionSnapshot | None = None

    for _ in range(_MAX_ADVANCE_STEPS):
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

        if snapshot.status != AnalysisSessionStatus.ACTIVE.value:
            return snapshot
        if result.decision.action in {
            SupervisorAction.COMPLETE,
            SupervisorAction.FAIL,
            SupervisorAction.RESPOND_UNSUPPORTED,
        }:
            return snapshot
        if snapshot.agent_state is not None and snapshot.agent_state.phase in {
            AgentPhase.COMPLETED,
            AgentPhase.FAILED,
        }:
            return snapshot

    raise InvalidRoutingError("Supervisor workflow did not reach a terminal state")
