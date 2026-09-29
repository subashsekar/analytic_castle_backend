"""Deterministic supervisor routing rules."""

from __future__ import annotations

from app.ai.supervisor.errors import InvalidRoutingError
from app.ai.supervisor.models import (
    ClassificationConfidence,
    RequestCategory,
    RequestClassification,
    SupervisorAction,
)
from app.enums import AgentPhase

_NO_DATA_SOURCE_MESSAGE = (
    "This request requires a connected data source, but none is configured."
)

# INTENT → PLANNING → EXECUTING → RESPONDING → COMPLETED
_PHASE_PROGRESSION: dict[AgentPhase, tuple[SupervisorAction, AgentPhase | None]] = {
    AgentPhase.INTENT: (SupervisorAction.RUN_PLANNER, AgentPhase.PLANNING),
    AgentPhase.PLANNING: (SupervisorAction.ADVANCE_WORKFLOW, AgentPhase.EXECUTING),
    AgentPhase.EXECUTING: (SupervisorAction.ADVANCE_WORKFLOW, AgentPhase.RESPONDING),
    AgentPhase.RESPONDING: (SupervisorAction.COMPLETE, AgentPhase.COMPLETED),
}

_INITIAL_ROUTING: dict[RequestCategory, tuple[SupervisorAction, AgentPhase | None]] = {
    RequestCategory.ANALYTICAL_QUERY: (
        SupervisorAction.RUN_INTENT_AGENT,
        AgentPhase.INTENT,
    ),
    RequestCategory.SCHEMA_QUESTION: (
        SupervisorAction.RUN_INTENT_AGENT,
        AgentPhase.INTENT,
    ),
    RequestCategory.GENERAL: (SupervisorAction.RUN_INTENT_AGENT, AgentPhase.INTENT),
    RequestCategory.UNSUPPORTED: (SupervisorAction.RESPOND_UNSUPPORTED, None),
    RequestCategory.AMBIGUOUS: (
        SupervisorAction.REQUEST_CLARIFICATION,
        AgentPhase.AWAITING_CLARIFICATION,
    ),
    RequestCategory.UNKNOWN: (
        SupervisorAction.REQUEST_CLARIFICATION,
        AgentPhase.AWAITING_CLARIFICATION,
    ),
}


def resolve_routing(
    *,
    phase: AgentPhase,
    classification: RequestClassification,
    has_data_source: bool,
) -> tuple[SupervisorAction, AgentPhase | None, str | None]:
    if phase is AgentPhase.INITIAL:
        return _route_initial(classification, has_data_source=has_data_source)
    if phase in _PHASE_PROGRESSION:
        return _route_phase_progression(phase)
    if phase is AgentPhase.AWAITING_CLARIFICATION:
        raise InvalidRoutingError(
            "Awaiting clarification; send a new user message to continue"
        )
    if phase in {AgentPhase.COMPLETED, AgentPhase.FAILED}:
        raise InvalidRoutingError(f"Cannot route from terminal phase {phase.value}")
    raise InvalidRoutingError(f"No routing rule for phase {phase.value}")


def resolve_advance_routing(
    phase: AgentPhase,
) -> tuple[SupervisorAction, AgentPhase | None, str | None]:
    if phase not in _PHASE_PROGRESSION:
        raise InvalidRoutingError(f"Phase {phase.value} cannot be advanced")
    action, next_phase = _PHASE_PROGRESSION[phase]
    return action, next_phase, None


def _route_initial(
    classification: RequestClassification,
    *,
    has_data_source: bool,
) -> tuple[SupervisorAction, AgentPhase | None, str | None]:
    if classification.requires_data_access and not has_data_source:
        return (
            SupervisorAction.RESPOND_UNSUPPORTED,
            None,
            _NO_DATA_SOURCE_MESSAGE,
        )

    if classification.requires_clarification:
        return (
            SupervisorAction.REQUEST_CLARIFICATION,
            AgentPhase.AWAITING_CLARIFICATION,
            classification.clarification_question,
        )

    if (
        classification.category is RequestCategory.UNKNOWN
        and classification.confidence is ClassificationConfidence.LOW
    ):
        return (
            SupervisorAction.REQUEST_CLARIFICATION,
            AgentPhase.AWAITING_CLARIFICATION,
            classification.clarification_question
            or "Could you clarify what you would like to analyze?",
        )

    try:
        action, next_phase = _INITIAL_ROUTING[classification.category]
    except KeyError as exc:
        raise InvalidRoutingError(
            f"Unknown request category {classification.category.value}"
        ) from exc

    message = (
        classification.reason
        if action is SupervisorAction.RESPOND_UNSUPPORTED
        else None
    )
    return action, next_phase, message


def _route_phase_progression(
    phase: AgentPhase,
) -> tuple[SupervisorAction, AgentPhase | None, str | None]:
    action, next_phase = _PHASE_PROGRESSION[phase]
    return action, next_phase, None
