"""Typed state transition application.

Phase transitions follow a forward-only graph for the analysis lifecycle, with
explicit resumable edges for multi-turn and clarification:

INITIAL -> INTENT -> PLANNING
INITIAL -> AWAITING_CLARIFICATION
AWAITING_CLARIFICATION -> INITIAL
PLANNING -> INITIAL (new user turn) | EXECUTING
EXECUTING -> RESPONDING | PLANNING (analysis failed; stay multi-turn ready)
RESPONDING -> COMPLETED | PLANNING (answer delivered; stay multi-turn ready)

COMPLETED and FAILED are terminal phases. Session completion and failure may set
those phases directly without going through ``apply_state_transition``.
"""

from __future__ import annotations

from app.ai.state.errors import InvalidStateTransitionError
from app.ai.state.models import (
    AgentStatePayload,
    AgentStateSnapshot,
    MergePayloadTransition,
    ReplacePayloadTransition,
    SetPhaseTransition,
    StateTransition,
)
from app.ai.state.serialization import deserialize_payload, serialize_payload
from app.enums import AgentPhase

TERMINAL_PHASES = frozenset({AgentPhase.COMPLETED, AgentPhase.FAILED})

# Phases that may accept a new user message after resetting to INITIAL.
RESUMABLE_MESSAGE_PHASES = frozenset(
    {
        AgentPhase.INITIAL,
        AgentPhase.AWAITING_CLARIFICATION,
        AgentPhase.PLANNING,
    }
)

ALLOWED_PHASE_TRANSITIONS: dict[AgentPhase, frozenset[AgentPhase]] = {
    AgentPhase.INITIAL: frozenset(
        {AgentPhase.INTENT, AgentPhase.AWAITING_CLARIFICATION}
    ),
    AgentPhase.INTENT: frozenset({AgentPhase.PLANNING}),
    AgentPhase.PLANNING: frozenset({AgentPhase.INITIAL, AgentPhase.EXECUTING}),
    AgentPhase.AWAITING_CLARIFICATION: frozenset({AgentPhase.INITIAL}),
    AgentPhase.EXECUTING: frozenset({AgentPhase.RESPONDING, AgentPhase.PLANNING}),
    AgentPhase.RESPONDING: frozenset({AgentPhase.COMPLETED, AgentPhase.PLANNING}),
    AgentPhase.COMPLETED: frozenset(),
    AgentPhase.FAILED: frozenset(),
}


def apply_state_transition(
    current: AgentStateSnapshot,
    transition: StateTransition,
) -> AgentStateSnapshot:
    if isinstance(transition, SetPhaseTransition):
        return _apply_set_phase(current, transition)
    if isinstance(transition, MergePayloadTransition):
        return _apply_merge_payload(current, transition)
    if isinstance(transition, ReplacePayloadTransition):
        return _apply_replace_payload(current, transition)
    raise InvalidStateTransitionError("Unsupported state transition")


def _apply_set_phase(
    current: AgentStateSnapshot,
    transition: SetPhaseTransition,
) -> AgentStateSnapshot:
    _ensure_not_terminal(current.phase)
    allowed_targets = ALLOWED_PHASE_TRANSITIONS.get(current.phase, frozenset())
    if transition.phase not in allowed_targets:
        raise InvalidStateTransitionError(
            f"Cannot transition from {current.phase.value} to {transition.phase.value}"
        )
    return AgentStateSnapshot(
        phase=transition.phase,
        payload=current.payload,
        version=current.version + 1,
    )


def _apply_merge_payload(
    current: AgentStateSnapshot,
    transition: MergePayloadTransition,
) -> AgentStateSnapshot:
    _ensure_not_terminal(current.phase)
    merged = current.payload.model_dump()
    merged.update(transition.updates)
    return AgentStateSnapshot(
        phase=current.phase,
        payload=deserialize_payload(merged),
        version=current.version + 1,
    )


def _apply_replace_payload(
    current: AgentStateSnapshot,
    transition: ReplacePayloadTransition,
) -> AgentStateSnapshot:
    _ensure_not_terminal(current.phase)
    return AgentStateSnapshot(
        phase=current.phase,
        payload=transition.payload,
        version=current.version + 1,
    )


def _ensure_not_terminal(phase: AgentPhase) -> None:
    if phase in TERMINAL_PHASES:
        raise InvalidStateTransitionError(
            f"Cannot transition from terminal phase {phase.value}"
        )


def initial_agent_state_snapshot() -> AgentStateSnapshot:
    return AgentStateSnapshot(
        phase=AgentPhase.INITIAL,
        payload=AgentStatePayload(),
        version=1,
    )


def snapshot_from_row(
    *, phase: AgentPhase, payload: dict, version: int
) -> AgentStateSnapshot:
    return AgentStateSnapshot(
        phase=phase,
        payload=deserialize_payload(payload),
        version=version,
    )


def payload_to_storage(payload: AgentStatePayload) -> dict:
    return serialize_payload(payload)
