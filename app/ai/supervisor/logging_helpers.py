"""Safe observability helpers that never emit user message content."""

from __future__ import annotations

from app.ai.supervisor.models import (
    RequestClassification,
    SupervisorAction,
    SupervisorDecision,
)
from app.enums import AgentPhase


def classification_log_context(
    classification: RequestClassification,
) -> dict[str, object]:
    return {
        "request_category": classification.category.value,
        "classification_confidence": classification.confidence.value,
        "classification_source": classification.source,
        "requires_data_access": classification.requires_data_access,
        "requires_clarification": classification.requires_clarification,
        "has_reason": classification.reason is not None,
    }


def routing_log_context(
    *,
    action: SupervisorAction,
    current_phase: AgentPhase,
    next_phase: AgentPhase | None = None,
) -> dict[str, object]:
    context: dict[str, object] = {
        "supervisor_action": action.value,
        "agent_phase": current_phase.value,
    }
    if next_phase is not None:
        context["next_agent_phase"] = next_phase.value
    return context


def decision_log_context(decision: SupervisorDecision) -> dict[str, object]:
    return {
        **classification_log_context(decision.classification),
        **routing_log_context(
            action=decision.action,
            current_phase=decision.current_phase,
            next_phase=decision.next_phase,
        ),
    }
