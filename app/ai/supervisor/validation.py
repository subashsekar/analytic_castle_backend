"""Validation helpers for supervisor LLM output and routing decisions."""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.ai.llm.errors import LLMResponseValidationError
from app.ai.supervisor.errors import (
    InvalidRoutingError,
    SupervisorClassificationError,
)
from app.ai.supervisor.models import (
    ClassificationConfidence,
    LLMRequestClassification,
    RequestCategory,
    RequestClassification,
    SupervisorAction,
    SupervisorDecision,
)
from app.ai.supervisor.routing import resolve_routing
from app.enums import AgentPhase


def parse_llm_classification(raw_content: str) -> LLMRequestClassification:
    try:
        from app.ai.llm.content import extract_json_object

        payload = extract_json_object(raw_content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMResponseValidationError(
            "Classification response is not valid JSON",
            raw_content=raw_content,
        ) from exc
    try:
        return LLMRequestClassification.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            "Classification response failed schema validation",
            raw_content=raw_content,
        ) from exc


def normalize_classification(
    llm_result: LLMRequestClassification,
) -> RequestClassification:
    category = llm_result.category
    confidence = llm_result.confidence

    if (
        category is RequestCategory.UNKNOWN
        and confidence is ClassificationConfidence.LOW
    ):
        category = RequestCategory.AMBIGUOUS

    if llm_result.requires_clarification and category not in {
        RequestCategory.AMBIGUOUS,
        RequestCategory.UNKNOWN,
    }:
        category = RequestCategory.AMBIGUOUS

    if category is RequestCategory.UNSUPPORTED and not llm_result.reason:
        raise SupervisorClassificationError(
            "Unsupported classification requires a reason"
        )

    if llm_result.requires_clarification and not llm_result.clarification_question:
        raise SupervisorClassificationError(
            "Clarification required but no question provided"
        )

    return RequestClassification(
        category=category,
        confidence=confidence,
        requires_data_access=llm_result.requires_data_access,
        requires_clarification=llm_result.requires_clarification,
        clarification_question=llm_result.clarification_question,
        reason=llm_result.reason,
        source="llm",
    )


def classification_from_policy(reason: str) -> RequestClassification:
    return RequestClassification(
        category=RequestCategory.UNSUPPORTED,
        confidence=ClassificationConfidence.HIGH,
        requires_data_access=False,
        reason=reason,
        source="policy",
    )


def ensure_routing_allowed(
    *,
    action: SupervisorAction,
    current_phase: AgentPhase,
    next_phase: AgentPhase | None,
) -> None:
    if action is SupervisorAction.FAIL:
        return
    if action is SupervisorAction.RESPOND_UNSUPPORTED:
        return
    if action is SupervisorAction.COMPLETE and current_phase is AgentPhase.RESPONDING:
        return
    if action is SupervisorAction.REQUEST_CLARIFICATION:
        if next_phase is not AgentPhase.AWAITING_CLARIFICATION:
            raise InvalidRoutingError(
                "Clarification routing must enter AWAITING_CLARIFICATION"
            )
    elif next_phase is None:
        raise InvalidRoutingError(
            f"Action {action.value} requires a next phase from {current_phase.value}"
        )
    if next_phase is None:
        return
    from app.ai.state.transitions import ALLOWED_PHASE_TRANSITIONS

    allowed = ALLOWED_PHASE_TRANSITIONS.get(current_phase, frozenset())
    if next_phase not in allowed:
        raise InvalidRoutingError(
            f"Cannot route from {current_phase.value} to {next_phase.value}"
        )


def build_decision(
    *,
    classification: RequestClassification,
    current_phase: AgentPhase,
    has_data_source: bool,
) -> SupervisorDecision:
    if current_phase is not AgentPhase.INITIAL:
        raise InvalidRoutingError(
            f"Classification routing is only valid from INITIAL phase, "
            f"got {current_phase.value}"
        )
    action, next_phase, message = resolve_routing(
        phase=current_phase,
        classification=classification,
        has_data_source=has_data_source,
    )
    ensure_routing_allowed(
        action=action,
        current_phase=current_phase,
        next_phase=next_phase,
    )
    return SupervisorDecision(
        action=action,
        classification=classification,
        current_phase=current_phase,
        next_phase=next_phase,
        message=message,
    )
