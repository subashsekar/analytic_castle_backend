"""Intent refinement helpers for planner output."""

from __future__ import annotations

from app.ai.intent_types import AIIntent, AIIntentType
from app.ai.planner_agent.models import AnalysisPlan


def apply_planner_intent(base_intent: AIIntent, plan: AnalysisPlan) -> AIIntent:
    """Apply planner clarification/unsupported flags without rewriting intent.

    The classified intent from the intent agent remains authoritative. The planner
    may only mark unsupported or clarification outcomes on the request plan.
    """
    updates: dict[str, object] = {}
    if plan.request_plan.unsupported:
        updates["intent"] = AIIntentType.UNSUPPORTED
        updates["requires_clarification"] = False
        updates["clarification_question"] = None
    elif plan.request_plan.requires_clarification:
        updates["requires_clarification"] = True
        if plan.request_plan.clarification_question:
            updates["clarification_question"] = plan.request_plan.clarification_question
    if not updates:
        return base_intent
    return base_intent.model_copy(update=updates)
