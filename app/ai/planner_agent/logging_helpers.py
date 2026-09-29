"""Safe observability helpers that never emit user message content."""

from __future__ import annotations

from app.ai.intent_types import AIIntentType
from app.ai.planner_agent.models import AnalysisPlan


def plan_log_context(plan: AnalysisPlan) -> dict[str, object]:
    return {
        "detected_intent": plan.detected_intent.value,
        "plan_version": plan.plan_version,
        "requires_clarification": plan.request_plan.requires_clarification,
        "unsupported": plan.request_plan.unsupported,
        "operation_count": len(plan.request_plan.operations),
        "capability_count": len(plan.request_plan.required_capabilities),
        "required_data_count": len(plan.required_data),
        "action_step_count": len(plan.action_steps),
        "missing_information_count": len(plan.missing_information),
    }


def intent_log_context(intent: AIIntentType) -> dict[str, object]:
    return {"intent": intent.value}
