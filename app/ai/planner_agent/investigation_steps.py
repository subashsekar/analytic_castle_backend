"""Attach schema-grounded investigation steps to an analysis plan."""

from __future__ import annotations

from app.ai.analysis_profile import build_analysis_profile
from app.ai.intent_types import AIIntent
from app.ai.investigation.plan import build_investigation_plan
from app.ai.metadata_types import ResolvedMetadataContext
from app.ai.planner_agent.models import (
    AnalysisPlan,
    PlannerActionKind,
    PlannerActionStep,
    PlannerToolRef,
)

_KIND_ACTION = {
    "overall_change": PlannerActionKind.TREND,
    "breakdown": PlannerActionKind.COMPARE,
    "drill": PlannerActionKind.RANK,
    "year_over_year": PlannerActionKind.COMPARE,
    "funnel": PlannerActionKind.AGGREGATE,
    "cohort": PlannerActionKind.TREND,
}


def attach_investigation_plan(
    plan: AnalysisPlan,
    *,
    message: str,
    metadata: ResolvedMetadataContext,
    intent: AIIntent,
) -> AnalysisPlan:
    profile = build_analysis_profile(message, intent=intent, plan=plan.request_plan)
    steps = build_investigation_plan(message, metadata, intent=intent, profile=profile)
    if not steps:
        return plan
    base_order = max((item.step_order for item in plan.action_steps), default=0)
    extra = [
        PlannerActionStep(
            step_order=base_order + index,
            action=_KIND_ACTION.get(step.kind.value, PlannerActionKind.COMPARE),
            description=f"Investigation step {step.step_order}: {step.description}",
            tool=PlannerToolRef.POSTGRES_QUERY,
        )
        for index, step in enumerate(steps, start=1)
    ]
    return plan.model_copy(
        update={
            "investigation_steps": steps,
            "action_steps": sorted(
                [*plan.action_steps, *extra],
                key=lambda item: item.step_order,
            ),
        }
    )
