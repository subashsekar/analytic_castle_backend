"""Build an ordered investigation plan for complex analytical questions."""

from __future__ import annotations

from datetime import date

from app.ai.analysis_profile import AnalysisProfile, build_analysis_profile
from app.ai.intent_types import AIIntent
from app.ai.investigation.models import InvestigationPlanStep, InvestigationStepKind
from app.ai.investigation.sql_build import build_investigation_sqls
from app.ai.metadata_types import ResolvedMetadataContext
from app.core.config import settings


def build_investigation_plan(
    message: str,
    metadata: ResolvedMetadataContext,
    *,
    intent: AIIntent | None = None,
    profile: AnalysisProfile | None = None,
    max_steps: int | None = None,
    today: date | None = None,
) -> list[InvestigationPlanStep]:
    profile = profile or build_analysis_profile(message, intent=intent)
    cap = max_steps if max_steps is not None else settings.AI_INVESTIGATION_MAX_QUERIES
    cap = max(0, min(cap, 5))
    if cap == 0:
        return []
    raw = build_investigation_sqls(
        message,
        metadata,
        intent=intent,
        profile=profile,
        max_steps=cap,
        today=today,
    )
    steps: list[InvestigationPlanStep] = []
    for order, (kind, description, sql) in enumerate(raw, start=1):
        steps.append(
            InvestigationPlanStep(
                step_order=order,
                kind=InvestigationStepKind(kind),
                description=description,
                question=description,
                sql=sql,
            )
        )
    return steps


def investigation_plan_summary(steps: list[InvestigationPlanStep]) -> str:
    if not steps:
        return ""
    lines = ["Investigation plan (execute in order):"]
    for step in steps:
        lines.append(f"{step.step_order}. [{step.kind.value}] {step.description}")
    return "\n".join(lines)
