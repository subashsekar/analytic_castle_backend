"""Serialize analysis plans into agent-state custom payload entries."""

from __future__ import annotations

import json

from app.ai.planner_agent.models import AnalysisPlan, PlannerActionStep, RequiredDataRef


def plan_custom_entries(plan: AnalysisPlan) -> dict[str, str]:
    """Compact JSON snapshots for required data, actions, and missing info."""
    required = [
        {"kind": item.kind.value, "name": item.name} for item in plan.required_data
    ]
    steps = [
        {
            "step_order": step.step_order,
            "action": step.action.value,
            "description": step.description,
            "tool": step.tool.value if step.tool is not None else None,
        }
        for step in plan.action_steps
    ]
    return {
        "plan_required_data": json.dumps(required, separators=(",", ":")),
        "plan_action_steps": json.dumps(steps, separators=(",", ":")),
        "plan_missing_information": json.dumps(
            plan.missing_information, separators=(",", ":")
        ),
    }


def metadata_refs_from_plan(plan: AnalysisPlan) -> list[str]:
    """Stable refs that preserve required-data kind and name."""
    refs = [f"{item.kind.value}:{item.name}" for item in plan.required_data]
    return refs[:20]


def parse_required_data(raw: str) -> list[RequiredDataRef]:
    from app.ai.planner_agent.models import RequiredDataKind

    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise TypeError("plan_required_data must be a JSON array")
    refs: list[RequiredDataRef] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        refs.append(
            RequiredDataRef(
                name=str(item["name"]),
                kind=RequiredDataKind(str(item["kind"])),
            )
        )
    return refs


def parse_action_steps(raw: str) -> list[PlannerActionStep]:
    from app.ai.planner_agent.models import PlannerActionKind, PlannerToolRef

    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise TypeError("plan_action_steps must be a JSON array")
    steps: list[PlannerActionStep] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        tool_value = item.get("tool")
        steps.append(
            PlannerActionStep(
                step_order=int(item["step_order"]),
                action=PlannerActionKind(str(item["action"])),
                description=str(item["description"]),
                tool=PlannerToolRef(str(tool_value)) if tool_value else None,
            )
        )
    return steps
