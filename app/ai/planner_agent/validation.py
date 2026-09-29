"""Validation helpers for planner LLM output and analysis plan normalization."""

from __future__ import annotations

import json

from pydantic import ValidationError

from app.ai.intent_types import (
    AIIntent,
    AIIntentType,
    AIPlanCapability,
    AIPlanOperation,
    AIRequestPlan,
)
from app.ai.llm.errors import LLMResponseValidationError
from app.ai.planner import AIRequestPlanner
from app.ai.planner_agent.errors import PlannerValidationError
from app.ai.planner_agent.models import (
    AnalysisPlan,
    LLMPlannerOutput,
    PlannerActionStep,
    RequiredDataRef,
)
from app.ai.planner_agent.prompts import PLANNER_BUNDLE_VERSION


def parse_llm_plan(raw_content: str) -> LLMPlannerOutput:
    try:
        from app.ai.llm.content import extract_json_object

        payload = extract_json_object(raw_content)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LLMResponseValidationError(
            "Planner response is not valid JSON",
            raw_content=raw_content,
        ) from exc
    try:
        return LLMPlannerOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            "Planner response failed schema validation",
            raw_content=raw_content,
        ) from exc


def normalize_plan(
    *,
    llm_result: LLMPlannerOutput,
    intent: AIIntent,
) -> AnalysisPlan:
    if llm_result.unsupported:
        if not llm_result.unsupported_reason:
            raise PlannerValidationError("Unsupported planner output requires a reason")
        request_plan = AIRequestPlan(
            intent=AIIntentType.UNSUPPORTED,
            unsupported=True,
        )
        return AnalysisPlan(
            request_plan=request_plan,
            detected_intent=AIIntentType.UNSUPPORTED,
            plan_version=PLANNER_BUNDLE_VERSION.value,
        )

    if llm_result.requires_clarification:
        question = (
            llm_result.clarification_question
            or intent.clarification_question
            or _default_clarification_question(llm_result.missing_information)
        )
        if not question:
            raise PlannerValidationError(
                "Clarification required but no question provided"
            )
        request_plan = AIRequestPlan(
            intent=llm_result.intent,
            requires_clarification=True,
            clarification_question=question,
            requires_metadata=intent.requires_metadata,
        )
        return AnalysisPlan(
            request_plan=request_plan,
            detected_intent=llm_result.intent,
            missing_information=list(llm_result.missing_information),
            plan_version=PLANNER_BUNDLE_VERSION.value,
        )

    if llm_result.missing_information and not llm_result.requires_clarification:
        request_plan = AIRequestPlan(
            intent=llm_result.intent,
            requires_clarification=True,
            clarification_question=_default_clarification_question(
                llm_result.missing_information
            ),
            requires_metadata=intent.requires_metadata,
        )
        return AnalysisPlan(
            request_plan=request_plan,
            detected_intent=llm_result.intent,
            missing_information=list(llm_result.missing_information),
            plan_version=PLANNER_BUNDLE_VERSION.value,
        )

    deterministic = AIRequestPlanner().plan(intent)
    request_plan = _merge_request_plans(deterministic, llm_result, intent=intent)
    required_data = [
        RequiredDataRef(name=item.name, kind=item.kind)
        for item in llm_result.required_data
    ]
    action_steps = [
        PlannerActionStep(
            step_order=item.step_order,
            action=item.action,
            description=item.description,
            tool=item.tool,
        )
        for item in sorted(llm_result.action_steps, key=lambda step: step.step_order)
    ]
    return AnalysisPlan(
        request_plan=request_plan,
        detected_intent=intent.intent,
        required_data=required_data,
        action_steps=action_steps,
        missing_information=list(llm_result.missing_information),
        plan_version=PLANNER_BUNDLE_VERSION.value,
    )


def plan_from_intent(intent: AIIntent) -> AnalysisPlan:
    """Deterministic plan without LLM for unsupported or clarification intents."""
    request_plan = AIRequestPlanner().plan(intent)
    return AnalysisPlan(
        request_plan=request_plan,
        detected_intent=intent.intent,
        plan_version=PLANNER_BUNDLE_VERSION.value,
    )


def _merge_request_plans(
    deterministic: AIRequestPlan,
    llm_result: LLMPlannerOutput,
    *,
    intent: AIIntent,
) -> AIRequestPlan:
    operations = _merge_operations(deterministic.operations, llm_result.operations)
    capabilities = _merge_capabilities(
        deterministic.required_capabilities,
        llm_result.required_capabilities,
    )
    return AIRequestPlan(
        intent=intent.intent,
        requires_clarification=False,
        clarification_question=None,
        operations=operations,
        required_capabilities=capabilities,
        requires_metadata=deterministic.requires_metadata,
        requires_database=deterministic.requires_database,
        requires_sample_data=deterministic.requires_sample_data,
        requires_aggregation=deterministic.requires_aggregation,
        requires_time_filter=deterministic.requires_time_filter,
        requires_relationships=deterministic.requires_relationships,
        unsupported=False,
    )


def _merge_operations(
    deterministic: list[AIPlanOperation],
    llm_operations: list[AIPlanOperation],
) -> list[AIPlanOperation]:
    merged: list[AIPlanOperation] = []
    seen: set[AIPlanOperation] = set()
    for item in (*deterministic, *llm_operations):
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged or list(deterministic)


def _merge_capabilities(
    deterministic: list[AIPlanCapability],
    llm_capabilities: list[AIPlanCapability],
) -> list[AIPlanCapability]:
    merged: list[AIPlanCapability] = []
    seen: set[AIPlanCapability] = set()
    for item in (*deterministic, *llm_capabilities):
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged or list(deterministic)


def _default_clarification_question(missing_information: list[str]) -> str | None:
    if not missing_information:
        return None
    if len(missing_information) == 1:
        return f"Please clarify: {missing_information[0]}"
    items = ", ".join(missing_information[:3])
    return f"I need more detail about: {items}."
